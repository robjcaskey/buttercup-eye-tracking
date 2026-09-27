//! Offline native-RAW bright-reflection evidence. No learned detector or sign labels.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

// Local-contrast components connect across small gaps in a reflected screen
// border. They remain reflection hypotheses, not identified anatomical glints.
fn local_candidates(p: &[f64], w: usize, h: usize, threshold: f64, origin: [f64; 2]) -> Vec<Value> {
    if w < 3 || h < 3 {
        return Vec::new();
    }
    let mut background = vec![0.0; p.len()];
    let mut mask = vec![false; p.len()];
    for y in 0..h {
        for x in 0..w {
            let mut ring = Vec::new();
            for dy in -5isize..=5 {
                for dx in -5isize..=5 {
                    if dx.abs().max(dy.abs()) < 3 {
                        continue;
                    }
                    let (xx, yy) = (x as isize + dx, y as isize + dy);
                    if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                        ring.push(p[yy as usize * w + xx as usize]);
                    }
                }
            }
            ring.sort_by(f64::total_cmp);
            let index = y * w + x;
            background[index] = ring[ring.len() / 2];
            mask[index] = p[index] >= threshold && p[index] - background[index] >= 45.0;
        }
    }
    let mut seen = vec![false; p.len()];
    let mut result = Vec::new();
    for first in 0..p.len() {
        if seen[first] || !mask[first] {
            continue;
        }
        let mut queue = vec![first];
        seen[first] = true;
        let (mut sum, mut sx, mut sy) = (0.0, 0.0, 0.0);
        let (mut minx, mut miny, mut maxx, mut maxy) = (w, h, 0, 0);
        let mut cursor = 0;
        while cursor < queue.len() {
            let index = queue[cursor];
            cursor += 1;
            let (x, y) = (index % w, index / w);
            let weight = p[index] - background[index];
            sum += weight;
            sx += (x as f64 * 4.0 + 1.5) * weight;
            sy += (y as f64 * 4.0 + 1.5) * weight;
            minx = minx.min(x);
            miny = miny.min(y);
            maxx = maxx.max(x);
            maxy = maxy.max(y);
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    let (xx, yy) = (x as isize + dx, y as isize + dy);
                    if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                        continue;
                    }
                    let next = yy as usize * w + xx as usize;
                    if mask[next] && !seen[next] {
                        seen[next] = true;
                        queue.push(next);
                    }
                }
            }
        }
        if queue.len() < 3 {
            continue;
        }
        let border = minx == 0 || miny == 0 || maxx + 1 == w || maxy + 1 == h;
        let bw = (maxx + 1 - minx) * 4;
        let bh = (maxy + 1 - miny) * 4;
        if border || bw > w * 4 / 3 || bh > h * 4 / 3 {
            continue;
        }
        let contrast = sum / queue.len() as f64;
        let score = contrast * (queue.len() as f64).sqrt();
        result.push(json!({"center_roi_px":[sx/sum,sy/sum],"center_sensor_px":[sx/sum+origin[0],sy/sum+origin[1]],"bbox_roi_px":[minx*4,miny*4,(maxx+1)*4,(maxy+1)*4],"area_native_px":queue.len()*16,"local_contrast":contrast,"score":score,"touches_border":false}));
    }
    // Disconnected bright sides of the same rectangular screen reflection
    // can have overlapping boxes. Merge those before assigning track identity.
    let mut changed = true;
    while changed {
        changed = false;
        'outer: for i in 0..result.len() {
            for j in i + 1..result.len() {
                let bbox = |v: &Value| -> [f64; 4] {
                    std::array::from_fn(|k| v["bbox_roi_px"][k].as_f64().unwrap())
                };
                let a = bbox(&result[i]);
                let b = bbox(&result[j]);
                if a[0] > b[2] || b[0] > a[2] || a[1] > b[3] || b[1] > a[3] {
                    continue;
                }
                let bounds = [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ];
                if bounds[2] - bounds[0] > w as f64 * 4.0 / 3.0
                    || bounds[3] - bounds[1] > h as f64 * 4.0 / 3.0
                {
                    continue;
                }
                let aa = result[i]["area_native_px"].as_f64().unwrap();
                let ab = result[j]["area_native_px"].as_f64().unwrap();
                let wa = aa * result[i]["local_contrast"].as_f64().unwrap();
                let wb = ab * result[j]["local_contrast"].as_f64().unwrap();
                let center: [f64; 2] = std::array::from_fn(|k| {
                    (wa * result[i]["center_roi_px"][k].as_f64().unwrap()
                        + wb * result[j]["center_roi_px"][k].as_f64().unwrap())
                        / (wa + wb)
                });
                let contrast = (wa + wb) / (aa + ab);
                result[i] = json!({"center_roi_px":center,"center_sensor_px":[center[0]+origin[0],center[1]+origin[1]],"bbox_roi_px":bounds,"area_native_px":aa+ab,"local_contrast":contrast,"score":contrast*((aa+ab)/16.0).sqrt(),"touches_border":false});
                result.remove(j);
                changed = true;
                break 'outer;
            }
        }
    }
    result.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["score"].as_f64().unwrap())
    });
    result.truncate(8);
    result
}

struct Track {
    last_ns: u64,
    center: [f64; 2],
    velocity: [f64; 2],
    area: f64,
    hits: usize,
}
fn associate(candidates: &[Value], ts: u64, state: &mut Option<Track>) -> Value {
    if state
        .as_ref()
        .is_some_and(|s| ts <= s.last_ns || ts - s.last_ns > 300_000_000)
    {
        *state = None;
    }
    let mut ranked = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        let center = [
            c["center_sensor_px"][0].as_f64().unwrap(),
            c["center_sensor_px"][1].as_f64().unwrap(),
        ];
        let mut cost = -(c["score"].as_f64().unwrap() / 1000.0).ln();
        if let Some(s) = state {
            let dt = (ts - s.last_ns) as f64 / 1e9;
            let distance = ((center[0] - s.center[0] - s.velocity[0] * dt).powi(2)
                + (center[1] - s.center[1] - s.velocity[1] * dt).powi(2))
            .sqrt();
            if distance > 60.0 {
                continue;
            }
            cost +=
                distance / 15.0 + 0.5 * (c["area_native_px"].as_f64().unwrap() / s.area).ln().abs();
        }
        ranked.push((cost, i, center));
    }
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some(&(cost, index, center)) = ranked.first() else {
        return json!({"fresh":false,"reason":"no-compatible-reflection","center_sensor_px":null});
    };
    let margin = ranked.get(1).map(|next| next.0 - cost);
    if margin.is_some_and(|m| m < 0.35) {
        return json!({"fresh":false,"reason":"ambiguous-reflections","margin":margin,"center_sensor_px":null});
    }
    let (hits, velocity) = if let Some(s) = state {
        let dt = (ts - s.last_ns) as f64 / 1e9;
        (
            s.hits + 1,
            [
                (center[0] - s.center[0]) / dt,
                (center[1] - s.center[1]) / dt,
            ],
        )
    } else {
        (1, [0.0, 0.0])
    };
    *state = Some(Track {
        last_ns: ts,
        center,
        velocity,
        area: candidates[index]["area_native_px"].as_f64().unwrap(),
        hits,
    });
    json!({"fresh":hits>=3,"reason":if hits>=3 {"tracked-reflection-unverified"}else{"acquiring-reflection"},"candidate_index":index,"center_sensor_px":if hits>=3 {Some(center)}else{None},"margin":margin,"supporting_matches":hits})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: buttercup_glint_evidence SOURCE_JSONL OUTPUT_JSONL".into());
    }
    let native_dir = std::path::Path::new(&args[1]).is_dir();
    let source = if native_dir {
        std::path::Path::new(&args[1]).join("frames.jsonl")
    } else {
        args[1].clone().into()
    };
    let mut output = BufWriter::new(File::create(&args[2])?);
    let mut tracks: HashMap<(String, u64), Option<Track>> = HashMap::new();
    for (index, line) in BufReader::new(File::open(source)?).lines().enumerate() {
        let mut row: Value = serde_json::from_str(&line?)?;
        if native_dir {
            let path = std::path::Path::new(&args[1])
                .join(row["stream"].as_str().ok_or("missing native stream")?)
                .canonicalize()?;
            row = json!({"id":index,"input":{"raw_file":path,"raw_offset":row["offset"],"raw_length":row["length"],"clock_lineage":row["source_clock"]["source_key"]["stream_epoch"],"frame":row}});
        }
        let input = &row["input"];
        let frame = &input["frame"];
        let integer = |v: &Value| -> Result<u64, Box<dyn std::error::Error>> {
            v.as_u64()
                .ok_or_else(|| "missing unsigned source metadata".into())
        };
        let width = integer(&frame["width"])? as usize;
        let height = integer(&frame["height"])? as usize;
        let length = integer(&input["raw_length"])? as usize;
        if width == 0
            || height == 0
            || width % 4 != 0
            || height % 4 != 0
            || width
                .checked_mul(height)
                .and_then(|n| n.checked_mul(5))
                .map(|n| n / 4)
                != Some(length)
            || length > 128 * 1024 * 1024
        {
            return Err("invalid packed RAW10 dimensions/length".into());
        }
        let mut file = File::open(input["raw_file"].as_str().ok_or("missing RAW path")?)?;
        file.seek(SeekFrom::Start(integer(&input["raw_offset"])?))?;
        let mut packed = vec![0u8; length];
        file.read_exact(&mut packed)?;
        let digest = format!("{:x}", Sha256::digest(&packed));
        let w = width / 4;
        let h = height / 4;
        let mut pixels = vec![0.0f64; w * h];
        let mut clipped = 0usize;
        for (i, group) in packed.chunks_exact(5).enumerate() {
            let word = group
                .iter()
                .enumerate()
                .fold(0u64, |v, (j, b)| v | ((*b as u64) << (8 * j)));
            for lane in 0..4 {
                let sample = ((word >> (10 * lane)) & 1023) as usize;
                clipped += usize::from(sample >= 1020);
                let position = i * 4 + lane;
                pixels[(position / width / 4) * w + (position % width / 4)] += sample as f64 / 16.0;
            }
        }
        let mut sorted = pixels.clone();
        sorted.sort_by(f64::total_cmp);
        let quantile = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
        let median = quantile(0.5);
        let threshold = quantile(0.99).max(median + 40.0);
        let mut visited = vec![false; pixels.len()];
        let mut candidates = Vec::new();
        for first in 0..pixels.len() {
            if visited[first] || pixels[first] < threshold {
                continue;
            }
            let mut queue = vec![first];
            visited[first] = true;
            let (mut total, mut sx, mut sy) = (0.0, 0.0, 0.0);
            let (mut minx, mut miny, mut maxx, mut maxy) = (w, h, 0, 0);
            let mut cursor = 0;
            while cursor < queue.len() {
                let p = queue[cursor];
                cursor += 1;
                let (x, y) = (p % w, p / w);
                let weight = pixels[p] - median;
                total += weight;
                sx += (x as f64 * 4.0 + 1.5) * weight;
                sy += (y as f64 * 4.0 + 1.5) * weight;
                minx = minx.min(x);
                miny = miny.min(y);
                maxx = maxx.max(x);
                maxy = maxy.max(y);
                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let n = ny as usize * w + nx as usize;
                    if !visited[n] && pixels[n] >= threshold {
                        visited[n] = true;
                        queue.push(n);
                    }
                }
            }
            if queue.len() < 2 || total <= 0.0 {
                continue;
            }
            candidates.push(json!({"center_roi_px":[sx/total,sy/total],"center_sensor_px":[sx/total+integer(&frame["sensor_x"])? as f64,sy/total+integer(&frame["sensor_y"])? as f64],"area_native_px":queue.len()*16,"excess_brightness":total,"bbox_roi_px":[minx*4,miny*4,(maxx+1)*4,(maxy+1)*4],"touches_border":minx==0||miny==0||maxx+1==w||maxy+1==h}));
        }
        candidates.sort_by(|a, b| {
            b["excess_brightness"]
                .as_f64()
                .unwrap()
                .total_cmp(&a["excess_brightness"].as_f64().unwrap())
        });
        candidates.truncate(8);
        let local = local_candidates(
            &pixels,
            w,
            h,
            quantile(0.97).max(median + 30.0),
            [
                integer(&frame["sensor_x"])? as f64,
                integer(&frame["sensor_y"])? as f64,
            ],
        );
        let key = (
            input["clock_lineage"]
                .as_str()
                .ok_or("missing source clock")?
                .to_owned(),
            integer(&frame["eye_id"])?,
        );
        let track = associate(
            &local,
            integer(&frame["timestamp_ns"])?,
            tracks.entry(key).or_insert(None),
        );
        writeln!(
            output,
            "{}",
            json!({"schema":"buttercup-raw-reflection-evidence-v1","id":row["id"],"frame":frame,"clock":input["clock_lineage"],"raw_sha256":digest,"raw_source":input,"selection":if native_dir {"all-native-capture-frames"}else{"external-evaluation-requests"},"median_raw":median,"threshold_raw":threshold,"clipped_fraction":clipped as f64/(width*height) as f64,"candidates":candidates,"local_candidates":local,"track":track,"sign":null,"status":"bright-components-not-identified-cornea-glints"})
        )?;
    }
    output.flush()?;
    Ok(())
}
