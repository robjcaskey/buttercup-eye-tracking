//! Bounded offline RAW-only discovery. Does not load predictions or checkpoints.
#[path = "../glasses_parallax.rs"]
mod glasses_parallax;
#[path = "../glasses_parallax_inventory.rs"]
mod glasses_parallax_inventory;
use glasses_parallax::{analyze, decode, Image};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Command,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
struct Bundle {
    path: PathBuf,
    entries: BTreeMap<String, (u64, u64)>,
}
impl Bundle {
    fn open(path: &Path) -> Result<Self> {
        let mut entries = BTreeMap::new();
        if !path.is_dir() {
            let mut f = File::open(path)?;
            let total = f.metadata()?.len();
            let mut pos = 0;
            while pos + 512 <= total {
                f.seek(SeekFrom::Start(pos))?;
                let mut h = [0u8; 512];
                f.read_exact(&mut h)?;
                if h.iter().all(|&x| x == 0) {
                    break;
                }
                let txt = |b: &[u8]| {
                    String::from_utf8_lossy(b)
                        .trim_matches(|c: char| c == '\0' || c.is_ascii_whitespace())
                        .to_string()
                };
                let name = txt(&h[..100]);
                let prefix = txt(&h[345..500]);
                let name = if prefix.is_empty() {
                    name
                } else {
                    format!("{prefix}/{name}")
                };
                let size = u64::from_str_radix(&txt(&h[124..136]), 8)?;
                if pos + 512 + size > total {
                    return Err("tar truncated".into());
                }
                if h[156] == 0 || h[156] == b'0' {
                    if entries.insert(name, (pos + 512, size)).is_some() {
                        return Err("duplicate tar member".into());
                    }
                }
                pos += 512 + size.div_ceil(512) * 512;
            }
        }
        Ok(Self {
            path: path.to_owned(),
            entries,
        })
    }
    fn read(&self, name: &str, offset: u64, length: Option<usize>) -> Result<Vec<u8>> {
        if name != "frames.jsonl"
            && name != "subject-left.raw10"
            && name != "subject-right.raw10"
            && name != "thumbnails.jsonl"
            && name != "thumbnails.oic1"
        {
            return Err("non-RAW input forbidden".into());
        }
        let (path, start, size) = if self.path.is_dir() {
            let p = self.path.join(name);
            let size = fs::metadata(&p)?.len();
            (p, 0, size)
        } else {
            let &(start, size) = self.entries.get(name).ok_or("missing stream")?;
            (self.path.clone(), start, size)
        };
        let n = length.unwrap_or(size as usize);
        if offset.checked_add(n as u64).is_none_or(|end| end > size) || n > 64 * 1024 * 1024 {
            return Err("invalid/unbounded member range".into());
        }
        let mut f = File::open(path)?;
        f.seek(SeekFrom::Start(start + offset))?;
        let mut bytes = vec![0; n];
        f.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn u(v: &Value, k: &str) -> Result<u64> {
    v[k].as_u64()
        .ok_or_else(|| format!("missing numeric {k}").into())
}
fn epoch_key(v: &Value) -> Value {
    json!([
        v["source_clock"]["source_key"]["stream_epoch"],
        v["region"]["session"],
        v["source_clock"]["source_key"]["viewer_session_id"]
    ])
}
fn ppm(path: &Path, w: usize, h: usize, rgb: &[u8]) -> Result<()> {
    let mut f = File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    Ok(())
}
fn display(image: &Image) -> Vec<u8> {
    let mut a = image.pixels.clone();
    a.sort_by(f64::total_cmp);
    let lo = a[a.len() / 200];
    let hi = a[a.len() * 199 / 200];
    image
        .pixels
        .iter()
        .map(|v| ((v - lo) * 255. / (hi - lo).max(8.)).clamp(0., 255.) as u8)
        .collect()
}
fn render(out: &Path, images: &[Image], report: &Value) -> Result<Value> {
    let w = images[0].w;
    let h = images[0].h;
    let indices = [
        0,
        images.len() / 4,
        images.len() / 2,
        images.len() * 3 / 4,
        images.len() - 1,
    ];
    let mut sheet = vec![0; w * 5 * h * 3];
    let frames: Vec<_> = images.iter().map(display).collect();
    for (panel, &t) in indices.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let p = (y * w * 5 + panel * w + x) * 3;
                sheet[p..p + 3].fill(frames[t][y * w + x]);
            }
        }
        for pair in report["pairs"].as_array().unwrap().iter().take(8) {
            for (key, col) in [
                ("a_quad_cell", [255, 215, 0]),
                ("b_quad_cell", [255, 30, 200]),
            ] {
                let p = &pair[key][t];
                let x = p[0].as_f64().unwrap() as isize;
                let y = p[1].as_f64().unwrap() as isize;
                for dy in -3isize..=3 {
                    for dx in -3isize..=3 {
                        if dx.abs() != 3 && dy.abs() != 3 {
                            continue;
                        }
                        let xx = x + dx;
                        let yy = y + dy;
                        if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                            let k = (yy as usize * w * 5 + panel * w + xx as usize) * 3;
                            sheet[k..k + 3].copy_from_slice(&col);
                        }
                    }
                }
            }
        }
    }
    let path = out.join("contact.ppm");
    ppm(&path, w * 5, h, &sheet)?;
    let png = out.join("contact.png");
    let converted = Command::new("ffmpeg")
        .args(["-v", "error", "-threads", "1", "-i"])
        .arg(&path)
        .args(["-frames:v", "1", "-threads", "1"])
        .arg(&png)
        .status()
        .is_ok_and(|s| s.success());
    let data =
        json!({"w":w,"h":h,"frames":frames,"pairs":report["pairs"],"sources":report["sources"]});
    let html = format!(
        r#"<!doctype html><meta charset="utf-8"><title>RAW anonymous copy motion</title><style>body{{background:#191919;color:#eee;font:16px monospace}}canvas{{image-rendering:pixelated;width:840px}}pre{{white-space:pre-wrap}}</style><p>Native RAW temporal order; yellow/magenta copy A/B are anonymous. This is NOT a surface-depth animation. Per-frame contrast is stretched. Space pauses.</p><canvas id="c"></canvas><p id="info"></p><input id="slider" type="range" min="0" value="0"><pre id="source"></pre><script>const d={data};const c=document.getElementById('c'),ctx=c.getContext('2d');c.width=d.w;c.height=d.h;const slider=document.getElementById('slider');slider.max=d.frames.length-1;let t=0,play=true;function draw(){{let im=ctx.createImageData(d.w,d.h);d.frames[t].forEach((v,i)=>{{im.data[4*i]=v;im.data[4*i+1]=v;im.data[4*i+2]=v;im.data[4*i+3]=255}});ctx.putImageData(im,0,0);d.pairs.slice(0,8).forEach((p,i)=>{{[['a_quad_cell','#ffd700'],['b_quad_cell','#ff1ec8']].forEach(([key,color])=>{{const xy=p[key][t];ctx.strokeStyle=color;ctx.strokeRect(xy[0]-3,xy[1]-3,6,6);ctx.fillStyle=color;ctx.fillText(i,xy[0]+4,xy[1])}})}});document.getElementById('info').textContent='Frame '+t+' / '+(d.frames.length-1)+'; first 8 pairs in report order';document.getElementById('source').textContent=JSON.stringify(d.sources[t],null,2);slider.value=t}}slider.oninput=()=>{{t=+slider.value;play=false;draw()}};onkeydown=e=>{{if(e.code==='Space'){{play=!play;e.preventDefault()}}}};setInterval(()=>{{if(play){{t=(t+1)%d.frames.length;draw()}}}},150);draw();</script>"#
    );
    fs::write(out.join("motion.html"), html)?;
    Ok(
        json!({"contact_ppm":path,"contact_png":if converted{Some(png)}else{None},"motion_html":out.join("motion.html"),"displayed_pair_limit":8,"sampled_contact_frame_indices":indices}),
    )
}
fn run_bundle(
    path: &Path,
    out: &Path,
    allowed: Option<&std::collections::BTreeSet<String>>,
    dense: bool,
) -> Result<Value> {
    let bundle = Bundle::open(path)?;
    let index = bundle.read("frames.jsonl", 0, None)?;
    let records: Vec<Value> = String::from_utf8(index.clone())?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let mut eyes = Vec::new();
    let mut errors = Vec::new();
    for eye in ["subject-right", "subject-left"] {
        let mut epochs: Vec<Vec<&Value>> = Vec::new();
        let mut previous: Option<&Value> = None;
        let mut seen = std::collections::BTreeSet::new();
        let mut duplicate = 0;
        for r in records.iter().filter(|r| r["label"] == eye) {
            if allowed.is_some_and(|set| !set.contains(&glasses_parallax_inventory::identity(r))) {
                continue;
            }
            if [
                "sequence",
                "timestamp_ns",
                "sensor_x",
                "sensor_y",
                "width",
                "height",
                "stride",
                "offset",
                "length",
            ]
            .iter()
            .any(|k| u(r, k).is_err())
            {
                errors.push(json!({"eye":eye,"reason":"missing_numeric_acquisition_metadata","sequence":r["sequence"]}));
                previous = None;
                continue;
            }
            if u(r, "width")? > 16384
                || u(r, "height")? > 16384
                || u(r, "sensor_x")? > 16384
                || u(r, "sensor_y")? > 16384
            {
                errors.push(json!({"eye":eye,"reason":"unbounded_sensor_geometry","sequence":r["sequence"]}));
                previous = None;
                continue;
            }
            let id = json!([
                r["source_clock"]["source_key"],
                r["sequence"],
                r["timestamp_ns"]
            ])
            .to_string();
            if !seen.insert(id) {
                duplicate += 1;
                continue;
            }
            let valid = r["pixel_format"] == "RAW10_LE40_1X1"
                && r["stream"] == format!("{eye}.raw10")
                && r["recording_start_snapshot"] != true;
            if !valid {
                errors.push(json!({"eye":eye,"reason":"unsupported_format_stream_or_snapshot","sequence":r["sequence"]}));
                continue;
            }
            let split = previous.is_none_or(|p| {
                epoch_key(p) != epoch_key(r)
                    || u(r, "timestamp_ns").unwrap_or(0) <= u(p, "timestamp_ns").unwrap_or(0)
                    || u(r, "timestamp_ns").unwrap_or(0) - u(p, "timestamp_ns").unwrap_or(0)
                        > 250_000_000
            });
            if split {
                epochs.push(Vec::new());
            }
            epochs.last_mut().unwrap().push(r);
            previous = Some(r);
        }
        let mut windows = Vec::new();
        let mut skipped = Vec::new();
        let eligible: Vec<_> = epochs
            .iter()
            .enumerate()
            .filter(|(_, e)| e.len() >= 17)
            .collect();
        // Fixed acquisition-only bounded sample, no image/model selector:
        // first/middle/last 17-source windows of longest eligible source epoch.
        let mut chosen = eligible.clone();
        chosen.sort_by_key(|(i, e)| (std::cmp::Reverse(e.len()), *i));
        let mut sampled = Vec::new();
        for (ei, epoch) in chosen {
            for start in [0, (epoch.len() - 17) / 2, epoch.len() - 17] {
                if sampled
                    .iter()
                    .any(|&(e, s): &(usize, usize)| e == ei && s == start)
                {
                    continue;
                }
                sampled.push((ei, start));
                if sampled.len() == 3 {
                    break;
                }
            }
            if sampled.len() == 3 {
                break;
            }
        }
        sampled.sort();
        if dense {
            sampled.clear();
            for (ei, epoch) in &eligible {
                for start in (0..=epoch.len() - 17).step_by(8) {
                    sampled.push((*ei, start));
                }
                let final_start = epoch.len() - 17;
                if !sampled.contains(&(*ei, final_start)) {
                    sampled.push((*ei, final_start));
                }
            }
        }
        for (ei, epoch) in epochs.iter().enumerate() {
            if epoch.len() < 17 {
                skipped.push(json!({"epoch":ei,"frames":epoch.len(),"reason":"short_stable_roi_or_source_epoch"}));
            }
        }
        for (ei, start) in sampled {
            let epoch = &epochs[ei];
            let selected = &epoch[start..start + 17];
            let x0 = selected
                .iter()
                .map(|r| u(r, "sensor_x").unwrap().div_ceil(4) * 4)
                .max()
                .unwrap();
            let y0 = selected
                .iter()
                .map(|r| u(r, "sensor_y").unwrap().div_ceil(4) * 4)
                .max()
                .unwrap();
            let x1 = selected
                .iter()
                .map(|r| (u(r, "sensor_x").unwrap() + u(r, "width").unwrap()) / 4 * 4)
                .min()
                .unwrap();
            let y1 = selected
                .iter()
                .map(|r| (u(r, "sensor_y").unwrap() + u(r, "height").unwrap()) / 4 * 4)
                .min()
                .unwrap();
            if x1 < x0 + 64 || y1 < y0 + 64 {
                errors.push(json!({"eye":eye,"epoch":ei,"start":start,"reason":"insufficient_common_sensor_roi_overlap","intersection":[x0,y0,x1,y1]}));
                continue;
            }
            let mut images = Vec::new();
            let mut sources = Vec::new();
            let mut failure = None;
            for r in selected {
                let result = (|| -> Result<(Image, Value)> {
                    let bytes = bundle.read(
                        r["stream"].as_str().ok_or("stream")?,
                        u(r, "offset")?,
                        Some(u(r, "length")? as usize),
                    )?;
                    let decoded = decode(
                        &bytes,
                        u(r, "width")? as usize,
                        u(r, "height")? as usize,
                        u(r, "stride")? as usize,
                        u(r, "sensor_x")? as usize,
                        u(r, "sensor_y")? as usize,
                    )?;
                    let left = ((x0 - u(r, "sensor_x")?.div_ceil(4) * 4) / 4) as usize;
                    let top = ((y0 - u(r, "sensor_y")?.div_ceil(4) * 4) / 4) as usize;
                    let ww = ((x1 - x0) / 4) as usize;
                    let hh = ((y1 - y0) / 4) as usize;
                    let crop = |pixels: &[f64]| {
                        (0..hh)
                            .flat_map(|y| {
                                pixels[(top + y) * decoded.w + left
                                    ..(top + y) * decoded.w + left + ww]
                                    .iter()
                                    .copied()
                            })
                            .collect::<Vec<_>>()
                    };
                    let image = Image {
                        w: ww,
                        h: hh,
                        pixels: crop(&decoded.pixels),
                        green: crop(&decoded.green),
                    };
                    let source = json!({"stream":r["stream"],"offset":r["offset"],"length":r["length"],"raw_sha256":hash(&bytes),"sequence":r["sequence"],"timestamp_ns":r["timestamp_ns"],"sensor_x":r["sensor_x"],"sensor_y":r["sensor_y"],"width":r["width"],"height":r["height"],"stride":r["stride"],"pixel_format":r["pixel_format"],"region":r["region"],"source_clock":r["source_clock"]});
                    Ok((image, source))
                })();
                match result {
                    Ok((im, src)) => {
                        images.push(im);
                        sources.push(src);
                    }
                    Err(e) => {
                        failure = Some(e.to_string());
                        break;
                    }
                }
            }
            if let Some(reason) = failure {
                errors.push(json!({"eye":eye,"epoch":ei,"reason":reason}));
                continue;
            }
            let times: Vec<_> = selected
                .iter()
                .map(|r| u(r, "timestamp_ns").unwrap())
                .collect();
            let mut report = analyze(&images, &times);
            report["epoch"] = json!(ei);
            report["epoch_frames"] = json!(epoch.len());
            report["window_start_within_epoch"] = json!(start);
            report["sources"] = json!(sources);
            report["eye"] = json!(eye);
            report["archive"] = json!(path);
            report["analysis_sensor_rectangle"] = json!({"x":x0,"y":y0,"width":x1-x0,"height":y1-y0,"native_pixel_center_offset":1.5,"native_pixels_per_cell":4,"policy":"common absolute-sensor intersection; margins outside overlap not analyzed"});
            report["roi_reframes"] = json!(selected
                .windows(2)
                .filter(|r| r[0]["sensor_x"] != r[1]["sensor_x"]
                    || r[0]["sensor_y"] != r[1]["sensor_y"])
                .count());
            for pair in report["pairs"].as_array_mut().unwrap() {
                for key in ["a", "b"] {
                    let native: Vec<_> = pair[format!("{key}_quad_cell")]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(selected)
                        .map(|(p, _r)| {
                            json!([
                                x0 as f64 + 4. * p[0].as_f64().unwrap() + 1.5,
                                y0 as f64 + 4. * p[1].as_f64().unwrap() + 1.5
                            ])
                        })
                        .collect();
                    pair[format!("{key}_sensor_native")] = json!(native);
                }
            }
            let destination = out.join(format!("{eye}-epoch-{ei}-start-{start}"));
            fs::create_dir_all(&destination)?;
            report["visuals"] = render(&destination, &images, &report)?;
            fs::write(
                destination.join("window.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            windows.push(json!({"report":destination.join("window.json"),"eye":eye,"epoch":ei,"source_frames":images.len(),"pairs":report["pairs"].as_array().unwrap().len(),"persistent_pairs":report["pairs"].as_array().unwrap().iter().filter(|p|p["temporal_copy_support_fraction"].as_f64().unwrap()>=0.75).count(),"anonymous_differential_candidates":report["pairs"].as_array().unwrap().iter().filter(|p|p["anonymous_differential_candidate"]==true).count(),"complete_tracks":report["complete_tracks"],"state":"unknown","strong_glasses_interval":false}));
        }
        let used_epochs: std::collections::BTreeSet<_> =
            windows.iter().filter_map(|w| w["epoch"].as_u64()).collect();
        eyes.push(json!({"eye":eye,"source_frames":epochs.iter().map(Vec::len).sum::<usize>(),"duplicate_sources_skipped":duplicate,"roi_source_epochs":epochs.len(),"eligible_epochs":eligible.len(),"sampled_windows":windows,"short_epochs":skipped,"missing_eye":epochs.is_empty(),"unsampled_eligible_epochs":eligible.len().saturating_sub(used_epochs.len())}));
    }
    Ok(
        json!({"archive":path,"frames_index_sha256":hash(&index),"index_records":records.len(),"eyes":eyes,"input_errors":errors,"scope":"at most three windows per eye,17 consecutive source frames, common absolute-sensor ROI intersection; remaining corpus not evaluated"}),
    )
}
fn run() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() < 3 {
        return Err("usage: buttercup_glasses_parallax OUTPUT_DIR BUNDLE.tar [BUNDLE.tar ...]; bounded trial, at most 3 windows/eye/bundle".into());
    }
    let out = Path::new(&args[1]);
    if out.exists() {
        return Err("output directory already exists; use a fresh path".into());
    }
    let resolved = fs::canonicalize(out.parent().ok_or("output parent")?)?;
    if !resolved.starts_with("/mnt/bulk_data/buttercup-eye-tracking") {
        return Err("outputs must be beneath checked bulk runtime links".into());
    }
    fs::create_dir(out)?;
    if args[2] == "--tracks" {
        if args.len() != 5 {
            return Err("usage: OUTPUT --tracks WINDOW.json ASSISTANT_SEEDS.json".into());
        }
        return seeded_tracks(Path::new(&args[3]), Path::new(&args[4]), out);
    }
    let start = std::time::Instant::now();
    let mut captures = Vec::new();
    let corpus = args[2] == "--corpus";
    let dense = args[2] == "--dense";
    let inventory = if corpus {
        let roots: Vec<_> = if args.len() > 3 {
            args[3..].iter().map(PathBuf::from).collect()
        } else {
            ["outputs", "data/recordings", "data/labeled-corpus"]
                .iter()
                .map(PathBuf::from)
                .collect()
        };
        Some(glasses_parallax_inventory::inventory(&roots, out)?)
    } else {
        None
    };
    let jobs: Vec<_> = if let Some(inv) = &inventory {
        inv.selected
            .iter()
            .map(|c| (c.path.clone(), Some(&c.allowed)))
            .collect()
    } else {
        args[if dense { 3 } else { 2 }..]
            .iter()
            .map(|p| (PathBuf::from(p), None))
            .collect()
    };
    let mut progress = File::create(out.join("evaluation-progress.jsonl"))?;
    for (number, (path, allowed)) in jobs.iter().enumerate() {
        let suffix = if corpus {
            format!("-{}", &hash(path.to_string_lossy().as_bytes())[..8])
        } else {
            String::new()
        };
        let destination = out.join(format!(
            "{}{suffix}",
            path.file_stem().ok_or("stem")?.to_string_lossy()
        ));
        fs::create_dir(&destination)?;
        eprintln!(
            "RAW bounded evaluation {}/{}: {}",
            number + 1,
            jobs.len(),
            path.display()
        );
        let result = match run_bundle(path, &destination, *allowed, dense) {
            Ok(v) => v,
            Err(e) => json!({"archive":path,"status":"input_error","error":e.to_string()}),
        };
        writeln!(progress, "{}", serde_json::to_string(&result)?)?;
        progress.flush()?;
        captures.push(result);
    }
    let mut report = json!({"schema":"buttercup-glasses-parallax-trial-v1","captures":captures,"command":args,"elapsed_seconds":start.elapsed().as_secs_f64(),"algorithm":"versioned native corner-patch NCC candidate generator; historical SIFT not equivalent","thresholds":{"window_frames":17,"max_windows_per_eye":3,"initial_patch_ncc":0.90,"temporal_copy_fraction":0.75,"green_only_ncc":0.85,"green_support_fraction":0.75,"minimum_common_tracks":12,"minimum_differential_rms_native_px":3.0,"heldout_velocity_skill":0.5,"cyclic_null_margin":0.15},"inputs":"frames.jsonl acquisition fields and exact RAW ranges only; no reviews, predictions, installed models, masks, SAM or learned selectors read","scope":"single-user bounded development trial; no human reflection labels or calibrated probabilities","strong_interval_count":0,"reason":"motion layers remain anonymous; independent eye/material provenance and full camera-motion model unestablished","limitations":["new patch candidate generator, not numeric SIFT parity","image-plane similarity common motion does not remove full projective motion or establish material provenance","physical Quad-Bayer RG/GB phase follows the existing sensor decoder; green-only replicate is correlated, not independent truth","tracked fixed-scale patches miss blur, glare saturation and deformation","short/moving ROI epochs remain unknown","no anatomical/lens depth or glasses-off output","no training or full-corpus sweep"]});
    report["source_receipt"] = json!({
        "compiled_driver_sha256":hash(include_bytes!("buttercup_glasses_parallax.rs")),
        "compiled_algorithm_sha256":hash(include_bytes!("../glasses_parallax.rs")),
        "compiled_inventory_sha256":hash(include_bytes!("../glasses_parallax_inventory.rs")),
        "cargo_lock_sha256":hash(include_bytes!("../../Cargo.lock")),
        "executable_sha256":hash(&fs::read(env::current_exe()?)?),
        "scope":"exact compiled source inputs and executable; not a whole-checkout cold-bootstrap certificate"
    });
    if dense {
        report["scope"]=json!("targeted known-on diagnostic:17-frame acquisition windows every8frames, unchanged gates, no full-corpus rerun");
        report["thresholds"]["max_windows_per_eye"] = Value::Null;
        report["thresholds"]["dense_window_step_frames"] = json!(8);
        let mut contexts = Vec::new();
        for (path, _) in &jobs {
            contexts.push(match native_context(path, out) {
                Ok(v) => v,
                Err(e) => json!({"archive":path,"context_error":e.to_string()}),
            });
        }
        report["native_context"] = json!(contexts);
    }
    if let Some(inv) = inventory {
        report["scope"]=json!("corpus-wide candidate discovery with bounded per-container acquisition-only sampling; not a validated glasses detector or exhaustive per-frame sweep");
        report["inventory"] = json!({"path":out.join("inventory.json"),"sha256":hash(&fs::read(out.join("inventory.json"))?),"unique_source_manifest":out.join("unique-sources.jsonl"),"unique_source_manifest_sha256":hash(&fs::read(out.join("unique-sources.jsonl"))?),"discovered_containers":inv.report["discovered_canonical_bundles"],"selected_unique_containers":inv.report["selected_unique_source_containers"],"unique_compatible_sources":inv.report["unique_compatible_source_crops"]});
        report["limitations"]
            .as_array_mut()
            .unwrap()
            .retain(|x| x != "no training or full-corpus sweep");
        report["limitations"].as_array_mut().unwrap().push(json!("corpus coverage is bounded sampling, not all frames; no glasses labels promoted and no training"));
    }
    let mut summaries = Vec::new();
    let mut source_ids = std::collections::BTreeSet::new();
    let mut frame_usages = 0;
    let mut window_count = 0;
    let mut low_track = 0;
    let mut reframe_count = 0;
    for capture in report["captures"].as_array().unwrap() {
        if let Some(eyes) = capture["eyes"].as_array() {
            for eye in eyes {
                for window in eye["sampled_windows"].as_array().unwrap() {
                    let path = Path::new(window["report"].as_str().unwrap());
                    let data: Value = serde_json::from_slice(&fs::read(path)?)?;
                    window_count += 1;
                    low_track += usize::from(data["complete_tracks"].as_u64().unwrap_or(0) < 12);
                    reframe_count += data["roi_reframes"].as_u64().unwrap_or(0);
                    for source in data["sources"].as_array().unwrap() {
                        source_ids.insert(glasses_parallax_inventory::identity(&json!({"timestamp_ns":source["timestamp_ns"],"sequence":source["sequence"],"label":data["eye"],"sensor_x":source["sensor_x"],"sensor_y":source["sensor_y"],"width":source["width"],"height":source["height"],"stride":source["stride"],"pixel_format":source["pixel_format"]})));
                        frame_usages += 1;
                    }
                    for (pair_index, pair) in data["pairs"].as_array().unwrap().iter().enumerate() {
                        summaries.push(json!({"window":path,"archive":capture["archive"],"eye":data["eye"],"pair_index":pair_index,"sources_start_end":[data["sources"][0],data["sources"].as_array().unwrap().last()],"temporal_copy_support_fraction":pair["temporal_copy_support_fraction"],"green_only_copy_support_fraction":pair["green_only_copy_support_fraction"],"anonymous_differential_candidate":pair["anonymous_differential_candidate"],"motion":pair["motion"],"rejection":pair["rejection"],"visuals":data["visuals"]}));
                    }
                }
            }
        }
    }
    summaries.sort_by(|a, b| {
        b["anonymous_differential_candidate"]
            .as_bool()
            .cmp(&a["anonymous_differential_candidate"].as_bool())
            .then_with(|| {
                b["motion"]["heldout_velocity_skill_vs_zero"]
                    .as_f64()
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(
                        &a["motion"]["heldout_velocity_skill_vs_zero"]
                            .as_f64()
                            .unwrap_or(f64::NEG_INFINITY),
                    )
            })
    });
    fs::write(
        out.join("candidate-ranking.json"),
        serde_json::to_vec_pretty(
            &json!({"ranking":"anonymous gate pass first, then heldout velocity skill; diagnostic ranking, not probability or glasses truth","candidates":summaries}),
        )?,
    )?;
    report["summary"] = json!({"windows":window_count,"frame_usages":frame_usages,"unique_sampled_source_crops":source_ids.len(),"windows_with_less_than12_complete_tracks":low_track,"roi_origin_transitions":reframe_count,"initial_pair_instances":summaries.len(),"persistent_pair_instances":summaries.iter().filter(|p|p["temporal_copy_support_fraction"].as_f64().unwrap_or(0.)>=0.75).count(),"anonymous_differential_instances":summaries.iter().filter(|p|p["anonymous_differential_candidate"]==true).count(),"glasses_positive_labels":0,"candidate_ranking":out.join("candidate-ranking.json")});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", out.join("report.json").display());
    Ok(())
}
fn native_context(path: &Path, out: &Path) -> Result<Value> {
    let bundle = Bundle::open(path)?;
    let raw_index = bundle.read("frames.jsonl", 0, None)?;
    let mut source_index = File::create(out.join("native-raw-source-index.jsonl"))?;
    for (i, line) in String::from_utf8(raw_index)?.lines().enumerate() {
        let r: Value = serde_json::from_str(line)?;
        let stream = r["stream"].as_str().ok_or("stream")?;
        let (raw_file, base) = if path.is_dir() {
            (path.join(stream), 0)
        } else {
            (
                path.to_owned(),
                bundle.entries.get(stream).ok_or("RAW member")?.0,
            )
        };
        writeln!(
            source_index,
            "{}",
            json!({"index":i,"frame":r,"raw_file":raw_file,"raw_offset":base+u(&r,"offset")?,"raw_length":u(&r,"length")?,"provenance":"native acquisition only; no learned overlay or shape"})
        )?;
    }
    let index = bundle.read("thumbnails.jsonl", 0, None)?;
    let rows: Vec<Value> = String::from_utf8(index.clone())?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let mut images = Vec::new();
    let mut sources = Vec::new();
    let mut fresh_globals = 0;
    let mut snapshots = 0;
    for r in &rows {
        if r["recording_start_snapshot"] == true {
            snapshots += 1;
            continue;
        }
        let c = &r["camera"];
        if c["frame_kind"] == "global_sensor" {
            fresh_globals += 1;
        }
        if c["frame_kind"] != "sensor_band" || c["encoding"] != "GRAY16LE" {
            continue;
        }
        let bytes = bundle.read(
            "thumbnails.oic1",
            u(r, "offset")?,
            Some(u(r, "length")? as usize),
        )?;
        if bytes.len() < 24 || &bytes[..4] != b"OIC1" {
            return Err("invalid context envelope".into());
        }
        let le = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        let metadata_len = le(8);
        let camera_header_len = le(12);
        let payload_len = le(16);
        let offset = 24 + metadata_len + camera_header_len;
        if camera_header_len != 64 || offset + payload_len != bytes.len() {
            return Err("invalid native context lengths".into());
        }
        let w = u(c, "width_px")? as usize;
        let h = u(c, "height_px")? as usize;
        let stride = u(c, "stride_bytes")? as usize;
        if stride != w * 2 || payload_len != h * stride {
            return Err("invalid context shape".into());
        }
        let pixels: Vec<_> = bytes[offset..]
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]) as f64)
            .collect();
        images.push(Image {
            w,
            h,
            green: pixels.clone(),
            pixels,
        });
        sources.push(json!({"archive":path,"stream":"thumbnails.oic1","offset":r["offset"],"length":r["length"],"packet_sha256":hash(&bytes),"camera":c,"recording_start_snapshot":false}));
    }
    let destination = out.join("fresh-sensor-context");
    fs::create_dir(&destination)?;
    let mut record = json!({"archive":path,"index_sha256":hash(&index),"fresh_sensor_bands":images.len(),"fresh_global_count":fresh_globals,"excluded_start_snapshots":snapshots,"sources":sources,"pairs":[],"meaning":"native camera-average context confirms frame/eyewear appearance; same exposure as individual RAW ROI not assumed"});
    if !images.is_empty() {
        record["visuals"] = render(&destination, &images, &record)?;
    }
    fs::write(
        destination.join("context.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(record)
}
fn seeded_tracks(window: &Path, seeds_file: &Path, out: &Path) -> Result<()> {
    let original = fs::read(window)?;
    let record: Value = serde_json::from_slice(&original)?;
    let seed_bytes = fs::read(seeds_file)?;
    let spec: Value = serde_json::from_slice(&seed_bytes)?;
    if spec["annotation_source"] != "assistant-diagnostic-seeds" {
        return Err("diagnostic seed provenance must be explicit".into());
    }
    let archive = Path::new(record["archive"].as_str().ok_or("archive")?);
    let bundle = Bundle::open(archive)?;
    let rect = &record["analysis_sensor_rectangle"];
    let x0 = u(rect, "x")?;
    let y0 = u(rect, "y")?;
    let ww = (u(rect, "width")? / 4) as usize;
    let hh = (u(rect, "height")? / 4) as usize;
    let mut images = Vec::new();
    for r in record["sources"].as_array().ok_or("sources")? {
        let bytes = bundle.read(
            r["stream"].as_str().ok_or("stream")?,
            u(r, "offset")?,
            Some(u(r, "length")? as usize),
        )?;
        if hash(&bytes) != r["raw_sha256"].as_str().ok_or("RAW hash")? {
            return Err("source RAW no longer matches original diagnostic window".into());
        }
        let decoded = decode(
            &bytes,
            u(r, "width")? as usize,
            u(r, "height")? as usize,
            u(r, "stride")? as usize,
            u(r, "sensor_x")? as usize,
            u(r, "sensor_y")? as usize,
        )?;
        let left = ((x0 - u(r, "sensor_x")?.div_ceil(4) * 4) / 4) as usize;
        let top = ((y0 - u(r, "sensor_y")?.div_ceil(4) * 4) / 4) as usize;
        let crop = |p: &[f64]| {
            (0..hh)
                .flat_map(|y| {
                    p[(top + y) * decoded.w + left..(top + y) * decoded.w + left + ww]
                        .iter()
                        .copied()
                })
                .collect()
        };
        images.push(Image {
            w: ww,
            h: hh,
            pixels: crop(&decoded.pixels),
            green: crop(&decoded.green),
        });
    }
    let mut tracks = Vec::new();
    for seed in spec["seeds"].as_array().ok_or("seeds")? {
        let xy = &seed["sensor_native"];
        let p = [
            (xy[0].as_f64().ok_or("seed x")? - x0 as f64 - 1.5) / 4.,
            (xy[1].as_f64().ok_or("seed y")? - y0 as f64 - 1.5) / 4.,
        ];
        let p = [p[0].round(), p[1].round()];
        let (path, failure) = glasses_parallax::track(&images, p);
        let native: Vec<_> = path
            .iter()
            .map(|q| [x0 as f64 + 4. * q[0] + 1.5, y0 as f64 + 4. * q[1] + 1.5])
            .collect();
        let residual: Vec<_> = path
            .iter()
            .enumerate()
            .map(|(t, q)| {
                let m = &record["common_similarity_by_frame"][t]["scale_cos_scale_sin_tx_ty"];
                let a = m[0].as_f64().unwrap();
                let b = m[1].as_f64().unwrap();
                [
                    4. * (q[0] - (a * p[0] - b * p[1] + m[2].as_f64().unwrap())),
                    4. * (q[1] - (b * p[0] + a * p[1] + m[3].as_f64().unwrap())),
                ]
            })
            .collect();
        tracks.push(json!({"name":seed["name"],"proposed_role":seed["proposed_role"],"role_provenance":"assistant visual interpretation, not established physical surface","seed_sensor_native":[x0 as f64+4.*p[0]+1.5,y0 as f64+4.*p[1]+1.5],"path_quad_cell":path,"path_sensor_native":native,"residual_to_common_similarity_native_px":residual,"failure":failure,"first_failed_frame":failure.map(|_|native.len()),"fresh_frames":native.len()}));
    }
    let mut relative = Vec::new();
    for i in 0..tracks.len() {
        for j in i + 1..tracks.len() {
            let a = tracks[i]["path_sensor_native"].as_array().unwrap();
            let b = tracks[j]["path_sensor_native"].as_array().unwrap();
            let n = a.len().min(b.len());
            let delta: Vec<_> = (0..n)
                .map(|t| {
                    let f = |k| {
                        a[t][k].as_f64().unwrap()
                            - b[t][k].as_f64().unwrap()
                            - (a[0][k].as_f64().unwrap() - b[0][k].as_f64().unwrap())
                    };
                    [f(0usize), f(1usize)]
                })
                .collect();
            relative.push(json!({"a":tracks[i]["name"],"b":tracks[j]["name"],"relative_displacement_change_native_px":delta,"fresh_overlapping_frames":n}));
        }
    }
    let data = json!({"w":ww,"h":hh,"frames":images.iter().map(display).collect::<Vec<_>>(),"sources":record["sources"],"tracks":tracks});
    let html = format!(
        r#"<!doctype html><meta charset="utf-8"><style>body{{background:#222;color:white;font:15px monospace}}canvas{{width:840px;image-rendering:pixelated}}pre{{white-space:pre-wrap}}</style><p>Assistant diagnostic seeds, not physical surface labels. Tracks disappear when correspondence fails. Space pauses.</p><canvas id="c"></canvas><input type="range" id="s" min="0" value="0"><pre id="p"></pre><script>const d={data},c=document.getElementById('c'),ctx=c.getContext('2d'),s=document.getElementById('s');c.width=d.w;c.height=d.h;s.max=d.frames.length-1;let t=0,play=true;function draw(){{const im=ctx.createImageData(d.w,d.h);d.frames[t].forEach((v,i)=>{{im.data[4*i]=v;im.data[4*i+1]=v;im.data[4*i+2]=v;im.data[4*i+3]=255}});ctx.putImageData(im,0,0);d.tracks.forEach((r,i)=>{{if(t<r.path_quad_cell.length){{ctx.strokeStyle=['#00ffff','#ffff00','#ff55ff','#44ff44'][i%4];let q=r.path_quad_cell[t];ctx.strokeRect(q[0]-3,q[1]-3,6,6);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(i,q[0]+4,q[1]);}}}});document.getElementById('p').textContent='Frame '+t+'\n'+d.tracks.map((r,i)=>i+': '+r.name+' ['+r.proposed_role+']'+(t>=r.fresh_frames?' LOST':'')).join('\n')+'\n'+JSON.stringify(d.sources[t],null,2);s.value=t}}s.oninput=()=>{{t=+s.value;play=false;draw()}};onkeydown=e=>{{if(e.code==='Space'){{play=!play;e.preventDefault()}}}};setInterval(()=>{{if(play){{t=(t+1)%d.frames.length;draw()}}}},150);draw()</script>"#
    );
    fs::write(out.join("seeded-motion.html"), html)?;
    let mut pixels = vec![0; ww * 5 * hh * 3];
    for (panel, t) in [0, 4, 8, 12, 16].iter().copied().enumerate() {
        let frame = display(&images[t]);
        for y in 0..hh {
            for x in 0..ww {
                let k = (y * ww * 5 + panel * ww + x) * 3;
                pixels[k..k + 3].fill(frame[y * ww + x]);
            }
        }
        for (i, tr) in tracks.iter().enumerate() {
            if let Some(p) = tr["path_quad_cell"].as_array().unwrap().get(t) {
                let x = p[0].as_f64().unwrap() as isize;
                let y = p[1].as_f64().unwrap() as isize;
                for dy in -3isize..=3 {
                    for dx in -3isize..=3 {
                        if dx.abs() != 3 && dy.abs() != 3 {
                            continue;
                        }
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && xx < ww as isize && yy < hh as isize {
                            let k = (yy as usize * ww * 5 + panel * ww + xx as usize) * 3;
                            pixels[k..k + 3].copy_from_slice(
                                &[[0, 255, 255], [255, 255, 0], [255, 85, 255], [68, 255, 68]]
                                    [i % 4],
                            );
                        }
                    }
                }
            }
        }
    }
    ppm(&out.join("seeded-contact.ppm"), ww * 5, hh, &pixels)?;
    let _ = Command::new("ffmpeg")
        .args(["-v", "error", "-threads", "1", "-i"])
        .arg(out.join("seeded-contact.ppm"))
        .args(["-frames:v", "1", "-threads", "1"])
        .arg(out.join("seeded-contact.png"))
        .status()?;
    fs::write(
        out.join("tracks.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"buttercup-assistant-seeded-RAW-diagnostics-v1","window":window,"window_sha256":hash(&original),"seed_spec":spec,"seed_sha256":hash(&seed_bytes),"sources":record["sources"],"tracks":tracks,"relative":relative,"common_fit":record["common_similarity_by_frame"],"observation":"single reflection and texture correspondence only; no reflective-copy pair or physical surface assignment established","algorithm_sha256":hash(include_bytes!("../glasses_parallax.rs")),"driver_sha256":hash(include_bytes!("buttercup_glasses_parallax.rs"))}),
        )?,
    )?;
    println!("{}", out.join("tracks.json").display());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("glasses parallax: {e}");
        std::process::exit(1);
    }
}
