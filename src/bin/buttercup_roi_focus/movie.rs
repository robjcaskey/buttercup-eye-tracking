//! Presentation of every previously classified ambiguous RAW exposure.
//! No new fit, target, sign decision, calibration, or training is introduced.
use super::{
    archive::{self, Frame, Manifest},
    replay, Result,
};
use buttercup_eye_tracking::{focus_region::*, raw10, recorded_bundle::BundleSource};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    process::{Command, Stdio},
};
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
use canvas::*;

const FPS: usize = 25;
const COLORS: [[f64; 3]; 2] = [CYAN, PINK];
type Key = (u32, u32);
struct Eye {
    id: usize,
    class: String,
    rays: Option<TheoreticalEllipseExplanations>,
    candidates: Vec<RayClassification>,
}
struct Exposure {
    ns: u64,
    eyes: [Option<Eye>; 2],
}
struct Clip {
    key: Key,
    regions: Vec<FocusRegion>,
    all: Vec<Exposure>,
    selected: Vec<usize>,
    first_ns: u64,
}

fn raw(bundle: &BundleSource, manifest: &Manifest, f: &Frame) -> Result<Vec<u8>> {
    let s = &manifest.sources[f.source as usize];
    Ok(bundle.read_range(
        &format!("{}{}", s.prefix, s.streams[f.stream as usize]),
        f.offset,
        f.length as usize,
    )?)
}
fn unpack(bytes: &[u8], f: &Frame) -> Result<Vec<u16>> {
    Ok(raw10::try_unpack_raw10(
        bytes,
        f.width as usize,
        f.height as usize,
        f.stride as usize,
    )?)
}
fn preview(p: &[u16], tone: [f64; 2]) -> Vec<u8> {
    p.iter()
        .flat_map(|v| {
            let q =
                (255. * ((*v as f64 - tone[0]) / (tone[1] - tone[0]).max(1.)).clamp(0., 1.)) as u8;
            [q, q, q, 255]
        })
        .collect()
}
fn label(class: &str) -> &str {
    match class {
        "multiple" => "BOTH interpretations compatible",
        "one" => "One interpretation compatible",
        "zero" => "Neither interpretation compatible",
        "missing_ellipse" => "No ellipse exported for this frame",
        "invalid_projection" => "Circle unprojection unavailable",
        "unusable_evidence" => "Recorded evidence rejected",
        _ => "Focus region unresolved",
    }
}
fn provider(p: u8) -> &'static str {
    match p {
        1 => "SAM fit",
        2 => "Archived semantic fit",
        3 => "Archived virtual-contact reconstruction",
        _ => "No exported ellipse",
    }
}

// An identical camera projection is shared by both interpretations. Project a
// short actual 3-D normal, not a freely positioned screen cursor.
fn project(p: V3) -> Option<[f64; 2]> {
    (p[2] < -0.1).then(|| [4000. + 4000. * p[0] / -p[2], 3000. + 4000. * p[1] / -p[2]])
}
fn plot(
    c: &mut Canvas,
    rect: [f64; 4],
    axes: [usize; 2],
    bounds: [V3; 2],
    regions: &[FocusRegion],
    eye: Option<&Eye>,
    order: [usize; 2],
) {
    let [x, y, w, h] = rect;
    c.rect(x, y, w, h, [0.075, 0.10, 0.13]);
    let a = axes[0];
    let b = axes[1];
    let s =
        ((w - 35.) / (bounds[1][a] - bounds[0][a])).min((h - 50.) / (bounds[1][b] - bounds[0][b]));
    let pp = |p: V3| {
        [
            x + w * 0.5 + (p[a] - (bounds[0][a] + bounds[1][a]) * 0.5) * s,
            y + h * 0.5 + 8. + (p[b] - (bounds[0][b] + bounds[1][b]) * 0.5) * s,
        ]
    };
    c.text(
        x + 10.,
        y + 22.,
        16.,
        MUTED,
        if a == 0 {
            "Top: X / Z (equal units)"
        } else {
            "Side: Y / Z (equal units)"
        },
    );
    for r in regions {
        let lo = pp(r.lower);
        let hi = pp(r.upper);
        c.rect(
            lo[0],
            lo[1],
            (hi[0] - lo[0]).max(2.),
            (hi[1] - lo[1]).max(2.),
            [0.18, 0.32, 0.25],
        );
        let center = pp(r.center);
        c.text(
            center[0] + 3.,
            center[1] - 3.,
            13.,
            GREEN,
            &format!("{}", r.id),
        );
    }
    let cam = pp([0.; 3]);
    c.cross(cam[0], cam[1], 5., WHITE);
    c.text(cam[0] + 7., cam[1] - 6., 12., MUTED, "camera");
    if let Some(e) = eye {
        if let Some(rays) = e.rays {
            for (display, &branch) in order.iter().enumerate() {
                let r = rays.rays[branch];
                let t = e.candidates[branch]
                    .forward_iris_radii
                    .unwrap_or(30.)
                    .min(250.);
                let start = pp(r.origin_iris_radii);
                let end = pp(add(r.origin_iris_radii, scale(r.direction, t)));
                c.clipped(x, y + 28., w, h - 28., |c| {
                    c.arrow(start, end, COLORS[display]);
                    c.dot(end[0], end[1], 4., COLORS[display], true);
                });
            }
        }
    }
}

fn draw_eye(
    c: &mut Canvas,
    x: f64,
    which: usize,
    eye: Option<&Eye>,
    f: Option<&Frame>,
    pixels: Option<&[u8]>,
    order: [usize; 2],
    regions: &[FocusRegion],
    bounds: [V3; 2],
) {
    let class = eye
        .map(|e| e.class.as_str())
        .unwrap_or("No simultaneous exposure");
    c.text(
        x,
        179.,
        22.,
        if class == "multiple" { ORANGE } else { MUTED },
        &format!(
            "Eye {}  |  {}",
            which + 1,
            if eye.is_some() { label(class) } else { class }
        ),
    );
    let rect = [x, 200., 672., 448.];
    c.rect(rect[0], rect[1], rect[2], rect[3], [0.025, 0.035, 0.045]);
    if let (Some(e), Some(f), Some(pixels)) = (eye, f, pixels) {
        let s = (rect[2] / f.width as f64).min(rect[3] / f.height as f64);
        let w = f.width as f64 * s;
        let h = f.height as f64 * s;
        let ix = x + (rect[2] - w) * 0.5;
        let iy = rect[1] + (rect[3] - h) * 0.5;
        c.image(pixels, f.width as usize, f.height as usize, ix, iy, w, h);
        let pp = |p: [f64; 2]| {
            [
                ix + (p[0] - f.origin[0] as f64) * s,
                iy + (p[1] - f.origin[1] as f64) * s,
            ]
        };
        if let Some(shape) = f.shape() {
            let mut outline = shape
                .dense_points(240)
                .iter()
                .map(|p| pp([p.0, p.1]))
                .collect::<Vec<_>>();
            outline.push(outline[0]);
            c.clipped(ix, iy, w, h, |c| {
                c.path(&outline, 2., WHITE);
                if let Some(rays) = e.rays {
                    for (display, &branch) in order.iter().enumerate() {
                        let r = rays.rays[branch];
                        let length = 0.8_f64.min(-r.origin_iris_radii[2] * 0.1);
                        if let (Some(start), Some(end)) = (
                            project(r.origin_iris_radii),
                            project(add(r.origin_iris_radii, scale(r.direction, length))),
                        ) {
                            c.arrow(pp(start), pp(end), COLORS[display]);
                        }
                    }
                }
            });
        }
        c.text(
            x,
            677.,
            19.,
            WHITE,
            &format!("Exposure {}  |  {}", f.sequence, provider(f.provider)),
        );
        c.text(
            x,
            706.,
            17.,
            MUTED,
            "White: fitted ellipse. Cyan / pink: both possible directions.",
        );
        let tx = x + 692.;
        for (display, &branch) in order.iter().enumerate() {
            let yy = 235. + display as f64 * 122.;
            c.text(
                tx,
                yy,
                21.,
                COLORS[display],
                if display == 0 {
                    "Candidate A"
                } else {
                    "Candidate B"
                },
            );
            if let Some(q) = e.candidates.get(branch) {
                c.text(tx, yy + 30., 20., WHITE, q.status);
                c.text(
                    tx,
                    yy + 57.,
                    17.,
                    MUTED,
                    &format!("Miss {:.2} deg", q.miss_degrees.unwrap_or(0.)),
                );
                c.text(
                    tx,
                    yy + 81.,
                    16.,
                    MUTED,
                    &format!(
                        "Region {}",
                        q.region.map(|v| v.to_string()).unwrap_or("--".into())
                    ),
                );
            } else {
                c.text(tx, yy + 30., 17., MUTED, "Unavailable");
            }
        }
        if let Some(r) = e.rays {
            c.text(
                tx,
                513.,
                17.,
                WHITE,
                &format!("{:.1} deg apart", r.separation_degrees),
            );
        }
        c.text(tx, 554., 17., MUTED, "Post-affine area");
        c.text(
            tx,
            581.,
            19.,
            WHITE,
            &if f.area.flags & 1 != 0 {
                format!("{:.0} px2", f.area.frontal_equivalent_disk_px2)
            } else {
                "Unavailable".into()
            },
        );
        c.text(tx, 610., 15., MUTED, "Independent scale");
        c.text(tx, 632., 15., MUTED, "unavailable");
    } else {
        c.text(
            x + 25.,
            410.,
            22.,
            MUTED,
            "No exposure at this source timestamp",
        );
    }
    plot(
        c,
        [x, 745., 442., 253.],
        [0, 2],
        bounds,
        regions,
        eye,
        order,
    );
    plot(
        c,
        [x + 456., 745., 442., 253.],
        [1, 2],
        bounds,
        regions,
        eye,
        order,
    );
}

fn verify_index(
    bundle: &BundleSource,
    manifest: &Manifest,
    frames: &[Frame],
    clip: &Clip,
) -> Result<()> {
    let s = &manifest.sources[clip.key.0 as usize];
    let bytes = bundle.read_entry(&format!("{}frames.jsonl", s.prefix))?;
    if archive::digest(&bytes) != s.frames_sha256 {
        return Err("source frame index changed".into());
    }
    // Independently bind the binary offsets/crops to the original native index.
    let rows = bytes
        .split(|v| *v == b'\n')
        .filter(|row| !row.is_empty())
        .collect::<Vec<_>>();
    for &i in &clip.selected {
        for e in clip.all[i].eyes.iter().flatten() {
            let f = &frames[e.id];
            let row: Value = serde_json::from_slice(
                rows.get(f.index as usize)
                    .ok_or("source index row missing")?,
            )?;
            let n = super::pack::number;
            for (key, expected) in [
                ("eye_id", f.eye as u64),
                ("sequence", f.sequence),
                ("sensor_x", f.origin[0] as u64),
                ("sensor_y", f.origin[1] as u64),
                ("width", f.width as u64),
                ("height", f.height as u64),
            ] {
                if n(&row[key]) != Some(expected) {
                    return Err(
                        format!("native source mismatch at record {}, key {key}", e.id).into(),
                    );
                }
            }
        }
    }
    Ok(())
}

fn load(
    input: &str,
    replay_dir: &Path,
) -> Result<(Manifest, Vec<Frame>, Vec<Clip>, BTreeSet<usize>)> {
    let summary_bytes = fs::read(replay_dir.join("summary.json"))?;
    let summary: Value = serde_json::from_slice(&summary_bytes)?;
    if archive::digest(&fs::read(input)?)
        != summary["binary_sha256"]
            .as_str()
            .ok_or("missing binary identity")?
    {
        return Err("binary differs from classified replay".into());
    }
    let (manifest, frames) = archive::read(Path::new(input))?;
    let mut groups = BTreeMap::new();
    for g in summary["per_group"].as_array().ok_or("missing groups")? {
        if g["classes"]["multiple"].as_u64().unwrap_or(0) > 0 {
            let key = (
                g["source"].as_u64().unwrap() as u32,
                g["epoch"].as_u64().unwrap() as u32,
            );
            groups.insert(
                key,
                serde_json::from_value::<Vec<FocusRegion>>(g["regions"].clone())?,
            );
        }
    }
    let mut expected = BTreeSet::new();
    let mut saved = HashMap::new();
    for line in BufReader::new(fs::File::open(replay_dir.join("classifications.jsonl"))?).lines() {
        let l = line?;
        if !l.contains("\"classification\":\"multiple\"") {
            continue;
        }
        let r: Value = serde_json::from_str(&l)?;
        let id = r["record"].as_u64().ok_or("record")? as usize;
        let f = frames.get(id).ok_or("record outside binary")?;
        if r["source"].as_u64() != Some(f.source as u64)
            || r["epoch"].as_u64() != Some(f.epoch as u64)
            || r["source_ns"].as_str() != Some(f.ns.to_string().as_str())
            || r["eye"].as_u64() != Some(f.eye as u64)
            || r["sequence"].as_u64() != Some(f.sequence)
        {
            return Err("classification/binary source mismatch".into());
        }
        if !expected.insert(id) {
            return Err("duplicate ambiguous record".into());
        }
        saved.insert(id, r);
    }
    if expected.len() as u64
        != summary["classes"]["multiple"]
            .as_u64()
            .ok_or("summary count")?
    {
        return Err("ambiguous count mismatch".into());
    }
    let mut timeline: BTreeMap<Key, BTreeMap<u64, [Option<usize>; 2]>> = BTreeMap::new();
    for (id, f) in frames.iter().enumerate() {
        if !groups.contains_key(&(f.source, f.epoch)) || (1..=2).contains(&f.eye) == false {
            continue;
        }
        let eyes = timeline
            .entry((f.source, f.epoch))
            .or_default()
            .entry(f.ns)
            .or_insert([None; 2]);
        if eyes[f.eye as usize - 1].replace(id).is_some() {
            return Err("duplicate same-eye source exposure; cannot silently drop one".into());
        }
    }
    let mut clips = vec![];
    let mut actual = BTreeSet::new();
    for (key, timeline) in timeline {
        let regions = groups.remove(&key).unwrap();
        let mut all = vec![];
        for (ns, ids) in timeline {
            let mut eyes = [None, None];
            for (which, id) in ids.into_iter().enumerate() {
                if let Some(id) = id {
                    let rays = replay::explanations(&frames[id], 4000.);
                    let (class, candidates) = replay::verdict(&frames[id], &rays, &regions);
                    if class == "multiple" {
                        actual.insert(id);
                        let old = saved
                            .get(&id)
                            .ok_or("recomputed ambiguity differs from saved replay")?;
                        let prior: TheoreticalEllipseExplanations =
                            serde_json::from_value(old["explanations"].clone())?;
                        let now = rays.unwrap();
                        for b in 0..2 {
                            if norm(sub(now.rays[b].direction, prior.rays[b].direction)) > 1e-12
                                || norm(sub(
                                    now.rays[b].origin_iris_radii,
                                    prior.rays[b].origin_iris_radii,
                                )) > 1e-10
                            {
                                return Err("candidate geometry changed".into());
                            }
                            if old["interpretations"][b]["status"].as_str()
                                != Some(candidates[b].status)
                            {
                                return Err("candidate classification changed".into());
                            }
                        }
                    }
                    eyes[which] = Some(Eye {
                        id,
                        class: class.into(),
                        rays,
                        candidates,
                    });
                }
            }
            all.push(Exposure { ns, eyes });
        }
        let mut selected = BTreeSet::new();
        for (i, e) in all.iter().enumerate() {
            if !e.eyes.iter().flatten().any(|e| e.class == "multiple") {
                continue;
            }
            selected.insert(i);
            // One neighbor on either side, only across short native intervals.
            if i > 0 && e.ns - all[i - 1].ns <= 350_000_000 {
                selected.insert(i - 1);
            }
            if i + 1 < all.len() && all[i + 1].ns - e.ns <= 350_000_000 {
                selected.insert(i + 1);
            }
        }
        let first_ns = all[0].ns;
        clips.push(Clip {
            key,
            regions,
            all,
            selected: selected.into_iter().collect(),
            first_ns,
        });
    }
    if actual != expected {
        return Err("movie does not contain the exact ambiguous record set".into());
    }
    Ok((manifest, frames, clips, expected))
}

pub fn run(input: &str, replay_dir: &str, output: &str, recovery: Option<&str>) -> Result<()> {
    let out = Path::new(output);
    if out.exists() {
        return Err("movie output exists".into());
    }
    fs::create_dir_all(out)?;
    let replay_dir = Path::new(replay_dir);
    let (manifest, frames, clips, expected) = load(input, replay_dir)?;
    let sources = clips.iter().map(|c| c.key.0).collect::<BTreeSet<_>>();
    let nsources = sources.len();
    let nevents = clips.iter().map(|c| c.selected.len()).sum::<usize>();
    eprintln!("MOVIE PLAN {} ambiguous eye frames, {nsources} recordings, {} clock groups, {nevents} exposures including context",expected.len(),clips.len());
    let encoded = out.join("ambiguous-frames-encoded.mp4");
    let mut encoder = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgra",
            "-video_size",
            "1920x1080",
            "-framerate",
            "25",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "6",
            "-movflags",
            "+faststart",
        ])
        .arg(&encoded)
        .stdin(Stdio::piped())
        .stderr(fs::File::create(out.join("encode.log"))?)
        .spawn()?;
    let mut sink = encoder.stdin.take().ok_or("encoder input")?;
    let mut c = Canvas::new(1920, 1080)?;
    let mut video_frame = 0usize;
    let mut written = BTreeSet::new();
    let mut events = BufWriter::new(fs::File::create(out.join("movie-frames.jsonl"))?);
    let mut chapters = vec![];
    let mut selected_providers = BTreeMap::<u8, usize>::new();
    for (ci, clip) in clips.iter().enumerate() {
        let source = &manifest.sources[clip.key.0 as usize];
        let basename = Path::new(&source.path)
            .file_name()
            .unwrap()
            .to_string_lossy();
        let number = sources.iter().position(|s| *s == clip.key.0).unwrap() + 1;
        let mut raw_path = source.path.as_str();
        let mut bundle = BundleSource::open(Path::new(raw_path))?;
        let present = |bundle: &BundleSource| {
            source.streams.iter().all(|name| {
                let name = format!("{}{}", source.prefix, name);
                match bundle {
                    BundleSource::Directory(root) => root.join(name).is_file(),
                    BundleSource::Tar { entries, .. } => entries.contains_key(&name),
                }
            })
        };
        if !present(&bundle) {
            raw_path = recovery.ok_or("original bundle lacks RAW; supply a recovery directory with an identical frame index")?;
            bundle = BundleSource::open(Path::new(raw_path))?;
            if !present(&bundle) {
                return Err("recovery directory lacks required RAW streams".into());
            }
            eprintln!(
                "RAW RECOVERY source {}: {raw_path}; requiring identical index hash before use",
                clip.key.0
            );
        }
        verify_index(&bundle, &manifest, &frames, clip)?;
        let mut tones = [[0.; 2]; 2];
        for (which, tone) in tones.iter_mut().enumerate() {
            let mut samples = vec![];
            for k in 0..8 {
                let i = clip.selected[k * (clip.selected.len() - 1) / 7];
                if let Some(e) = &clip.all[i].eyes[which] {
                    let f = &frames[e.id];
                    let mut pixels = unpack(&raw(&bundle, &manifest, f)?, f)?;
                    pixels.sort_unstable();
                    samples.push([
                        pixels[pixels.len() / 100] as f64,
                        pixels[pixels.len() * 99 / 100] as f64,
                    ]);
                }
            }
            *tone = if samples.is_empty() {
                [0., 1023.]
            } else {
                std::array::from_fn(|a| {
                    samples.iter().map(|v| v[a]).sum::<f64>() / samples.len() as f64
                })
            };
        }
        let mut bounds: [V3; 2] = [[0.; 3]; 2];
        for p in clip.regions.iter().flat_map(|r| [r.lower, r.upper]).chain(
            clip.selected.iter().flat_map(|&i| {
                clip.all[i]
                    .eyes
                    .iter()
                    .flatten()
                    .filter_map(|e| e.rays)
                    .flat_map(|r| r.rays.map(|r| r.origin_iris_radii))
            }),
        ) {
            for a in 0..3 {
                bounds[0][a] = bounds[0][a].min(p[a]);
                bounds[1][a] = bounds[1][a].max(p[a]);
            }
        }
        for a in 0..3 {
            let pad = ((bounds[1][a] - bounds[0][a]) * 0.08).max(2.);
            bounds[0][a] -= pad;
            bounds[1][a] += pad;
        }
        let start = video_frame;
        let mut previous: [Option<[GazeRay; 2]>; 2] = [None, None];
        let mut last_ns = None;
        let mut saved_preview = false;
        for (j, &i) in clip.selected.iter().enumerate() {
            let exposure = &clip.all[i];
            let mut order = [[0, 1]; 2];
            let mut pixels: [Option<Vec<u8>>; 2] = [None, None];
            let mut hashes: [Option<String>; 2] = [None, None];
            for which in 0..2 {
                if let Some(e) = &exposure.eyes[which] {
                    let f = &frames[e.id];
                    let bytes = raw(&bundle, &manifest, f)?;
                    hashes[which] = Some(archive::digest(&bytes));
                    pixels[which] = Some(preview(&unpack(&bytes, f)?, tones[which]));
                    if let Some(r) = e.rays {
                        if let Some(prev) = previous[which] {
                            let straight = norm(sub(prev[0].direction, r.rays[0].direction))
                                + norm(sub(prev[1].direction, r.rays[1].direction));
                            let swapped = norm(sub(prev[0].direction, r.rays[1].direction))
                                + norm(sub(prev[1].direction, r.rays[0].direction));
                            if swapped < straight {
                                order[which] = [1, 0];
                            }
                        }
                        previous[which] = Some(order[which].map(|b| r.rays[b]));
                    }
                    if e.class == "multiple" {
                        if !written.insert(e.id) {
                            return Err("ambiguous exposure rendered twice as new evidence".into());
                        }
                        *selected_providers.entry(f.provider).or_default() += 1;
                    }
                }
            }
            c.clear();
            c.text(24., 43., 30., WHITE, "When both gaze interpretations fit");
            c.text(
                24.,
                81.,
                21.,
                WHITE,
                &format!("Recording {number}/{nsources}  |  {basename}"),
            );
            c.text(24.,117.,18.,MUTED,&format!("Source +{:07.2}s  |  2x slow motion; long gaps shortened  |  {} / {} ambiguous eye frames",(exposure.ns-clip.first_ns) as f64*1e-9,written.len(),expected.len()));
            if let Some(prev) = last_ns {
                let gap = exposure.ns - prev;
                if gap > 350_000_000 {
                    c.text(
                        1320.,
                        43.,
                        20.,
                        ORANGE,
                        &format!("Source jump +{:.2}s", gap as f64 * 1e-9),
                    );
                }
            }
            for which in 0..2 {
                let eye = exposure.eyes[which].as_ref();
                draw_eye(
                    &mut c,
                    24. + which as f64 * 960.,
                    which,
                    eye,
                    eye.map(|e| &frames[e.id]),
                    pixels[which].as_deref(),
                    order[which],
                    &clip.regions,
                    bounds,
                );
            }
            c.text(24.,1030.,18.,GREEN,"Green boxes: inferred focus regions, not a measured screen. Both directions can fit different regions.");
            c.text(24.,1060.,16.,MUTED,"Compatibility is not verified gaze. Nearby tolerance: 2 deg. Camera geometry is nominal. Candidate colors follow continuity for display only.");
            if !saved_preview
                && exposure
                    .eyes
                    .iter()
                    .flatten()
                    .any(|e| e.class == "multiple")
            {
                c.png(&out.join(format!("recording-{:02}-epoch-{}.png", number, clip.key.1)))?;
                saved_preview = true;
            }
            let next = clip.selected.get(j + 1).map(|&k| clip.all[k].ns);
            let dt = next
                .map(|ns| (ns - exposure.ns) as f64 * 1e-9)
                .unwrap_or(0.2);
            // Every selected exposure is visible; CFR holds are presentation,
            // not fresh evidence. Long gaps cannot masquerade as continuous RAW.
            let repeat = (dt.mul_add(2., 0.).clamp(0.12, 0.6) * FPS as f64).round() as usize;
            let from = video_frame;
            for _ in 0..repeat {
                sink.write_all(c.bytes())?;
                video_frame += 1;
            }
            let eyes=exposure.eyes.iter().enumerate().map(|(which,e)|e.as_ref().map(|e|json!({"record":e.id,"eye":which+1,"sequence":frames[e.id].sequence,"class":e.class,"provider":frames[e.id].provider,"source_index_row":frames[e.id].index,"raw_sha256":hashes[which],"display_branches":order[which],"explanations":e.rays,"interpretations":e.candidates,"disk_area":frames[e.id].area.json()}))).collect::<Vec<_>>();
            serde_json::to_writer(
                &mut events,
                &json!({"movie_start_frame":from,"movie_end_frame":video_frame,"source":clip.key.0,"epoch":clip.key.1,"source_ns":exposure.ns.to_string(),"source_seconds":(exposure.ns-clip.first_ns) as f64*1e-9,"eyes":eyes}),
            )?;
            writeln!(events)?;
            last_ns = Some(exposure.ns);
        }
        chapters.push(json!({"recording":number,"source":clip.key.0,"epoch":clip.key.1,"path":source.path,"raw_source_path":raw_path,"title":format!("{number:02} {basename} (epoch {})",clip.key.1),"start_frame":start,"end_frame":video_frame,"tone_raw10":tones,"plot_bounds_iris_radii":bounds}));
        eprintln!(
            "MOVIE {}/{}: source {} complete, {} ambiguous exposures included, {:.1}s encoded",
            ci + 1,
            clips.len(),
            clip.key.0,
            written.len(),
            video_frame as f64 / FPS as f64
        );
    }
    drop(sink);
    let status = encoder.wait()?;
    if !status.success() {
        return Err(format!("video encoding failed: {status}").into());
    }
    events.flush()?;
    if written != expected {
        return Err("not every ambiguous frame was rendered".into());
    }
    let mut metadata =
        String::from(";FFMETADATA1\ntitle=Ambiguous ROI frames - both interpretations\n");
    for chapter in &chapters {
        metadata.push_str(&format!(
            "[CHAPTER]\nTIMEBASE=1/{FPS}\nSTART={}\nEND={}\ntitle={}\n",
            chapter["start_frame"],
            chapter["end_frame"],
            chapter["title"].as_str().unwrap()
        ));
    }
    fs::write(out.join("chapters.ffmetadata"), metadata)?;
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&encoded)
        .args(["-i"])
        .arg(out.join("chapters.ffmetadata"))
        .args([
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-codec",
            "copy",
            "-movflags",
            "+faststart",
        ])
        .arg(out.join("ambiguous-frames.mp4"))
        .status()?;
    if !status.success() {
        return Err("chapter mux failed".into());
    }
    fs::remove_file(encoded)?;
    let result = json!({"schema":"buttercup-ambiguous-roi-movie-v1","binary":input,"binary_sha256":archive::digest(&fs::read(input)?),"replay":replay_dir,"summary_sha256":archive::digest(&fs::read(replay_dir.join("summary.json"))?),"recordings":nsources,"clock_groups":clips.len(),"ambiguous_eye_frames":written.len(),"ambiguous_by_provider":selected_providers,"all_ambiguous_records_rendered_exactly_once":true,"source_exposures_including_context":nevents,"fps":FPS,"video_frames":video_frame,"duration_seconds":video_frame as f64/FPS as f64,"chapters":chapters,"notes":["Source index hashes/crops/sequences checked. Each displayed RAW exposure has a recorded SHA-256.","Two eyes appear together only at exactly equal source timestamps. No stale fit or other-eye RAW is substituted.","All 2473 ambiguity classifications are unchanged from the completed replay. Context is explicitly classified separately.","Colors track nearest prior normals for presentation only; original candidate indices are retained in movie-frames.jsonl.","Native RAW10 mosaic in monochrome, no 4x4 averaging, fixed brightness range for each eye/clock group.","CFR presentation is 2x slower within 0.12-0.6-second holds; source gaps are shortened and labeled.","Regions, signs and areas are historical diagnostic estimates. No measured screen, independent scale or physical sign truth is claimed."]});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    fs::write(out.join("README.md"),format!("# Ambiguous ROI review movie\n\n[ambiguous-frames.mp4](ambiguous-frames.mp4) includes all {} ambiguous eye-frame records from {nsources} recordings, with short neighboring context. Runtime {:.2} minutes, 1920x1080 at {FPS} fps. Recording/clock chapters support seeking.\n\nWhite curves are the original fitted ellipses; cyan/pink arrows are their two nominal 3-D circle normals. Green boxes are inferred competing focus regions, not the measured monitor. Both directions may fit different regions. Candidate labels follow nearest normal continuity for display, without selecting a correct sign.\n\nThe source timestamp is displayed. Source gaps are shortened and called out; small intervals play 2x slower. Repeated presentation frames are not new observations. Missing other-eye exposures/ellipses remain absent. RAW10 is shown at native pixel resolution with fixed per-recording-eye contrast and without 4x4 averaging.\n\nAll classifications/geometry match the saved replay. Most fits are historical virtual-contact reconstructions, explicitly labeled; SAM fits are marked separately. Post-affine disk area is unnormalized pi*a*a in pixels squared. No independent scale or verified gaze truth is available.\n\nsummary.json and movie-frames.jsonl contain the exact recording, clock, source index, RAW hashes, area, candidate identity and frame-to-video timeline.\n",written.len(),video_frame as f64/FPS as f64/60.))?;
    eprintln!(
        "MOVIE DONE {} frames, {:.2}s, all {} ambiguous records covered",
        video_frame,
        video_frame as f64 / FPS as f64,
        written.len()
    );
    Ok(())
}
