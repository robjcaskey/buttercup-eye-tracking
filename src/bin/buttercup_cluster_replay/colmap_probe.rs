//! Bounded, offline standard COLMAP baseline on source-matched RAW eye crops.
//! Classical CPU SIFT; full-ROI versus SAM sclera-center feature masks.
use super::{json, quantiles, read_rows, BundleSource, Error, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, process::Command, time::Instant};
#[path = "../../bootstrapability.rs"]
#[allow(dead_code)]
mod boot;
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
#[path = "../../sclera_splat_input_recipe.rs"]
mod recipe;
use canvas::*;
#[path = "colmap_playback.rs"]
mod playback;
type Result<T> = std::result::Result<T, Error>;
const W: usize = 420;
const H: usize = 280;
// Existing nominal sensor intrinsics, NOT a measured calibration. Pixel centers
// in this repository are integer; COLMAP uses centers at (0.5, 0.5).
const FOCAL: f64 = 4000.;
const PRINCIPAL: [i64; 2] = [4000, 3000];
#[derive(Clone, Copy)]
enum Matching {
    Exhaustive,
    Sequential,
}
impl Matching {
    fn name(self) -> &'static str {
        match self {
            Self::Exhaustive => "exhaustive",
            Self::Sequential => "sequential",
        }
    }
    fn max_frames(self) -> usize {
        match self {
            Self::Exhaustive => 128,
            Self::Sequential => 512,
        }
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn write(path: impl AsRef<Path>, v: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}
fn num(v: &Value) -> u64 {
    v.as_u64().expect("integer acquisition metadata")
}
struct Frame {
    row: Value,
    bgra: Vec<u8>,
    mask: Vec<u8>,
}
impl Frame {
    fn origin(&self) -> [i64; 2] {
        [
            num(&self.row["source"]["sensor_x"]) as i64,
            num(&self.row["source"]["sensor_y"]) as i64,
        ]
    }
}
#[derive(Clone, Copy)]
struct Extent {
    origin: [i64; 2],
    width: usize,
    height: usize,
}
impl Extent {
    fn from_origins(origins: &[[i64; 2]]) -> Self {
        // Include the nominal optical center so COLMAP's camera validity gates
        // remain meaningful. Padding is masked out, not invented scene content.
        let lo = std::array::from_fn(|k| {
            origins
                .iter()
                .map(|p| p[k])
                .min()
                .unwrap()
                .min(PRINCIPAL[k] - 16)
        });
        let hi: [i64; 2] = std::array::from_fn(|k| {
            origins
                .iter()
                .map(|p| p[k] + [W, H][k] as i64)
                .max()
                .unwrap()
                .max(PRINCIPAL[k] + 16)
        });
        Self {
            origin: lo,
            width: (hi[0] - lo[0]) as usize,
            height: (hi[1] - lo[1]) as usize,
        }
    }
    fn offset(&self, origin: [i64; 2]) -> [usize; 2] {
        std::array::from_fn(|k| (origin[k] - self.origin[k]) as usize)
    }
    fn camera_params(&self) -> String {
        format!(
            "{FOCAL},{FOCAL},{},{}",
            (PRINCIPAL[0] - self.origin[0]) as f64 + 0.5,
            (PRINCIPAL[1] - self.origin[1]) as f64 + 0.5
        )
    }
}
fn png(path: &Path, data: &[u8], w: usize, h: usize) -> Result<()> {
    let mut c = Canvas::new(w, h)?;
    c.image(data, w, h, 0., 0., w as f64, h as f64);
    c.png(path)
}
fn execute(dir: &Path, label: &str, arguments: &[String]) -> Result<Value> {
    let start = Instant::now();
    let output = Command::new("/usr/bin/colmap")
        .args(arguments)
        .env("QT_QPA_PLATFORM", "offscreen")
        .env("OMP_NUM_THREADS", "4")
        .output()?;
    fs::write(dir.join(format!("{label}.stdout.log")), &output.stdout)?;
    fs::write(dir.join(format!("{label}.stderr.log")), &output.stderr)?;
    let no_model = label == "mapper"
        && expected_no_model(
            output.status.code(),
            &String::from_utf8_lossy(&output.stderr),
        );
    let receipt = json!({"program":"/usr/bin/colmap","arguments":arguments,"status":output.status.code(),"no_model":no_model,"seconds":start.elapsed().as_secs_f64()});
    write(dir.join(format!("{label}.command.json")), &receipt)?;
    if !output.status.success() && !no_model {
        return Err(format!("COLMAP {label} failed: {}", dir.display()).into());
    }
    Ok(receipt)
}
fn expected_no_model(code: Option<i32>, stderr: &str) -> bool {
    code == Some(1)
        && stderr.contains("failed to create sparse model")
        && (stderr.contains("No good initial image pair found")
            || stderr.contains("No images with matches found"))
}
fn sql(db: &Path, query: &str) -> Result<Vec<Value>> {
    let output = Command::new("/usr/bin/sqlite3")
        .arg("-readonly")
        .arg("-json")
        .arg(db)
        .arg(query)
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string().into());
    }
    if output.stdout.is_empty() {
        Ok(Vec::new())
    } else {
        Ok(serde_json::from_slice(&output.stdout)?)
    }
}
fn unhex(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err("odd SQLite blob hex".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| Ok(u8::from_str_radix(&s[i..i + 2], 16)?))
        .collect()
}
fn keypoints(db: &Path) -> Result<BTreeMap<String, Vec<[f64; 2]>>> {
    let mut result = BTreeMap::new();
    for row in sql(db,"SELECT images.name,keypoints.rows,keypoints.cols,hex(keypoints.data) AS data FROM images JOIN keypoints USING(image_id) ORDER BY images.name")? {
        let bytes=unhex(row["data"].as_str().ok_or("keypoint blob")?)?;
        let cols=num(&row["cols"]) as usize;
        let rows=num(&row["rows"]) as usize;
        if cols<2 || bytes.len()!=rows*cols*4 { return Err("invalid keypoint dimensions".into()); }
        let points=bytes.chunks_exact(cols*4).map(|chunk| std::array::from_fn(|k| f32::from_le_bytes(chunk[4*k..4*k+4].try_into().unwrap()) as f64)).collect();
        result.insert(row["name"].as_str().ok_or("image name")?.to_string(),points);
    }
    Ok(result)
}
fn geometry(db: &Path) -> Result<Vec<Value>> {
    sql(db,"SELECT a.name AS first,b.name AS second,g.rows AS inliers,g.config,hex(g.data) AS data FROM two_view_geometries g JOIN images a ON a.image_id=(g.pair_id-g.pair_id%2147483647)/2147483647 JOIN images b ON b.image_id=g.pair_id%2147483647 WHERE g.rows>0 ORDER BY g.rows DESC,g.pair_id")
}
fn reconstruct(
    dir: &Path,
    images: &Path,
    mask: &Path,
    extent: Extent,
    matching: Matching,
) -> Result<Value> {
    fs::create_dir(dir)?;
    fs::create_dir(dir.join("sparse"))?;
    let db = dir.join("database.db");
    let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut commands = Vec::new();
    commands.push(execute(
        dir,
        "features",
        &args(&[
            "feature_extractor",
            "--database_path",
            &db.to_string_lossy(),
            "--image_path",
            &images.to_string_lossy(),
            "--ImageReader.mask_path",
            &mask.to_string_lossy(),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            &extent.camera_params(),
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.num_threads",
            "4",
            "--SiftExtraction.max_image_size",
            "3200",
        ]),
    )?);
    let mut matcher_args = args(&[
        match matching {
            Matching::Exhaustive => "exhaustive_matcher",
            Matching::Sequential => "sequential_matcher",
        },
        "--database_path",
        &db.to_string_lossy(),
        "--SiftMatching.use_gpu",
        "0",
        "--SiftMatching.num_threads",
        "4",
    ]);
    if matches!(matching, Matching::Sequential) {
        // All acquisition frames remain inputs. Match nearby frames and
        // exponentially spaced wider baselines without a learned vocabulary.
        matcher_args.extend(args(&[
            "--SequentialMatching.overlap",
            "20",
            "--SequentialMatching.quadratic_overlap",
            "1",
            "--SequentialMatching.loop_detection",
            "0",
        ]));
    }
    commands.push(execute(dir, "matches", &matcher_args)?);
    commands.push(execute(
        dir,
        "mapper",
        &args(&[
            "mapper",
            "--database_path",
            &db.to_string_lossy(),
            "--image_path",
            &images.to_string_lossy(),
            "--output_path",
            &dir.join("sparse").to_string_lossy(),
            "--Mapper.num_threads",
            "4",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ]),
    )?);
    let mut models = Vec::new();
    for entry in fs::read_dir(dir.join("sparse"))? {
        let model = entry?.path();
        if !model.join("images.bin").exists() {
            continue;
        }
        let name = model.file_name().unwrap().to_string_lossy();
        execute(
            dir,
            &format!("model-{name}"),
            &args(&[
                "model_converter",
                "--input_path",
                &model.to_string_lossy(),
                "--output_path",
                &model.to_string_lossy(),
                "--output_type",
                "TXT",
            ]),
        )?;
        execute(
            dir,
            &format!("analysis-{name}"),
            &args(&["model_analyzer", "--path", &model.to_string_lossy()]),
        )?;
        let lines = fs::read_to_string(model.join("images.txt"))?;
        // COLMAP images.txt has a pose row and an observation row per image;
        // preserve empty observation rows rather than counting whitespace lines.
        let registered = lines
            .lines()
            .filter(|s| !s.starts_with('#'))
            .enumerate()
            .filter(|(i, s)| i % 2 == 0 && !s.trim().is_empty())
            .count();
        let points = fs::read_to_string(model.join("points3D.txt"))?
            .lines()
            .filter(|s| !s.starts_with('#') && !s.trim().is_empty())
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let errors = points
            .iter()
            .filter_map(|s| s.split_whitespace().nth(7)?.parse::<f64>().ok())
            .collect();
        models.push(json!({"path":model,"registered_images":registered,"points3d":points.len(),"reprojection_error_px":quantiles(errors)}));
    }
    let features=sql(&db,"SELECT images.name,keypoints.rows AS features FROM images JOIN keypoints USING(image_id) ORDER BY images.name")?;
    let pairs = geometry(&db)?;
    let raw_matches=sql(&db,"SELECT count(*) AS pairs,coalesce(sum(rows),0) AS matches,max(rows) AS max_pair_matches FROM matches WHERE rows>0")?;
    let report = json!({"commands":commands,"features":features,"feature_count":features.iter().map(|v|num(&v["features"])).sum::<u64>(),"descriptor_matches":raw_matches,
        "verified_pairs":pairs.len(),"max_pair_inliers":pairs.iter().map(|v|num(&v["inliers"])).max().unwrap_or(0),
        "models":models,"pairwise_geometry":pairs.iter().map(|v|{let mut v=v.clone();v.as_object_mut().unwrap().remove("data");v}).collect::<Vec<_>>()});
    write(dir.join("report.json"), &report)?;
    Ok(report)
}
fn review(out: &Path, eye: u64, frames: &[Frame], extent: Extent) -> Result<()> {
    let full_db = out.join(format!("eye-{eye}/full/database.db"));
    let sclera_db = out.join(format!("eye-{eye}/sclera/database.db"));
    let full = keypoints(&full_db)?;
    let has_sclera = sclera_db.exists();
    let sclera = if has_sclera {
        keypoints(&sclera_db)?
    } else {
        BTreeMap::new()
    };
    for (index, frame) in frames.iter().enumerate() {
        let name = format!("{index:04}.png");
        let mut c = Canvas::new(1284, 370)?;
        c.clear();
        c.text(
            10.,
            24.,
            20.,
            WHITE,
            &format!(
                "COLMAP CPU SIFT | eye {eye} | source sequence {}",
                frame.row["source"]["sequence"]
            ),
        );
        let offset = extent.offset(frame.origin());
        for (col, title, points) in [
            (0, "Native RAW preview", None),
            (1, "Full-ROI feature centers", full.get(&name)),
            (
                2,
                if has_sclera {
                    "Sclera-mask feature centers"
                } else {
                    "Sclera-only pass not run"
                },
                sclera.get(&name),
            ),
        ] {
            let x = col as f64 * 428.;
            c.text(x + 5., 49., 16., WHITE, title);
            c.image(&frame.bgra, W, H, x, 59., W as f64, H as f64);
            if col == 2 {
                // Darken unsupported pixels without changing the actual input
                // to SIFT. These are unreviewed SAM proposals, not human truth.
                let mut shade = frame.bgra.clone();
                for (i, pixel) in shade.chunks_exact_mut(4).enumerate() {
                    if frame.mask[i] < 179 {
                        for color in &mut pixel[..3] {
                            *color = (*color as f64 * 0.2) as u8;
                        }
                    }
                }
                c.image(&shade, W, H, x, 59., W as f64, H as f64);
            }
            if let Some(points) = points {
                for p in points {
                    c.dot(
                        x + p[0] - offset[0] as f64 - 0.5,
                        59. + p[1] - offset[1] as f64 - 0.5,
                        2.4,
                        if col == 2 { PINK } else { CYAN },
                        false,
                    );
                }
                c.text(
                    x + 5.,
                    361.,
                    14.,
                    MUTED,
                    &format!(
                        "{} SIFT descriptors; centers only, not vessel labels",
                        points.len()
                    ),
                );
            }
        }
        c.png(&out.join(format!("eye-{eye}-{index:02}.png")))?;
    }
    for (variant, db, points) in [("full", &full_db, &full), ("sclera", &sclera_db, &sclera)] {
        if !db.exists() {
            continue;
        }
        for (rank, pair) in geometry(db)?.iter().take(3).enumerate() {
            let first = pair["first"].as_str().ok_or("first image")?;
            let second = pair["second"].as_str().ok_or("second image")?;
            let ia: usize = first.trim_end_matches(".png").parse()?;
            let ib: usize = second.trim_end_matches(".png").parse()?;
            let a = &frames[ia];
            let b = &frames[ib];
            let oa = extent.offset(a.origin());
            let ob = extent.offset(b.origin());
            let mut c = Canvas::new(864, 376)?;
            c.clear();
            c.text(
                10.,
                25.,
                19.,
                WHITE,
                &format!(
                    "{variant}: {} geometric inliers | sequences {} -> {}",
                    pair["inliers"], a.row["source"]["sequence"], b.row["source"]["sequence"]
                ),
            );
            c.image(&a.bgra, W, H, 0., 50., W as f64, H as f64);
            c.image(&b.bgra, W, H, 444., 50., W as f64, H as f64);
            let bytes = unhex(pair["data"].as_str().ok_or("match blob")?)?;
            for pair in bytes.chunks_exact(8) {
                let i = u32::from_le_bytes(pair[..4].try_into().unwrap()) as usize;
                let j = u32::from_le_bytes(pair[4..].try_into().unwrap()) as usize;
                let pa = points[first][i];
                let pb = points[second][j];
                let pa = [pa[0] - oa[0] as f64 - 0.5, 50. + pa[1] - oa[1] as f64 - 0.5];
                let pb = [
                    444. + pb[0] - ob[0] as f64 - 0.5,
                    50. + pb[1] - ob[1] as f64 - 0.5,
                ];
                c.line(pa, pb, 0.6, CYAN);
                c.dot(pa[0], pa[1], 2., PINK, true);
                c.dot(pb[0], pb[1], 2., PINK, true);
            }
            c.text(
                8.,
                359.,
                14.,
                MUTED,
                "Pairwise geometric inliers are not proof of correct vessel identity or 3D shape.",
            );
            c.png(&out.join(format!("eye-{eye}-{variant}-matches-{rank}.png")))?;
        }
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("--colmap-probe VERIFIED_SCLERA_INPUTS NEW_OUTPUT_DIR".into());
    }
    let input = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("new directory beneath checked outputs required".into());
    }
    let source = boot::current_source(Path::new("."))?;
    let summary: Value = serde_json::from_slice(&fs::read(input.join("summary.json"))?)?;
    let manifest = fs::read(input.join("frames.jsonl"))?;
    if summary["complete"] != true
        || summary["device"] != "cpu"
        || summary["preparation_recipe_sha256"] != recipe::stamp()?
        || summary["frames_sha256"] != hash(&manifest)
    {
        return Err("completed, unchanged CPU mask preparation required".into());
    }
    let info: Value = serde_json::from_slice(&fs::read(input.join("inputs.json"))?)?;
    let bundle = BundleSource::open(Path::new(info["bundle"].as_str().ok_or("bundle path")?))?;
    fs::create_dir(out)?;
    let graph = json!({"schema":boot::SCHEMA,"source":source,"targets":["colmap_probe"],"nodes":[
        {"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]},
        {"id":"raw","kind":"raw","sha256":hash(&fs::read(input.join("raw-inventory.json"))?),"dependencies":[]},
        {"id":"sam31","kind":"sam3","sha256":info["teacher"]["checkpoint_sha256"],"dependencies":[]},
        {"id":"export","kind":"export","sha256":info["teacher"]["model_sha256"],"dependencies":["sam31","source"]},
        {"id":"masks","kind":"derived_data","sha256":hash(&manifest),"dependencies":["raw","source","export"]},
        {"id":"colmap_probe","kind":"evaluation","planned":true,"sha256":null,"dependencies":["raw","source","masks"]}]});
    let parsed: boot::Manifest = serde_json::from_value(graph.clone())?;
    write(
        out.join("bootstrap-preflight.json"),
        &boot::validate(&parsed, &source).map_err(|e| format!("COLMAP preflight: {e:?}"))?,
    )?;
    write(out.join("bootstrap-graph.json"), &graph)?;
    write(out.join("mask-provenance.json"), &info)?;
    fs::copy(
        input.join("raw-inventory.json"),
        out.join("raw-inventory.json"),
    )?;
    let mut groups = BTreeMap::<u64, Vec<Frame>>::new();
    for row in read_rows(&input.join("frames.jsonl"))? {
        let s = &row["source"];
        let packed = bundle.read_range(
            s["stream"].as_str().ok_or("RAW stream")?,
            num(&s["offset"]),
            num(&s["length"]) as usize,
        )?;
        let mask = fs::read(input.join(row["mask"].as_str().ok_or("mask")?))?;
        if num(&s["width"]) != W as u64
            || num(&s["height"]) != H as u64
            || mask.len() != W * H
            || row["raw_sha256"] != hash(&packed)
            || row["mask_sha256"] != hash(&mask)
            || row["prompt"] != "exposed white sclera"
        {
            return Err("source RAW or mask identity mismatch".into());
        }
        let raw = buttercup_eye_tracking::raw10::try_unpack_raw10(
            &packed,
            W,
            H,
            num(&s["stride"]) as usize,
        )?;
        let pixels = super::raw_preview::color_preview(
            &raw,
            W,
            H,
            num(&s["sensor_x"]) as u32,
            num(&s["sensor_y"]) as u32,
            100,
            None,
        );
        let bgra = pixels
            .iter()
            .flat_map(|p| [*p as u8, (p >> 8) as u8, (p >> 16) as u8, 255])
            .collect();
        groups
            .entry(num(&s["eye_id"]))
            .or_default()
            .push(Frame { row, bgra, mask });
    }
    let reports = process_groups(&groups, out, true, Matching::Exhaustive)?;
    let version = Command::new("/usr/bin/colmap").arg("-h").output()?;
    if boot::current_source(Path::new("."))? != source {
        return Err("source changed during COLMAP probe".into());
    }
    write(
        out.join("report.json"),
        &json!({"schema":"buttercup-colmap-probe-v1","source":source,"input":input,"eyes":reports,
        "colmap_version":String::from_utf8_lossy(&version.stdout),"colmap_sha256":hash(&fs::read("/usr/bin/colmap")?),
        "runner_sha256":hash(&fs::read(std::env::current_exe()?)?),"device":"cpu","intrinsics":"nominal fx=fy=4000, cx=4000, cy=3000, zero distortion; fixed during bundle adjustment",
        "mask_policy":"full: 8px ROI boundary; sclera: same plus 4px-eroded SAM mask >=179; feature centers only, descriptors may extend outside",
        "image_policy":"native RAW10 unpack, phase-correct native RGB preview, lossless PNG, no rescaling; padded in sensor coordinates and includes nominal optical center; outside ROI excluded",
        "limitations":["Offline structure from motion, not a run of real-time visual SLAM","Two eyes are separate objects and separate reconstructions","12-frame intervals previously inspected; Rob only; no untouched test","Unreviewed SAM masks include known reflection failures; not anatomical ground truth","Fixed nominal intrinsics, no measured scale or 3D vessel labels","Lids, skin, globe and reflections do not form one rigid scene","Pairwise geometric verification does not prove correct identity or 3D reconstruction"],
        "sn_feida":null,"human_label_localization_error":null,"measured_3d_error":null}),
    )?;
    println!("COLMAP probe complete: {}", out.display());
    Ok(())
}
fn process_groups(
    groups: &BTreeMap<u64, Vec<Frame>>,
    out: &Path,
    use_sclera: bool,
    matching: Matching,
) -> Result<Vec<Value>> {
    // Independent eye scenes are reconstructed concurrently, four COLMAP
    // workers per eye. No correspondences are created between different eyes.
    let results = std::thread::scope(|scope| {
        let workers = groups
            .iter()
            .map(|(eye, frames)| {
                scope.spawn(move || {
                    process_eye(*eye, frames, out, use_sclera, matching).map_err(|e| e.to_string())
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| "COLMAP eye worker panicked".to_string())?
            })
            .collect::<std::result::Result<Vec<_>, String>>()
    });
    results.map_err(Into::into)
}
fn process_eye(
    eye: u64,
    frames: &[Frame],
    out: &Path,
    use_sclera: bool,
    matching: Matching,
) -> Result<Value> {
    let dir = out.join(format!("eye-{eye}"));
    fs::create_dir(&dir)?;
    for child in ["images", "full-masks", "sclera-masks"] {
        fs::create_dir(dir.join(child))?;
    }
    let extent = Extent::from_origins(&frames.iter().map(Frame::origin).collect::<Vec<_>>());
    if extent.width > 3200 || extent.height > 3200 {
        return Err("canvas would trigger SIFT resizing; explicit adaptation needed".into());
    }
    let mut inventory = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let offset = extent.offset(frame.origin());
        let mut pixels = vec![0u8; extent.width * extent.height * 4];
        let mut full = pixels.clone();
        let mut sclera = pixels.clone();
        for buffer in [&mut pixels, &mut full, &mut sclera] {
            for pixel in buffer.chunks_exact_mut(4) {
                pixel[3] = 255;
            }
        }
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let j = (y + offset[1]) * extent.width + x + offset[0];
                pixels[4 * j..4 * j + 4].copy_from_slice(&frame.bgra[4 * i..4 * i + 4]);
                // Same ROI boundary exclusion for both variants. COLMAP masks
                // keypoint centers; descriptor support can extend past a mask.
                if x >= 8 && x < W - 8 && y >= 8 && y < H - 8 {
                    full[4 * j..4 * j + 3].fill(255);
                    if (-4..=4).all(|dy| {
                        (-4..=4).all(|dx| {
                            frame.mask[(y as i32 + dy) as usize * W + (x as i32 + dx) as usize]
                                >= 179
                        })
                    }) {
                        sclera[4 * j..4 * j + 3].fill(255);
                    }
                }
            }
        }
        let name = format!("{index:04}.png");
        png(
            &dir.join("images").join(&name),
            &pixels,
            extent.width,
            extent.height,
        )?;
        png(
            &dir.join("full-masks").join(format!("{name}.png")),
            &full,
            extent.width,
            extent.height,
        )?;
        png(
            &dir.join("sclera-masks").join(format!("{name}.png")),
            &sclera,
            extent.width,
            extent.height,
        )?;
        inventory.push(json!({"name":name,"source":frame.row["source"],"raw_sha256":frame.row["raw_sha256"],"mask_sha256":frame.row["mask_sha256"],"offset":offset,
                "image_sha256":hash(&fs::read(dir.join("images").join(&name))?)}));
    }
    write(
        dir.join("inventory.json"),
        &json!({"images":inventory,"canvas_origin":extent.origin,"canvas_size":[extent.width,extent.height],"camera_params":extent.camera_params(),"intrinsics_role":"fixed nominal pinhole, no measured calibration"}),
    )?;
    let mut result = json!({"eye_id":eye,"frames":frames.len()});
    for variant in ["full", "sclera"] {
        if variant == "sclera" && !use_sclera {
            continue;
        }
        eprintln!("COLMAP eye {eye} {variant}: {} RAW frames", frames.len());
        result[variant] = reconstruct(
            &dir.join(variant),
            &dir.join("images"),
            &dir.join(format!("{variant}-masks")),
            extent,
            matching,
        )?;
    }
    review(out, eye, frames, extent)?;
    Ok(result)
}
pub fn raw_run(args: &[String]) -> Result<()> {
    if !(5..=6).contains(&args.len()) {
        return Err(
            "--colmap-raw-probe BUNDLE NEW_OUTPUT_DIR STEP_PER_EYE [exhaustive|sequential]".into(),
        );
    }
    let matching = match args.get(5).map(String::as_str).unwrap_or("exhaustive") {
        "exhaustive" => Matching::Exhaustive,
        "sequential" => Matching::Sequential,
        _ => return Err("matching must be exhaustive or sequential".into()),
    };
    let step: usize = args[4].parse()?;
    if step == 0 {
        return Err("nonzero sampling step required".into());
    }
    let input = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("new checked outputs directory required".into());
    }
    let source = boot::current_source(Path::new("."))?;
    let bundle = BundleSource::open(input)?;
    let rows = bundle.read_entry("frames.jsonl")?;
    let mut groups = BTreeMap::<u64, Vec<Frame>>::new();
    let mut counts = BTreeMap::<u64, usize>::new();
    for line in std::str::from_utf8(&rows)?.lines() {
        let s: Value = serde_json::from_str(line)?;
        let eye = num(&s["eye_id"]);
        if !(1..=2).contains(&eye) {
            return Err("native subject eye IDs 1 and 2 required".into());
        }
        let count = counts.entry(eye).or_default();
        let take = *count % step == 0;
        *count += 1;
        if !take {
            continue;
        }
        if num(&s["width"]) != W as u64
            || num(&s["height"]) != H as u64
            || s["pixel_format"] != "RAW10_LE40_1X1"
        {
            return Err("native 420x280 RAW10 required".into());
        }
        let packed = bundle.read_range(
            s["stream"].as_str().ok_or("stream")?,
            num(&s["offset"]),
            num(&s["length"]) as usize,
        )?;
        let raw = buttercup_eye_tracking::raw10::try_unpack_raw10(
            &packed,
            W,
            H,
            num(&s["stride"]) as usize,
        )?;
        let pixels = super::raw_preview::color_preview(
            &raw,
            W,
            H,
            num(&s["sensor_x"]) as u32,
            num(&s["sensor_y"]) as u32,
            100,
            None,
        );
        groups.entry(eye).or_default().push(Frame {
            row: json!({"source":s,"raw_sha256":hash(&packed),"mask_role":"none"}),
            bgra: pixels
                .iter()
                .flat_map(|p| [*p as u8, (p >> 8) as u8, (p >> 16) as u8, 255])
                .collect(),
            mask: vec![0; W * H],
        });
    }
    if groups.is_empty()
        || groups
            .values()
            .any(|v| v.len() < 2 || v.len() > matching.max_frames())
    {
        return Err(format!(
            "bounded {} run requires 2..{} sampled frames per eye",
            matching.name(),
            matching.max_frames()
        )
        .into());
    }
    fs::create_dir(out)?;
    let reports = process_groups(&groups, out, false, matching)?;
    if boot::current_source(Path::new("."))? != source {
        return Err("source changed during full RAW probe".into());
    }
    let version = Command::new("/usr/bin/colmap").arg("-h").output()?;
    write(
        out.join("report.json"),
        &json!({"schema":"buttercup-colmap-raw-probe-v1","source":source,"bundle":input,"step_per_eye":step,"available_frames_per_eye":counts,
        "colmap_version":String::from_utf8_lossy(&version.stdout),"colmap_sha256":hash(&fs::read("/usr/bin/colmap")?),"runner_sha256":hash(&fs::read(std::env::current_exe()?)?),
        "device":"cpu","eyes":reports,"matching":matching.name(),"intrinsics":"fixed nominal fx=fy=4000 cx=4000 cy=3000, not measured; zero distortion",
        "input_policy":"every Nth acquisition frame per eye, no model/target-based selection; native RAW decoded to phase-correct RGB and lossless PNG; sensor-coordinate padding masked out",
        "limitations":["Full ROI only; no sclera segmentation or anatomical labels supplied in this extended run","Eyes reconstructed separately; not stereo","Single-user Rob development recordings","Nominal unmeasured intrinsics and arbitrary reconstruction scale","Lids, skin, eye and reflections have distinct motion","No proof of vessel identity, gaze accuracy or anatomical 3D depth"],"sn_feida":null,"human_label_error":null,"measured_3d_error":null}),
    )?;
    println!("COLMAP full RAW probe complete: {}", out.display());
    Ok(())
}
/// Display COLMAP's actual triangulated observations, never all detected SIFT
/// centers or a sphere prior. The largest component is selected by point count.
pub fn model_review(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("--colmap-model-review RAW_PROBE_DIR NEW_OUTPUT_DIR".into());
    }
    let input = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("new checked output required".into());
    }
    let report: Value = serde_json::from_slice(&fs::read(input.join("report.json"))?)?;
    if report["schema"] != "buttercup-colmap-raw-probe-v1" {
        return Err("completed full RAW probe required".into());
    }
    let bundle = BundleSource::open(Path::new(report["bundle"].as_str().ok_or("bundle")?))?;
    fs::create_dir(out)?;
    let mut models = Vec::new();
    for eye in report["eyes"].as_array().ok_or("eye reports")? {
        let id = num(&eye["eye_id"]);
        let inventory: Value =
            serde_json::from_slice(&fs::read(input.join(format!("eye-{id}/inventory.json")))?)?;
        let chosen = eye["full"]["models"]
            .as_array()
            .ok_or("models")?
            .iter()
            .max_by_key(|m| num(&m["points3d"]));
        let mut observations = BTreeMap::<String, Vec<[f64; 2]>>::new();
        let mut points = Vec::<Vec<f64>>::new();
        let mut q = [1., 0., 0., 0.];
        if let Some(model) = chosen {
            let path = Path::new(model["path"].as_str().ok_or("model path")?);
            for line in fs::read_to_string(path.join("points3D.txt"))?
                .lines()
                .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            {
                let a = line.split_whitespace().collect::<Vec<_>>();
                let mut p = a[1..8]
                    .iter()
                    .map(|s| s.parse::<f64>())
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                p.push((a.len() - 8) as f64 / 2.);
                points.push(p);
            }
            let images = fs::read_to_string(path.join("images.txt"))?;
            let mut lines = images.lines().filter(|l| !l.starts_with('#'));
            let mut first = true;
            while let Some(pose) = lines.next() {
                if pose.trim().is_empty() {
                    return Err("empty model pose".into());
                }
                let a = pose.split_whitespace().collect::<Vec<_>>();
                if first {
                    for k in 0..4 {
                        q[k] = a[k + 1].parse()?;
                    }
                    first = false;
                }
                let data = lines
                    .next()
                    .ok_or("missing model observations")?
                    .split_whitespace()
                    .collect::<Vec<_>>();
                let mut obs = Vec::new();
                for a in data.chunks_exact(3) {
                    if a[2] != "-1" {
                        obs.push([a[0].parse()?, a[1].parse()?]);
                    }
                }
                observations.insert(a[9].to_string(), obs);
            }
        }
        for (index, frame) in inventory["images"]
            .as_array()
            .ok_or("images")?
            .iter()
            .enumerate()
        {
            let s = &frame["source"];
            let packed = bundle.read_range(
                s["stream"].as_str().ok_or("stream")?,
                num(&s["offset"]),
                num(&s["length"]) as usize,
            )?;
            if hash(&packed) != frame["raw_sha256"] {
                return Err("model review RAW identity mismatch".into());
            }
            let raw = buttercup_eye_tracking::raw10::try_unpack_raw10(
                &packed,
                W,
                H,
                num(&s["stride"]) as usize,
            )?;
            let rgb = super::raw_preview::color_preview(
                &raw,
                W,
                H,
                num(&s["sensor_x"]) as u32,
                num(&s["sensor_y"]) as u32,
                100,
                None,
            );
            let bgra = rgb
                .iter()
                .flat_map(|p| [*p as u8, (p >> 8) as u8, (p >> 16) as u8, 255])
                .collect::<Vec<_>>();
            let obs = observations.get(frame["name"].as_str().ok_or("name")?);
            let mut c = Canvas::new(864, 382)?;
            c.clear();
            c.text(
                8.,
                25.,
                19.,
                WHITE,
                &format!(
                    "COLMAP triangulated observations | eye {id} | sequence {}",
                    s["sequence"]
                ),
            );
            c.text(8., 48., 14., MUTED, "Native RAW preview");
            c.text(
                450.,
                48.,
                14.,
                if obs.is_some() { GREEN } else { ORANGE },
                &format!(
                    "{}",
                    obs.map_or("NOT REGISTERED IN THIS COMPONENT".to_string(), |p| format!(
                        "{} observations of 3D points",
                        p.len()
                    ))
                ),
            );
            c.image(&bgra, W, H, 0., 60., W as f64, H as f64);
            c.image(&bgra, W, H, 444., 60., W as f64, H as f64);
            if let Some(obs) = obs {
                for p in obs {
                    c.dot(
                        444. + p[0] - num(&frame["offset"][0]) as f64 - 0.5,
                        60. + p[1] - num(&frame["offset"][1]) as f64 - 0.5,
                        2.1,
                        PINK,
                        false,
                    );
                }
            }
            c.text(8.,366.,14.,MUTED,"Largest component only. Pink marks are observed points, not anatomical or vessel labels.");
            c.png(&out.join(format!("eye-{id}-{index:03}.png")))?;
        }
        if !points.is_empty() {
            let [w, x, y, z] = q;
            let rotated = points
                .iter()
                .map(|p| {
                    [
                        (1. - 2. * y * y - 2. * z * z) * p[0]
                            + (2. * x * y - 2. * z * w) * p[1]
                            + (2. * x * z + 2. * y * w) * p[2],
                        (2. * x * y + 2. * z * w) * p[0]
                            + (1. - 2. * x * x - 2. * z * z) * p[1]
                            + (2. * y * z - 2. * x * w) * p[2],
                        (2. * x * z - 2. * y * w) * p[0]
                            + (2. * y * z + 2. * x * w) * p[1]
                            + (1. - 2. * x * x - 2. * y * y) * p[2],
                    ]
                })
                .collect::<Vec<_>>();
            let lo: [f64; 3] =
                std::array::from_fn(|k| rotated.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min));
            let hi: [f64; 3] = std::array::from_fn(|k| {
                rotated
                    .iter()
                    .map(|p| p[k])
                    .fold(f64::NEG_INFINITY, f64::max)
            });
            let scale = 440. / (0..3).map(|k| hi[k] - lo[k]).fold(1e-9, f64::max);
            let mut c = Canvas::new(1040, 570)?;
            c.clear();
            c.text(
                12.,
                24.,
                20.,
                WHITE,
                &format!(
                    "COLMAP eye {id}: {} points | nominal calibration | arbitrary scale",
                    points.len()
                ),
            );
            for (panel, angle) in [0_f64, 0.7].iter().enumerate() {
                c.text(
                    12. + panel as f64 * 520.,
                    49.,
                    16.,
                    MUTED,
                    if panel == 0 {
                        "Reference camera orientation"
                    } else {
                        "Same points rotated 40 degrees"
                    },
                );
                for (p, color) in rotated.iter().zip(&points) {
                    let a = p[0] - (lo[0] + hi[0]) * 0.5;
                    let b = p[1] - (lo[1] + hi[1]) * 0.5;
                    let d = p[2] - (lo[2] + hi[2]) * 0.5;
                    c.dot(
                        260. + panel as f64 * 520. + scale * (angle.cos() * a + angle.sin() * d),
                        300. + scale * b,
                        1.5,
                        std::array::from_fn(|k| (45. + 210. * color[k + 3] / 255.) / 255.),
                        true,
                    );
                }
            }
            c.text(12.,553.,14.,MUTED,"Full-crop sparse reconstruction. No imposed eye surface; anatomical depth and vessel identity unverified.");
            c.png(&out.join(format!("eye-{id}-cloud.png")))?;
        }
        let first_registered_frame = observations
            .keys()
            .filter_map(|name| name.trim_end_matches(".png").parse::<usize>().ok())
            .min();
        let mut model = json!({"eye_id":id,"component":chosen,"points":points,"view_quaternion":q,
            "observed_frames":observations.len(),"sampled_frames":inventory["images"].as_array().unwrap().len(),
            "first_registered_frame":first_registered_frame,
            "component_count":eye["full"]["models"].as_array().unwrap().len()});
        playback::enrich(&mut model, &inventory)?;
        models.push(model);
    }
    let html = include_str!("colmap_viewer.html");
    fs::write(
        out.join("viewer.html"),
        html.replace("MODEL_DATA", &serde_json::to_string(&models)?),
    )?;
    write(
        out.join("models.json"),
        &json!({"input":input,"source":boot::current_source(Path::new("."))?,"models":models,"selection":"largest point count per eye; not ground-truth quality"}),
    )?;
    println!("COLMAP model review: {}", out.display());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shifted_crops_preserve_sensor_rays_and_mask_origin() {
        let origins = [[3040, 1294], [3008, 1294]];
        let extent = Extent::from_origins(&origins);
        let sensor_point = [3200, 1400];
        for origin in origins {
            let offset = extent.offset(origin);
            for k in 0..2 {
                let local = sensor_point[k] - origin[k];
                let png_center = local as f64 + offset[k] as f64 + 0.5;
                let principal = (PRINCIPAL[k] - extent.origin[k]) as f64 + 0.5;
                assert_eq!(
                    (png_center - principal) / FOCAL,
                    (sensor_point[k] - PRINCIPAL[k]) as f64 / FOCAL
                );
            }
        }
        assert!(extent.width <= 3200 && extent.height <= 3200);
        assert_eq!(
            extent.offset(origins[0])[0] - extent.offset(origins[1])[0],
            32
        );
    }
    #[test]
    fn sqlite_feature_blob_round_trip() {
        let bytes = unhex("0000803F00000040").unwrap();
        assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), 1.);
        assert_eq!(f32::from_le_bytes(bytes[4..].try_into().unwrap()), 2.);
        assert!(unhex("000").is_err());
        assert!(unhex("zz").is_err());
    }
    #[test]
    fn mapping_abstention_does_not_hide_tool_errors() {
        assert!(expected_no_model(
            Some(1),
            "No good initial image pair found.\nfailed to create sparse model"
        ));
        assert!(!expected_no_model(Some(1), "Failed to parse options"));
        assert!(!expected_no_model(
            None,
            "No good initial image pair found.\nfailed to create sparse model"
        ));
        assert!(!expected_no_model(Some(1), "failed to create sparse model"));
    }
}
