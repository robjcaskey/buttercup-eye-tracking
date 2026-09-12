//! Offline supervised patch training and source-matched evaluation. Runtime data
//! only; no camera, annotation UI, recorded gaze, or completed human ellipse.
use super::*;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

type Result<T> = std::result::Result<T, String>;
fn read(path: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn write_new(path: &Path, value: &Value) -> Result<()> {
    let root = std::fs::canonicalize("data").map_err(|e| e.to_string())?;
    let parent = std::fs::canonicalize(path.parent().ok_or("output needs parent")?)
        .map_err(|e| e.to_string())?;
    if !parent.starts_with(root) {
        return Err("runtime output must be beneath data/outputs".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer(&mut file, value).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())
}
fn point(v: &Value) -> Option<(f64, f64)> {
    Some((v[0].as_f64()?, v[1].as_f64()?))
}
fn ellipse(v: &Value) -> Option<Ellipse> {
    Some(Ellipse {
        center: point(&v["center"])?,
        major_radius: v["major_radius"].as_f64()?,
        minor_radius: v["minor_radius"].as_f64()?,
        angle: v["angle"].as_f64()?,
    })
}
fn ellipse_json(e: Ellipse) -> Value {
    json!({"center":e.center,"major_radius":e.major_radius,
    "minor_radius":e.minor_radius,"angle":e.angle})
}

struct Frame {
    metadata: Value,
    baseline: Value,
    raw: Vec<u16>,
    width: usize,
    height: usize,
    group: i64,
    shape: Option<Ellipse>,
    context: Context,
}

fn source_raw(src: &Value) -> Result<Vec<u16>> {
    let f = &src["frame"];
    let width = f["width"].as_u64().ok_or("width")? as usize;
    let height = f["height"].as_u64().ok_or("height")? as usize;
    let length = src["raw_length"]
        .as_u64()
        .filter(|n| *n <= 16 * 1024 * 1024)
        .ok_or("bounded RAW length")? as usize;
    let mut file =
        File::open(src["raw_file"].as_str().ok_or("RAW path")?).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(
        src["raw_offset"].as_u64().ok_or("RAW offset")?,
    ))
    .map_err(|e| e.to_string())?;
    let mut payload = vec![0u8; length];
    file.read_exact(&mut payload).map_err(|e| e.to_string())?;
    // Hash the bounded native payload, not a potentially huge containing tar.
    let mut hash = std::process::Command::new("sha256sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    hash.stdin
        .take()
        .ok_or("hash stdin")?
        .write_all(&payload)
        .map_err(|e| e.to_string())?;
    let digest = hash.wait_with_output().map_err(|e| e.to_string())?;
    let key = src["raw_sha256"].as_str().ok_or("RAW identity")?;
    if !digest.status.success()
        || !String::from_utf8_lossy(&digest.stdout).starts_with(&format!("{key} "))
    {
        return Err("RAW bytes changed since preparation".into());
    }
    crate::raw10::try_unpack_raw10(
        &payload,
        width,
        height,
        f["stride"].as_u64().ok_or("stride")? as usize,
    )
}
fn source_context(src: &Value) -> Context {
    let hint = &src["scale_hint"];
    let scale = (|| {
        Some(Support {
            estimate: hint["pixels_per_10mm"].as_f64()? / 10.0,
            half_width: (hint["bounds_px_per_10mm"][1].as_f64()?
                - hint["bounds_px_per_10mm"][0].as_f64()?)
                / 20.0,
        })
    })();
    Context::from_coarse_scale(scale, 4000.0)
}

/// Weak generic-rim supervision with independent scale, never apex/depth labels.
fn prepare_aux(teacher: &Path, human: &Path, output: &Path) -> Result<()> {
    let human = read(human)?;
    let receipts = human["frames"].as_array().ok_or("human frames")?;
    let mut masks = File::open(teacher.join("masks.u8")).map_err(|e| e.to_string())?;
    let mut records = vec![];
    let mut eligible = 0;
    for (index, line) in
        BufReader::new(File::open(teacher.join("records.jsonl")).map_err(|e| e.to_string())?)
            .lines()
            .enumerate()
    {
        let row: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let src = &row["input"];
        if src["student_split"] != "train"
            || row["teacher"]["raw_admitted"] != true
            || source_context(src).pixels_per_mm.is_none()
        {
            continue;
        }
        let timestamp = src["frame"]["timestamp_ns"].as_u64().ok_or("source time")?;
        if receipts.iter().any(|r| {
            r["source"]["raw_sha256"] == src["raw_sha256"]
                || r["source"]["frame"]["timestamp_ns"]
                    .as_u64()
                    .is_some_and(|t| t.abs_diff(timestamp) <= 300_000_000_000)
        }) {
            continue;
        }
        eligible += 1;
        source_raw(src)?;
        masks
            .seek(SeekFrom::Start((index * 6 * 384 * 256) as u64))
            .map_err(|e| e.to_string())?;
        let mut mask = vec![0u8; 384 * 256];
        masks.read_exact(&mut mask).map_err(|e| e.to_string())?;
        let Some(review) = crate::sam31_outer::diagnostic_fit_single_frame_mask(&mask, 384, 256)
        else {
            continue;
        };
        let scale = src["frame"]["width"].as_f64().ok_or("width")? / 384.0;
        let retained: Vec<_> = review
            .retained_points
            .iter()
            .map(|p| (p.0 * scale, p.1 * scale))
            .collect();
        let shape = Ellipse {
            center: (
                review.ellipse.center.0 * scale,
                review.ellipse.center.1 * scale,
            ),
            major_radius: review.ellipse.major_radius * scale,
            minor_radius: review.ellipse.minor_radius * scale,
            angle: review.ellipse.angle,
        };
        let observations:Vec<_>=(0..retained.len().min(24)).map(|k| {
            let p=retained[k*retained.len()/retained.len().min(24)];
            json!({"anchor":p,"targets":{"rim":p},"weights":{"rim":0.03},"occluded":[],"kind":"weak-SAM-contour"})
        }).collect();
        records.push(json!({"source":src,"group":-3,"supervision":"SAM-only-coarse-context",
            "label":format!("SAM-only-source-{index}"),"observations":observations,
            "baseline":{"input":src,"accepted":true,"source_identity_verified":true,
                "candidates":[{"baseline_ellipse":ellipse_json(shape),"baseline_retained":retained,
                    "baseline_censored":review.flat_tire_points.iter().map(|p|(p.0*scale,p.1*scale)).collect::<Vec<_>>()}]}}));
    }
    eprintln!(
        "auxiliary scale-bearing teacher frames: {} / {} eligible",
        records.len(),
        eligible
    );
    write_new(
        output,
        &json!({"schema":"buttercup-limbus-weak-context-v1","frames":records,
        "scope":"SAM generic rim only; 3% label weight; no human source-neighborhood overlap; 4000px uncalibrated focal prior"}),
    )
}
fn frames(dataset: &Path, reference: &Path) -> Result<Vec<Frame>> {
    let data = read(dataset)?;
    if data["schema"] != "buttercup-limbus-patch-dataset-v1" || data["roles"] != json!(ROLES) {
        return Err("unsupported patch dataset".into());
    }
    let mut baselines = BTreeMap::new();
    for line in BufReader::new(File::open(reference).map_err(|e| e.to_string())?).lines() {
        let row: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let key = row["input"]["raw_sha256"]
            .as_str()
            .ok_or("baseline RAW identity")?
            .to_owned();
        if baselines.insert(key, row).is_some() {
            return Err("duplicate baseline identity".into());
        }
    }
    let mut result: Vec<Frame> = data["frames"]
        .as_array()
        .ok_or("missing frames")?
        .iter()
        .map(|row| {
            let src = &row["source"];
            let f = &src["frame"];
            let key = src["raw_sha256"].as_str().ok_or("missing RAW hash")?;
            let baseline = baselines.remove(key).ok_or("missing matching baseline")?;
            if baseline["input"]["frame"] != *f || baseline["source_identity_verified"] != true {
                return Err("baseline/source metadata mismatch".into());
            }
            let width = f["width"].as_u64().ok_or("width")? as usize;
            let height = f["height"].as_u64().ok_or("height")? as usize;
            let raw = source_raw(src)?;
            let shape = ellipse(&baseline["candidates"][0]["baseline_ellipse"]);
            Ok(Frame {
                metadata: row.clone(),
                baseline,
                raw,
                width,
                height,
                group: row["group"].as_i64().ok_or("group")?,
                shape,
                context: source_context(src),
            })
        })
        .collect::<Result<_>>()?;
    if let Some(path) = std::env::var_os("BUTTERCUP_LIMBUS_AUX") {
        let aux = read(Path::new(&path))?;
        if aux["schema"] != "buttercup-limbus-weak-context-v1" {
            return Err("unknown auxiliary context dataset".into());
        }
        for row in aux["frames"].as_array().ok_or("aux frames")? {
            let src = &row["source"];
            let baseline = row["baseline"].clone();
            let shape = ellipse(&baseline["candidates"][0]["baseline_ellipse"]);
            let timestamp = src["frame"]["timestamp_ns"]
                .as_u64()
                .ok_or("aux timestamp")?;
            if result.iter().any(|f| {
                f.metadata["source"]["raw_sha256"] == src["raw_sha256"]
                    || (f.group >= 0
                        && f.metadata["source"]["frame"]["timestamp_ns"]
                            .as_u64()
                            .is_some_and(|t| t.abs_diff(timestamp) <= 300_000_000_000))
            }) {
                return Err("auxiliary RAW overlaps human source neighborhood".into());
            }
            result.push(Frame {
                metadata: row.clone(),
                baseline,
                raw: source_raw(src)?,
                width: src["frame"]["width"].as_u64().ok_or("width")? as usize,
                height: src["frame"]["height"].as_u64().ok_or("height")? as usize,
                group: -3,
                shape,
                context: source_context(src),
            });
        }
    }
    Ok(result)
}

struct Example {
    patch: Patch,
    target: Vec<f32>,
    weight: [f32; 6],
    group: i64,
}
fn examples(frames: &[Frame], augment: bool) -> Vec<Example> {
    let mut samples = vec![];
    for frame in frames {
        let Some(shape) = frame.shape else {
            continue;
        };
        let mut variants_raw = vec![frame.raw.clone()];
        if augment {
            for gain in [0.7, 1.3] {
                variants_raw.push(
                    frame
                        .raw
                        .iter()
                        .map(|x| (f64::from(*x) * gain).min(1023.0).round() as u16)
                        .collect(),
                );
            }
            let mut blurred = frame.raw.clone();
            for y in 1..frame.height - 1 {
                for x in 1..frame.width - 1 {
                    let mut sum = 0u32;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            sum += u32::from(frame.raw[(y + dy - 1) * frame.width + x + dx - 1]);
                        }
                    }
                    blurred[y * frame.width + x] = (sum / 9) as u16;
                }
            }
            variants_raw.push(blurred);
        }
        for observation in frame.metadata["observations"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let Some(anchor) = point(&observation["anchor"]) else {
                continue;
            };
            let Some(n) = normal_at(shape, anchor) else {
                continue;
            };
            let variants = if augment { 28 } else { 7 };
            for variant in 0..variants {
                let shift = (variant % 7) as f64 * 2.0 - 6.0;
                let tangent = if augment {
                    (variant / 7) as f64 - 1.5
                } else {
                    0.0
                };
                let center = (
                    anchor.0 + n.0 * shift - n.1 * tangent,
                    anchor.1 + n.1 * shift + n.0 * tangent,
                );
                let Some(patch) = extract_patch(
                    &variants_raw[variant / 7],
                    frame.width,
                    frame.height,
                    shape,
                    center,
                    if augment && variant % 2 == 0 {
                        Context::default()
                    } else {
                        frame.context
                    },
                    SAMPLE_STEP_PX,
                ) else {
                    continue;
                };
                let mut target = vec![0.0; OUTPUTS];
                let mut weight = [0.0; 6];
                for (role, name) in ROLES.iter().enumerate() {
                    if let Some(p) = point(&observation["targets"][name]) {
                        let offset = (p.0 - patch.center.0) * patch.normal.0
                            + (p.1 - patch.center.1) * patch.normal.1;
                        let bin = offset / patch.step_px + 7.5;
                        // Out of this patch is NOT a label of occlusion.
                        if !(0.0..=15.0).contains(&bin) {
                            continue;
                        }
                        let sigma = if observation["kind"] == "paired_midpoint" && role == 0 {
                            1.2
                        } else {
                            0.65
                        };
                        let mut sum = 0.0;
                        for j in 0..16 {
                            let p = (-0.5 * ((j as f64 - bin) / sigma).powi(2)).exp() as f32;
                            target[role * BINS + j] = p;
                            sum += p;
                        }
                        for j in 0..16 {
                            target[role * BINS + j] /= sum;
                        }
                        weight[role] = observation["weights"][name].as_f64().unwrap_or(1.0) as f32;
                    } else if observation["occluded"]
                        .as_array()
                        .is_some_and(|a| a.contains(&json!(name)))
                    {
                        target[role * BINS + 16] = 1.0;
                        weight[role] = 1.0;
                    }
                }
                if weight.iter().any(|w| *w > 0.0) {
                    samples.push(Example {
                        patch,
                        target,
                        weight,
                        group: frame.group,
                    });
                }
            }
        }
    }
    samples
}

struct Network {
    first: nn::Linear,
    second: nn::Linear,
}
impl Network {
    fn new(path: &nn::Path) -> Self {
        Self {
            first: nn::linear(
                path / "first",
                INPUTS as i64,
                HIDDEN as i64,
                Default::default(),
            ),
            second: nn::linear(
                path / "second",
                HIDDEN as i64,
                OUTPUTS as i64,
                Default::default(),
            ),
        }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        self.second.forward(&self.first.forward(x).relu())
    }
    fn serialize(&self, meta: Value) -> Result<Value> {
        let numbers = |x: &Tensor| -> Result<Vec<f32>> {
            Vec::<f32>::try_from(x.flatten(0, -1).to_device(Device::Cpu)).map_err(|e| e.to_string())
        };
        let mut v = json!({"architecture":ARCHITECTURE,"roles":ROLES,"inputs":INPUTS,"hidden":HIDDEN,"bins":BINS,
            "preprocess":"native-linear-tent-normal-patch-v1", "sample_step_px":SAMPLE_STEP_PX,
            "trained_optional_context":[false,false,false],
            "first_weight":numbers(&self.first.ws)?,"first_bias":numbers(self.first.bs.as_ref().unwrap())?,
            "second_weight":numbers(&self.second.ws)?,"second_bias":numbers(self.second.bs.as_ref().unwrap())?});
        v["training"] = meta;
        Ok(v)
    }
}
fn tensor_examples(examples: &[&Example], device: Device) -> (Tensor, Tensor, Tensor) {
    let x: Vec<_> = examples
        .iter()
        .flat_map(|e| e.patch.features.iter().copied())
        .collect();
    let y: Vec<_> = examples
        .iter()
        .flat_map(|e| e.target.iter().copied())
        .collect();
    let weights: Vec<_> = examples.iter().flat_map(|e| e.weight).collect();
    (
        Tensor::from_slice(&x)
            .reshape([-1, INPUTS as i64])
            .to_device(device),
        Tensor::from_slice(&y)
            .reshape([-1, 6, BINS as i64])
            .to_device(device),
        Tensor::from_slice(&weights)
            .reshape([-1, 6])
            .to_device(device),
    )
}
fn loss(logits: Tensor, target: &Tensor, weights: &Tensor) -> Tensor {
    let per_role = -(logits
        .reshape([-1, 6, BINS as i64])
        .log_softmax(-1, Kind::Float)
        * target)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
    (per_role * weights).sum(Kind::Float) / weights.sum(Kind::Float).clamp_min(1.0)
}

fn train(
    dataset: &Path,
    reference: &Path,
    output: &Path,
    epochs: usize,
    test: i64,
    validation: i64,
) -> Result<()> {
    if !(1..=1000).contains(&epochs) || test == validation {
        return Err("invalid training split/epochs".into());
    }
    if output.exists() {
        return Err("output already exists".into());
    }
    if read(dataset)?["evaluation_only"] == true {
        return Err("evaluation-only sequence cannot train the model".into());
    }
    let train_all = test == -1 && validation == -2;
    let frames = frames(dataset, reference)?;
    let examples = examples(&frames, true);
    let train: Vec<_> = examples
        .iter()
        .filter(|s| s.group != test && s.group != validation)
        .collect();
    let valid: Vec<_> = examples.iter().filter(|s| s.group == validation).collect();
    if train.is_empty() || (!train_all && valid.is_empty()) {
        return Err("empty training/validation split".into());
    }
    // Match the existing SAM/student runtime: Rust's CPU-only symbol use lets
    // the linker discard torch_cuda unless its dispatch library is loaded.
    // Keep the library alive until all tensors/stores below have been dropped.
    let _cuda_dispatch = unsafe {
        libloading::os::unix::Library::open(
            Some("libtorch_cuda.so"),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
        )
    }
    .map_err(|e| e.to_string())?;
    if !tch::Cuda::is_available() {
        return Err("CUDA required for training".into());
    }
    tch::set_num_threads(2);
    tch::manual_seed(73017);
    let device = Device::Cuda(0);
    let store = nn::VarStore::new(device);
    let network = Network::new(&store.root());
    let mut optimizer = nn::AdamW::default()
        .wd(0.02)
        .build(&store, 0.001)
        .map_err(|e| e.to_string())?;
    let (x, y, w) = tensor_examples(&train, device);
    let validation_tensors = (!train_all).then(|| tensor_examples(&valid, device));
    let context_flags = [267, 270, 272].map(|i| train.iter().any(|e| e.patch.features[i] > 0.5));
    let mut best_loss = f64::INFINITY;
    let mut best = None;
    let mut best_epoch = 0;
    let started = std::time::Instant::now();
    for epoch in 1..=epochs {
        let order = Tensor::randperm(train.len() as i64, (Kind::Int64, device));
        for start in (0..train.len()).step_by(128) {
            let ids = order.narrow(0, start as i64, 128.min(train.len() - start) as i64);
            let objective = loss(
                network.forward(&x.index_select(0, &ids)),
                &y.index_select(0, &ids),
                &w.index_select(0, &ids),
            );
            optimizer.backward_step_clip_norm(&objective, 5.0);
        }
        if epoch == 1 || epoch % 5 == 0 || epoch == epochs {
            let validation_loss = validation_tensors
                .as_ref()
                .map(|(vx, vy, vw)| {
                    tch::no_grad(|| loss(network.forward(vx), vy, vw).double_value(&[]))
                })
                .unwrap_or(0.0);
            eprintln!("epoch={epoch} validation_loss={validation_loss:.6} train_patches={} validation_patches={}",train.len(),valid.len());
            if train_all || validation_loss < best_loss {
                best_loss = validation_loss;
                best_epoch = epoch;
                best = Some(network.serialize(Value::Null)?);
            }
        }
    }
    let mut best = best.ok_or("no checkpoint")?;
    best["trained_optional_context"] = json!(context_flags);
    best["training"] = json!({"dataset":dataset,"reference":reference,"seed":73017,"epochs":epochs,"selected_epoch":best_epoch,
        "test_group":test,"validation_group":validation,"train_patches":train.len(),"validation_patches":valid.len(),
        "validation_cross_entropy":best_loss,"elapsed_seconds":started.elapsed().as_secs_f64(),
        "train_raw_sha256":frames.iter().filter(|f|f.group!=test&&f.group!=validation)
            .map(|f|f.metadata["source"]["raw_sha256"].clone()).collect::<Vec<_>>(),
        "all_reviewed_development_data":train_all,"auxiliary_context_dataset":std::env::var("BUTTERCUP_LIMBUS_AUX").ok(),
        "physical_context":if context_flags[0] {"coarse external scale; camera range derived with uncalibrated 4000px focal prior; SAM-only weak generic-rim support; no VCM supervision"}
            else {"independent scale/distance and VCM labels missing; optional physical inputs disabled"},
        "learned_context":"apparent ellipse axis ratio, radius and patch bearing; measured local RAW contrast, focus energy, illumination"});
    let model = Model::from_json(best.clone())?;
    // Exact portable CPU path must agree with the checkpoint tensor graph.
    let first = &train[0].patch;
    let cpu = model.logits(first).ok_or("portable model invalid")?;
    let tensor_logits = Tensor::from_slice(&first.features)
        .reshape([1, INPUTS as i64])
        .linear(
            &Tensor::from_slice(&model.first_weight).reshape([HIDDEN as i64, INPUTS as i64]),
            Some(&Tensor::from_slice(&model.first_bias)),
        )
        .relu()
        .linear(
            &Tensor::from_slice(&model.second_weight).reshape([OUTPUTS as i64, HIDDEN as i64]),
            Some(&Tensor::from_slice(&model.second_bias)),
        );
    let expected = Vec::<f32>::try_from(tensor_logits.flatten(0, -1)).map_err(|e| e.to_string())?;
    let max_error = cpu
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if max_error > 1e-4 || cpu.iter().any(|x| !x.is_finite()) {
        return Err("portable/Torch checkpoint mismatch".into());
    }
    best["training"]["portable_torch_max_logit_error"] = json!(max_error);
    write_new(output, &best)
}

fn distance(p: (f64, f64), e: Ellipse) -> f64 {
    let (s, c) = e.angle.sin_cos();
    let dx = p.0 - e.center.0;
    let dy = p.1 - e.center.1;
    let (x, y) = ((c * dx + s * dy).abs(), (-s * dx + c * dy).abs());
    let d =
        |t: f64| (e.major_radius * t.cos() - x).powi(2) + (e.minor_radius * t.sin() - y).powi(2);
    let best = (0usize..65)
        .min_by(|a, b| {
            d(*a as f64 * std::f64::consts::FRAC_PI_2 / 64.0)
                .total_cmp(&d(*b as f64 * std::f64::consts::FRAC_PI_2 / 64.0))
        })
        .unwrap();
    let mut lo = best.saturating_sub(1) as f64 * std::f64::consts::FRAC_PI_2 / 64.0;
    let mut hi = (best + 1).min(64) as f64 * std::f64::consts::FRAC_PI_2 / 64.0;
    for _ in 0..32 {
        let a = lo + (hi - lo) * 0.381966;
        let b = hi - (hi - lo) * 0.381966;
        if d(a) < d(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    d((lo + hi) / 2.0).sqrt()
}
fn metrics(frame: &Frame, e: Ellipse) -> Value {
    let mut errors: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for o in frame.metadata["observations"]
        .as_array()
        .into_iter()
        .flatten()
    {
        for role in ["rim", "surface_apex"] {
            if let Some(p) = point(&o["targets"][role]) {
                errors.entry(role.into()).or_default().push(distance(p, e));
            }
        }
    }
    json!(errors
        .into_iter()
        .map(|(key, values)| (
            key,
            json!({"n":values.len(),
        "rms_px":(values.iter().map(|x|x*x).sum::<f64>()/values.len() as f64).sqrt(),
        "mean_px":values.iter().sum::<f64>()/values.len() as f64})
        ))
        .collect::<BTreeMap<_, _>>())
}
fn area_diagnostic(
    frame: &Frame,
    candidate: Option<Ellipse>,
    control: Option<Ellipse>,
) -> Option<Value> {
    let relative = &frame.metadata["source"]["relative_scale_hint"];
    let (s, reference, units) = if let Some(s) = frame.context.pixels_per_mm {
        (
            s,
            json!([
                frame.metadata["source"]["clock_lineage"],
                frame.metadata["source"]["scale_hint"]["reacquisitions"]
            ]),
            "coarse_mm2",
        )
    } else if relative["raw_bytes_verified"] == true {
        let estimate = relative["scale"].as_f64()?;
        (
            Support {
                estimate,
                half_width: estimate
                    * relative["cumulative_fractional_uncertainty_heuristic"].as_f64()?,
            },
            json!([
                relative["report"],
                relative["provenance"],
                relative["reference_timestamp_ns"]
            ]),
            "reference_px2",
        )
    } else {
        return None;
    };
    if !s.valid() || s.half_width >= s.estimate {
        return None;
    }
    let area = |e: Ellipse| std::f64::consts::PI * (e.major_radius / s.estimate).powi(2);
    let bounds = |e: Ellipse| {
        [
            std::f64::consts::PI * (e.major_radius / (s.estimate + s.half_width)).powi(2),
            std::f64::consts::PI * (e.major_radius / (s.estimate - s.half_width)).powi(2),
        ]
    };
    Some(
        json!({"baseline":frame.shape.map(area),"candidate":candidate.map(area),"unmodified_refit":control.map(area),
        "units":units,"scale_reference":reference,"independent_scale":s.estimate,"half_width":s.half_width,
        "baseline_scale_only_bounds":frame.shape.map(bounds),"candidate_scale_only_bounds":candidate.map(bounds),
        "limitations":"scale-only heuristic bounds; radius and optical uncertainty excluded; no calibrated confidence claim"}),
    )
}
fn evaluate(dataset: &Path, reference: &Path, model_path: &Path, output: &Path) -> Result<()> {
    let model = Model::load(model_path)?;
    let frames = frames(dataset, reference)?;
    // Optional ablation is explicit in the report; scale used for SN-FEIDA
    // remains the same independent measurement in both arms.
    let ablate_context = std::env::var("BUTTERCUP_LIMBUS_ABLATE_CONTEXT").as_deref() == Ok("1");
    let mut results = vec![];
    for frame in &frames {
        // Annotation coordinates are consumed only after this full-frame
        // candidate is fixed. They never select a patch along the live contour.
        let field = frame.shape.map(|baseline| {
            let retained: Vec<_> = frame.baseline["candidates"][0]["baseline_retained"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(point)
                .collect();
            refine(
                &model,
                &frame.raw,
                frame.width,
                frame.height,
                baseline,
                &retained,
                if ablate_context {
                    Context::default()
                } else {
                    frame.context
                },
            )
        });
        let candidate = field.as_ref().and_then(|r| r.candidate);
        let control = field.as_ref().and_then(|r| r.unmodified_refit);
        let raw_admitted = frame.baseline["accepted"] == true;
        results.push(json!({"label":frame.metadata["label"],"source":frame.metadata["source"],"group":frame.group,
            "split":if model.manifest["training"]["test_group"]==frame.group {"test"}
                else if model.manifest["training"]["validation_group"]==frame.group {"validation"} else {"train"},
            "baseline_raw_admitted":raw_admitted,"baseline":frame.shape.map(ellipse_json),"candidate":candidate.map(ellipse_json),
            "candidate_geometry_available":candidate.is_some(),"candidate_is_gaze_authority":false,
            "baseline_errors":frame.shape.map(|e|metrics(frame,e)),"candidate_errors":candidate.map(|e|metrics(frame,e)),
            "unmodified_refit":control.map(ellipse_json),"unmodified_refit_errors":control.map(|e|metrics(frame,e)),
            "status":field.as_ref().map(|r|r.status),"elapsed_ms":field.as_ref().map(|r|r.elapsed_ms),
            "patches":field.as_ref().map(|r|r.samples.len()),
            "corrected_patches":field.as_ref().map(|r|r.samples.iter().filter(|p|p.correction_px.is_some()).count()),
            "field":field.as_ref().map(|r|r.samples.iter().map(|p|json!({"origin":p.origin,"normal":p.normal,
                "correction_px":p.correction_px,"landmarks":p.landmarks.iter().map(|l|json!({"offset_px":l.offset_px,
                    "spread_px":l.spread_px,"visible_mass":l.visible_mass})).collect::<Vec<_>>() })).collect::<Vec<_>>()),
            "supervision":frame.metadata["supervision"],
            "sn_feida":area_diagnostic(frame,candidate,control),
            "scale_reason":if frame.context.pixels_per_mm.is_some() {"coarse acquisition hint, not calibrated metric truth"}
                else if frame.metadata["source"]["relative_scale_hint"].is_object() {"RAW-registered independent relative image scale; not metric depth"}
                else {"no independent metric or relative scale matched to these RAW labels"}}));
    }
    // Conditional patch-localization diagnostic, separately named: centers
    // are sampled around labels here, unlike the full-frame experiment above.
    let mut patch_metrics: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut diagnostic_frames = frames;
    if ablate_context {
        for frame in &mut diagnostic_frames {
            frame.context = Context::default();
        }
    }
    for e in examples(&diagnostic_frames, false) {
        let Some(predicted) = model.predict(&e.patch) else {
            continue;
        };
        for (role, prediction) in predicted.iter().enumerate() {
            if e.weight[role] <= 0.0 || e.target[role * BINS + 16] > 0.5 {
                continue;
            }
            let target = e.target[role * BINS..role * BINS + 16]
                .iter()
                .enumerate()
                .map(|(i, p)| f64::from(*p) * (i as f64 - 7.5) * e.patch.step_px)
                .sum::<f64>();
            patch_metrics
                .entry(format!("group{}/{}", e.group, ROLES[role]))
                .or_default()
                .push((prediction.offset_px - target).abs());
        }
    }
    write_new(
        output,
        &json!({"schema":"buttercup-limbus-refiner-evaluation-v1","model":model_path,
        "model_training":model.manifest["training"],"frames":results,"physical_context_ablated":ablate_context,
        "conditional_patch_mae":patch_metrics.into_iter().map(|(k,v)|(k,json!({"n":v.len(),
            "mean_px":v.iter().sum::<f64>()/v.len() as f64}))).collect::<BTreeMap<_,_>>(),
        "limitations":["Sixteen labeled images are not sixteen independent sessions.",
            "This preview preserves SAM RAW admissions; new pupil/gaze accuracy is not established.",
            "Heightmap is an optical normal-displacement distribution, not a measured 3D surface."]}),
    )
}
pub fn run_cli(args: impl Iterator<Item = String>) -> Result<()> {
    let a: Vec<_> = args.collect();
    match a.first().map(String::as_str) {
        Some("prepare-aux") if a.len()==4=>prepare_aux(Path::new(&a[1]),Path::new(&a[2]),Path::new(&a[3])),
        Some("train-all") if a.len()==5=>train(Path::new(&a[1]),Path::new(&a[2]),Path::new(&a[3]),
            a[4].parse().map_err(|_|"epochs")?,-1,-2),
        Some("train") if a.len()==7=>train(Path::new(&a[1]),Path::new(&a[2]),Path::new(&a[3]),
            a[4].parse().map_err(|_|"epochs")?,a[5].parse().map_err(|_|"test group")?,a[6].parse().map_err(|_|"validation group")?),
        Some("evaluate") if a.len()==5=>evaluate(Path::new(&a[1]),Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4])),
        _=>Err("usage: prepare-aux TEACHER_DIR DATASET.json AUX.json | train DATASET.json SAM.jsonl MODEL.json EPOCHS TEST_GROUP VALIDATION_GROUP | train-all DATASET.json SAM.jsonl MODEL.json EPOCHS | evaluate DATASET.json SAM.jsonl MODEL.json OUTPUT.json".into()),
    }
}
