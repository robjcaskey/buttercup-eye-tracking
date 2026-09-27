//! Standalone cold-trained local RAW landmark component. Human band vectors
//! define patch orientation only; no hidden contour, gaze target, calibration,
//! existing model or SAM prediction is loaded. This is single-user development.
use crate::{bootstrapability as boot, training_refiner_data as prep};
use buttercup_eye_tracking::{geometry::Ellipse, limbus_refiner::*, raw10};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const GEOMETRY: [usize; 5] = [260, 261, 262, 263, 275];
const MAX_EXAMPLES: usize = 4096;
const MAX_FOLD_MS: u128 = 120_000;
const SEED: u64 = 73017;
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(p: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn write(p: impl AsRef<Path>, v: &Value) -> Result<()> {
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(p)?;
    serde_json::to_writer_pretty(&mut f, v)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn point(v: &Value) -> Option<[f64; 2]> {
    Some([v[0].as_f64()?, v[1].as_f64()?])
}
struct Example {
    x: Vec<f32>,
    y: Vec<f32>,
    weight: [f32; 6],
    group: i64,
    source: String,
    observation: usize,
    shift: f64,
}
#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn shuffle(&mut self, a: &mut [usize]) {
        for i in (1..a.len()).rev() {
            a.swap(i, self.next() as usize % (i + 1));
        }
    }
}
fn examples(dataset: &Value, augment: bool) -> Result<(Vec<Example>, Value)> {
    let mut result = vec![];
    let mut excluded = BTreeMap::<&str, usize>::new();
    for row in dataset["frames"]
        .as_array()
        .ok_or("missing canonical frames")?
    {
        let src = &row["source"];
        let f = &src["frame"];
        let bytes = fs::read(src["raw_file"].as_str().ok_or("RAW path")?)?;
        if hash(&bytes) != src["raw_sha256"] {
            return Err("RAW changed since preparation".into());
        }
        let w = f["width"].as_u64().ok_or("width")? as usize;
        let h = f["height"].as_u64().ok_or("height")? as usize;
        let raw =
            raw10::try_unpack_raw10(&bytes, w, h, f["stride"].as_u64().ok_or("stride")? as usize)?;
        let origin = [
            f["sensor_x"].as_u64().ok_or("x")? as usize,
            f["sensor_y"].as_u64().ok_or("y")? as usize,
        ];
        let mut variants = vec![raw.clone()];
        if augment {
            // Fixed front-loaded channel gains, applied on actual Quad-Bayer
            // sites. Clipping is real; occluded labels are never made visible.
            for gains in [[0.6, 0.6, 0.6], [1.3, 0.8, 0.65], [0.65, 0.85, 1.3]] {
                variants.push(
                    raw.iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let x = (i % w + origin[0]) % 4;
                            let y = (i / w + origin[1]) % 4;
                            let band = if x < 2 && y < 2 {
                                0
                            } else if x >= 2 && y >= 2 {
                                2
                            } else {
                                1
                            };
                            (f64::from(v) * gains[band]).round().clamp(0.0, 1023.0) as u16
                        })
                        .collect(),
                );
            }
        }
        for (index, obs) in row["observations"]
            .as_array()
            .ok_or("observations")?
            .iter()
            .enumerate()
        {
            let Some(anchor) = point(&obs["anchor"]) else {
                continue;
            };
            let Some((inner, outer)) =
                point(&obs["targets"]["band_inner"]).zip(point(&obs["targets"]["band_outer"]))
            else {
                *excluded.entry("no-reviewed-band-orientation").or_default() += 1;
                continue;
            };
            let delta = [outer[0] - inner[0], outer[1] - inner[1]];
            let length = delta[0].hypot(delta[1]);
            if !(0.5..=30.0).contains(&length) {
                *excluded.entry("degenerate-band-vector").or_default() += 1;
                continue;
            }
            let normal = [delta[0] / length, delta[1] / length];
            for pixels in &variants {
                for shift in [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0] {
                    let center = (anchor[0] + normal[0] * shift, anchor[1] + normal[1] * shift);
                    // An algebraic circle supplies the reviewed outward vector to
                    // the shared extractor. Its size/bearing are zeroed and cannot
                    // teach an imagined completed human ellipse.
                    let carrier = Ellipse {
                        center: (center.0 - 100.0 * normal[0], center.1 - 100.0 * normal[1]),
                        major_radius: 100.0,
                        minor_radius: 100.0,
                        angle: 0.0,
                    };
                    let Some(patch) = extract_patch(
                        pixels,
                        w,
                        h,
                        carrier,
                        center,
                        Context::default(),
                        SAMPLE_STEP_PX,
                    ) else {
                        *excluded.entry("patch-outside-native-RAW").or_default() += 1;
                        continue;
                    };
                    let mut x = patch.features;
                    for i in GEOMETRY {
                        x[i] = 0.0;
                    }
                    let mut y = vec![0.0; OUTPUTS];
                    let mut weight = [0.0; 6];
                    for (role, name) in ROLES.iter().enumerate() {
                        if let Some(p) = point(&obs["targets"][*name]) {
                            let offset =
                                (p[0] - center.0) * normal[0] + (p[1] - center.1) * normal[1];
                            let bin = offset / SAMPLE_STEP_PX + 7.5;
                            if !(0.0..=15.0).contains(&bin) {
                                continue;
                            }
                            let sigma = if obs["kind"] == "paired_midpoint" && role == 0 {
                                1.2
                            } else {
                                0.65
                            };
                            let mut sum = 0.0;
                            for j in 0..16 {
                                y[role * BINS + j] =
                                    (-0.5 * ((j as f64 - bin) / sigma).powi(2)).exp() as f32;
                                sum += y[role * BINS + j];
                            }
                            for j in 0..16 {
                                y[role * BINS + j] /= sum;
                            }
                            weight[role] = obs["weights"][*name].as_f64().unwrap_or(1.0) as f32;
                        } else if obs["occluded"]
                            .as_array()
                            .is_some_and(|a| a.contains(&json!(name)))
                        {
                            y[role * BINS + 16] = 1.0;
                            weight[role] = 1.0;
                        }
                    }
                    if weight.iter().sum::<f32>() > 0.0 {
                        result.push(Example {
                            x,
                            y,
                            weight,
                            group: row["group"].as_i64().ok_or("group")?,
                            source: src["raw_sha256"].as_str().ok_or("RAW hash")?.into(),
                            observation: index,
                            shift,
                        });
                    }
                }
            }
        }
    }
    if result.len() > MAX_EXAMPLES {
        return Err(format!(
            "{} patches exceed the fixed {MAX_EXAMPLES} budget",
            result.len()
        )
        .into());
    }
    Ok((result, json!(excluded)))
}
#[derive(Clone)]
struct Network {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
impl Network {
    fn zero() -> Self {
        Self {
            w1: vec![0.; INPUTS * HIDDEN],
            b1: vec![0.; HIDDEN],
            w2: vec![0.; HIDDEN * OUTPUTS],
            b2: vec![0.; OUTPUTS],
        }
    }
    fn new() -> Self {
        let mut n = Self::zero();
        let mut r = Rng(SEED);
        for (i, w) in n.w1.iter_mut().enumerate() {
            if !GEOMETRY.contains(&(i % INPUTS)) {
                *w = (r.uniform() * 2. - 1.) / (INPUTS as f32).sqrt();
            }
        }
        for w in &mut n.w2 {
            *w = (r.uniform() * 2. - 1.) / (HIDDEN as f32).sqrt();
        }
        n
    }
    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let h = self
            .w1
            .chunks_exact(INPUTS)
            .zip(&self.b1)
            .map(|(w, b)| (dot(w, x) + b).max(0.))
            .collect::<Vec<_>>();
        let o = self
            .w2
            .chunks_exact(HIDDEN)
            .zip(&self.b2)
            .map(|(w, b)| dot(w, &h) + b)
            .collect();
        (h, o)
    }
    fn value(&self, metadata: Value) -> Value {
        json!({"architecture":ARCHITECTURE,"roles":ROLES,"inputs":INPUTS,"hidden":HIDDEN,"bins":BINS,"preprocess":"native-linear-tent-normal-patch-v1","sample_step_px":SAMPLE_STEP_PX,"trained_optional_context":[false,false,false],"first_weight":self.w1,"first_bias":self.b1,"second_weight":self.w2,"second_bias":self.b2,"training":metadata})
    }
    fn slices(&mut self) -> [&mut [f32]; 4] {
        [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2]
    }
    fn gradient(&self, e: &Example, grad: &mut Self) -> f32 {
        let (h, logits) = self.forward(&e.x);
        let (loss, d) = loss_gradient(&logits, &e.y, &e.weight);
        let mut dh = vec![0.; HIDDEN];
        for k in 0..OUTPUTS {
            grad.b2[k] += d[k];
            for j in 0..HIDDEN {
                grad.w2[k * HIDDEN + j] += d[k] * h[j];
                dh[j] += self.w2[k * HIDDEN + j] * d[k];
            }
        }
        for j in 0..HIDDEN {
            if h[j] <= 0. {
                continue;
            }
            grad.b1[j] += dh[j];
            for i in 0..INPUTS {
                grad.w1[j * INPUTS + i] += dh[j] * e.x[i];
            }
        }
        loss
    }
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.; 4];
    let n = a.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        for j in 0..4 {
            sums[j] += a[i + j] * b[i + j];
        }
    }
    sums.into_iter().sum::<f32>() + a[n..].iter().zip(&b[n..]).map(|(a, b)| a * b).sum::<f32>()
}
fn loss_gradient(logits: &[f32], target: &[f32], weights: &[f32; 6]) -> (f32, Vec<f32>) {
    let mut d = vec![0.; OUTPUTS];
    let mut loss = 0.;
    let total = weights.iter().sum::<f32>().max(1e-6);
    for role in 0..6 {
        if weights[role] <= 0. {
            continue;
        }
        let row = &logits[role * BINS..(role + 1) * BINS];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = row.iter().map(|x| (x - max).exp()).sum::<f32>();
        let logz = sum.ln() + max;
        for j in 0..BINS {
            let k = role * BINS + j;
            let w = weights[role] / total;
            loss -= w * target[k] * (logits[k] - logz);
            d[k] = w * ((logits[k] - max).exp() / sum - target[k]);
        }
    }
    (loss, d)
}
fn evaluate(n: &Network, examples: &[Example], group: i64) -> Value {
    let mut errors = vec![Vec::<f64>::new(); 6];
    let mut gated_errors = vec![Vec::<f64>::new(); 6];
    let mut corrections = [0usize; 6];
    let mut identity_errors = vec![Vec::<f64>::new(); 6];
    let mut sources = BTreeSet::new();
    let mut rows = vec![];
    for e in examples.iter().filter(|e| e.group == group) {
        let (_, logits) = n.forward(&e.x);
        sources.insert(e.source.clone());
        for role in 0..6 {
            if e.weight[role] <= 0. || e.y[role * BINS + 16] > 0.5 {
                continue;
            }
            let max = logits[role * BINS..(role + 1) * BINS]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let p = (0..16)
                .map(|j| (logits[role * BINS + j] - max).exp())
                .collect::<Vec<_>>();
            let sum = p.iter().sum::<f32>();
            let target = (0..16)
                .map(|j| e.y[role * BINS + j] * (j as f32 - 7.5) * 2.)
                .sum::<f32>();
            let estimate = p
                .iter()
                .enumerate()
                .map(|(j, p)| p * (j as f32 - 7.5) * 2.)
                .sum::<f32>()
                / sum.max(1e-9);
            let error = (estimate - target).abs() as f64;
            let visible_mass = sum / (sum + (logits[role * BINS + 16] - max).exp());
            let spread = (p
                .iter()
                .enumerate()
                .map(|(j, p)| p * ((j as f32 - 7.5) * 2. - estimate).powi(2))
                .sum::<f32>()
                / sum.max(1e-9))
            .sqrt();
            let supported = visible_mass >= 0.85
                && spread <= 4.0
                && estimate.abs() <= 4.0
                && e.x[257] >= 0.015
                && e.x[259] < 0.2;
            corrections[role] += usize::from(supported);
            gated_errors[role].push(
                (if supported {
                    estimate - target
                } else {
                    -target
                })
                .abs() as f64,
            );
            errors[role].push(error);
            identity_errors[role].push(target.abs() as f64);
            if role == 0 {
                rows.push(json!({"raw_sha256":e.source,"observation":e.observation,"shift_px":e.shift,"estimated_offset_px":estimate,"target_offset_px":target,"abs_error_px":error,"correction_supported":supported,"visible_mass":visible_mass,"spread_px":spread}));
            }
        }
    }
    let stats = |v: &Vec<f64>| json!({"n":v.len(),"mae_px":if v.is_empty(){None}else{Some(v.iter().sum::<f64>()/v.len() as f64)}});
    json!({"test_group":group,"sources":sources,"roles":ROLES.iter().enumerate().map(|(i,name)|json!({"role":name,"trained":stats(&errors[i]),"zero_correction":stats(&identity_errors[i]),"gated_offset":stats(&gated_errors[i]),"supported_corrections":corrections[i]})).collect::<Vec<_>>(),"rim_samples":rows,
 "contract":"source-group held-out normal-offset error at canonical reviewed bands, including fixed offset perturbations; correlated patches are not independent observations; no contour completion, gaze truth, scale or SN-FEIDA labels"})
}
fn train(
    examples: &[Example],
    evaluation: &[Example],
    test: i64,
    validation: i64,
    epochs: usize,
    augment: bool,
) -> Result<(Network, Value, Value)> {
    let ids = examples
        .iter()
        .enumerate()
        .filter(|(_, e)| e.group != test && e.group != validation)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let valid = evaluation
        .iter()
        .filter(|e| e.group == validation)
        .collect::<Vec<_>>();
    let train_all = test == -1 && validation == -2;
    if ids.is_empty() || (!train_all && valid.is_empty()) {
        return Err("empty training or validation partition".into());
    }
    let mut n = Network::new();
    let mut m = Network::zero();
    let mut v = Network::zero();
    let mut order = ids;
    let mut rng = Rng(SEED);
    let mut step = 0i32;
    let mut best = n.clone();
    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0;
    let started = Instant::now();
    let mut history = vec![];
    for epoch in 1..=epochs {
        rng.shuffle(&mut order);
        for batch in order.chunks(64) {
            if started.elapsed().as_millis() > MAX_FOLD_MS {
                return Err(
                    format!("CPU fold exceeded {MAX_FOLD_MS}ms; no model published").into(),
                );
            }
            let mut g = Network::zero();
            for &i in batch {
                n.gradient(&examples[i], &mut g);
            }
            let norm =
                g.w1.iter()
                    .chain(&g.b1)
                    .chain(&g.w2)
                    .chain(&g.b2)
                    .map(|x| (*x / batch.len() as f32).powi(2))
                    .sum::<f32>()
                    .sqrt();
            let scale = (5.0 / norm.max(5.0)) / batch.len() as f32;
            step += 1;
            for (((weight, first), second), gradient) in n
                .slices()
                .into_iter()
                .zip(m.slices())
                .zip(v.slices())
                .zip(g.slices())
            {
                for i in 0..weight.len() {
                    let grad = gradient[i] * scale;
                    first[i] = 0.9 * first[i] + 0.1 * grad;
                    second[i] = 0.999 * second[i] + 0.001 * grad * grad;
                    let update = (first[i] / (1. - 0.9f32.powi(step)))
                        / ((second[i] / (1. - 0.999f32.powi(step))).sqrt() + 1e-8);
                    weight[i] -= 0.001 * (update + 0.02 * weight[i]);
                }
            }
        }
        if epoch == 1 || epoch % 5 == 0 || epoch == epochs {
            let val = if train_all {
                0.0
            } else {
                valid
                    .iter()
                    .map(|e| loss_gradient(&n.forward(&e.x).1, &e.y, &e.weight).0 as f64)
                    .sum::<f64>()
                    / valid.len() as f64
            };
            if !val.is_finite() {
                return Err("nonfinite validation objective".into());
            }
            history.push(json!({"epoch":epoch,"validation_loss":val}));
            if train_all || val < best_loss {
                best = n.clone();
                best_loss = val;
                best_epoch = epoch;
            }
            eprintln!("cpu fold={test} augment={augment} epoch={epoch} validation={val:.5} elapsed={:.1}s",started.elapsed().as_secs_f64());
        }
    }
    for row in best.w1.chunks_exact(INPUTS) {
        if GEOMETRY.iter().any(|i| row[*i] != 0.0) {
            return Err("geometry context leakage".into());
        }
    }
    let metrics = evaluate(&best, evaluation, test);
    let meta = json!({"recipe":"cold-canonical-band-cpu-v1","seed":SEED,"epochs":epochs,"selected_epoch":best_epoch,"validation_loss":best_loss,"test_group":test,"validation_group":validation,"channel_gain_augmentation":augment,"train_patches":order.len(),"validation_patches":valid.len(),"cpu_elapsed_ms":started.elapsed().as_millis(),"maximum_update_ms":MAX_FOLD_MS,"maximum_examples":MAX_EXAMPLES,"device":"native-rust-cpu","cuda_dependency":false,"geometry_context":"disabled-zero-weights","custom_ancestors":[],"train_raw_sha256":order.iter().map(|i|examples[*i].source.clone()).collect::<BTreeSet<_>>(),"history":history});
    Ok((best, meta, metrics))
}
pub fn run(args: Vec<String>) -> Result<()> {
    if args.len() != 4 || args[0] != "cold" {
        return Err(
            "usage: buttercup_limbus_cpu cold CANONICAL_INVENTORY.json NEW_OUTPUT_DIR EPOCHS"
                .into(),
        );
    }
    let epochs: usize = args[3].parse()?;
    if !(5..=80).contains(&epochs) {
        return Err("epochs must be 5..80".into());
    }
    let out = Path::new(&args[2]);
    let data = fs::canonicalize("data")?;
    let parent = fs::canonicalize(out.parent().ok_or("output parent")?)?;
    if !parent.starts_with(data) {
        return Err("output must be under checked runtime links".into());
    }
    fs::create_dir(out)?;
    let start = Instant::now();
    let source = boot::current_source(Path::new("."))?;
    let mut inventory = read(&args[1])?;
    if inventory["schema"] == "buttercup-limbus-patch-dataset-v1" {
        if inventory["evaluation_only"] == true {
            return Err("evaluation-only material cannot become a training inventory".into());
        }
        // Legacy canonical sets are accepted as label PATH inventories only.
        // Reopen every human annotation and RAW; never reuse cached coordinates,
        // grouping, model predictions or a completed fitted ellipse.
        let labels = inventory["frames"]
            .as_array()
            .ok_or("canonical inventory frames")?
            .iter()
            .map(|r| r["label"].clone())
            .collect::<Vec<_>>();
        inventory = json!({"labels":labels});
    }
    let dataset = prep::prepare(&inventory)?;
    let labels = dataset["frames"]
        .as_array()
        .ok_or("frames")?
        .iter()
        .map(|r| {
            let p = r["label"].as_str().ok_or("label path")?;
            Ok(json!({"path":p,"sha256":hash(&fs::read(p)?)}))
        })
        .collect::<Result<Vec<_>>>()?;
    let roots = json!({"sources":dataset["frames"].as_array().unwrap().iter().map(|r|r["source"].clone()).collect::<Vec<_>>(),"canonical_labels":labels});
    let mut graph = json!({"schema":boot::SCHEMA,"source":source,"targets":["evaluation"],"nodes":[
  {"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]},
  {"id":"raw","kind":"raw","sha256":hash(&serde_json::to_vec(&roots["sources"])?),"dependencies":[]},
  {"id":"human","kind":"human_labels","sha256":hash(&serde_json::to_vec(&roots["canonical_labels"])?),"dependencies":[]},
  {"id":"prepared","kind":"derived_data","sha256":hash(&serde_json::to_vec(&dataset)?),"dependencies":["source","raw","human"]},
  {"id":"patches","kind":"features","sha256":null,"planned":true,"dependencies":["prepared","source"]},
  {"id":"models","kind":"custom_model","sha256":null,"planned":true,"dependencies":["patches","source"]},
  {"id":"evaluation","kind":"evaluation","sha256":null,"planned":true,"dependencies":["models","patches","human","source"]}]});
    let manifest: boot::Manifest = serde_json::from_value(graph.clone())?;
    let certificate =
        boot::validate(&manifest, &source).map_err(|e| format!("preflight: {e:?}"))?;
    write(out.join("bootstrap-graph-planned.json"), &graph)?;
    write(
        out.join("bootstrap-preflight.json"),
        &serde_json::to_value(certificate)?,
    )?;
    write(out.join("roots.json"), &roots)?;
    write(out.join("dataset.json"), &dataset)?;
    let (control, excluded) = examples(&dataset, false)?;
    let (augmented, aug_excluded) = examples(&dataset, true)?;
    let groups = control
        .iter()
        .map(|e| e.group)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if groups.len() < 4 {
        return Err(
            "four independent conservative groups required for train/validation/test".into(),
        );
    }
    let mut results = vec![];
    let mut models = vec![];
    for (i, &test) in groups.iter().enumerate() {
        let validation = groups[(i + 1) % groups.len()];
        for augmented_arm in [false, true] {
            let examples = if augmented_arm { &augmented } else { &control };
            let (n, mut meta, metrics) =
                train(examples, &control, test, validation, epochs, augmented_arm)?;
            meta["source"] = serde_json::to_value(&source)?;
            let model = n.value(meta);
            let portable = Model::from_json(model.clone())?;
            for e in control.iter().filter(|e| e.group == test).take(16) {
                let fake = Patch {
                    features: e.x.clone(),
                    center: (0., 0.),
                    normal: (1., 0.),
                    step_px: 2.,
                    mean: 0.,
                    contrast: 1.,
                    focus_energy: 0.,
                    saturated_fraction: 0.,
                };
                let p = portable.logits(&fake).ok_or("portable CPU output")?;
                let r = n.forward(&e.x).1;
                if p.iter().zip(&r).any(|(a, b)| (a - b).abs() > 1e-4) {
                    return Err("native training/portable inference mismatch".into());
                }
            }
            let name = format!(
                "fold-{test}-{}.json",
                if augmented_arm {
                    "augmented"
                } else {
                    "control"
                }
            );
            write(out.join(&name), &model)?;
            models.push(json!({"path":name,"sha256":hash(&fs::read(out.join(&name))?)}));
            results.push(json!({"test_group":test,"augmented":augmented_arm,"metrics":metrics,"training":model["training"]}));
        }
    }
    let mut pooled = Vec::new();
    for arm in [false, true] {
        let rows = results
            .iter()
            .filter(|r| r["augmented"] == arm)
            .collect::<Vec<_>>();
        let count = rows
            .iter()
            .map(|r| {
                r["metrics"]["roles"][0]["gated_offset"]["n"]
                    .as_u64()
                    .unwrap()
            })
            .sum::<u64>();
        let sum = rows
            .iter()
            .map(|r| {
                let m = &r["metrics"]["roles"][0]["gated_offset"];
                m["mae_px"].as_f64().unwrap() * m["n"].as_u64().unwrap() as f64
            })
            .sum::<f64>();
        let coverage = rows
            .iter()
            .map(|r| {
                r["metrics"]["roles"][0]["supported_corrections"]
                    .as_u64()
                    .unwrap()
            })
            .sum::<u64>();
        pooled.push(json!({"augmented":arm,"rim_gated_mae_px":sum/count as f64,"rim_samples":count,"supported_corrections":coverage}));
        // Full development exports use only the fold VALIDATION stopping epochs.
        // Held-out test errors above do not choose this schedule or checkpoint.
        let mut schedule = rows
            .iter()
            .map(|r| r["training"]["selected_epoch"].as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        schedule.sort_unstable();
        let selected_epochs = schedule[schedule.len() / 2];
        let (n, mut meta, _) = train(
            if arm { &augmented } else { &control },
            &control,
            -1,
            -2,
            selected_epochs,
            arm,
        )?;
        meta["source"] = serde_json::to_value(&source)?;
        meta["all_reviewed_development_data"] = json!(true);
        meta["epoch_rule"] =
            json!("upper median of grouped validation-selected epochs; test errors unused");
        let name = if arm {
            "full-augmented.json"
        } else {
            "full-control.json"
        };
        let model = n.value(meta);
        Model::from_json(model.clone())?;
        write(out.join(name), &model)?;
        models.push(json!({"path":name,"sha256":hash(&fs::read(out.join(name))?)}));
    }
    let improvement = 1.0
        - pooled[1]["rim_gated_mae_px"].as_f64().unwrap()
            / pooled[0]["rim_gated_mae_px"].as_f64().unwrap();
    let max_group_regression = groups
        .iter()
        .map(|g| {
            let metric = |arm| {
                results
                    .iter()
                    .find(|r| r["test_group"] == *g && r["augmented"] == arm)
                    .unwrap()["metrics"]["roles"][0]["gated_offset"]["mae_px"]
                    .as_f64()
                    .unwrap()
            };
            metric(true) - metric(false)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let coverage_ratio = pooled[1]["supported_corrections"].as_u64().unwrap() as f64
        / pooled[0]["supported_corrections"].as_u64().unwrap().max(1) as f64;
    let assessment = json!({"pooled":pooled,"relative_gated_rim_improvement":improvement,"maximum_group_regression_px":max_group_regression,"correction_coverage_ratio":coverage_ratio,
        "criterion":"at least 5% pooled gated offset improvement, no group regression above 0.25px, at least 80% control coverage",
        "passes_development_comparison":improvement>=0.05&&max_group_regression<=0.25&&coverage_ratio>=0.8,
        "deployment_authorized_by_this_report":false});
    let finish = boot::current_source(Path::new("."))?;
    if finish != source {
        return Err("source changed during cold training; receipt refused".into());
    }
    let result = json!({"schema":"buttercup-limbus-cold-cpu-result-v1","source":source,"models":models,"results":results,"assessment":assessment,"control_patches":control.len(),"augmented_patches":augmented.len(),"excluded":excluded,"augmented_excluded":aug_excluded,"total_elapsed_seconds":start.elapsed().as_secs_f64(),"contract":"cold human-band-only local optical landmark experiment, CPU training and inference; no custom teacher/SAM/cache/personal-calibration input; source-group development holdouts for Rob only; sparse-label offset error is not gaze accuracy or a complete iris-surface label; no automatic promotion"});
    write(out.join("results.json"), &result)?;
    let mut features_hash = Sha256::new();
    for arm in [&control, &augmented] {
        features_hash.update((arm.len() as u64).to_le_bytes());
        for e in arm {
            features_hash.update(e.source.as_bytes());
            features_hash.update(e.group.to_le_bytes());
            features_hash.update((e.observation as u64).to_le_bytes());
            features_hash.update(e.shift.to_le_bytes());
            for value in e.x.iter().chain(&e.y).chain(&e.weight) {
                features_hash.update(value.to_le_bytes());
            }
        }
    }
    graph["nodes"][4]["sha256"] = json!(format!("{:x}", features_hash.finalize()));
    graph["nodes"][5]["sha256"] = json!(hash(&serde_json::to_vec(&models)?));
    graph["nodes"][6]["sha256"] = json!(hash(&fs::read(out.join("results.json"))?));
    for i in 4..7 {
        graph["nodes"][i]["planned"] = json!(false);
    }
    let complete: boot::Manifest = serde_json::from_value(graph.clone())?;
    let certificate =
        boot::validate(&complete, &finish).map_err(|e| format!("final graph: {e:?}"))?;
    write(out.join("bootstrap-graph.json"), &graph)?;
    write(
        out.join("bootstrap-final-structural-check.json"),
        &serde_json::to_value(certificate)?,
    )?;
    println!(
        "cold CPU training completed: {} models in {:.1}s; inspect results before any promotion",
        models.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masked_softmax_gradient_matches_finite_difference() {
        let l = (0..OUTPUTS).map(|i| i as f32 * 0.013).collect::<Vec<_>>();
        let mut y = vec![0.; OUTPUTS];
        y[3] = 1.;
        y[BINS + 7] = 1.;
        let w = [1., 0.3, 0., 0., 0., 0.];
        let (_, g) = loss_gradient(&l, &y, &w);
        for k in [0, 3, 16, 17, 24, 40] {
            let mut plus = l.clone();
            let mut minus = l.clone();
            plus[k] += 0.001;
            minus[k] -= 0.001;
            let fd = (loss_gradient(&plus, &y, &w).0 - loss_gradient(&minus, &y, &w).0) / 0.002;
            assert!((fd - g[k]).abs() < 0.0003, "{k} {fd} {}", g[k]);
        }
        assert!(g[2 * BINS..].iter().all(|x| *x == 0.));
    }
    #[test]
    fn zero_geometry_features_cannot_leak_anatomy() {
        let n = Network::new();
        let mut a = vec![0.1; INPUTS];
        let mut b = a.clone();
        for i in GEOMETRY {
            a[i] = 0.;
            b[i] = 12.;
        }
        assert_eq!(n.forward(&a).1, n.forward(&b).1);
    }
    #[test]
    fn network_gradient_reaches_each_layer() {
        let n = Network::new();
        let mut e = Example {
            x: vec![0.2; INPUTS],
            y: vec![0.; OUTPUTS],
            weight: [1., 0., 0., 0., 0., 0.],
            group: 0,
            source: "fixture".into(),
            observation: 0,
            shift: 0.,
        };
        e.y[4] = 1.;
        let mut g = Network::zero();
        n.gradient(&e, &mut g);
        let mut p = n.clone();
        let mut m = n.clone();
        p.w1[5] += 0.001;
        m.w1[5] -= 0.001;
        let f = |n: &Network| loss_gradient(&n.forward(&e.x).1, &e.y, &e.weight).0;
        assert!(((f(&p) - f(&m)) / 0.002 - g.w1[5]).abs() < 0.0005);
    }
}
