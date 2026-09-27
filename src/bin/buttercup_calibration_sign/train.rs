use super::{bootstrapability as boot, branch_labels, data, native, Result};
use buttercup_eye_tracking::{
    calibration_sign_model::{
        self as net, BranchHead, Model, CLASSES, GRID, HEIGHT, HIDDEN, INPUTS, PIXELS, WIDTH,
    },
    recorded_bundle::BundleSource,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

const EPOCHS: usize = 32;
const BATCH: usize = 64;
const MAX_TRAIN_SECONDS: u64 = 180;
const SEED: u64 = 829_416;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.
    }
    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.next() as usize % (i + 1));
        }
    }
}
pub(super) struct Frame {
    pub pixels: Vec<f32>,
    pub hash: String,
    pub source: Value,
    pub conic: Option<native::NativeFit>,
}
pub(super) struct Example {
    pub frames: [usize; 2],
    pub class: usize,
    pub uv: [f32; 2],
    pub day: u64,
    pub group: String,
    pub visit: String,
    pub source: Value,
    pub branch: Option<usize>,
}
pub(super) struct Dataset {
    pub frames: Vec<Frame>,
    pub examples: Vec<Example>,
    roots: Value,
    excluded: Value,
    pub branch_audit: Option<Value>,
    pub sam_teacher: Option<Value>,
}
fn prepare(sources: &[data::Source], out: &Path, branches: bool) -> Result<Dataset> {
    let mut frames = Vec::<Frame>::new();
    let mut examples = Vec::<Example>::new();
    let mut hashes = HashMap::<String, usize>::new();
    let mut labels = HashMap::<String, BTreeSet<usize>>::new();
    let mut skipped = Vec::new();
    let mut roots = Vec::new();
    let mut measurements = Vec::new();
    let mut pairs = BTreeSet::new();
    for (si, s) in sources
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.eligible.is_empty())
    {
        let bundle = BundleSource::open(Path::new(&s.archive))?;
        let mut cache = HashMap::<usize, Option<usize>>::new();
        for (indices, &target) in s.eligible.iter().zip(&s.targets) {
            let mut ids = [0; 2];
            let mut good = true;
            for j in 0..2 {
                let fi = indices[j];
                if let Some(value) = cache.get(&fi) {
                    if let Some(value) = value {
                        ids[j] = *value;
                    } else {
                        good = false;
                    }
                    continue;
                }
                let f = &s.frames[fi];
                let get =
                    |key: &str| data::num(&f[key]).ok_or_else(|| format!("missing RAW {key}"));
                let raw = bundle.read_range(
                    f["stream"].as_str().ok_or("RAW stream")?,
                    get("offset")?,
                    get("length")? as usize,
                )?;
                let hash = data::digest(&raw);
                let source = json!({"archive":s.archive,"frame":f,"raw_sha256":hash});
                let geometry_key = native::identity(&hash, f);
                let idx = if let Some(&i) = hashes.get(&geometry_key) {
                    Some(i)
                } else {
                    match net::image(
                        &raw,
                        get("width")? as usize,
                        get("height")? as usize,
                        get("stride")? as usize,
                    ) {
                        Ok(pixels) => {
                            let idx = frames.len();
                            frames.push(Frame {
                                pixels,
                                hash: hash.clone(),
                                source: source.clone(),
                                conic: if branches {
                                    native::for_supervision(&native::unpack(&raw, f)?, f)
                                } else {
                                    None
                                },
                            });
                            hashes.insert(geometry_key, idx);
                            Some(idx)
                        }
                        Err(e) => {
                            skipped.push(json!({"source":source,"reason":e}));
                            None
                        }
                    }
                };
                roots.push(source);
                cache.insert(fi, idx);
                if let Some(idx) = idx {
                    ids[j] = idx;
                } else {
                    good = false;
                }
            }
            if !good {
                continue;
            }
            let t = &s.spans[target];
            let class = net::class(t.uv);
            for &idx in &ids {
                labels
                    .entry(frames[idx].hash.clone())
                    .or_default()
                    .insert(class);
            }
            if !pairs.insert((frames[ids[0]].hash.clone(), frames[ids[1]].hash.clone())) {
                continue;
            }
            examples.push(Example{frames:ids,class,uv:t.uv,day:s.day,group:s.group.clone(),visit:format!("{}:{target}",s.session),branch:None,source:json!({"session":s.session,"archive":s.archive,"target":t,"previous":s.frames[indices[0]],"current":s.frames[indices[1]],"raw_sha256":[frames[ids[0]].hash,frames[ids[1]].hash],"day":s.day,"viewer_group":s.group,"target_class":class})});
        }
        measurements.push(json!({"session":s.session,"session_sha256":s.session_sha256,"metadata_sha256":s.metadata_sha256,"frame_index_sha256":s.frame_index_sha256,"target_spans":s.spans}));
        eprintln!(
            "RAW preparation {si}: {} unique frames, {} pairs",
            frames.len(),
            examples.len()
        );
    }
    let before = examples.len();
    examples.retain(|e| e.frames.iter().all(|&i| labels[&frames[i].hash].len() == 1));
    let mut ownership = HashMap::<String, BTreeSet<u64>>::new();
    for e in &examples {
        for &i in &e.frames {
            ownership
                .entry(frames[i].hash.clone())
                .or_default()
                .insert(e.day);
        }
    }
    let before_days = examples.len();
    examples.retain(|e| {
        e.frames
            .iter()
            .all(|&i| ownership[&frames[i].hash].len() == 1)
    });
    let mut writer = BufWriter::new(fs::File::create(out.join("pairs.jsonl"))?);
    for e in &examples {
        serde_json::to_writer(&mut writer, &e.source)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    let excluded = json!({"unusable_images":skipped,"conflicting_target_pairs":before-before_days,"cross_day_duplicate_pairs":before_days-examples.len(),"duplicate_pairs_removed":sources.iter().map(|s|s.eligible.len()).sum::<usize>()-before});
    let roots = json!({"native_raw":roots,"target_measurements":measurements});
    Ok(Dataset {
        frames,
        examples,
        roots,
        excluded,
        branch_audit: None,
        sam_teacher: None,
    })
}
fn zero() -> Model {
    Model {
        schema: "buttercup-two-frame-target-sign-v1".into(),
        w1: vec![0.; INPUTS * HIDDEN],
        b1: vec![0.; HIDDEN],
        w2: vec![0.; HIDDEN * CLASSES],
        b2: vec![0.; CLASSES],
        branch: None,
        provenance: Value::Null,
    }
}
fn init(branches: bool) -> Model {
    let mut m = zero();
    let mut r = Rng(SEED);
    for w in &mut m.w1 {
        *w = (2. * r.unit() - 1.) * (6. / INPUTS as f32).sqrt();
    }
    for w in &mut m.w2 {
        *w = (2. * r.unit() - 1.) * (6. / HIDDEN as f32).sqrt();
    }
    if branches {
        let mut head = BranchHead::zero();
        for w in &mut head.weights {
            *w = (2. * r.unit() - 1.) * (6. / HIDDEN as f32).sqrt();
        }
        m.branch = Some(head);
    }
    m
}
fn input(d: &Dataset, e: &Example, current_only: bool, x: &mut [f32]) {
    x[..PIXELS].copy_from_slice(&d.frames[e.frames[usize::from(current_only)]].pixels);
    x[PIXELS..].copy_from_slice(&d.frames[e.frames[1]].pixels);
}
fn augment(x: &mut [f32], r: &mut Rng) {
    let shared_gain = 0.85 + 0.3 * r.unit();
    let shared_bias = (r.unit() - 0.5) * 0.16;
    for frame in x.chunks_exact_mut(PIXELS) {
        let g = shared_gain * (0.95 + 0.1 * r.unit());
        let b = shared_bias + (r.unit() - 0.5) * 0.04;
        for p in frame {
            *p = (*p * g + b).clamp(-1., 1.);
        }
    }
}
fn weights(d: &Dataset, ids: &[usize]) -> Vec<f32> {
    let mut sizes = HashMap::<&str, usize>::new();
    let mut classes = [0usize; CLASSES];
    for &i in ids {
        *sizes.entry(&d.examples[i].visit).or_default() += 1;
        classes[d.examples[i].class] += 1;
    }
    let mut w: Vec<f32> = ids
        .iter()
        .map(|&i| {
            let e = &d.examples[i];
            1. / (sizes[e.visit.as_str()] as f32 * classes[e.class] as f32).sqrt()
        })
        .collect();
    let mean = w.iter().sum::<f32>() / w.len() as f32;
    for w in &mut w {
        *w = (*w / mean).clamp(0.1, 5.);
    }
    w
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0
}
fn metrics(
    m: &Model,
    d: &Dataset,
    ids: &[usize],
    current_only: bool,
    save: Option<&Path>,
) -> Result<Value> {
    let mut x = vec![0.; INPUTS];
    let mut h = [0.; HIDDEN];
    let mut confusion = [[0usize; CLASSES]; CLASSES];
    let mut visits = HashMap::<&str, (f64, usize)>::new();
    let mut nll = 0f64;
    let mut error = 0f64;
    let mut signs = [[0usize; 2]; 2];
    let mut confident = [[0usize; 2]; 2];
    let mut axes_correct = [0usize; 2];
    let mut rows = Vec::new();
    let mut per_session = BTreeMap::<String, (usize, usize)>::new();
    let mut branch_confusion = [[0usize; 2]; 2];
    let mut branch_loss = 0.;
    let mut branch_visits = BTreeMap::<String, (f64, usize)>::new();
    let mut branch_supported = [0usize; 2];
    for &i in ids {
        let e = &d.examples[i];
        input(d, e, current_only, &mut x);
        let p = m.forward(&x, &mut h);
        let cls = argmax(&p);
        let bp = m.branch.as_ref().map(|b| b.forward(&h));
        if let (Some(label), Some(bp)) = (e.branch, bp) {
            let pred = argmax(&bp);
            branch_confusion[label][pred] += 1;
            let l = -(bp[label].max(1e-8) as f64).ln();
            branch_loss += l;
            let v = branch_visits.entry(e.visit.clone()).or_default();
            v.0 += l;
            v.1 += 1;
            if bp[pred] >= 0.8 {
                branch_supported[1] += 1;
                branch_supported[0] += usize::from(pred == label);
            }
        }
        confusion[e.class][cls] += 1;
        let loss = -(p[e.class].max(1e-8).ln() as f64);
        nll += loss;
        let v = visits.entry(&e.visit).or_default();
        v.0 += loss;
        v.1 += 1;
        let mut uv = [0.; 2];
        let mut a = [[0.; 3]; 2];
        for j in 0..CLASSES {
            uv[0] += p[j] * GRID[j][0];
            uv[1] += p[j] * GRID[j][1];
            a[0][j % 3] += p[j];
            a[1][j / 3] += p[j];
        }
        error += ((uv[0] - e.uv[0]).powi(2) + (uv[1] - e.uv[1]).powi(2)) as f64;
        for axis in 0..2 {
            let label = if axis == 0 { e.class % 3 } else { e.class / 3 };
            let pred = argmax(&a[axis]);
            axes_correct[axis] += usize::from(pred == label);
            if label != 1 {
                signs[axis][1] += 1;
                signs[axis][0] += usize::from(pred == label);
                if a[axis][pred] >= 0.8 && pred != 1 {
                    confident[axis][1] += 1;
                    confident[axis][0] += usize::from(pred == label);
                }
            }
        }
        let session = e.source["session"].as_str().unwrap();
        let s = per_session.entry(session.into()).or_default();
        s.0 += usize::from(cls == e.class);
        s.1 += 1;
        if save.is_some() {
            rows.push(json!({"pair_index":i,"source":e.source,"predicted_uv":uv,"class_scores":p,"predicted_class":cls,"correct_target_class":cls==e.class,"axis_scores":a,"conditional_branch_scores":bp,"conditional_branch_label":e.branch,"label_status":"intended fixation and conditional RAW conic teacher; neither is independent physical sign truth"}));
        }
    }
    if let Some(path) = save {
        let mut w = BufWriter::new(fs::File::create(path)?);
        for row in rows {
            serde_json::to_writer(&mut w, &row)?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
    }
    let n = ids.len().max(1) as f64;
    let correct = (0..CLASSES).map(|i| confusion[i][i]).sum::<usize>();
    let branch_n = branch_confusion.iter().flatten().sum::<usize>();
    let branch_correct = branch_confusion[0][0] + branch_confusion[1][1];
    let ratio = |a: usize, b: usize| {
        if b > 0 {
            Some(a as f64 / b as f64)
        } else {
            None
        }
    };
    let mut report = json!({"pairs":ids.len(),"target_class_correct":correct,"target_class_accuracy":correct as f64/n,"confusion_true_rows_predicted_columns":confusion,"mean_cross_entropy":nll/n,"visit_macro_cross_entropy":visits.values().map(|(l,n)|l/(*n as f64)).sum::<f64>()/visits.len().max(1)as f64,"target_uv_rmse":(error/n).sqrt(),"horizontal_3way_accuracy":axes_correct[0]as f64/n,"vertical_3way_accuracy":axes_correct[1]as f64/n,"noncentral_signs":(0..2).map(|a|json!({"axis":a,"correct":signs[a][0],"n":signs[a][1],"accuracy":ratio(signs[a][0],signs[a][1]),"support_ge_0_8":confident[a][1],"supported_correct":confident[a][0],"supported_accuracy":ratio(confident[a][0],confident[a][1])})).collect::<Vec<_>>(),"by_session":per_session.iter().map(|(s,(c,n))|json!({"session":s,"pairs":n,"correct":c,"accuracy":ratio(*c,*n)})).collect::<Vec<_>>(),"signed_3d_accuracy":null});
    report["conditional_branch"] = json!({"labeled_pairs":branch_n,"coverage":branch_n as f64/n,"matches":branch_correct,"agreement":ratio(branch_correct,branch_n),"confusion_teacher_rows_prediction_columns":branch_confusion,"mean_cross_entropy":if branch_n>0{Some(branch_loss/branch_n as f64)}else{None},"visit_macro_cross_entropy":if branch_visits.is_empty(){None}else{Some(branch_visits.values().map(|(l,n)|l/(*n as f64)).sum::<f64>()/branch_visits.len()as f64)},"support_ge_0_8":branch_supported[1],"supported_matches":branch_supported[0],"supported_agreement":ratio(branch_supported[0],branch_supported[1]),"scope":"agreement with conditional RAW/target teacher; not independent sign accuracy"});
    Ok(report)
}
fn adam(w: &mut [f32], g: &mut [f32], a: &mut [f32], b: &mut [f32], step: usize, n: usize) {
    let c1 = 1. - 0.9f32.powi(step as i32);
    let c2 = 1. - 0.999f32.powi(step as i32);
    for i in 0..w.len() {
        let grad = (g[i] / n as f32 + 0.0001 * w[i]).clamp(-2., 2.);
        a[i] = 0.9 * a[i] + 0.1 * grad;
        b[i] = 0.999 * b[i] + 0.001 * grad * grad;
        w[i] -= 0.001 * (a[i] / c1) / ((b[i] / c2).sqrt() + 1e-7);
        g[i] = 0.;
    }
}
fn fit(
    d: &Dataset,
    train: &[usize],
    validation: &[usize],
    current_only: bool,
    epochs: usize,
) -> Result<(Model, Value)> {
    let start = Instant::now();
    let mut branch_counts = [0usize; 2];
    for &i in train {
        if let Some(k) = d.examples[i].branch {
            branch_counts[k] += 1;
        }
    }
    // An unseen day may be the only source of conditional labels. Never
    // expose random or single-class head weights as a trained sign model.
    let branches = d.branch_audit.is_some() && branch_counts.iter().all(|n| *n >= 10);
    let mut m = init(branches);
    let mut g = zero();
    let mut a = zero();
    let mut b = zero();
    if branches {
        g.branch = Some(BranchHead::zero());
        a.branch = Some(BranchHead::zero());
        b.branch = Some(BranchHead::zero());
    }
    let mut best = m.clone();
    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0;
    let ws = weights(d, train);
    let branch_total = branch_counts.iter().sum::<usize>();
    let mut order: Vec<_> = (0..train.len()).collect();
    let mut rng = Rng(SEED);
    let mut step = 0;
    let mut x = vec![0.; INPUTS];
    let mut h = [0.; HIDDEN];
    let mut history = Vec::new();
    let mut completed = 0;
    for epoch in 1..=epochs {
        rng.shuffle(&mut order);
        for batch in order.chunks(BATCH) {
            for &k in batch {
                let e = &d.examples[train[k]];
                input(d, e, current_only, &mut x);
                augment(&mut x, &mut rng);
                let p = m.forward(&x, &mut h);
                // Bounded generalized CE downweights noisy weak targets. It is
                // not proof that a difficult example was an actual distraction.
                let robust = p[e.class].max(1e-6).powf(0.4) * ws[k];
                let mut gh = [0.; HIDDEN];
                for j in 0..CLASSES {
                    let dz = robust * (p[j] - if j == e.class { 1. } else { 0. });
                    g.b2[j] += dz;
                    for k in 0..HIDDEN {
                        g.w2[j * HIDDEN + k] += dz * h[k];
                        gh[k] += dz * m.w2[j * HIDDEN + k];
                    }
                }
                if let (Some(label), Some(head), Some(grad)) =
                    (e.branch, m.branch.as_ref(), g.branch.as_mut())
                {
                    let bp = head.forward(&h);
                    let balance = (branch_total as f32 / (2. * branch_counts[label].max(1) as f32))
                        .sqrt()
                        .clamp(0.25, 4.);
                    let weight = 2. * balance * ws[k] * bp[label].max(1e-6).powf(0.4);
                    for j in 0..2 {
                        let dz = weight * (bp[j] - if j == label { 1. } else { 0. });
                        grad.biases[j] += dz;
                        for t in 0..HIDDEN {
                            grad.weights[j * HIDDEN + t] += dz * h[t];
                            gh[t] += dz * head.weights[j * HIDDEN + t];
                        }
                    }
                }
                for j in 0..HIDDEN {
                    let dh = if h[j] > 0. { gh[j] } else { 0. };
                    g.b1[j] += dh;
                    let row = &mut g.w1[j * INPUTS..(j + 1) * INPUTS];
                    for k in 0..INPUTS {
                        row[k] += dh * x[k];
                    }
                }
            }
            step += 1;
            adam(
                &mut m.w1,
                &mut g.w1,
                &mut a.w1,
                &mut b.w1,
                step,
                batch.len(),
            );
            adam(
                &mut m.b1,
                &mut g.b1,
                &mut a.b1,
                &mut b.b1,
                step,
                batch.len(),
            );
            adam(
                &mut m.w2,
                &mut g.w2,
                &mut a.w2,
                &mut b.w2,
                step,
                batch.len(),
            );
            adam(
                &mut m.b2,
                &mut g.b2,
                &mut a.b2,
                &mut b.b2,
                step,
                batch.len(),
            );
            if let (Some(m), Some(g), Some(a), Some(b)) =
                (&mut m.branch, &mut g.branch, &mut a.branch, &mut b.branch)
            {
                adam(
                    &mut m.weights,
                    &mut g.weights,
                    &mut a.weights,
                    &mut b.weights,
                    step,
                    batch.len(),
                );
                adam(
                    &mut m.biases,
                    &mut g.biases,
                    &mut a.biases,
                    &mut b.biases,
                    step,
                    batch.len(),
                );
            }
        }
        completed = epoch;
        if !validation.is_empty() {
            let v = metrics(&m, d, validation, current_only, None)?;
            let loss = v["conditional_branch"]["visit_macro_cross_entropy"]
                .as_f64()
                .unwrap_or_else(|| v["visit_macro_cross_entropy"].as_f64().unwrap());
            if loss < best_loss {
                best_loss = loss;
                best = m.clone();
                best_epoch = epoch;
            }
            eprintln!(
                "{} epoch {epoch}/{epochs}: validation macro CE {loss:.4}, target accuracy {:.3}",
                if current_only {
                    "current-only"
                } else {
                    "two-frame"
                },
                v["target_class_accuracy"].as_f64().unwrap()
            );
            history.push(json!({"epoch":epoch,"validation":v}));
        } else {
            best = m.clone();
            best_epoch = epoch;
            eprintln!("full-corpus epoch {epoch}/{epochs}");
        }
        if start.elapsed().as_secs() >= MAX_TRAIN_SECONDS {
            break;
        }
    }
    best.validate()?;
    Ok((
        best,
        json!({"epochs_completed":completed,"selected_epoch":best_epoch,"selection":if branches{"lowest validation visit-macro conditional branch CE; target CE only if no validation branch labels; final median fold epoch"}else{"lowest held-out validation visit-macro target cross entropy; final median fold epoch"},"conditional_training_labels":branch_counts,"branch_head_trained":branches,"branch_head_minimum_per_training_class":10,"seconds":start.elapsed().as_secs_f64(),"current_frame_only":current_only,"history":history}),
    ))
}
fn artifact_hash(path: &Path) -> Result<String> {
    Ok(data::digest(&fs::read(path)?))
}

pub fn cold(corpus: &str, out: &str) -> Result<()> {
    cold_mode(corpus, out, false, false)
}
pub fn cold_mode(corpus: &str, out: &str, branches: bool, labels_only: bool) -> Result<()> {
    cold_inner(corpus, out, branches, labels_only, None, None)
}
#[cfg(feature = "sam31")]
pub fn cold_sam(args: &[String]) -> Result<()> {
    if !(4..=5).contains(&args.len()) || (args[0] == "cold-sam-branches" && args.len() != 4) {
        return Err("cold-sam-branches CORPUS CHECKPOINT NEW_OUT; sam-branch-labels CORPUS CHECKPOINT NEW_OUT [ARCHIVE]".into());
    }
    cold_inner(
        &args[1],
        &args[3],
        true,
        args[0] == "sam-branch-labels",
        Some(&args[2]),
        args.get(4).map(String::as_str),
    )
}
fn cold_inner(
    corpus: &str,
    out: &str,
    branches: bool,
    labels_only: bool,
    sam_checkpoint: Option<&str>,
    archive: Option<&str>,
) -> Result<()> {
    let out = data::output(out)?;
    let start = Instant::now();
    let source = boot::current_source(Path::new("."))?;
    let version = |name: &str, arg: &str| {
        std::process::Command::new(name)
            .arg(arg)
            .output()
            .ok()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_owned()
            })
    };
    data::write(
        out.join("runtime.json"),
        &json!({"command":std::env::args().collect::<Vec<_>>(),"rustc":version("rustc","--version"),"cargo":version("cargo","--version"),"ffmpeg":version("ffmpeg","-version"),"architecture":std::env::consts::ARCH,"cpu":fs::read_to_string("/proc/cpuinfo").ok().and_then(|s|s.lines().find(|l|l.starts_with("model name")).map(str::to_owned)),"model_training_and_inference_device":"cpu","shared_sam_teacher_device":if sam_checkpoint.is_some(){Some("cuda:0")}else{None},"random_seed":SEED,"max_seconds_per_fit":MAX_TRAIN_SECONDS,"model_training_inputs":"fresh RAW, recorded presentations, seeded random weights; optional explicitly pinned SAM root is freshly exported; no custom pretrained checkpoint or derived cache"}),
    )?;
    let (mut sources, inventory) = data::scan(corpus)?;
    sources.retain(|s| archive.is_none_or(|a| s.archive.contains(a)));
    data::write(out.join("inventory.json"), &inventory)?;
    let mut d = prepare(&sources, &out, branches && sam_checkpoint.is_none())?;
    data::write(out.join("roots.json"), &d.roots)?;
    data::write(out.join("exclusions.json"), &d.excluded)?;
    if d.examples.is_empty() {
        return Err("no target-attested fresh frame pairs".into());
    }
    let days: Vec<_> = d
        .examples
        .iter()
        .map(|e| e.day)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut feature_hash = Sha256::new();
    for f in &d.frames {
        feature_hash.update(f.hash.as_bytes());
        for p in &f.pixels {
            feature_hash.update(p.to_le_bytes());
        }
    }
    let feature_hash = format!("{:x}", feature_hash.finalize());
    let mut graph = json!({"schema":boot::SCHEMA,"source":source,"targets":["evaluation"],"nodes":[
        {"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]},
        {"id":"raw","kind":"raw","sha256":data::digest(&serde_json::to_vec(&d.roots["native_raw"])?),"dependencies":[]},
        {"id":"targets","kind":"measurements","sha256":data::digest(&serde_json::to_vec(&d.roots["target_measurements"])?),"dependencies":[]},
        {"id":"images","kind":"features","sha256":feature_hash,"dependencies":["source","raw","targets"]},
        {"id":"model","kind":"custom_model","planned":true,"sha256":null,"dependencies":["images","source","targets"]},
        {"id":"evaluation","kind":"evaluation","planned":true,"sha256":null,"dependencies":["model","images","source","targets"]}]});
    if branches {
        graph["nodes"].as_array_mut().unwrap().push(json!({"id":"conic-labels","kind":"derived_data","planned":true,"sha256":null,"dependencies":["raw","targets","source"]}));
        graph["nodes"][4]["dependencies"]
            .as_array_mut()
            .unwrap()
            .push(json!("conic-labels"));
        graph["nodes"][5]["dependencies"]
            .as_array_mut()
            .unwrap()
            .push(json!("conic-labels"));
    }
    if let Some(checkpoint) = sam_checkpoint {
        if super::sam_export::hash(Path::new(checkpoint))? != super::sam_export::CHECKPOINT_SHA {
            return Err("official SAM checkpoint hash mismatch".into());
        }
        let nodes = graph["nodes"].as_array_mut().unwrap();
        nodes.push(json!({"id":"sam31","kind":"sam3","sha256":super::sam_export::CHECKPOINT_SHA,"dependencies":[]}));
        nodes.push(json!({"id":"sam-export","kind":"export","planned":true,"sha256":null,"dependencies":["source","sam31"]}));
        nodes
            .iter_mut()
            .find(|n| n["id"] == "conic-labels")
            .unwrap()["dependencies"]
            .as_array_mut()
            .unwrap()
            .push(json!("sam-export"));
    }
    let manifest: boot::Manifest = serde_json::from_value(graph.clone())?;
    let certificate =
        boot::validate(&manifest, &source).map_err(|e| format!("bootstrap preflight: {e:?}"))?;
    data::write(out.join("bootstrap-graph-planned.json"), &graph)?;
    data::write(out.join("bootstrap-preflight.json"), &certificate)?;
    if let Some(checkpoint) = sam_checkpoint {
        #[cfg(feature = "sam31")]
        {
            let export = out.join("sam-export");
            super::sam_export::run(&[
                "sam-export".into(),
                checkpoint.into(),
                export.to_string_lossy().into_owned(),
            ])?;
            let teacher = super::sam_native::Teacher::open(&export)?;
            d.sam_teacher = Some(super::sam_native::prepare(&mut d, &out, &teacher)?);
            let node = graph["nodes"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|n| n["id"] == "sam-export")
                .unwrap();
            node["sha256"] = teacher.receipt["model_sha256"].clone();
            node["planned"] = json!(false);
        }
        #[cfg(not(feature = "sam31"))]
        {
            let _ = checkpoint;
            return Err("SAM cold training requires the sam31 build feature".into());
        }
    }
    if branches {
        d.branch_audit = Some(branch_labels::assign(&mut d, &out)?);
        let node = graph["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|n| n["id"] == "conic-labels")
            .unwrap();
        node["sha256"] = json!(data::digest(&serde_json::to_vec(
            d.branch_audit.as_ref().unwrap()
        )?));
        node["planned"] = json!(false);
    }
    // A second preflight records the actually regenerated teacher before any
    // custom optimizer consumes its labels. The original recipe stays intact.
    let prepared: boot::Manifest = serde_json::from_value(graph.clone())?;
    data::write(
        out.join("bootstrap-prepared.json"),
        &boot::validate(&prepared, &boot::current_source(Path::new("."))?)
            .map_err(|e| format!("prepared graph: {e:?}"))?,
    )?;
    data::write(out.join("bootstrap-graph-prepared.json"), &graph)?;
    if labels_only {
        if boot::current_source(Path::new("."))? != source {
            return Err("source changed during label generation".into());
        }
        data::write(out.join("label-source.json"), &source)?;
        eprintln!("LABELS DONE: {}", out.display());
        return Ok(());
    }
    if days.len() < 3 {
        return Err("three recording days required for held-out day evaluation".into());
    }
    if branches && d.examples.iter().filter(|e| e.branch.is_some()).count() < 50 {
        return Err(
            "fewer than 50 conditional branch labels; refusing to present an untrained sign head"
                .into(),
        );
    }
    let contract = json!({"schema":"buttercup-two-frame-sign-training-contract-v1","scope":"single-user Rob development; no deployment or cross-user claim","source":source,"input":"two consecutive native RAW ROI frames; 32x24 grayscale each after CFA-cell averaging, fractional box resize, 5-tap binomial blur, per-frame p5/p95 contrast normalization","model":"1536 -> 32 ReLU -> 9 softmax; target grid [.1,.5,.9]^2","seed":SEED,"training_device":"cpu","inference_device":"cpu","epochs_max":EPOCHS,"seconds_per_fit_max":MAX_TRAIN_SECONDS,"batch":BATCH,"learning_rate":0.001,"loss":"generalized cross entropy q=.4; inverse sqrt target-visit and class frequency weighting, clamped .1..5; not verified distraction separation","augmentation":"shared brightness gain .85..1.15 and offset +/-.08; additional per-frame gain .95..1.05 and offset +/-.02; no test adaptation","checkpoint":"validation target-visit macro cross entropy; full-data median selected epoch","evaluation":"each day wholly unseen in test; both eyes and all viewer sessions held together; validation uses separate entire viewer sessions on other days; current-only matched control","promotion_rule":"experimental artifact only; no signed-3D truth or deployment promotion from target accuracy","limits":["Commanded target is weak supervision: actual fixation is unmeasured.","Saved gaze affine, display pose, sign states and segmentation are not training inputs.","Calibration success is not an example-selection filter.","Known screen target alone cannot supply camera-relative 3D branch truth without extrinsics.","No available independent per-frame signed 3D truth; report target-direction accuracy separately.","No ellipse or measured point is modified: SN-FEIDA and localization are unchanged/unscored."]});
    let mut contract = contract;
    contract["limits"].as_array_mut().unwrap().push(json!("This corpus has been used for method development. Held-out days are disjoint from model fitting, but are not an untouched final benchmark."));
    if branches {
        contract["model"] = json!("1536 -> 32 ReLU -> 9 target softmax + optional 2-way conic branch head, canonical low/high camera-normal X");
        contract["conditional_branch_teacher"] = json!(branch_labels::CONTRACT);
        contract["conditional_branch_loss"] = json!("GCE q=.4, weight 2 times target-visit weight times sqrt(total/2/class_count), clamped .25..4; at least 10 training labels per class or head absent");
        contract["checkpoint"] = json!("lowest validation visit-macro branch CE when labeled; target CE otherwise; final median fold epoch");
        contract["geometry_provider"] =
            d.branch_audit.as_ref().unwrap()["geometry_provider"].clone();
        contract["limits"].as_array_mut().unwrap().push(json!("Branch labels are conditional fresh RAW geometry plus target/pose assumptions; their agreement is not independent 3D sign accuracy. Missing human localization and independent scale prevent a SN-FEIDA quality claim."));
    }
    data::write(out.join("contract.json"), &contract)?;
    contact(&d, &out, None)?;
    let mut results = Vec::new();
    let mut models = Vec::new();
    let mut selected_epochs = Vec::new();
    let mut held_predictions = Vec::new();
    for &day in &days {
        let test: Vec<_> = d
            .examples
            .iter()
            .enumerate()
            .filter(|(_, e)| e.day == day)
            .map(|(i, _)| i)
            .collect();
        let mut group_sizes = BTreeMap::<String, usize>::new();
        for e in d.examples.iter().filter(|e| e.day != day) {
            *group_sizes.entry(e.group.clone()).or_default() += 1;
        }
        let mut ranked: Vec<_> = group_sizes.into_iter().collect();
        ranked.sort_by_key(|(g, n)| (std::cmp::Reverse(*n), g.clone()));
        let validation_group = ranked
            .get(1)
            .or(ranked.first())
            .ok_or("missing independent validation group")?
            .0
            .clone();
        let validation: Vec<_> = d
            .examples
            .iter()
            .enumerate()
            .filter(|(_, e)| e.day != day && e.group == validation_group)
            .map(|(i, _)| i)
            .collect();
        let train: Vec<_> = d
            .examples
            .iter()
            .enumerate()
            .filter(|(_, e)| e.day != day && e.group != validation_group)
            .map(|(i, _)| i)
            .collect();
        if train.is_empty() || validation.is_empty() || test.is_empty() {
            return Err("empty held-out partition".into());
        }
        let raw_set = |ids: &[usize]| {
            ids.iter()
                .flat_map(|&i| {
                    d.examples[i]
                        .frames
                        .iter()
                        .map(|&f| d.frames[f].hash.as_str())
                })
                .collect::<BTreeSet<_>>()
        };
        let train_raw = raw_set(&train);
        let validation_raw = raw_set(&validation);
        let test_raw = raw_set(&test);
        if !train_raw.is_disjoint(&validation_raw)
            || !train_raw.is_disjoint(&test_raw)
            || !validation_raw.is_disjoint(&test_raw)
        {
            return Err("RAW content overlaps training, validation or test".into());
        }
        let test_groups: BTreeSet<_> = test.iter().map(|&i| &d.examples[i].group).collect();
        if train
            .iter()
            .chain(&validation)
            .any(|&i| test_groups.contains(&d.examples[i].group))
        {
            return Err("viewer session crosses day split".into());
        }
        let mut majority = [0usize; CLASSES];
        for &i in &train {
            majority[d.examples[i].class] += 1;
        }
        let majority_class = majority
            .iter()
            .enumerate()
            .max_by_key(|(_, n)| *n)
            .unwrap()
            .0;
        let mut branch_train = [0usize; 2];
        let mut branch_test = [0usize; 2];
        for &i in &train {
            if let Some(k) = d.examples[i].branch {
                branch_train[k] += 1;
            }
        }
        for &i in &test {
            if let Some(k) = d.examples[i].branch {
                branch_test[k] += 1;
            }
        }
        let majority_branch = usize::from(branch_train[1] > branch_train[0]);
        let branch_baseline = json!({"training_counts":branch_train,"test_counts":branch_test,"chosen_class":majority_branch,"matches":branch_test[majority_branch],"labeled_pairs":branch_test.iter().sum::<usize>(),"scope":"constant training-majority conditional branch; no test-based choice"});
        for current_only in [true, false] {
            eprintln!(
                "FIT day {day} current_only={current_only} train={} validation={} test={}",
                train.len(),
                validation.len(),
                test.len()
            );
            let (mut m, training) = fit(&d, &train, &validation, current_only, EPOCHS)?;
            m.provenance = json!({"source":source,"input_feature_sha256":feature_hash,"training_scope":"Rob-only","test_day":day,"validation_group":validation_group,"current_only":current_only});
            m.provenance["conditional_branch_teacher"] =
                contract["conditional_branch_teacher"].clone();
            m.provenance["geometry_provider"] = contract["geometry_provider"].clone();
            m.provenance["conditional_training_labels"] =
                training["conditional_training_labels"].clone();
            m.provenance["branch_labels_sha256"] = d
                .branch_audit
                .as_ref()
                .map(|a| a["label_file_sha256"].clone())
                .unwrap_or(Value::Null);
            let name = format!(
                "day-{day}-{}.json",
                if current_only {
                    "current-only"
                } else {
                    "two-frame"
                }
            );
            let path = out.join(&name);
            data::write(&path, &m)?;
            let predictions = out.join(format!("predictions-{name}l"));
            let test_metrics = metrics(&m, &d, &test, current_only, Some(&predictions))?;
            if !current_only {
                selected_epochs.push(training["selected_epoch"].as_u64().unwrap() as usize);
                held_predictions.extend(test.iter().map(|&i| {
                    (
                        i,
                        m.predict(
                            &d.frames[d.examples[i].frames[0]].pixels,
                            &d.frames[d.examples[i].frames[1]].pixels,
                        )
                        .unwrap(),
                    )
                }));
            }
            let reloaded: Model = serde_json::from_slice(&fs::read(&path)?)?;
            reloaded.validate()?;
            if serde_json::to_value(&reloaded)? != serde_json::to_value(&m)? {
                return Err("model reload differs".into());
            }
            models.push(json!({"path":name,"sha256":artifact_hash(&path)?}));
            results.push(json!({"test_day":day,"training_pairs":train.len(),"validation_pairs":validation.len(),"test_pairs":test.len(),"validation_group":validation_group,"arm":if current_only{"current-only"}else{"two-frame"},"training":training,"test":test_metrics,"majority_class":majority_class,"majority_test_accuracy":test.iter().filter(|&&i|d.examples[i].class==majority_class).count()as f64/test.len()as f64}));
            results.last_mut().unwrap()["majority_branch_baseline"] = branch_baseline.clone();
            data::write(out.join("folds-progress.json"), &results)?;
        }
    }
    selected_epochs.sort_unstable();
    let final_epochs = selected_epochs[selected_epochs.len() / 2];
    let all: Vec<_> = (0..d.examples.len()).collect();
    let (mut final_model, full_training) = fit(&d, &all, &[], false, final_epochs)?;
    if branches && final_model.branch.is_none() {
        return Err("full-corpus training could not train both sign classes; no binary sign model to publish".into());
    }
    final_model.provenance = json!({"source":source,"features_sha256":feature_hash,"supervision":"intended screen target; no independent signed-3D truth","training_scope":"Rob-only all eligible calibration corpus","training_pairs":all.len(),"requires_two_fresh_source_aligned_frames":true,"inference_device":"cpu","epochs":final_epochs,"calibrated_probabilities":false});
    final_model.provenance["conditional_branch_teacher"] =
        contract["conditional_branch_teacher"].clone();
    final_model.provenance["geometry_provider"] = contract["geometry_provider"].clone();
    final_model.provenance["conditional_training_labels"] =
        full_training["conditional_training_labels"].clone();
    final_model.provenance["branch_labels_sha256"] = d
        .branch_audit
        .as_ref()
        .map(|a| a["label_file_sha256"].clone())
        .unwrap_or(Value::Null);
    data::write(out.join("model.json"), &final_model)?;
    let reload: Model = serde_json::from_slice(&fs::read(out.join("model.json"))?)?;
    reload.validate()?;
    let mut max_diff = 0f32;
    let latency = Instant::now();
    for &i in all.iter().take(1000) {
        let e = &d.examples[i];
        let a =
            final_model.predict(&d.frames[e.frames[0]].pixels, &d.frames[e.frames[1]].pixels)?;
        let b = reload.predict(&d.frames[e.frames[0]].pixels, &d.frames[e.frames[1]].pixels)?;
        for j in 0..CLASSES {
            max_diff = max_diff.max((a.target_scores[j] - b.target_scores[j]).abs());
        }
        if let (Some(a), Some(b)) = (a.conditional_branch_scores, b.conditional_branch_scores) {
            for j in 0..2 {
                max_diff = max_diff.max((a[j] - b[j]).abs());
            }
        } else if a.conditional_branch_scores.is_some() != b.conditional_branch_scores.is_some() {
            return Err("reloaded branch head missing".into());
        }
    }
    let two_predictions_mean_ms =
        latency.elapsed().as_secs_f64() * 1000. / all.len().min(1000) as f64;
    if max_diff > 1e-7 {
        return Err("loaded model inference drift".into());
    }
    models.push(json!({"path":"model.json","sha256":artifact_hash(&out.join("model.json"))?}));
    contact(&d, &out, Some(&held_predictions))?;
    let pooled = |arm: &str| {
        let rows: Vec<_> = results.iter().filter(|v| v["arm"] == arm).collect();
        let n: usize = rows
            .iter()
            .map(|v| v["test"]["pairs"].as_u64().unwrap() as usize)
            .sum();
        let correct: usize = rows
            .iter()
            .map(|v| v["test"]["target_class_correct"].as_u64().unwrap() as usize)
            .sum();
        json!({"pairs":n,"target_class_correct":correct,"target_class_accuracy":correct as f64/n as f64,"axes":(0..2).map(|a|{let total:usize=rows.iter().map(|v|v["test"]["noncentral_signs"][a]["n"].as_u64().unwrap()as usize).sum();let correct:usize=rows.iter().map(|v|v["test"]["noncentral_signs"][a]["correct"].as_u64().unwrap()as usize).sum();let support:usize=rows.iter().map(|v|v["test"]["noncentral_signs"][a]["support_ge_0_8"].as_u64().unwrap()as usize).sum();let sc:usize=rows.iter().map(|v|v["test"]["noncentral_signs"][a]["supported_correct"].as_u64().unwrap()as usize).sum();json!({"axis":a,"n":total,"correct":correct,"accuracy":correct as f64/total.max(1)as f64,"support_ge_0_8":support,"supported_correct":sc,"supported_accuracy":if support>0{Some(sc as f64/support as f64)}else{None}})}).collect::<Vec<_>>()})
    };
    let report = json!({"schema":"buttercup-calibration-sign-results-v1","source":source,"inventory":inventory,"training_pairs":d.examples.len(),"unique_decoded_raw":d.frames.len(),"excluded":d.excluded,"days":days,"folds":results,"pooled_current_only":pooled("current-only"),"pooled_two_frame":pooled("two-frame"),"final_full_corpus_training":full_training,"model_artifacts":models,"reload_max_absolute_difference":max_diff,"two_predictions_mean_ms":two_predictions_mean_ms,"wall_seconds":start.elapsed().as_secs_f64(),"signed_3d_accuracy":null,"limitations":contract["limits"],"required_future_evidence":"independent camera/display projection or camera-relative sign labels for physical conic-sign accuracy; additional people and lighting transitions for generalization"});
    let mut report = report;
    report["branch_label_audit"] = d.branch_audit.clone().unwrap_or(Value::Null);
    for arm in ["current-only", "two-frame"] {
        let rows: Vec<_> = results.iter().filter(|v| v["arm"] == arm).collect();
        let sum = |key: &str| {
            rows.iter()
                .map(|r| r["test"]["conditional_branch"][key].as_u64().unwrap_or(0))
                .sum::<u64>()
        };
        let n = sum("labeled_pairs");
        let c = sum("matches");
        let support = sum("support_ge_0_8");
        let sc = sum("supported_matches");
        let total_labels = rows
            .iter()
            .map(|r| {
                r["majority_branch_baseline"]["labeled_pairs"]
                    .as_u64()
                    .unwrap_or(0)
            })
            .sum::<u64>();
        let majority = rows
            .iter()
            .map(|r| {
                r["majority_branch_baseline"]["matches"]
                    .as_u64()
                    .unwrap_or(0)
            })
            .sum::<u64>();
        let ratio = |c: u64, n: u64| {
            if n > 0 {
                Some(c as f64 / n as f64)
            } else {
                None
            }
        };
        let dst = if arm == "two-frame" {
            "pooled_two_frame"
        } else {
            "pooled_current_only"
        };
        report[dst]["conditional_branch"] = json!({"available_labels":total_labels,"evaluated_labels":n,"matches":c,"agreement":ratio(c,n),"support_ge_0_8":support,"supported_matches":sc,"supported_agreement":ratio(sc,support),"training_majority_matches":majority,"training_majority_agreement":ratio(majority,total_labels),"teacher_coverage":ratio(total_labels,d.examples.len()as u64),"evaluated_coverage":ratio(n,d.examples.len()as u64),"scope":"conditional teacher agreement; absent training labels cause an absent head, not a random prediction"});
    }
    data::write(out.join("results.json"), &report)?;
    graph["nodes"][4]["sha256"] = json!(data::digest(&serde_json::to_vec(&models)?));
    graph["nodes"][4]["planned"] = json!(false);
    graph["nodes"][5]["sha256"] = json!(artifact_hash(&out.join("results.json"))?);
    graph["nodes"][5]["planned"] = json!(false);
    let finish = boot::current_source(Path::new("."))?;
    let complete: boot::Manifest = serde_json::from_value(graph.clone())?;
    let cert =
        boot::validate(&complete, &finish).map_err(|e| format!("final bootstrap check: {e:?}"))?;
    data::write(out.join("bootstrap-graph.json"), &graph)?;
    data::write(out.join("bootstrap-final-structural-check.json"), &cert)?;
    eprintln!("DONE: {}", out.display());
    Ok(())
}

fn contact(
    d: &Dataset,
    out: &Path,
    predictions: Option<&[(usize, net::Prediction)]>,
) -> Result<()> {
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, e) in d.examples.iter().enumerate() {
        if seen.insert((e.day, e.class)) {
            selected.push(i);
        }
    }
    let scale = 4;
    let w = WIDTH * scale * 2;
    let rowh = HEIGHT * scale + 24;
    let h = rowh * selected.len();
    let mut rgb = vec![24u8; w * h * 3];
    let mut labels = Vec::new();
    for (row, &i) in selected.iter().enumerate() {
        let e = &d.examples[i];
        for t in 0..2 {
            let p = &d.frames[e.frames[t]].pixels;
            for y in 0..HEIGHT * scale {
                for x in 0..WIDTH * scale {
                    let gray =
                        ((p[(y / scale) * WIDTH + x / scale] + 1.) * 127.5).clamp(0., 255.) as u8;
                    let pos = ((row * rowh + y) * w + t * WIDTH * scale + x) * 3;
                    rgb[pos..pos + 3].fill(gray);
                }
            }
        }
        let label = if let Some(p) = predictions
            .and_then(|p| p.iter().find(|(j, _)| *j == i))
            .map(|(_, p)| p)
        {
            format!(
                "day{} target({:.1},{:.1}) pred({:.2},{:.2})",
                e.day, e.uv[0], e.uv[1], p.uv[0], p.uv[1]
            )
        } else {
            format!(
                "day{} target({:.1},{:.1}) prev | current",
                e.day, e.uv[0], e.uv[1]
            )
        };
        labels.push(json!({"row":row,"pair_index":i,"label":label,"source":e.source}));
    }
    let name = if predictions.is_some() {
        "held-out-contact"
    } else {
        "training-input-contact"
    };
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.extend(rgb);
    fs::write(out.join(format!("{name}.ppm")), ppm)?;
    data::write(out.join(format!("{name}.json")), &labels)?;
    let filters: Vec<_> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| {
            format!(
                "drawtext=text='{}':x=3:y={}:fontsize=10:fontcolor=white",
                l["label"].as_str().unwrap(),
                i * rowh + HEIGHT * scale + 5
            )
        })
        .collect();
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-y",
            "-i",
        ])
        .arg(out.join(format!("{name}.ppm")))
        .args(["-vf", &filters.join(","), "-frames:v", "1", "-threads", "1"])
        .arg(out.join(format!("{name}.png")))
        .status()?;
    if !status.success() {
        return Err("contact rendering failed".into());
    }
    Ok(())
}
