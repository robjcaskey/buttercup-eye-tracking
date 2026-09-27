//! CPU-only synthetic sign-expert arbitration trial. Not a live gaze model.
//! RAW reflection statistics set missingness/clipping scenarios, never sign truth.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
#[path = "../bootstrapability.rs"]
mod bootstrapability;
use std::path::Path;

#[allow(dead_code)]
#[path = "../geometry.rs"]
mod geometry;
#[allow(dead_code)]
#[path = "../eye_scene_model/perspective_sign.rs"]
mod perspective_sign;
#[allow(dead_code)]
#[path = "../raw10.rs"]
mod raw10;
#[allow(dead_code)]
#[path = "../eye_scene_model/sign_kinematic_beam.rs"]
mod sign_kinematic_beam;
#[allow(dead_code)]
#[path = "../"]
mod native {
    pub(crate) mod binocular_coordinator;
    pub(crate) mod conic_solver;
    pub(crate) mod outline_conic_segments;
    pub(crate) mod roi_evidence;
    pub(crate) mod eye_scene_model {
        pub(crate) mod binocular_pose;
    }
}
use native::{
    binocular_coordinator, conic_solver, eye_scene_model, outline_conic_segments, roi_evidence,
};
#[path = "buttercup_sign_expert_trial/scene.rs"]
mod scene;
use scene::generate;
const EXPERTS: usize = 6;
const N: usize = EXPERTS * 2;
const EXPERT_NAMES: [&str; EXPERTS] = [
    "nearest",
    "velocity",
    "pivot",
    "relative_glint",
    "native_kinematic_beam",
    "native_perspective_history",
];
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}
struct Sample {
    x: [f64; N],
    label: f64,
    regime: usize,
    session: usize,
    normal_errors_deg: [f64; 2],
    separation_deg: f64,
    scene: Value,
}
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-40.0, 40.0)).exp())
}
fn metrics(samples: &[Sample], weights: &[f64; N]) -> Value {
    let mut correct = [0usize; 6];
    let mut total = [0usize; 6];
    let mut loss = 0.0;
    let mut angular_error = 0.0;
    let mut separated = 0;
    let mut separated_correct = 0;
    for s in samples {
        let p = sigmoid(s.x.iter().zip(weights).map(|(x, w)| x * w).sum());
        angular_error += s.normal_errors_deg[usize::from(p < 0.5)];
        if s.separation_deg >= 3.0 {
            separated += 1;
            separated_correct += usize::from((p >= 0.5) == (s.label == 1.0));
        }
        correct[s.regime] += usize::from((p >= 0.5) == (s.label == 1.0));
        total[s.regime] += 1;
        loss -= s.label * p.max(1e-12).ln() + (1.0 - s.label) * (1.0 - p).max(1e-12).ln();
    }
    json!({"mean_normal_error_deg":angular_error/samples.len() as f64,"separated_count":separated,"separated_accuracy":separated_correct as f64/separated.max(1) as f64,"count":samples.len(),"accuracy":correct.iter().sum::<usize>() as f64/samples.len() as f64,"log_loss":loss/samples.len() as f64,"regimes":(0..6).map(|r|json!({"regime":r,"count":total[r],"accuracy":correct[r] as f64/total[r] as f64})).collect::<Vec<_>>()})
}
fn evidence(
    out: &Path,
    samples: &[Sample],
    weights: &[f64; N],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut trace = std::io::BufWriter::new(File::create(out.join("heldout-evidence.jsonl"))?);
    let mut sessions = std::collections::BTreeMap::<usize, (usize, usize, usize)>::new();
    for s in samples {
        let p = sigmoid(s.x.iter().zip(weights).map(|(x, w)| x * w).sum());
        let chosen = usize::from(p < 0.5);
        let score = sessions.entry(s.session).or_default();
        score.0 += 1;
        score.1 += usize::from((p >= 0.5) == (s.label == 1.0));
        score.2 += usize::from((s.x[2] >= 0.0) == (s.label == 1.0));
        writeln!(
            trace,
            "{}",
            json!({"session":s.session,"regime":s.regime,"truth_candidate":if s.label==1.0 {0}else{1},"selected_candidate":chosen,"model_score":p,"expert_evidence":s.x,"normal_errors_deg":s.normal_errors_deg,"separation_deg":s.separation_deg,"scene":s.scene})
        )?;
    }
    trace.flush()?;
    let differences: Vec<_> = sessions
        .values()
        .map(|(n, c, b)| (*c as f64 - *b as f64) / *n as f64)
        .collect();
    let mean = differences.iter().sum::<f64>() / differences.len() as f64;
    let se = (differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
        / ((differences.len() - 1) * differences.len()) as f64)
        .sqrt();
    fs::write(
        out.join("session-comparison.json"),
        serde_json::to_vec_pretty(
            &json!({"session_count":sessions.len(),"mean_accuracy_gain_over_pivot":mean,"approx_normal_95_interval":[mean-1.96*se,mean+1.96*se],"sessions":sessions.iter().map(|(id,(n,c,b))|json!({"session":id,"count":n,"combination_correct":c,"pivot_correct":b})).collect::<Vec<_>>(),"limitation":"interval summarizes independent synthetic sessions, not real-user uncertainty"}),
        )?,
    )?;
    let mut svg=String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='820' viewBox='0 0 1000 820'><rect width='100%' height='100%' fill='#101c25'/><g fill='white' font-family='sans-serif'><text x='20' y='28' font-size='19'>Synthetic perspective/glint inspection — worst selected error in four regimes</text><text x='20' y='50' font-size='13'>Yellow: observed conic. Blue/magenta: candidate normals and predicted glints. Red: observed glint.</text>");
    for (panel, regime) in [0, 3, 4, 5].into_iter().enumerate() {
        let s = samples
            .iter()
            .filter(|s| s.regime == regime)
            .max_by(|a, b| {
                let err = |s: &Sample| {
                    s.normal_errors_deg[usize::from(
                        sigmoid(s.x.iter().zip(weights).map(|(x, w)| x * w).sum()) < 0.5,
                    )]
                };
                err(a).total_cmp(&err(b))
            })
            .unwrap();
        let x = (panel % 2) as f64 * 500.0;
        let y = (panel / 2) as f64 * 370.0 + 75.0;
        let v = &s.scene;
        let cx = v["ellipse"][0].as_f64().unwrap();
        let cy = v["ellipse"][1].as_f64().unwrap();
        let scale = 1.8;
        let px = |a: f64| x + 250.0 + (a - cx) * scale;
        let py = |a: f64| y + 180.0 + (a - cy) * scale;
        svg.push_str(&format!("<text x='{}' y='{}'>Regime {} · session {} · errors {:.1}/{:.1}°</text><ellipse cx='{}' cy='{}' rx='{}' ry='{}' transform='rotate({} {} {})' fill='none' stroke='#ffe54b'/>",x+15.0,y+20.0,regime,s.session,s.normal_errors_deg[0],s.normal_errors_deg[1],px(cx),py(cy),v["ellipse"][2].as_f64().unwrap()*scale,v["ellipse"][3].as_f64().unwrap()*scale,v["ellipse"][4].as_f64().unwrap().to_degrees(),px(cx),py(cy)));
        for (j, color) in ["#00caff", "#ff47cf"].iter().enumerate() {
            let a = &v["normal_endpoints"][j];
            let g = &v["predicted_glints"][j];
            svg.push_str(&format!("<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='{}' stroke-width='2'/><circle cx='{}' cy='{}' r='5' fill='none' stroke='{}'/>",px(cx),py(cy),px(a[0].as_f64().unwrap()),py(a[1].as_f64().unwrap()),color,px(g[0].as_f64().unwrap()),py(g[1].as_f64().unwrap()),color));
        }
        let g = &v["observed_glint"];
        if v["glint_available"] == true {
            svg.push_str(&format!(
                "<circle cx='{}' cy='{}' r='4' fill='#ff645c'/>",
                px(g[0].as_f64().unwrap()),
                py(g[1].as_f64().unwrap())
            ));
        }
    }
    svg.push_str("</g></svg>");
    fs::write(out.join("synthetic-failures.svg"), svg)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err(
            "usage: buttercup_sign_expert_trial RAW_GLINT_JSONL NEW_OUTPUT_DIRECTORY".into(),
        );
    }
    let out = Path::new(&args[2]);
    fs::create_dir(out)?;
    let mut stats = Vec::new();
    let mut inventory = Vec::new();
    for line in BufReader::new(File::open(&args[1])?).lines() {
        let v: Value = serde_json::from_str(&line?)?;
        if v["schema"] != "buttercup-raw-reflection-evidence-v1" {
            return Err("unexpected RAW reflection schema".into());
        }
        if v["selection"] != "all-native-capture-frames" {
            return Err("training rejects model-selected/external request subsets".into());
        }
        let raw = &v["raw_source"];
        let path = raw["raw_file"].as_str().ok_or("missing native source")?;
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(
            raw["raw_offset"].as_u64().ok_or("missing native offset")?,
        ))?;
        let length = raw["raw_length"].as_u64().ok_or("missing native length")?;
        if length > 128 * 1024 * 1024 {
            return Err("native frame too large".into());
        }
        let mut bytes = vec![0; length as usize];
        file.read_exact(&mut bytes)?;
        if v["raw_sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
            return Err("native RAW hash mismatch".into());
        }
        inventory.push(json!({"source":raw,"sha256":v["raw_sha256"]}));
        let missing = v["candidates"]
            .as_array()
            .ok_or("missing candidates")?
            .iter()
            .all(|p| p["touches_border"] == true);
        stats.push((
            missing,
            v["clipped_fraction"].as_f64().ok_or("missing clipping")?,
        ));
    }
    if stats.is_empty() {
        return Err("no native reflection evidence".into());
    }
    let source = bootstrapability::current_source(Path::new("."))?;
    let inventory_bytes = serde_json::to_vec_pretty(&inventory)?;
    fs::write(out.join("raw-inventory.json"), &inventory_bytes)?;
    let graph = json!({"schema":"buttercup-bootstrap-graph-v1","source":source,"targets":["model"],"nodes":[
        {"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]},
        {"id":"raw","kind":"raw","sha256":format!("{:x}",Sha256::digest(&inventory_bytes)),"dependencies":[]},
        {"id":"features","kind":"features","sha256":format!("{:x}",Sha256::digest(fs::read(&args[1])?)),"dependencies":["raw","source"]},
        {"id":"model","kind":"custom_model","sha256":null,"planned":true,"dependencies":["features","source"]}]});
    fs::write(out.join("graph.json"), serde_json::to_vec_pretty(&graph)?)?;
    let manifest =
        bootstrapability::parse(&serde_json::to_vec(&graph)?).map_err(|e| format!("{e:?}"))?;
    let certificate =
        bootstrapability::validate(&manifest, &source).map_err(|e| format!("{e:?}"))?;
    fs::write(
        out.join("preflight.json"),
        serde_json::to_vec_pretty(&certificate)?,
    )?;
    // RAW statistics condition training only. Validation/test use declared
    // synthetic missingness/clipping levels, not held-out real-user evidence.
    let train = generate(&stats, 0, 180);
    let validation_stats = [(false, 0.0), (false, 0.01), (true, 0.0)];
    let test_stats = [(false, 0.0), (false, 0.02), (true, 0.0), (false, 0.005)];
    let valid = generate(&validation_stats, 1000, 60);
    let test = generate(&test_stats, 5000, 60);
    let mut best = [0.0; N];
    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0;
    let mut best_mask = 0;
    let mut subsets = Vec::new();
    for mask in 1..(1 << EXPERTS) {
        let mut weights = [0.0; N];
        let mut local_best = weights;
        let mut local_loss = f64::INFINITY;
        let mut local_epoch = 0;
        for epoch in 0..400 {
            let mut gradient = [0.0; N];
            for s in &train {
                let p = sigmoid(s.x.iter().zip(weights).map(|(x, w)| x * w).sum());
                for j in 0..N {
                    gradient[j] += (p - s.label) * s.x[j];
                }
            }
            for j in 0..N {
                if mask & (1 << (j % EXPERTS)) != 0 {
                    weights[j] -= 0.2 * (gradient[j] / train.len() as f64 + 0.001 * weights[j]);
                }
            }
            let loss = metrics(&valid, &weights)["log_loss"].as_f64().unwrap();
            if loss < local_loss {
                local_loss = loss;
                local_best = weights;
                local_epoch = epoch;
            }
        }
        subsets.push(
            json!({"expert_mask":mask,"validation_log_loss":local_loss,"weights":local_best}),
        );
        if local_loss < best_loss {
            best_loss = local_loss;
            best = local_best;
            best_epoch = local_epoch;
            best_mask = mask;
        }
    }
    fs::write(
        out.join("validation-subsets.json"),
        serde_json::to_vec_pretty(&subsets)?,
    )?;
    let mut comparisons = Vec::new();
    for (j, name) in EXPERT_NAMES.iter().enumerate() {
        let mut w = [0.0; N];
        w[j] = 4.0;
        comparisons.push(json!({"method":name,"metrics":metrics(&test,&w)}));
    }
    comparisons.push(json!({"method":"equal_weight","metrics":metrics(&test,&[1.0;N])}));
    comparisons.push(json!({"method":"learned_combination","metrics":metrics(&test,&best)}));
    let model = json!({"schema":"buttercup-sign-expert-synthetic-trial-v1","weights":best,"bias":0,"features":EXPERT_NAMES.iter().map(|s|s.to_string()).chain(EXPERT_NAMES.iter().map(|s|format!("{s}_reliability"))).collect::<Vec<_>>(),"expert_mask":best_mask,"checkpoint_epoch":best_epoch,"device":"cpu","scope":"synthetic-perspective-twin-arbitration-not-real-gaze","input_sha256":format!("{:x}",Sha256::digest(fs::read(&args[1])?))});
    fs::write(out.join("model.json"), serde_json::to_vec_pretty(&model)?)?;
    let reloaded: Value = serde_json::from_slice(&fs::read(out.join("model.json"))?)?;
    let mut loaded_weights = [0.0; N];
    for j in 0..N {
        loaded_weights[j] = reloaded["weights"][j]
            .as_f64()
            .ok_or("invalid saved weight")?;
    }
    if loaded_weights != best {
        return Err("saved model round-trip differs".into());
    }
    evidence(out, &test, &best)?;
    let report = json!({"selected_expert_mask":best_mask,"model_reload_verified":true,"heldout_synthetic_sessions":[5000,5059],"train_samples":train.len(),"validation_samples":valid.len(),"test_samples":test.len(),"comparisons":comparisons,"train_session_range":[train.first().unwrap().session,train.last().unwrap().session],"regimes":["crossing","turnaround","rapid_turn","head_motion","flagged_glare","unflagged_glare"],"limitations":["spherical specular model assumes known light and sphere geometry; no refraction or glasses","reliability flags and absolute pivot support are simulated and may be easier than real estimation","no real sign labels or real gaze accuracy","no probability calibration","not promoted to live viewer"]});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
