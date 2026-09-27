//! Conditional sign labels from freshly regenerated RAW conics
//! and recorded targets. This is an engineering teacher, not physical truth.
use super::{data, train::Dataset, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::Path,
};

pub const CONTRACT: &str = "Weak conic-sign labels: fresh RAW geometry from the declared provider (classical gradient RANSAC or regenerated pinned SAM3.1 mask/shared contour fit), plus outward RAW contrast on left/bottom/right; nominal focal=4000px, principal=(4000,3000); stable approximately upright camera/display; camera yaw maps oppositely to screen X, pitch maps with screen Y, projected roll grid +/-15deg. Per-recording/eye target families fit positive-scale rotated angular maps. All near-best grid maps must agree; this is not a continuous certification, measured sign truth, verified fixation or human localization.";
#[derive(Clone, Serialize, Deserialize)]
pub struct Map {
    pub roll: f64,
    pub slope: [f64; 2],
    pub offset: [f64; 2],
    pub rmse: f64,
}
impl Map {
    pub fn project(&self, q: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.roll.sin_cos();
        let r = [c * q[0] + s * q[1], -s * q[0] + c * q[1]];
        [
            self.slope[0] * r[0] + self.offset[0],
            self.slope[1] * r[1] + self.offset[1],
        ]
    }
}
pub fn canonical(normals: [[f64; 3]; 2]) -> Option<[[f64; 2]; 2]> {
    if normals.iter().flatten().any(|x| !x.is_finite())
        || normals.iter().any(|n| n[2] <= 0.)
        || (normals[0][0] - normals[1][0]).abs() < 0.08
    {
        return None;
    }
    let mut n = normals;
    if n[0][0] > n[1][0] {
        n.swap(0, 1);
    }
    Some(n.map(|n| [-n[0].atan2(n[2]), n[1].clamp(-1., 1.).asin()]))
}
fn median(mut x: Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    x[x.len() / 2]
}
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
#[derive(Clone)]
struct Target {
    uv: [f64; 2],
    q: [[f64; 2]; 2],
    n: usize,
}
fn line(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let var = x.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
    if var < 0.001 {
        return None;
    }
    let a = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / var;
    (0.1..8.).contains(&a).then_some((a, my - a * mx))
}
fn maps(groups: &[Target]) -> Vec<Map> {
    if groups.len() < 4
        || !(0..2)
            .all(|a| groups.iter().any(|g| g.uv[a] < 0.2) && groups.iter().any(|g| g.uv[a] > 0.8))
    {
        return vec![];
    }
    let mut result = Vec::new();
    for mask in 0..(1usize << groups.len()) {
        for roll_deg in [-15f64, -7.5, 0., 7.5, 15.] {
            let roll = roll_deg.to_radians();
            let (s, c) = roll.sin_cos();
            let xy = groups
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let q = g.q[(mask >> i) & 1];
                    [c * q[0] + s * q[1], -s * q[0] + c * q[1]]
                })
                .collect::<Vec<_>>();
            let a = line(
                &xy.iter().map(|p| p[0]).collect::<Vec<_>>(),
                &groups.iter().map(|g| g.uv[0]).collect::<Vec<_>>(),
            );
            let b = line(
                &xy.iter().map(|p| p[1]).collect::<Vec<_>>(),
                &groups.iter().map(|g| g.uv[1]).collect::<Vec<_>>(),
            );
            if let (Some(a), Some(b)) = (a, b) {
                let mut m = Map {
                    roll,
                    slope: [a.0, b.0],
                    offset: [a.1, b.1],
                    rmse: 0.,
                };
                let errors = groups
                    .iter()
                    .enumerate()
                    .map(|(i, g)| dist(m.project(g.q[(mask >> i) & 1]), g.uv))
                    .collect::<Vec<_>>();
                m.rmse = (errors.iter().map(|e| e * e).sum::<f64>() / groups.len() as f64).sqrt();
                if m.rmse <= 0.14 && errors.iter().all(|e| *e < 0.30) {
                    result.push(m);
                }
            }
        }
    }
    result.sort_by(|a, b| a.rmse.total_cmp(&b.rmse));
    if let Some(best) = result.first().map(|m| m.rmse) {
        result.retain(|m| m.rmse.powi(2) <= best.powi(2) + 0.0025);
    }
    result
}
pub fn assign(d: &mut Dataset, out: &Path) -> Result<Value> {
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    for (i, e) in d.examples.iter().enumerate() {
        let f = &d.frames[e.frames[1]].source["frame"];
        let key = format!(
            "{}:{}:{}",
            e.source["archive"].as_str().ok_or("archive")?,
            f["eye_id"],
            f["source_clock"]["source_key"]["stream_epoch"]
        );
        families.entry(key).or_default().push(i);
    }
    let mut writer = BufWriter::new(fs::File::create(out.join("branch-labels.jsonl"))?);
    let mut geometry = BufWriter::new(fs::File::create(out.join("native-conics.jsonl"))?);
    for f in &d.frames {
        serde_json::to_writer(
            &mut geometry,
            &json!({"source":f.source,"fit":f.conic,"admissible":f.conic.as_ref().is_some_and(|c|c.admissible())}),
        )?;
        geometry.write_all(b"\n")?;
    }
    geometry.flush()?;
    let mut audits = Vec::new();
    let mut class_counts = [0; 2];
    let mut reasons = BTreeMap::<String, usize>::new();
    let mut by_day = BTreeMap::<u64, [usize; 3]>::new();
    for (family, ids) in families {
        let mut sites = BTreeMap::<usize, Vec<[[f64; 2]; 2]>>::new();
        for &i in &ids {
            let e = &d.examples[i];
            if let Some(f) = d.frames[e.frames[1]]
                .conic
                .as_ref()
                .filter(|f| f.admissible())
            {
                if let Some(q) = canonical(f.normals) {
                    sites.entry(e.class).or_default().push(q);
                }
            }
        }
        let targets = sites
            .iter()
            .filter(|(_, v)| v.len() >= 3)
            .map(|(class, v)| Target {
                uv: buttercup_eye_tracking::calibration_sign_model::GRID[*class].map(|v| v as f64),
                q: std::array::from_fn(|b| {
                    std::array::from_fn(|a| median(v.iter().map(|q| q[b][a]).collect()))
                }),
                n: v.len(),
            })
            .collect::<Vec<_>>();
        let hypotheses = maps(&targets);
        let before = class_counts.iter().sum::<usize>();
        for &i in &ids {
            let e = &d.examples[i];
            let f = &d.frames[e.frames[1]];
            let mut label = None;
            let reason = match f.conic.as_ref() {
                None => "no-native-conic",
                Some(c) if !c.admissible() => "insufficient-direct-rim-support",
                Some(c) => match canonical(c.normals) {
                    None => "camera-x-branch-order-uncertain",
                    Some(q) if hypotheses.is_empty() => {
                        let _ = q;
                        "no-coherent-multi-target-orientation"
                    }
                    Some(q) => {
                        let mut unanimous = None;
                        let mut supported = true;
                        for m in &hypotheses {
                            let uv = e.uv.map(|v| v as f64);
                            let errors = q.map(|q| dist(m.project(q), uv));
                            let best = usize::from(errors[1] < errors[0]);
                            if errors[best] > 0.20
                                || errors[1 - best] - errors[best] < 0.12
                                || unanimous.is_some_and(|k| k != best)
                            {
                                supported = false;
                                break;
                            }
                            unanimous = Some(best);
                        }
                        if supported {
                            label = unanimous;
                            "conditional-raw-target-label"
                        } else {
                            "orientation-family-or-frame-ambiguous"
                        }
                    }
                },
            };
            *reasons.entry(reason.into()).or_default() += 1;
            by_day.entry(e.day).or_default()[2] += 1;
            if let Some(k) = label {
                class_counts[k] += 1;
                by_day.entry(e.day).or_default()[k] += 1;
            }
            serde_json::to_writer(
                &mut writer,
                &json!({"pair_index":i,"source":e.source,"class":label,"reason":reason,"conditional":true,"family":family,"maps":hypotheses.len()}),
            )?;
            writer.write_all(b"\n")?;
            d.examples[i].branch = label;
        }
        audits.push(json!({"family":family,"pairs":ids.len(),"supported_targets":targets.iter().map(|g|json!({"uv":g.uv,"frames":g.n,"camera_angle_medians":g.q})).collect::<Vec<_>>(),"near_best_maps":hypotheses,"labeled_pairs":class_counts.iter().sum::<usize>()-before}));
    }
    writer.flush()?;
    let result = json!({"contract":CONTRACT,"geometry_provider":if d.sam_teacher.is_some(){"sam31-single"}else{"classical-raw"},"sam_teacher":d.sam_teacher,"pairs":d.examples.len(),"labels_low_high_camera_x":class_counts,"labels_by_day":by_day,"reasons":reasons,"families":audits,"independent_signed_accuracy":null,"label_file_sha256":data::digest(&fs::read(out.join("branch-labels.jsonl"))?),"native_conics_sha256":data::digest(&fs::read(out.join("native-conics.jsonl"))?),"all_images_retained_for_target_supervision":true,"labeling_is_retrospective_inference_is_two_frames_only":true,"missing_validation":["human limbus localization","independent 3D signs","measured intrinsics/extrinsics","independent scale/SN-FEIDA","off-target fixation identity"]});
    data::write(out.join("branch-label-audit.json"), &result)?;
    Ok(result)
}
