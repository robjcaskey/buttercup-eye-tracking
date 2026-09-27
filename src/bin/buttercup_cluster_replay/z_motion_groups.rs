//! RAW multi-frame shared-surface experiment; fitted and held frames are disjoint.
use super::{digest, frame, json, motion, png, reduced, sphere, Picture, Result, Value, P};
use std::{fs, path::Path};
#[path = "z_motion_groups_math.rs"]
mod engine;
#[path = "z_motion_groups_history.rs"]
mod history;
#[path = "z_motion_groups_pixels.rs"]
mod pixels;
#[path = "z_motion_scene.rs"]
mod scene;
pub use history::run as run_history;

fn fine_scores(pictures: &[Picture], d: &engine::Discovery) -> Vec<Value> {
    let images = pictures.iter().map(motion::normalized).collect::<Vec<_>>();
    let ps = pixels::patches(&images[0], 3);
    d.frames.iter().map(|f|{let mut one=0.;let mut separate=0.;let mut shared=0.;let mut n=0;
        for p in &ps {let q=p.p.map(|x|x/3.);let Some((id,dist))=d.points.iter().enumerate().map(|(i,p)|(i,(p[0]-q[0]).hypot(p[1]-q[1]))).min_by(|a,b|a.1.total_cmp(&b.1))else{continue;};if dist>0.8{continue;}
            let loss=|m:motion::Model| {pixels::error(p,&images[f.index],m.scaled(3.))};
            one+=loss(f.single);shared+=loss(f.shared_groups[d.labels[id]]);separate+=loss(f.groups[d.labels[id]]);n+=1;
        }json!({"index":f.index,"held":f.held,"single_error":one/n.max(1)as f64,"shared_error":shared/n.max(1)as f64,"grouped_error":separate/n.max(1)as f64,"samples":n})
    }).collect()
}
pub fn run(
    out: &Path,
    ultra: &[Picture],
    coarse: &[Picture],
    frames: Vec<Value>,
    crops: &[P],
    source: Value,
) -> Result<()> {
    if ultra.len() < 5
        || ultra.len() % 2 != 1
        || pixels::patches(&motion::normalized(&ultra[0]), 1).len() < 24
    {
        return Err(
            "odd sequence length >=5 and at least 24 textured source patches required".into(),
        );
    }
    let times = frames
        .iter()
        .map(|f| f["time"].as_f64().unwrap())
        .collect::<Vec<_>>();
    let d = engine::discover(ultra, crops, &times);
    let fine = fine_scores(coarse, &d);
    let held = d.frames.iter().filter(|f| f.held).collect::<Vec<_>>();
    let n = held.len() as f64;
    let single = held.iter().map(|f| f.single_error).sum::<f64>() / n;
    let shared_error = held.iter().map(|f| f.shared_error).sum::<f64>() / n;
    let grouped = held.iter().map(|f| f.grouped_error).sum::<f64>() / n;
    let swapped = held.iter().map(|f| f.swapped_error).sum::<f64>() / n;
    let fine_mean = |key: &str| {
        fine.iter()
            .filter(|f| f["held"] == true)
            .map(|f| f[key].as_f64().unwrap())
            .sum::<f64>()
            / n
    };
    let fine_single = fine_mean("single_error");
    let fine_grouped = fine_mean("grouped_error");
    let fine_shared = fine_mean("shared_error");
    let sizes = [
        d.labels.iter().filter(|&&x| x == 0).count(),
        d.labels.iter().filter(|&&x| x == 1).count(),
    ];
    let separated = held.iter().filter(|f| f.separation >= 0.15).count();
    let gains: [f64; 2] = std::array::from_fn(|g| {
        let mut a = 0.;
        let mut b = 0.;
        for f in &held {
            for (i, &label) in d.labels.iter().enumerate() {
                if label == g {
                    a += f.shared_losses[i];
                    b += f.grouped_losses[i];
                }
            }
        }
        1. - b / a.max(1e-12)
    });
    // Written before corpus evaluation; these are engineering gates, not probabilities.
    let gates = json!({"both_groups_have_12_patches":sizes.iter().all(|&n|n>=12),"held_18px_gain_at_least_15_percent":grouped<=shared_error*0.85,"held_53px_gain_at_least_5_percent":fine_grouped<=fine_shared*0.95,"swapping_models_worsens_10_percent":swapped>=grouped*1.1,"separation_over_point15px_in_10_held_frames":separated>=10,"each_group_held_gain_over_5_percent":gains.iter().all(|&v|v>=0.05)});
    let mut report = json!({"source":source,"schema":"persistent-shared-3d-motion-v1","training_frames":d.train,"held_frames":held.len(),"group_sizes":sizes,"held_single_error":single,"held_shared_error":shared_error,"held_shared_gain":1.-grouped/shared_error.max(1e-12),"held_grouped_error":grouped,"held_gain":1.-grouped/single.max(1e-12),"held_swapped_error":swapped,"fine_held_single_error":fine_single,"fine_held_shared_error":fine_shared,"fine_held_shared_gain":1.-fine_grouped/fine_shared.max(1e-12),"fine_held_grouped_error":fine_grouped,"fine_held_gain":1.-fine_grouped/fine_single.max(1e-12),"group_held_gains":gains,"separated_held_frames":separated,"demonstration_gates":gates,"passed":gates.as_object().unwrap().values().all(|v|v==true),"provenance":{"runner":digest(include_bytes!("z_motion_groups.rs")),"engine":digest(include_bytes!("z_motion_groups_math.rs")),"pixels":digest(include_bytes!("z_motion_groups_pixels.rs")),"projection":digest(include_bytes!("z_motion3d_math.rs")),"native_recipe":digest(include_bytes!("z_motion3d.rs")),"viewer":digest(include_bytes!("z_motion_groups_viewer.html"))},"limitations":"Full-frame anonymous two-group hypothesis. Boundary neighborhoods contain real in-bounds pixels only; partial target overlap must retain at least five of nine samples and pays an overlap penalty. A photometric split does not establish face versus eye segmentation. Group membership is constant in the reference image, fitted only on even target frames. Both groups share one fixed depth surface per group across time; poses rotate in 3D with a fixed assumed focal length. Odd frames use timestamp interpolation of rotation quaternions and translation, with known ROI crop offsets; their target images do not choose memberships or poses. Fine images are evaluation only. Photometric agreement is not measured tissue identity; lighting/reflections/occlusion can mimic motion. No calibrated depth, physical head/eye pose, anatomical labels or human correspondence labels. Membership has a 0.02 neighbor-agreement penalty with no anatomical location assumptions. Membership comparisons require common observed samples for both motions, so overlap loss alone is not a group difference. Core membership additionally requires a 0.015 training margin, 80% sign consistency and 80% absolute photometric support at error below 0.08. Final pose refinement and the common-motion baseline use the same training-supported patch subset. All patches remain in the held comparison; gray pixels are not discarded to improve it. Group discovery remains heuristic and may miss or hallucinate a split. No inference/training model assets. SN-FEIDA is not applicable."});
    let (w, h) = (ultra[0][0].w, ultra[0][0].h);
    let outer = |p: &P| p[0] < 2. || p[1] < 2. || p[0] >= w as f64 - 2. || p[1] >= h as f64 - 2.;
    let outer_ids = d
        .points
        .iter()
        .enumerate()
        .filter(|(_, p)| outer(p))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    report["coverage"] = json!({"image_pixels":w*h,"sampled_centers":d.points.len(),"previous_sampler_centers":(w-4)*(h-4),"outer_band_sampled":outer_ids.len(),"outer_band_core_sizes":(0..2).map(|g|outer_ids.iter().filter(|&&i|d.core[i]&&d.labels[i]==g).count()).collect::<Vec<_>>(),"note":"Outer band is a coverage audit, not a face label. Full-frame overlapping neighborhoods are correlated evidence."});
    report["anatomical_separation_verified"] = json!(false);
    let core_sizes: [usize; 2] = std::array::from_fn(|g| {
        d.labels
            .iter()
            .zip(&d.core)
            .filter(|(id, c)| **id == g && **c)
            .count()
    });
    report["core_sizes"] = json!(core_sizes);
    report["demonstration_gates"]["both_consistent_cores_have_12_patches"] =
        json!(core_sizes.iter().all(|&n| n >= 12));
    report["held_frames_improved"] = json!(held
        .iter()
        .filter(|f| f.grouped_error < f.shared_error)
        .count());
    if source["synthetic"] == true {
        report["synthetic_identity"] =
            truth_audit(&d, source["kind"].as_str().unwrap_or("synthetic"));
    }
    if source["synthetic"] == true && source["kind"] != "synthetic-null" {
        report["demonstration_gates"]["known_core_identity_at_least_90_percent"] = json!(
            report["synthetic_identity"]["label_accuracy_up_to_permutation"]
                .as_f64()
                .unwrap()
                >= 0.9
        );
        report["demonstration_gates"]["known_core_prediction_below_quarter_pixel"] = json!(
            report["synthetic_identity"]["mean_prediction_error_px18"]
                .as_f64()
                .unwrap_or(1.)
                < 0.25
        );
    }
    report["passed"] = json!(report["demonstration_gates"]
        .as_object()
        .unwrap()
        .values()
        .all(|v| v == true));
    let data = json!({"report":report,"frames":frames,"result":d,"fine":fine,"width":ultra[0][0].w,"height":ultra[0][0].h});
    fs::write(out.join("groups.json"), serde_json::to_vec(&data)?)?;
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&data["report"])?,
    )?;
    fs::write(
        out.join("viewer.html"),
        include_str!("z_motion_groups_viewer.html")
            .replace("GROUP_DATA", &serde_json::to_string(&data)?),
    )?;
    println!("{}", serde_json::to_string_pretty(&data["report"])?);
    Ok(())
}
fn truth_audit(d: &engine::Discovery, kind: &str) -> Value {
    let mirrored = kind == "synthetic-mirror";
    let same_motion = kind == "synthetic-null";
    let mut matches = [0usize; 2];
    let mut count = 0;
    let mut errors = Vec::new();
    for (i, p) in d.points.iter().enumerate() {
        if !d.core[i] || (p[0] - 79. / 9.).abs() < 2. {
            continue;
        }
        let x = if mirrored {
            158. - 9. * p[0]
        } else {
            9. * p[0]
        };
        let owner = usize::from(x >= 79.);
        matches[0] += usize::from(d.labels[i] == owner);
        matches[1] += usize::from(d.labels[i] != owner);
        count += 1;
        let ray = [(x - 79.) / 318., (p[1] * 9. - 52.) / 318., 1.];
        let aa = ray.iter().map(|x| x * x).sum::<f64>();
        let disc = 1.7 * 1.7 - aa * (1.7 * 1.7 - 0.9 * 0.9);
        if disc <= 0. {
            continue;
        }
        let depth = (1.7 - disc.sqrt()) / aa;
        let q = [ray[0] * depth, ray[1] * depth, depth - 1.7];
        for f in d.frames.iter().filter(|f| f.held) {
            let phase = f.index as f64 / 50. * std::f64::consts::PI;
            let (yaw, pitch) = if owner == 0 || same_motion {
                (0.015 * phase.sin(), 0.01 * (phase * 0.7).sin())
            } else {
                (-0.065 * phase.sin(), 0.025 * (phase * 0.7).sin())
            };
            let (sy, cy) = yaw.sin_cos();
            let (sp, cp) = pitch.sin_cos();
            let r = [q[0], cp * q[1] - sp * q[2], sp * q[1] + cp * q[2]];
            let n = [cy * r[0] + sy * r[2], r[1], -sy * r[0] + cy * r[2]];
            let target = [n[0] + 0.005 * phase.sin(), n[1], n[2] + 1.7];
            if n.iter().zip(target).map(|(a, b)| a * b).sum::<f64>() >= 0. {
                continue;
            }
            let px = 318. * target[0] / target[2] + 79.;
            let py = 318. * target[1] / target[2] + 52.;
            if usize::from(px >= 79.) != owner {
                continue;
            }
            let truth = [if mirrored { (158. - px) / 9. } else { px / 9. }, py / 9.];
            if truth[0] < 1. || truth[0] >= 16. || truth[1] < 1. || truth[1] >= 10. {
                continue;
            }
            let pred = f.groups[d.labels[i]].prepare().map(*p).unwrap();
            errors.push((truth[0] - pred[0]).hypot(truth[1] - pred[1]));
        }
    }
    json!({"core_reference_points_away_from_boundary":count,"label_accuracy_up_to_permutation":if count==0{0.}else{*matches.iter().max().unwrap()as f64/count as f64},"visible_held_predictions":errors.len(),"mean_prediction_error_px18":if errors.is_empty(){None}else{Some(errors.iter().sum::<f64>()/errors.len()as f64)},"null_scene_has_no_true_split":same_motion})
}
pub fn synthetic(out: &Path, kind: &str, history: bool) -> Result<()> {
    if kind.starts_with("synthetic-attached-") {
        return scene::synthetic_attached(out, kind);
    }
    let mut ultra = Vec::new();
    let mut coarse = Vec::new();
    let mut frames = Vec::new();
    for t in 0..51 {
        let time = t as f64 / 30.;
        let phase = t as f64 / 50. * std::f64::consts::PI;
        let delayed = kind.starts_with("synthetic-history-");
        let head = ((t as f64 - 10.) / 15.).clamp(0., 1.);
        let relative = ((t as f64 - 25.) / 25.).clamp(0., 1.);
        let a = if delayed {
            sphere(0.025 * head, 0.015 * head, 0.005 * head)
        } else {
            sphere(
                0.015 * phase.sin(),
                0.01 * (phase * 0.7).sin(),
                0.005 * phase.sin(),
            )
        };
        let b = if kind == "synthetic-null" || kind == "synthetic-history-null" {
            a.clone()
        } else if delayed {
            sphere(
                0.025 * head - 0.1 * relative,
                0.015 * head + 0.015 * relative,
                0.005 * head,
            )
        } else {
            sphere(
                -0.065 * phase.sin(),
                0.025 * (phase * 0.7).sin(),
                0.005 * phase.sin(),
            )
        };
        let image: Picture = std::array::from_fn(|c| {
            let mut im = a[c].clone();
            for y in 0..im.h {
                for x in im.w / 2..im.w {
                    im.v[y * im.w + x] = b[c].v[y * im.w + x];
                }
            }
            im
        });
        let image = if kind == "synthetic-mirror" {
            std::array::from_fn(|c| {
                let mut im = image[c].clone();
                for y in 0..im.h {
                    im.v[y * im.w..(y + 1) * im.w].reverse();
                }
                im
            })
        } else {
            image
        };
        let u = reduced(&image, 9);
        let f = reduced(&image, 3);
        frames.push(frame(out, t, &u, &f, time, "independent-two-spheres")?);
        png(out, &format!("raw-{t:03}.png"), &image)?;
        ultra.push(u);
        coarse.push(f);
    }
    let runner = if history { run_history } else { run };
    runner(
        out,
        &ultra,
        &coarse,
        frames,
        &vec![[0.; 2]; 51],
        json!({"synthetic":true,"kind":kind,"truth":"two separately rotating ray-intersected spherical surfaces, seen through fixed half-image apertures; exclude the aperture boundary when judging material identity"}),
    )
}
