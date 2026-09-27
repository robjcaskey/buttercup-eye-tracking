//! Prefix-only inference with explicit retrospective membership revisions.
use super::{digest, engine, json, motion, pixels, scene, Picture, Result, Value, P};
use std::{fs, path::Path};
fn revise(
    previous: &[Option<usize>],
    labels: &[usize],
    core: &[bool],
    supported: bool,
    retract: bool,
) -> Vec<Option<usize>> {
    previous
        .iter()
        .enumerate()
        .map(|(i, &old)| {
            if retract {
                None
            } else if supported && core[i] {
                Some(labels[i])
            } else {
                old
            }
        })
        .collect()
}
fn recent_core(input: &[Picture], d: &engine::Discovery) -> Vec<bool> {
    let pictures = input.iter().map(motion::normalized).collect::<Vec<_>>();
    let ps = pixels::patches(&pictures[0], 1);
    let train = d
        .train
        .iter()
        .copied()
        .filter(|&t| t > 0)
        .rev()
        .take(5)
        .collect::<Vec<_>>();
    ps.iter()
        .enumerate()
        .map(|(i, p)| {
            let g = d.labels[i];
            let mut margins = Vec::new();
            let mut own = Vec::new();
            for &t in &train {
                if let Some(e) = pixels::paired_error(
                    p,
                    &pictures[t],
                    d.groups[g].poses[t],
                    d.groups[1 - g].poses[t],
                ) {
                    margins.push(e[1] - e[0]);
                    own.push(e[0]);
                }
            }
            let n = margins.len();
            n >= 3
                && n as f64 >= 0.8 * train.len() as f64
                && margins.iter().sum::<f64>() / n as f64 >= 0.015
                && margins.iter().filter(|&&m| m > 0.003).count() as f64 >= 0.8 * n as f64
                && own.iter().filter(|&&e| e < 0.08).count() as f64 >= 0.8 * n as f64
        })
        .collect()
}
pub fn run(
    out: &Path,
    ultra: &[Picture],
    coarse: &[Picture],
    frames: Vec<Value>,
    crops: &[P],
    source: Value,
) -> Result<()> {
    if ultra.len() < 11 || ultra.len() % 2 != 1 {
        return Err("history requires odd sequence of at least 11 exposures".into());
    }
    let times = frames
        .iter()
        .map(|f| f["time"].as_f64().unwrap())
        .collect::<Vec<_>>();
    let mut ends = (10..ultra.len()).step_by(10).collect::<Vec<_>>();
    if ends.last().copied() != Some(ultra.len() - 1) {
        ends.push(ultra.len() - 1);
    }
    let mut history = Vec::<Value>::new();
    let mut previous = Vec::<Option<usize>>::new();
    let mut points = Vec::new();
    for end in ends {
        // Slicing happens before any fitting or confidence calculation. Future
        // target images cannot influence this version of the reconstruction.
        let mut d = engine::discover(&ultra[..=end], &crops[..=end], &times[..=end]);
        let coupled = scene::fit(&ultra[..=end], &coarse[..=end], &times[..=end], &d);
        d.labels = coupled.articulated.labels.clone();
        for g in 0..2 {
            d.groups[g].shape = coupled.articulated.shapes[g];
            d.groups[g].poses = (0..=end)
                .map(|t| coupled.articulated.models(t)[g])
                .collect();
        }
        let fine=coupled.frames.iter().map(|f|json!({"held":f["held"],"shared_error":f["fine_rigid_loss"],"grouped_error":f["fine_loss"]})).collect::<Vec<_>>();
        for f in &mut d.frames {
            let row = &coupled.frames[f.index - 1];
            f.groups = coupled.articulated.models(f.index).to_vec();
            f.shared_groups = coupled.rigid.models(f.index).to_vec();
            f.single = f.shared_groups[0];
            f.shared_error = row["rigid_loss"].as_f64().unwrap();
            f.grouped_error = row["loss"].as_f64().unwrap();
            f.swapped_error = row["swapped_loss"].as_f64().unwrap();
            f.separation = d
                .points
                .iter()
                .filter_map(|&p| {
                    let a = f.groups[0].prepare().map(p)?;
                    let b = f.groups[1].prepare().map(p)?;
                    Some((a[0] - b[0]).hypot(a[1] - b[1]))
                })
                .sum::<f64>()
                / d.points.len() as f64;
        }
        let core = recent_core(&ultra[..=end], &d);
        if previous.is_empty() {
            previous = vec![None; d.points.len()];
            points = d.points.clone();
        }
        assert_eq!(points, d.points);
        let mut agreement = [0usize; 2];
        for (i, old) in previous.iter().enumerate() {
            if let Some(g) = old {
                agreement[usize::from(*g != d.labels[i])] += 1;
            }
        }
        let swap = agreement[1] > agreement[0];
        let labels = d
            .labels
            .iter()
            .map(|&g| if swap { 1 - g } else { g })
            .collect::<Vec<_>>();
        let sizes: [usize; 2] = std::array::from_fn(|g| {
            labels
                .iter()
                .zip(&core)
                .filter(|(v, c)| **v == g && **c)
                .count()
        });
        let recent = d
            .frames
            .iter()
            .filter(|f| f.held)
            .rev()
            .take(5)
            .collect::<Vec<_>>();
        let mean = |key: fn(&engine::FrameResult) -> f64| {
            recent.iter().map(|f| key(f)).sum::<f64>() / recent.len() as f64
        };
        let a = mean(|f| f.shared_error);
        let b = mean(|f| f.grouped_error);
        let swapped = mean(|f| f.swapped_error);
        let fine_recent = fine
            .iter()
            .filter(|f| f["held"] == true)
            .rev()
            .take(5)
            .collect::<Vec<_>>();
        let fine_mean = |key: &str| {
            fine_recent
                .iter()
                .map(|f| f[key].as_f64().unwrap())
                .sum::<f64>()
                / fine_recent.len() as f64
        };
        let fa = fine_mean("shared_error");
        let fb = fine_mean("grouped_error");
        // The common motion may already explain one group perfectly: require
        // joint improvement and distinct supported groups, not improvement of
        // each group separately over the common-motion model.
        let gates = json!({"two_recent_supported_regions":sizes.iter().all(|&n|n>=12),"recent_held_gain_15_percent":b<a*0.85,"absolute_held_improvement":a-b>0.002,"fine_held_gain_5_percent":fb<fa*0.95,"swapping_worsens":swapped>b*1.1,"distinct_projected_motions":recent.iter().filter(|f|f.separation>0.15).count()>=3});
        let accepted = gates.as_object().unwrap().values().all(|v| v == true);
        let all_a = d
            .frames
            .iter()
            .filter(|f| f.held)
            .map(|f| f.shared_error)
            .sum::<f64>();
        let all_b = d
            .frames
            .iter()
            .filter(|f| f.held)
            .map(|f| f.grouped_error)
            .sum::<f64>();
        let fine_a = fine
            .iter()
            .filter(|f| f["held"] == true)
            .map(|f| f["shared_error"].as_f64().unwrap())
            .sum::<f64>();
        let fine_b = fine
            .iter()
            .filter(|f| f["held"] == true)
            .map(|f| f["grouped_error"].as_f64().unwrap())
            .sum::<f64>();
        let retract = !accepted && all_b > all_a * 1.1 && fine_b > fine_a * 1.1;
        let assignments = revise(&previous, &labels, &core, accepted, retract);
        let newly = assignments
            .iter()
            .zip(&previous)
            .filter(|(a, b)| a.is_some() && b.is_none())
            .count();
        let changed = assignments
            .iter()
            .zip(&previous)
            .filter(|(a, b)| a.is_some() && b.is_some() && a != b)
            .count();
        let withdrawn = assignments
            .iter()
            .zip(&previous)
            .filter(|(a, b)| a.is_none() && b.is_some())
            .count();
        let poses = d
            .frames
            .iter()
            .map(|f| {
                let mut groups = f.groups.clone();
                if swap {
                    groups.swap(0, 1);
                }
                let row=&coupled.frames[f.index-1];
                let owners=row["owners"].as_array().unwrap().iter().map(|v|v.as_u64().map(|g|if swap{1-g}else{g})).collect::<Vec<_>>();
                let mut rigid=f.shared_groups.clone();if swap{rigid.swap(0,1);}
                json!({"index":f.index,"held":f.held,"single":f.single,"groups":groups,"rigid_groups":rigid,"owners":owners,"depth":row["depth"],"collisions":row["collisions"]})
            })
            .collect::<Vec<_>>();
        let state = if accepted {
            "split supported"
        } else if assignments.iter().any(Option::is_some) {
            "previous split retained; no new split evidence"
        } else {
            "unresolved; shared motion is not shared identity"
        };
        eprintln!("history through {end}: {state}; new {newly}, reattributed {changed}, withdrawn {withdrawn}");
        history.push(json!({"end":end,"time":times[end],"state":state,"accepted":accepted,"retracted":retract,"assignments":assignments,"tentative_labels":labels,"recent_core":core,"recent_core_sizes":sizes,"newly_supported":newly,"reattributed":changed,"withdrawn":withdrawn,"gates":gates,"recent_held_gain":1.-b/a.max(1e-12),"recent_fine_gain":1.-fb/fa.max(1e-12),"poses":poses,"attachment":{"parent_group":usize::from(swap),"parent_origin":[0.,0.,1.],"child_pivot_offset":coupled.articulated.child_pivot,"child_depth_scale":coupled.articulated.child_depth_scale,"relative_rotations":coupled.articulated.relative_rotation,"alternative_hypotheses":coupled.alternative_orders}}));
        previous = assignments;
    }
    let data = json!({"schema":"retrospective-motion-history-v1","source":source,"frames":frames,"points":points,"history":history,"width":ultra[0][0].w,"height":ultra[0][0].h,"provenance":{"history":digest(include_bytes!("z_motion_groups_history.rs")),"runner":digest(include_bytes!("z_motion_groups.rs")),"native_recipe":digest(include_bytes!("z_motion3d.rs")),"engine":digest(include_bytes!("z_motion_groups_math.rs")),"pixels":digest(include_bytes!("z_motion_groups_pixels.rs")),"scene":digest(include_bytes!("z_motion_scene.rs")),"projection":digest(include_bytes!("z_motion3d_math.rs")),"viewer":digest(include_bytes!("z_motion_groups_history.html"))},"limitations":"Offline articulated prefix replay, not a live tracker. The initial rigid hypothesis permits simultaneous XYZ rotation and translation around one fixed reference origin [0,0,1]. This origin fixes the coordinate gauge; it is not a measured anatomical pivot. The alternative adds a child rotation around one pivot fixed in parent coordinates; it has no independent child translation. Both parent choices and front/back depth seeds are fitted using training images. Depth-buffered forward rendering chooses visible surfaces and boundary refinement revises the reference partition across the whole history. Depth and attachment remain competing uncalibrated hypotheses, not measured anatomy. Each version sees only images through its cutoff, and even frames fit the geometry. Recent odd frames test predictions; they are not fitting inputs. Later assignments explicitly revise older reconstructions. Gray means identity unresolved, not one physical object. Shared later motion alone does not erase a previously supported split; opposite supported membership can reattribute a patch, and a substantially better common-motion explanation can retract the split. Confidence thresholds are engineering heuristics. No anatomical labels, calibrated 3D pose, causal proof of tissue identity, or new-user claim. Raw patches may include reflections or multiple tissues. Full 18x12 image sampled; no eye-position prior."});
    fs::write(out.join("history.json"), serde_json::to_vec(&data)?)?;
    fs::write(
        out.join("viewer.html"),
        include_str!("z_motion_groups_history.html")
            .replace("HISTORY_DATA", &serde_json::to_string(&data)?),
    )?;
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"source":data["source"],"checkpoints":data["history"].as_array().unwrap().iter().map(|h|json!({"end":h["end"],"state":h["state"],"accepted":h["accepted"],"newly_supported":h["newly_supported"],"reattributed":h["reattributed"],"withdrawn":h["withdrawn"],"gates":h["gates"]})).collect::<Vec<_>>(),"anatomical_separation_verified":false}),
        )?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delayed_evidence_can_reattribute_the_past_without_forcing_early_identity() {
        let unknown = vec![None; 3];
        let labels = [0, 0, 1];
        let core = [true; 3];
        assert_eq!(revise(&unknown, &labels, &core, false, false), unknown);
        let split = revise(&unknown, &labels, &core, true, false);
        assert_eq!(split, vec![Some(0), Some(0), Some(1)]);
        assert_eq!(revise(&split, &[0, 1, 1], &core, false, false), split);
        assert_eq!(
            revise(&split, &[0, 1, 1], &core, true, false),
            vec![Some(0), Some(1), Some(1)]
        );
        assert_eq!(revise(&split, &labels, &core, false, true), unknown);
    }
}
