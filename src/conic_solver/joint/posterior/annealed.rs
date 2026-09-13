//! Offline annealed importance paths; never a publication confidence source.
//!
//! Neal (2001), https://arxiv.org/abs/physics/9803008, equations 3--11.
//! The bridge is q^(1-beta) p^beta. We accumulate incremental density ratios
//! BEFORE each invariant transition, not a final endpoint p/q ratio. Paths are
//! independent numerical trials; transitions within a path are not samples.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    pub(crate) paths: usize,
    pub(crate) steps: usize,
}

struct PathSample {
    parameters: Parameters,
    log_weight: f64,
    stratum: usize,
}

struct Run {
    paths: usize,
    endpoints: Vec<PathSample>,
    evaluations: usize,
    proposed: [usize; 2],
    feasible: [usize; 2],
    accepted: [usize; 2],
}

fn sample_paths(
    proposals: &[Proposal], log_target: &impl Fn(&mut Parameters) -> Option<f64>,
    config: Config, seed: u64,
) -> Run {
    assert!(!proposals.is_empty());
    assert!((1..=65536).contains(&config.paths) && config.steps <= 1024);
    assert!(proposals.iter().all(|q| q.profile_transport.is_none()),
        "reference kernels use fixed centered covariances");
    let moves = proposals.iter().map(|q| q.with_conditional_nuisance().unwrap()).collect::<Vec<_>>();
    let active = &proposals[0].active;
    assert!(proposals.iter().all(|q| q.active == *active && q.scales == proposals[0].scales));
    let paths = config.paths / proposals.len() * proposals.len();
    assert!(paths >= 2 * proposals.len(), "need at least two independent paths per initial stratum");
    let density = |p: &Parameters| log_mean_exp(proposals.iter().map(|q| q.log_density(p)));
    let mut run = Run {paths, endpoints: Vec::new(), evaluations: 0,
        proposed: [0; 2], feasible: [0; 2], accepted: [0; 2]};
    for path in 0..paths {
        // Each path keeps its initial draw across step-count controls. This
        // stream is separate from the caller's importance-sampling stream.
        let mut rng = Random(replica_seed(seed ^ 0x82d3_4b79_619c_7a51, path + 1));
        let stratum = path % proposals.len();
        let mut p = proposals[stratum].draw(&mut rng);
        run.evaluations += 1;
        // Infeasible starts are zero-weight paths, not conditioned-away draws.
        let Some(mut lp) = log_target(&mut p) else { continue; };
        let mut lq = density(&p);
        assert!(lp.is_finite() && lq.is_finite());
        let mut log_weight = 0.0;
        if config.steps == 0 { log_weight = lp - lq; }
        for step in 1..=config.steps {
            let beta = (step as f64 / config.steps as f64).powi(2);
            let previous = ((step - 1) as f64 / config.steps as f64).powi(2);
            log_weight += (beta - previous) * (lp - lq);
            // One fixed kernel per stage, common to ALL initial strata. Every
            // fourth stage uses an independence MH proposal from full q;
            // the others use a symmetric Gaussian random walk. Covariances
            // and widths are selected independently of the current state.
            let independence = step % 4 == 0;
            let kind = usize::from(independence);
            run.proposed[kind] += 1;
            let index = ((rng.uniform() * proposals.len() as f64) as usize).min(proposals.len()-1);
            let mut candidate = if independence { proposals[index].draw(&mut rng) } else {
                let q = &moves[index];
                let lower = q.conditional_lower.as_ref().unwrap();
                let scale = [0.25, 0.6, 1.25][step % 3];
                let z = (0..active.len()).map(|_| rng.normal()).collect::<Vec<_>>();
                let mut candidate = p;
                for (a, &i) in active.iter().enumerate() {
                    candidate[i] += scale * q.scales[i] * (0..=a).map(|b| lower[a][b] * z[b]).sum::<f64>();
                }
                candidate
            };
            run.evaluations += 1;
            // Hard bounds are rejected, never projected: projection would
            // destroy random-walk symmetry and the stated MH correction.
            let Some(next_lp) = log_target(&mut candidate) else { continue; };
            let next_lq = density(&candidate);
            assert!(next_lp.is_finite() && next_lq.is_finite());
            run.feasible[kind] += 1;
            let log_acceptance = if independence {
                beta * ((next_lp - next_lq) - (lp - lq))
            } else {
                beta * (next_lp - lp) + (1.0 - beta) * (next_lq - lq)
            };
            if rng.uniform().ln() < log_acceptance.min(0.0) {
                p = candidate; lp = next_lp; lq = next_lq;
                run.accepted[kind] += 1;
            }
        }
        assert!(log_weight.is_finite());
        run.endpoints.push(PathSample {parameters: p, log_weight, stratum});
    }
    assert_eq!(run.evaluations, run.paths + run.proposed.iter().sum::<usize>());
    assert!(run.evaluations <= paths * (1 + config.steps));
    run
}

pub(super) fn diagnose(
    model: &Problem<'_>, best: &JointConicSolution, best_parameters: &Parameters,
    integrated_inner_indices: &[usize], proposals: &[Proposal],
    log_target: &impl Fn(&mut Parameters) -> Option<f64>, config: Config, seed: u64,
) -> serde_json::Value {
    let run = sample_paths(proposals, log_target, config, seed);
    let mut json = serde_json::json!({"seed":seed.to_string(),"requested_paths":config.paths,"paths":run.paths,
        "steps":config.steps,"schedule":"beta=(stage/steps)^2; no adaptive stopping",
        "feasible_paths":run.endpoints.len(),"model_evaluations":run.evaluations,
        "kernel_order":["symmetric-gaussian-walk","independence-full-mixture"],
        "proposed":run.proposed,"feasible_moves":run.feasible,"accepted":run.accepted,
        "status":"no-feasible-paths","best_target":best.target_camera_mm,
        "sources":model.request.eyes.map(|e|e.map(|e|serde_json::json!({
            "roi":e.exposure.roi.0,"clock_domain":e.exposure.clock.domain.to_string(),
            "clock_epoch":e.exposure.clock.epoch.to_string(),"sequence":e.exposure.sequence.to_string(),
            "timestamp_ns":e.exposure.timestamp_ns.to_string()}))),
        "contract":"Independent extended-space importance paths with exact accumulated bridge weights. Each path counts once; invalid starts count as zero. Fixed kernels leave each bridge invariant. No resampling, contour deletion, MAP changes or independent-eye gaze averaging. Numerical diagnostics do not certify unvisited modes or calibrated gaze accuracy."});
    let logs = run.endpoints.iter().map(|s|s.log_weight).collect::<Vec<_>>();
    let Some((weights, effective)) = weights(&logs) else { return json; };
    let samples = run.endpoints.iter().map(|s| {
        let p = &s.parameters;
        let target = model.target(p).unwrap();
        let gazes = std::array::from_fn(|eye| {
            if !model.present[eye] { return None; }
            let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
            normalized3(sub3(target, [p[k], p[k+1], p[k+2]]))
        });
        (target, gazes, 0)
    }).collect::<Vec<PosteriorSample>>();
    let strata = run.endpoints.iter().map(|s|s.stratum).collect::<Vec<_>>();
    let numerics = direction_numerics(best, &samples, &weights, &strata, proposals.len(), run.paths);
    let radius = std::array::from_fn::<_, 2, _>(|eye| {
        let axis = best.eye_gaze_directions[eye]?;
        quantile(samples.iter().zip(&weights).filter_map(|(s,w)|
            Some((dot3(axis,s.1[eye]?).clamp(-1.0,1.0).acos().to_degrees(),*w))).collect(),0.9)
    });
    let maximum_mass = weights.iter().copied().fold(0.0, f64::max);
    json["status"] = serde_json::json!(if effective >= 24.0 && maximum_mass <= 0.2 {"estimated-paths"} else {"insufficient-paths"});
    json["effective_paths"] = serde_json::json!(effective);
    json["maximum_path_mass"] = serde_json::json!(maximum_mass);
    json["log_relative_normalizer"] = serde_json::json!(log_mean_exp(logs.iter().copied()) + (logs.len() as f64 / run.paths as f64).ln());
    json["gaze_radius_90_degrees"] = serde_json::json!(radius);
    json["direction_numerics"] = serde_json::json!(numerics.map(|n|n.map(|n|serde_json::json!({
        "mass_within_15_degrees":n.mass,"path_standard_error":n.standard_error}))));
    // Rejection-pattern mass exposes whether path motion reaches a different
    // explanation of the same contours, without changing the evidence model.
    let mut patterns = std::collections::BTreeMap::<Vec<usize>, f64>::new();
    for (sample, weight) in run.endpoints.iter().zip(&weights) {
        let conics = model.conics(&sample.parameters).unwrap();
        let rejected = model.rejected_groups(&conics, &model.select(&conics));
        let pattern = rejected.iter().enumerate().filter_map(|(i,&r)|r.then_some(i)).collect();
        *patterns.entry(pattern).or_default() += weight;
    }
    json["rejection_pattern_mass"] = serde_json::json!(patterns.into_iter().map(|(groups,mass)|
        serde_json::json!({"groups":groups,"mass":mass})).collect::<Vec<_>>());
    if std::env::var("BUTTERCUP_ANNEALED_GEOMETRY_TRACE").ok().as_deref() == Some("1") {
        assert!(config.paths <= 4096, "bounded offline geometry trace");
        assert_eq!(model.target(best_parameters), Some(best.target_camera_mm));
        // Read-only endpoint diagnostics run after sampling and use no RNG.
        // Preserve the integrated-inner embedding explicitly: its placeholder
        // radius and prior cost are NOT draws from that omitted conditional.
        let mut reference = *best_parameters;
        let reference_log_density = log_target(&mut reference).unwrap();
        let best_state = geometry_state(model, &reference, integrated_inner_indices);
        let states = run.endpoints.iter().zip(&weights).map(|(sample, weight)| {
            let mut p = sample.parameters;
            let log_density = log_target(&mut p).unwrap();
            assert_eq!(p, sample.parameters, "canonical embedding is stable");
            let mut state = geometry_state(model, &p, integrated_inner_indices);
            state["normalized_path_weight"] = serde_json::json!(weight);
            state["log_path_weight"] = serde_json::json!(sample.log_weight);
            state["initial_stratum"] = serde_json::json!(sample.stratum);
            state["log_target_density"] = serde_json::json!(log_density);
            state
        }).collect::<Vec<_>>();
        json["geometry_trace"] = serde_json::json!({
            "contract":"Read-only endpoint states of the same joint posterior paths. Use normalized path weights, not endpoint counts. Integrated inner-radius placeholders are excluded from sampled radii; embedded objective is not marginal negative log density. Scene priors are engineering assumptions, not independent native gaze truth.",
            "additional_log_target_evaluations":states.len()+1,
            "active_parameters":proposals[0].active,
            "integrated_inner_parameter_indices":integrated_inner_indices,
            "reference_log_target_density":reference_log_density,
            "reference_state":best_state,
            "scene_prior":geometry_prior(model),
            "group_alternatives":model.groups.iter().map(|g|serde_json::json!({
                "weight":g.weight,"alternatives":g.alternatives.iter().map(|a|serde_json::json!({
                    "eye":a.eye,"roi":model.request.eyes[a.eye].unwrap().exposure.roi.0,
                    "kind":format!("{:?}",a.kind),"group":a.group,"arc":a.index,
                    "weight":a.weight,"sigma_px":a.sigma,"support_length_px":a.length_px,
                })).collect::<Vec<_>>()
            })).collect::<Vec<_>>(),
            "states":states,
        });
    }
    json
}

fn geometry_state(model: &Problem<'_>, p: &Parameters, integrated_inner: &[usize]) -> serde_json::Value {
    let target = model.target(p).unwrap();
    let conics = model.conics(p).unwrap();
    let selected = model.select(&conics);
    let rejected = model.rejected_groups(&conics, &selected);
    let residuals = model.residuals(p, &selected).unwrap();
    let mut group_costs = Vec::new();
    let mut rms = Vec::new();
    for (index,(g, &choice)) in model.groups.iter().zip(&selected).enumerate() {
        let a = &g.alternatives[choice];
        let c = conics[a.eye][a.boundary].unwrap();
        group_costs.push(a.weight*a.mean_cost_at_level(c,selected.levels[index]).min(MAXIMUM_GROUP_COST)
            +(g.weight-a.weight)*MAXIMUM_GROUP_COST);
        rms.push(((0..a.points.len()).map(|i|c.residual_px(a.point_at_level(i,selected.levels[index])).powi(2)).sum::<f64>()/a.points.len() as f64).sqrt());
    }
    let total = squared_norm(&residuals);
    let contour = group_costs.iter().enumerate().filter(|(i,_)|!selected.in_family[*i]).map(|(_,c)|c).sum::<f64>()
        + selected.families.iter().map(|f|f.marginal_cost).sum::<f64>();
    assert!(total + 1e-8 >= contour);
    let eyes = std::array::from_fn::<_,2,_>(|eye| {
        if !model.present[eye] { return None; }
        let k = TARGET_PARAMETERS + eye*EYE_PARAMETERS;
        let (center, normal, _) = model.geometry(p, eye).unwrap();
        let prior = model.request.scene.eyes[eye].unwrap();
        Some(serde_json::json!({
            "center_camera_mm":center,"normal":normal,
            "gaze":normalized3(sub3(target,center)).unwrap(),
            "sampled_radii_mm":std::array::from_fn::<_,3,_>(|b|
                (!integrated_inner.contains(&(k+3+b))).then_some(p[k+3+b])),
            "pupil_decentration_mm":[p[k+6],p[k+7]],"pupil_inward_depth_mm":p[k+8],
            "surface_axis_alignment_degrees":[p[k+9].to_degrees(),p[k+10].to_degrees()],
            "center_prior_displacement_sigma":std::array::from_fn::<_,3,_>(|axis|
                prior.limbus_center.displacement(center)[axis]/prior.limbus_center.sigma_mm[axis]),
        }))
    });
    serde_json::json!({
        "target_camera_mm":target,"viewpoint_slopes":[p[0],p[1]],"axial_distance_mm":p[2].exp(),
        "eyes":eyes,"embedded_total_objective":total,"contour_objective":contour,
        "embedded_prior_objective":total-contour,"group_objectives":group_costs,
        "group_rms_px":rms,"rejected_groups":rejected,"selected_alternatives":selected.choices,
        "mask_level_families":mask_levels::diagnostics(model,&selected),
    })
}

fn geometry_prior(model: &Problem<'_>) -> serde_json::Value {
    let scalar = |s: ScalarSupport| serde_json::json!({
        "nominal":s.nominal,"minimum":s.minimum,"maximum":s.maximum,"sigma":s.sigma});
    let position = |p: PositionSupport| serde_json::json!({
        "camera_mm":p.camera_mm,"sigma_mm":p.sigma_mm,"maximum_displacement_mm":p.maximum_displacement_mm,
        "transverse_frame":format!("{:?}",p.transverse_frame)});
    let scene = model.request.scene;
    serde_json::json!({
        "focal_px":scene.camera.focal_px,"principal_px":scene.camera.principal_px,
        "target_reference_camera_mm":scene.target_reference_camera_mm,
        "fixation_axial_distance_mm":scalar(scene.fixation_axial_distance_mm),
        "maximum_gaze_slope":scene.maximum_gaze_slope,
        "interocular_distance_mm":scene.interocular_distance_mm.map(scalar),
        "eyes":scene.eyes.map(|p|p.map(|p|serde_json::json!({
            "limbus_center":position(p.limbus_center),"radii_mm":p.radii_mm.map(scalar),
            "pupil_inward_depth_mm":scalar(p.pupil_inward_depth_mm),
            "pupil_decentration_sigma_mm":p.pupil_decentration_sigma_mm,
            "pupil_maximum_decentration_mm":p.pupil_maximum_decentration_mm,
            "surface_axis_alignment":p.surface_axis_alignment.map(|a|serde_json::json!({
                "nominal_degrees":a.nominal_radians.map(f64::to_degrees),
                "sigma_degrees":a.sigma_radians.map(f64::to_degrees),
                "maximum_deviation_degrees":a.maximum_deviation_radians.map(f64::to_degrees)})),
            "effective_pivot":p.effective_pivot.map(position),"limbus_to_pivot_mm":p.limbus_to_pivot_mm,
        }))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annealed_mixture_paths_normalize_distinct_t_and_gaussian_nuisance_proposals() {
        let mut left=[0.0;PARAMETERS];left[0]=-1.0;left[3]=0.5;
        let first=Proposal::with_active(left,vec![0,3],[1.0;PARAMETERS],
            &[vec![0.8,0.3],vec![0.3,1.8]]).unwrap().with_conditional_nuisance().unwrap();
        let mut right=[0.0;PARAMETERS];right[0]=1.2;right[3]=-0.8;
        let second=Proposal::with_active(right,vec![0,1,3],[1.0;PARAMETERS],
            &[vec![1.5,0.2,-0.6],vec![0.2,1.0,0.1],vec![-0.6,0.1,1.0]])
            .unwrap().marginal(&[1],true).unwrap();
        assert!(second.original_sampler.is_some());
        let proposals=[first,second];
        let log_target=|p:&mut Parameters|[0,3].iter().all(|&i|(-1.0..=1.0).contains(&p[i])).then_some(0.0);
        for steps in [0,24] {
            let run=sample_paths(&proposals,&log_target,Config {paths:24000,steps},0x74ad_f3e1_9a22_3159);
            let logs=run.endpoints.iter().map(|s|s.log_weight).collect::<Vec<_>>();
            let (weights,effective)=weights(&logs).unwrap();assert!(effective>2000.0);
            for axis in [0,3] {
                let mean=run.endpoints.iter().zip(&weights).map(|(s,w)|w*s.parameters[axis]).sum::<f64>();
                let second=run.endpoints.iter().zip(&weights).map(|(s,w)|w*s.parameters[axis].powi(2)).sum::<f64>();
                assert!(mean.abs()<0.03,"steps={steps} axis={axis}: {mean}");
                assert!((second-1.0/3.0).abs()<0.03,"steps={steps} axis={axis}: {second}");
            }
            let cross=run.endpoints.iter().zip(&weights).map(|(s,w)|w*s.parameters[0]*s.parameters[3]).sum::<f64>();
            assert!(cross.abs()<0.03,"steps={steps}: {cross}");
            let normalizer=(log_mean_exp(logs.iter().copied())+(logs.len()as f64/run.paths as f64).ln()-t_log_normalizer(2)).exp();
            assert!((normalizer-4.0).abs()<0.18,"steps={steps}: {normalizer}");
        }
    }

    #[test]
    fn annealed_paths_recover_known_asymmetric_mixture_and_normalizer() {
        let mut center = [0.0; PARAMETERS];center[0] = -0.4;
        let q = Proposal::with_active(center,vec![0],[1.0;PARAMETERS],&[vec![1.0/9.0]]).unwrap();
        let log_target = |p: &mut Parameters| {
            let normal = |mean:f64,sd:f64|(-0.5*((p[0]-mean)/sd).powi(2)).exp()/(sd*(2.0*std::f64::consts::PI).sqrt());
            let value = (0.8*normal(-1.5,0.5)+0.2*normal(2.5,0.8)).ln();
            value.is_finite().then_some(value)
        };
        for steps in [0, 1, 32] {
            let run = sample_paths(&[q.clone()],&log_target,Config {paths:16000,steps},0x891b_efa8_23bc_3b19);
            let logs=run.endpoints.iter().map(|s|s.log_weight).collect::<Vec<_>>();
            let (weights,effective)=weights(&logs).unwrap();assert!(effective>1000.0);
            let mean=run.endpoints.iter().zip(&weights).map(|(s,w)|w*s.parameters[0]).sum::<f64>();
            let second=run.endpoints.iter().zip(&weights).map(|(s,w)|w*s.parameters[0].powi(2)).sum::<f64>();
            assert!((mean-(-0.7)).abs()<0.055,"steps={steps}: {mean}");
            assert!((second-3.378).abs()<0.10,"steps={steps}: {second}");
            let normalizer=(log_mean_exp(logs.iter().copied())+(logs.len()as f64/run.paths as f64).ln()-t_log_normalizer(1)).exp();
            assert!((normalizer-1.0).abs()<0.035,"steps={steps}: {normalizer}");
        }
    }

    #[test]
    fn annealed_paths_keep_invalid_starts_and_boundaries_in_the_measure() {
        let q=Proposal::with_active([0.0;PARAMETERS],vec![0,3],[1.0;PARAMETERS],
            &[vec![1.0,0.4],vec![0.4,1.0]]).unwrap().with_conditional_nuisance().unwrap();
        let log_target=|p:&mut Parameters|[0,3].iter().all(|&i|(-1.0..=1.0).contains(&p[i])).then_some(0.0);
        let run=sample_paths(&[q],&log_target,Config {paths:16000,steps:24},0x642b_ec32_f921_5397);
        assert!(run.endpoints.len()<run.paths);
        assert!(run.feasible.iter().sum::<usize>()<run.proposed.iter().sum::<usize>());
        let logs=run.endpoints.iter().map(|s|s.log_weight).collect::<Vec<_>>();
        let (weights,_)=weights(&logs).unwrap();
        for axis in [0,3] {
            let mean=run.endpoints.iter().zip(&weights).map(|(s,w)|w*s.parameters[axis]).sum::<f64>();
            let second=run.endpoints.iter().zip(&weights).map(|(s,w)|w*s.parameters[axis].powi(2)).sum::<f64>();
            assert!(mean.abs()<0.025,"axis={axis}: {mean}");
            assert!((second-1.0/3.0).abs()<0.025,"axis={axis}: {second}");
        }
        let normalizer=(log_mean_exp(logs.iter().copied())+(logs.len()as f64/run.paths as f64).ln()-t_log_normalizer(2)).exp();
        assert!((normalizer-4.0).abs()<0.15,"{normalizer}");
    }
}
