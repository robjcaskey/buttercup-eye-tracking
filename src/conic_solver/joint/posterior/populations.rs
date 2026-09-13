//! Bounded annealed SMC on the unchanged shared-fixation model.
//!
//! Del Moral, Doucet and Jasra (2006), sections 3.2 and 4:
//! https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf
//! Weight by gamma_beta/gamma_previous BEFORE resampling and an invariant
//! transition. Each population estimates Z and Z*f; pool these estimates,
//! not the population posterior ratios. Only populations are independent.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    pub(crate) particles: usize,
    pub(crate) steps: usize,
    pub(crate) populations: usize,
}

#[derive(Clone)]
struct Particle {
    p: Parameters,
    lp: f64,
    lq: f64,
    log_weight: f64,
    ancestor: usize,
}

struct Population {
    particles: Vec<Particle>,
    initial_draws: usize,
    initial_feasible: usize,
    log_normalizer: f64,
    evaluations: usize,
    resamplings: usize,
    proposed: [usize; 2],
    feasible: [usize; 2],
    accepted: [usize; 2],
}

fn log_sum(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() { return f64::NEG_INFINITY; }
    max + values.iter().map(|v| (v-max).exp()).sum::<f64>().ln()
}

fn resample(particles: &[Particle], count: usize, rng: &mut Random) -> Vec<Particle> {
    // Stratified uniforms give E[offspring_i | particles] = count * weight_i.
    // Keep ancestry to expose collapse; new descendants are not fresh data.
    let total = particles.iter().map(|p| p.log_weight.exp()).sum::<f64>();
    assert!(total.is_finite() && total > 0.0);
    let mut index = 0;
    let mut cumulative = particles[0].log_weight.exp() / total;
    (0..count).map(|j| {
        let u = (j as f64 + rng.uniform()) / count as f64;
        while u >= cumulative && index+1 < particles.len() {
            index += 1;
            cumulative += particles[index].log_weight.exp() / total;
        }
        let mut particle = particles[index].clone();
        particle.log_weight = -(count as f64).ln();
        particle
    }).collect()
}

fn sample(
    proposals: &[Proposal], log_target: &impl Fn(&mut Parameters) -> Option<f64>,
    config: Config, seed: u64,
) -> Vec<Population> {
    assert!(!proposals.is_empty());
    assert!((2..=64).contains(&config.populations));
    assert!((1..=4096).contains(&config.particles) && config.steps <= 256);
    assert!(proposals.iter().all(|q| q.profile_transport.is_none()),
        "fixed covariance kernels do not support profile transports");
    let active = &proposals[0].active;
    assert!(proposals.iter().all(|q| q.active == *active && q.scales == proposals[0].scales));
    let count = config.particles / proposals.len() * proposals.len();
    assert!(count >= 2*proposals.len(), "at least two draws per initial mixture stratum");
    let moves = proposals.iter().map(|q| q.with_conditional_nuisance().unwrap()).collect::<Vec<_>>();
    let density = |p: &Parameters| log_mean_exp(proposals.iter().map(|q| q.log_density(p)));
    (0..config.populations).map(|replica| {
        let mut rng = Random(replica_seed(seed ^ 0x6dc9_21af_57e2_b843, replica+1));
        let mut run = Population {particles: Vec::new(), initial_draws: count,
            initial_feasible: 0, log_normalizer: 0.0, evaluations: count,
            resamplings: 0, proposed: [0; 2], feasible: [0; 2], accepted: [0; 2]};
        for index in 0..count {
            let mut p = proposals[index % proposals.len()].draw(&mut rng);
            let Some(lp) = log_target(&mut p) else { continue; };
            let lq = density(&p);
            assert!(lp.is_finite() && lq.is_finite());
            // Divide by ALL attempted starts, preserving infeasible zero mass.
            run.particles.push(Particle {p, lp, lq, log_weight: -(count as f64).ln(), ancestor: index});
        }
        run.initial_feasible = run.particles.len();
        if run.particles.is_empty() {
            run.log_normalizer = f64::NEG_INFINITY;
            return run;
        }
        for step in 1..=config.steps.max(1) {
            let beta = (step as f64 / config.steps.max(1) as f64).powi(2);
            let previous = ((step-1) as f64 / config.steps.max(1) as f64).powi(2);
            for p in &mut run.particles { p.log_weight += (beta-previous)*(p.lp-p.lq); }
            let increment = log_sum(run.particles.iter().map(|p| p.log_weight));
            assert!(increment.is_finite());
            run.log_normalizer += increment;
            for p in &mut run.particles { p.log_weight -= increment; }
            let effective = run.particles.iter().map(|p| (2.0*p.log_weight).exp()).sum::<f64>().recip();
            // Do not add resampling noise at the final endpoint. A zero-step
            // control is exactly stratified importance sampling, with no moves.
            if step < config.steps && effective < 0.5*count as f64 {
                run.particles = resample(&run.particles, count, &mut rng);
                run.resamplings += 1;
            }
            if config.steps == 0 { continue; }
            for p in &mut run.particles {
                let independence = step % 4 == 0;
                let kind = usize::from(independence);
                let index = ((rng.uniform()*proposals.len() as f64) as usize).min(proposals.len()-1);
                let mut candidate = if independence { proposals[index].draw(&mut rng) } else {
                    let q = &moves[index];
                    let lower = q.conditional_lower.as_ref().unwrap();
                    let scale = [0.25, 0.6, 1.25][step % 3];
                    let z = (0..active.len()).map(|_| rng.normal()).collect::<Vec<_>>();
                    let mut candidate = p.p;
                    for (a, &i) in active.iter().enumerate() {
                        candidate[i] += scale*q.scales[i]*(0..=a).map(|b|lower[a][b]*z[b]).sum::<f64>();
                    }
                    candidate
                };
                run.evaluations += 1;
                run.proposed[kind] += 1;
                // Reject hard bounds, never project or repair the proposal.
                let Some(lp) = log_target(&mut candidate) else { continue; };
                let lq = density(&candidate);
                assert!(lp.is_finite() && lq.is_finite());
                run.feasible[kind] += 1;
                let acceptance = if independence {
                    beta*((lp-lq)-(p.lp-p.lq))
                } else {
                    beta*(lp-p.lp)+(1.0-beta)*(lq-p.lq)
                };
                if rng.uniform().ln() < acceptance.min(0.0) {
                    p.p = candidate; p.lp = lp; p.lq = lq;
                    run.accepted[kind] += 1;
                }
            }
        }
        assert_eq!(run.evaluations, count+run.proposed.iter().sum::<usize>());
        assert!(run.evaluations <= count*(1+config.steps));
        run
    }).collect()
}

pub(super) fn integrate(
    model: &Problem<'_>, best: &JointConicSolution, proposals: &[Proposal], original_count: usize,
    log_target: &impl Fn(&mut Parameters) -> Option<f64>, config: Config, seed: u64,
    mut result: ModelPosterior,
) -> ModelPosterior {
    let runs = sample(proposals, log_target, config, seed);
    let normalizers = runs.iter().map(|r|r.log_normalizer).collect::<Vec<_>>();
    result.samples += runs.iter().map(|r|r.evaluations).sum::<usize>();
    result.feasible_samples = runs.iter().map(|r|r.particles.len()).sum();
    result.replicas = runs.len();
    result.require_numerical_margin = true;
    result.population_integration = Some(serde_json::json!({
        "seed":seed.to_string(), "requested_particles_per_population":config.particles,
        "populations":config.populations, "steps":config.steps,
        "schedule":"beta=(stage/steps)^2; fixed work; resample below 50% initial-particle ESS before invariant moves",
        "model_evaluations":runs.iter().map(|r|r.evaluations).sum::<usize>(),
        "maximum_model_evaluations":config.populations*config.particles*(1+config.steps),
        "log_relative_normalizer":log_sum(normalizers.iter().copied())-(runs.len() as f64).ln(),
        "runs":runs.iter().map(|r|serde_json::json!({
            "initial_draws":r.initial_draws, "initial_feasible":r.initial_feasible,
            "endpoints":r.particles.len(), "resamplings":r.resamplings,
            "distinct_initial_ancestors":r.particles.iter().map(|p|p.ancestor).collect::<std::collections::BTreeSet<_>>().len(),
            "log_relative_normalizer":r.log_normalizer,
            "model_evaluations":r.evaluations,"proposed":r.proposed,
            "feasible_moves":r.feasible,"accepted":r.accepted,
        })).collect::<Vec<_>>(),
        "contract":"Source-local joint latent states, not temporal tracking particles. Invalid starts retain zero mass. Complete populations are independent numerical trials; descendants, accepted moves and repeated endpoints are not additional observations. Pooled endpoint mass includes each population's estimated normalizer. Same contours, priors, ROI association and selected joint MAP. Default-off experimental integration."
    }));
    let Some((shares, effective)) = weights(&normalizers) else {
        result.status = "no-feasible-samples";
        return result;
    };
    result.effective_samples = effective;
    result.maximum_sample_mass = shares.iter().copied().reduce(f64::max);
    let mut samples = Vec::new();
    let mut weights = Vec::new();
    let mut replicas = Vec::new();
    for (replica, run) in runs.iter().enumerate() {
        for p in &run.particles {
            let target = model.target(&p.p).unwrap();
            let gazes = std::array::from_fn(|eye| {
                if !model.present[eye] { return None; }
                let k = TARGET_PARAMETERS+eye*EYE_PARAMETERS;
                normalized3(sub3(target,[p.p[k],p.p[k+1],p.p[k+2]]))
            });
            let mode = (0..original_count).min_by(|&a,&b|
                (p.p[0]-proposals[a].mean[0]).hypot(p.p[1]-proposals[a].mean[1]).total_cmp(
                    &(p.p[0]-proposals[b].mean[0]).hypot(p.p[1]-proposals[b].mean[1]))).unwrap();
            samples.push((target,gazes,mode));
            weights.push(shares[replica]*p.log_weight.exp());
            replicas.push(replica);
        }
    }
    let counts = runs.iter().map(|r|r.initial_draws).collect::<Vec<_>>();
    result.replicate_direction_numerics = replicate_direction_numerics(best,&samples,&weights,&replicas,&counts);
    result.direction_numerics = result.replicate_direction_numerics.each_ref().map(|r|
        r.as_ref().map(|r|DirectionNumerics {mass:r.mass,standard_error:r.standard_error}));
    // A failed population cannot establish numerical agreement. Population
    // normalizer ESS is a degeneracy check, not an effective sample-size CLT.
    if runs.iter().any(|r|r.particles.is_empty()) || effective < 8.0
        || result.maximum_sample_mass.is_some_and(|w|w>0.25) {
        result.status = "insufficient-sampling";
        return result;
    }
    result.status = "estimated-conditional";
    let mean = std::array::from_fn(|i|samples.iter().zip(&weights).map(|(s,w)|w*s.0[i]).sum());
    result.target_mean_camera_mm = Some(mean);
    result.target_covariance_mm2 = Some(std::array::from_fn(|i|std::array::from_fn(|j|
        samples.iter().zip(&weights).map(|(s,w)|w*(s.0[i]-mean[i])*(s.0[j]-mean[j])).sum())));
    for (i,mode) in result.modes.iter_mut().enumerate() {
        mode.model_mass = Some(samples.iter().zip(&weights).filter(|(s,_)|s.2==i).map(|(_,w)|w).sum());
    }
    for eye in 0..2 {
        if let Some(axis) = best.eye_gaze_directions[eye] {
            result.gaze_radius_90_degrees[eye] = quantile(samples.iter().zip(&weights).filter_map(|(s,w)|
                Some((dot3(axis,s.1[eye]?).clamp(-1.0,1.0).acos().to_degrees(),*w))).collect(),0.9);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(runs: &[Population], dimension: usize) -> (f64,f64,f64) {
        let logs = runs.iter().flat_map(|r|r.particles.iter().map(move|p|r.log_normalizer+p.log_weight)).collect::<Vec<_>>();
        let (weights,_) = weights(&logs).unwrap();
        let points = runs.iter().flat_map(|r|&r.particles).collect::<Vec<_>>();
        let mean = points.iter().zip(&weights).map(|(p,w)|w*p.p[0]).sum();
        let second = points.iter().zip(&weights).map(|(p,w)|w*p.p[0].powi(2)).sum();
        let normalizer = (log_sum(runs.iter().map(|r|r.log_normalizer))-(runs.len() as f64).ln()-t_log_normalizer(dimension)).exp();
        (mean,second,normalizer)
    }

    #[test]
    fn populations_recover_asymmetric_mixture_and_normalizer() {
        let q = Proposal::with_active([0.0;PARAMETERS],vec![0],[1.0;PARAMETERS],&[vec![1.0/9.0]]).unwrap();
        let target = |p: &mut Parameters| {
            let normal = |m:f64,s:f64|(-0.5*((p[0]-m)/s).powi(2)).exp()/(s*(2.0*std::f64::consts::PI).sqrt());
            let value = (0.8*normal(-1.5,0.5)+0.2*normal(2.5,0.8)).ln();
            value.is_finite().then_some(value)
        };
        for steps in [0,1,32] {
            let runs = sample(&[q.clone()],&target,Config {particles:512,steps,populations:32},0x419b_79df_a2c7_5301);
            let (mean,second,z) = moments(&runs,1);
            assert!((mean+0.7).abs()<0.06,"steps={steps}: mean={mean}");
            assert!((second-3.378).abs()<0.12,"steps={steps}: second={second}");
            assert!((z-1.0).abs()<0.05,"steps={steps}: Z={z}");
        }
    }

    #[test]
    fn populations_keep_invalid_start_mass_and_hard_bounds() {
        let mut left = [0.0;PARAMETERS]; left[0]=-1.0;left[3]=0.5;
        let first = Proposal::with_active(left,vec![0,3],[1.0;PARAMETERS],&[vec![0.8,0.3],vec![0.3,1.8]])
            .unwrap().with_conditional_nuisance().unwrap();
        let mut right = [0.0;PARAMETERS];right[0]=1.2;right[3]=-0.8;
        let second = Proposal::with_active(right,vec![0,1,3],[1.0;PARAMETERS],
            &[vec![1.5,0.2,-0.6],vec![0.2,1.0,0.1],vec![-0.6,0.1,1.0]]).unwrap().marginal(&[1],true).unwrap();
        let target = |p:&mut Parameters|[0,3].iter().all(|&i|(-1.0..=1.0).contains(&p[i])).then_some(0.0);
        let runs = sample(&[first,second],&target,Config {particles:512,steps:24,populations:32},0x74ad_f3e1_9a22_3159);
        assert!(runs.iter().all(|r|r.initial_feasible<r.initial_draws && r.resamplings>0));
        assert!(runs.iter().all(|r|r.feasible.iter().sum::<usize>()<r.proposed.iter().sum::<usize>()));
        let (mean,second,z) = moments(&runs,2);
        assert!(mean.abs()<0.04 && (second-1.0/3.0).abs()<0.04,"{mean} {second}");
        assert!((z-4.0).abs()<0.2,"{z}");
    }

    #[test]
    fn population_precision_is_invariant_to_duplicating_descendants() {
        let points = [(0.05,true,0),(0.05,false,0),(0.1,false,1),(0.3,true,2),(0.5,true,3)];
        let first = replicate_indicator_numerics(&points,&[100;4]).unwrap();
        let duplicates = points.iter().flat_map(|&(w,b,r)|(0..100).map(move|_|(w/100.0,b,r))).collect::<Vec<_>>();
        let second = replicate_indicator_numerics(&duplicates,&[100;4]).unwrap();
        assert!((first.mass-0.85).abs()<1e-12,"pool Z*f over Z, not mean population ratios");
        assert!((first.mass-second.mass).abs()<1e-12);
        assert!((first.standard_error-second.standard_error).abs()<1e-12);
        assert!(first.standard_error>0.07);
    }

    #[test]
    fn populations_do_not_retry_extinct_initial_populations() {
        let q = Proposal::with_active([0.0;PARAMETERS],vec![0],[1.0;PARAMETERS],&[vec![1.0]]).unwrap();
        let runs = sample(&[q],&|_|None,Config {particles:32,steps:24,populations:4},1234);
        assert!(runs.iter().all(|r|r.particles.is_empty() && r.evaluations==32 && r.log_normalizer==f64::NEG_INFINITY));
    }
}
