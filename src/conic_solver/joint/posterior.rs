//! Bounded importance integration of the shared-target robust model.
//!
//! Local Hessians propose samples; they are not the answer. Every sample is
//! scored against the original current arcs and frozen priors, reconsidering
//! alternative arcs and outlier groups. A mixture proposal spans the explored
//! sign basins. Correcting by its density avoids counting optimizer starts as
//! independent evidence or interpreting cost gaps as posterior probabilities.
//!
//! This is a generalized posterior for an engineering robust loss. It is
//! conditional on one ROI association and uncalibrated camera/anatomy priors.
//! Finite importance sampling cannot certify unvisited modes or real coverage.
use super::*;

#[cfg(test)]
pub(crate) mod annealed;
pub(crate) mod populations;

const T_DOF: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TailProposal {
    Off,
    PilotOnly,
    Recenter,
    Refit,
    ConditionalRefit,
    OutlierRefit,
    BoundaryRefit,
    BoundaryDefensive,
    ProfileAffine,
    ProfileQuadratic,
}

impl TailProposal {
    fn explicit_boundaries(self) -> bool {
        matches!(self, Self::BoundaryRefit | Self::BoundaryDefensive)
    }

    fn profiled(self) -> bool {
        matches!(self, Self::ProfileAffine | Self::ProfileQuadratic)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct IntegrationConfig {
    pub(crate) budget: usize,
    pub(crate) seed: u64,
    pub(crate) adaptive: bool,
    pub(crate) tail_proposal: TailProposal,
    pub(crate) global_proposal: bool,
    pub(crate) conditional_global: bool,
    pub(crate) trace_tail: bool,
    pub(crate) conditional_nuisance: bool,
    pub(crate) numerical_admission: bool,
    pub(crate) marginalize_unobserved_inner: bool,
    pub(crate) preserve_marginal_draws: bool,
    pub(crate) replicas: usize,
    /// Partition the unchanged original draws into proposal-balanced batches.
    /// This isolates the diagnostic from changing seeds or the draw budget.
    pub(crate) preserve_replica_draws: bool,
    /// Proposal-only conditioning on coherent mask states. Experimental until
    /// matched native coverage and localization support live use.
    pub(crate) mask_state_proposals: bool,
    /// Extra bounded MAP initializations, each refined against the original
    /// full marginal model. Separate from proposal-only conditioning.
    pub(crate) mask_state_refinement: bool,
    /// Experimental likelihood change: integrate correlated arc alternatives
    /// at each geometry instead of selecting only their lowest capped cost.
    pub(crate) marginalize_arc_alternatives: bool,
    /// Experimental independent annealed populations. Never enabled by live().
    pub(crate) populations: Option<populations::Config>,
    #[cfg(test)]
    pub(crate) annealed_reference: Option<annealed::Config>,
    pub(crate) early_stop: bool,
}

// Keep the original integration as an explicit comparison control. Production
// enters through live(); offline experiments may override this baseline.
impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            budget: 8192,
            seed: 0xd1b5_4a32_d192_ed03,
            adaptive: false,
            tail_proposal: TailProposal::Off,
            global_proposal: false,
            conditional_global: false,
            trace_tail: false,
            conditional_nuisance: false,
            numerical_admission: false,
            marginalize_unobserved_inner: false,
            preserve_marginal_draws: false,
            replicas: 1,
            preserve_replica_draws: false,
            mask_state_proposals: false,
            mask_state_refinement: false,
            marginalize_arc_alternatives: false,
            populations: None,
            #[cfg(test)]
            annealed_reference: None,
            early_stop: true,
        }
    }
}

impl IntegrationConfig {
    pub(crate) fn live() -> Self {
        Self {
            tail_proposal: TailProposal::ConditionalRefit,
            global_proposal: true,
            conditional_nuisance: true,
            numerical_admission: true,
            marginalize_unobserved_inner: true,
            replicas: 4,
            preserve_replica_draws: true,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PosteriorMode {
    pub(crate) target_camera_mm: [f64; 3],
    pub(crate) model_mass: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct ModelPosterior {
    pub(crate) status: &'static str,
    pub(crate) samples: usize,
    pub(crate) feasible_samples: usize,
    /// Proposal fitting draws are discarded before posterior estimation.
    pub(crate) pilot_samples: usize,
    pub(crate) pilot_feasible_samples: usize,
    pub(crate) adapted_proposals: usize,
    pub(crate) global_proposals: usize,
    pub(crate) effective_samples: f64,
    pub(crate) maximum_sample_mass: Option<f64>,
    pub(crate) modeled_eyes: [bool; 2],
    pub(crate) modes: Vec<PosteriorMode>,
    pub(crate) target_mean_camera_mm: Option<[f64; 3]>,
    pub(crate) target_covariance_mm2: Option<[[f64; 3]; 3]>,
    /// Radius about the selected joint gaze ray, not a Gaussian sigma and not
    /// a screen accuracy claim. Includes sampled competing direction basins.
    pub(crate) gaze_radius_90_degrees: [Option<f64>; 2],
    pub(crate) direction_numerics: [Option<DirectionNumerics>; 2],
    pub(crate) require_numerical_margin: bool,
    pub(crate) marginalized_inner_radii: usize,
    pub(crate) marginal_draw_policy: &'static str,
    pub(crate) replicas: usize,
    pub(crate) replicate_direction_numerics: [Option<ReplicateNumerics>; 2],
    pub(crate) mask_state_proposals: Vec<serde_json::Value>,
    pub(crate) population_integration: Option<serde_json::Value>,
}

impl ModelPosterior {
    /// A bounded model-support gate, not a calibrated accuracy guarantee.
    /// A broad mirror pair or failed integration cannot authorize a cursor
    /// direction merely because the optimizer chose one of its modes.
    pub(crate) fn supports_direction(&self, eye: usize) -> bool {
        if self.require_numerical_margin
            && !self
                .admission_numerics(eye)
                .is_some_and(|n| n.mass - 2.0 * n.standard_error >= 0.9)
        {
            return false;
        }
        self.status == "estimated-conditional"
            && self
                .gaze_radius_90_degrees
                .get(eye)
                .copied()
                .flatten()
                .is_some_and(|radius| radius.is_finite() && (0.0..=15.0).contains(&radius))
    }

    pub(crate) fn admission_numerics(&self, eye: usize) -> Option<DirectionNumerics> {
        let mut n = self.direction_numerics.get(eye).copied().flatten()?;
        if self.require_numerical_margin && self.replicas > 1 {
            let replicated = self.replicate_direction_numerics.get(eye)?.as_ref()?;
            n.standard_error = n.standard_error.max(replicated.standard_error);
        }
        Some(n)
    }

    pub(crate) fn json(&self) -> serde_json::Value {
        #[allow(unused_mut)]
        let mut json = serde_json::json!({
            "status":self.status,"samples":self.samples,"feasible_samples":self.feasible_samples,
            "pilot_samples":self.pilot_samples,"pilot_feasible_samples":self.pilot_feasible_samples,
            "estimation_samples":self.samples.saturating_sub(self.pilot_samples),
            "adapted_proposals":self.adapted_proposals,
            "global_proposals":self.global_proposals,
            "effective_samples":self.effective_samples,"maximum_sample_mass":self.maximum_sample_mass,
            "modeled_eyes":self.modeled_eyes,
            "modes":self.modes.iter().map(|m|serde_json::json!({"target_camera_mm":m.target_camera_mm,
                "conditional_model_mass":m.model_mass})).collect::<Vec<_>>(),
            "target_mean_camera_mm":self.target_mean_camera_mm,
            "target_covariance_mm2":self.target_covariance_mm2,
            "gaze_radius_90_degrees":self.gaze_radius_90_degrees,
            "direction_supported":[self.supports_direction(0),self.supports_direction(1)],
            "direction_support_rule":if self.require_numerical_margin {
                "90% sampled model angular radius at most 15 degrees, with mass minus twice the larger within-stratum/between-batch standard error at least 90%; engineering admission, not calibrated accuracy"
            } else {
                "90% sampled model angular radius at most 15 degrees; engineering admission, not calibrated accuracy"
            },
            "density":"exp(-robust_loss/2) with bounded Gaussian engineering priors; uniform viewpoint slopes; Gaussian axial-distance prior",
            "integration":"self-normalized importance sampling; Student-t mixture from distinct joint basins; nuisance parameters integrated, arc alternatives and rejection reconsidered",
            "adaptation":"discard proposal-fitting pilot draws, freeze a mixture retaining every original proposal, then estimate using fresh draws and the complete frozen mixture density",
            "contract":"conditional on current contours, selected ROI association and fixed camera/model priors; model probabilities, not calibrated gaze accuracy; finite samples do not certify unvisited modes",
            "selected_fit":"unchanged joint MAP geometry; posterior mean is diagnostic only",
        });
        json["direction_numerics"]=serde_json::json!(self.direction_numerics.map(|n|n.map(|n|
            serde_json::json!({"mass_within_15_degrees":n.mass,"mass_standard_error":n.standard_error,
                "two_standard_error_lower":n.mass-2.0*n.standard_error,
                "two_standard_error_upper":n.mass+2.0*n.standard_error}))));
        json["direction_numerics_contract"]=serde_json::json!(
            "Stratified self-normalized importance delta-method error; approximate numerical precision of current sampled model mass, not calibrated gaze accuracy, finite-sample coverage or a bound on unvisited modes.");
        if self.require_numerical_margin {
            json["admission_direction_numerics"]=serde_json::json!([0,1].map(|eye|
                self.admission_numerics(eye).map(|n|serde_json::json!({
                    "mass_within_15_degrees":n.mass,"mass_standard_error":n.standard_error,
                    "two_standard_error_lower":n.mass-2.0*n.standard_error,
                    "two_standard_error_upper":n.mass+2.0*n.standard_error}))));
        }
        {
            json["require_numerical_margin"] = serde_json::json!(self.require_numerical_margin);
            json["marginalized_inner_radii"] = serde_json::json!(self.marginalized_inner_radii);
            json["marginal_draw_policy"] = serde_json::json!(self.marginal_draw_policy);
            json["replicas"] = serde_json::json!(self.replicas);
            json["replicate_direction_numerics"] = serde_json::json!(self.replicate_direction_numerics.each_ref().map(|r|r.as_ref().map(|r|
                serde_json::json!({"pooled_mass":r.mass,"between_replica_standard_error":r.standard_error,
                    "replica_masses":r.masses,"normalizer_shares":r.normalizer_shares,
                    "effective_samples":r.effective_samples,"draw_counts":r.draw_counts}))));
            json["replicate_contract"] = serde_json::json!("Pseudorandom batches sample the same frozen proposal mixture with balanced proposal counts within each batch. The paired control partitions the original draw stream; separate streams use equal work. Draw counts include infeasible zero-weight draws. Samples retain joint importance weights; batch posterior ratios are not averaged. The work-weighted between-batch delta-method error is an approximate numerical diagnostic, not calibrated coverage or a bound on unvisited modes. Optional precision admission uses the larger within/between error.");
        }
        if !self.mask_state_proposals.is_empty() {
            json["mask_state_proposals"] = serde_json::json!(self.mask_state_proposals);
            json["mask_state_proposal_contract"] = serde_json::json!(
                "At most 16 coherent mask assignments, refined from at most two joint MAP basins. Temporary conditional models only locate proposals; every estimation draw uses the original full marginal likelihood and complete frozen proposal density. The selected MAP, priors, arc mass and shared fixation are unchanged. This does not certify unvisited states or calibrated confidence.");
        }
        if let Some(populations) = &self.population_integration {
            json["population_integration"] = populations.clone();
            json["integration"] = serde_json::json!("independent annealed SMC populations; exact incremental bridge weights before invariant moves; stratified unbiased resampling below half the initial population size");
            json["direction_numerics_contract"] = serde_json::json!("Delta-method error across independent populations, pooled using their estimated normalizers. Descendants within a population are dependent and never independent numerical trials. Finite population count and unvisited modes can defeat this diagnostic; no calibrated accuracy claim.");
            json["replicate_contract"] = json["direction_numerics_contract"].clone();
            json["direction_support_rule"] = serde_json::json!("at least eight effective independent populations, maximum population normalizer share at most 25%, 90% model angular radius at most 15 degrees, and mass minus twice the between-population error at least 90%; engineering admission only");
            json["effective_samples_unit"] = serde_json::json!("independent population normalizer shares, not resampled particles");
        }
        json
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectionNumerics {
    pub(crate) mass: f64,
    pub(crate) standard_error: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ReplicateNumerics {
    mass: f64,
    standard_error: f64,
    masses: Vec<f64>,
    normalizer_shares: Vec<f64>,
    effective_samples: Vec<f64>,
    draw_counts: Vec<usize>,
}

/// Let A_r and Z_r be each independent batch's importance-weighted indicator
/// sum and normalizer, n_r its draws, N=sum(n_r), and R the batch count. All
/// batches balance the same proposal strata. The pooled ratio is m=sum(A)/sum(Z),
/// not mean(A_r/Z_r). Linearizing the ratio gives the between-batch variance
/// N/(R-1) * sum(((A_r-m*Z_r)/sum(Z))^2/n_r). With equal work this reduces to
/// R/(R-1) * sum(((A_r-m*Z_r)/sum(Z))^2). This is approximate, including at an
/// adaptive stopping time. Infeasible draws retain their allotted work; a batch
/// with no mass cannot establish agreement.
fn replicate_indicator_numerics(
    observations: &[(f64, bool, usize)],
    draw_counts: &[usize],
) -> Option<ReplicateNumerics> {
    let replicas = draw_counts.len();
    if replicas < 2 || observations.is_empty() || draw_counts.contains(&0) {
        return None;
    }
    let mut totals = vec![0.0; replicas];
    let mut squares = vec![0.0; replicas];
    let mut inside = vec![0.0; replicas];
    for &(weight, accepted, replica) in observations {
        if replica >= replicas || !weight.is_finite() || weight < 0.0 {
            return None;
        }
        totals[replica] += weight;
        squares[replica] += weight * weight;
        if accepted {
            inside[replica] += weight;
        }
    }
    if totals
        .iter()
        .chain(&squares)
        .any(|&w| w <= 0.0 || !w.is_finite())
    {
        return None;
    }
    let total = totals.iter().sum::<f64>();
    let mass = inside.iter().sum::<f64>() / total;
    let variance = draw_counts.iter().map(|&n| n as f64).sum::<f64>() / (replicas - 1) as f64
        * inside
            .iter()
            .zip(&totals)
            .zip(draw_counts)
            .map(|((a, z), &n)| ((a - mass * z) / total).powi(2) / n as f64)
            .sum::<f64>();
    if !mass.is_finite() || !variance.is_finite() {
        return None;
    }
    Some(ReplicateNumerics {
        mass,
        standard_error: variance.sqrt(),
        masses: inside.iter().zip(&totals).map(|(a, z)| a / z).collect(),
        normalizer_shares: totals.iter().map(|z| z / total).collect(),
        effective_samples: totals.iter().zip(squares).map(|(z, s)| z * z / s).collect(),
        draw_counts: draw_counts.to_vec(),
    })
}

/// Delta-method variance of a self-normalized importance ratio, stratified by
/// the proposal that GENERATED a draw, not the nearest posterior mode. Each
/// stratum includes its infeasible (zero-weight) draws. The frozen-mixture
/// pilot is excluded. This is an asymptotic numerical diagnostic; an unvisited
/// tail can still defeat it. See Owen, chapter 9, eq. 9.9 and section 9.12.
fn indicator_numerics(
    observations: &[(f64, bool, usize)],
    proposals: usize,
    draws_per_proposal: usize,
) -> Option<DirectionNumerics> {
    if proposals == 0 || draws_per_proposal < 2 || observations.is_empty() {
        return None;
    }
    let mass = observations
        .iter()
        .filter(|o| o.1)
        .map(|o| o.0)
        .sum::<f64>();
    let mut sums = vec![0.0; proposals];
    let mut squares = vec![0.0; proposals];
    for &(weight, inside, proposal) in observations {
        if proposal >= proposals || !weight.is_finite() || weight < 0.0 {
            return None;
        }
        let influence = weight * ((if inside { 1.0 } else { 0.0 }) - mass);
        sums[proposal] += influence;
        squares[proposal] += influence * influence;
    }
    let n = draws_per_proposal as f64;
    let variance = sums
        .iter()
        .zip(squares)
        .map(|(sum, square)| (square - sum * sum / n).max(0.0) * n / (n - 1.0))
        .sum::<f64>();
    (mass.is_finite() && variance.is_finite()).then_some(DirectionNumerics {
        mass,
        standard_error: variance.sqrt(),
    })
}

type PosteriorSample = ([f64; 3], [Option<[f64; 3]>; 2], usize);

fn direction_numerics(
    best: &JointConicSolution,
    samples: &[PosteriorSample],
    weights: &[f64],
    sample_proposals: &[usize],
    proposals: usize,
    draws: usize,
) -> [Option<DirectionNumerics>; 2] {
    if proposals == 0 || samples.len() != weights.len() || samples.len() != sample_proposals.len() {
        return [None; 2];
    }
    let cosine = 15.0_f64.to_radians().cos();
    std::array::from_fn(|eye| {
        let axis = best.eye_gaze_directions[eye]?;
        let mut observations = Vec::with_capacity(samples.len());
        for ((sample, &weight), &proposal) in samples.iter().zip(weights).zip(sample_proposals) {
            let gaze = sample.1[eye]?;
            observations.push((weight, dot3(axis, gaze) >= cosine, proposal));
        }
        indicator_numerics(&observations, proposals, draws / proposals)
    })
}

fn replicate_direction_numerics(
    best: &JointConicSolution,
    samples: &[PosteriorSample],
    weights: &[f64],
    sample_replicas: &[usize],
    draw_counts: &[usize],
) -> [Option<ReplicateNumerics>; 2] {
    if samples.len() != weights.len() || samples.len() != sample_replicas.len() {
        return [None, None];
    }
    let cosine = 15.0_f64.to_radians().cos();
    std::array::from_fn(|eye| {
        let axis = best.eye_gaze_directions[eye]?;
        let mut observations = Vec::with_capacity(samples.len());
        for ((sample, &weight), &replica) in samples.iter().zip(weights).zip(sample_replicas) {
            observations.push((weight, dot3(axis, sample.1[eye]?) >= cosine, replica));
        }
        replicate_indicator_numerics(&observations, draw_counts)
    })
}

fn replica_seed(seed: u64, replica: usize) -> u64 {
    // SplitMix64 finalizer gives reproducible, separated PRNG stream states.
    let mut x = seed.wrapping_add(0x9e37_79b9_7f4a_7c15u64.wrapping_mul(replica as u64));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (x ^ (x >> 31)).max(1)
}

#[derive(Clone)]
struct ProfileTransport {
    base: Box<Proposal>,
    target_lower: Vec<Vec<f64>>,
    steps: Vec<f64>,
    constant: Parameters,
    linear: Vec<Parameters>,
    quadratic: Vec<Parameters>,
    include_quadratic: bool,
}

impl ProfileTransport {
    fn apply(&self, input: Parameters, sign: f64) -> Parameters {
        // A triangular shear: target coordinates are unchanged and nuisance
        // variables receive a function of the target ONLY. Its Jacobian has
        // unit determinant, including with clipped shift features. The target
        // coordinates themselves are never clipped or otherwise changed here.
        // Inverting it requires subtracting that same shift, not ignoring it
        // when evaluating the importance density.
        let k = self.steps.len();
        let mut z = vec![0.0; k];
        for a in 0..k {
            let i = self.base.active[a];
            let delta = (input[i] - self.base.mean[i]) / self.base.scales[i];
            z[a] = (delta - (0..a).map(|b| self.target_lower[a][b] * z[b]).sum::<f64>())
                / self.target_lower[a][a];
        }
        let mut output = input;
        for &i in self.base.active.iter().filter(|&&i| i >= TARGET_PARAMETERS) {
            let mut delta = self.constant[i];
            for a in 0..k {
                if self.steps[a] <= 0.0 { continue; }
                let x = (z[a] / self.steps[a]).clamp(-2.0, 2.0);
                delta += self.linear[a][i] * x;
                if self.include_quadratic { delta += self.quadratic[a][i] * x * x; }
            }
            output[i] += sign * delta;
        }
        output
    }
}

#[derive(Clone)]
struct Proposal {
    mean: Parameters,
    active: Vec<usize>,
    scales: Parameters,
    information: Vec<Vec<f64>>,
    lower: Vec<Vec<f64>>,
    log_sqrt_determinant: f64,
    conditional_lower: Option<Vec<Vec<f64>>>,
    /// Preserve common draws with the full sampler; density still uses the
    /// retained marginal coordinates. Omitted coordinates are integrated out.
    original_sampler: Option<Box<Proposal>>,
    profile_transport: Option<ProfileTransport>,
}

impl Proposal {
    fn new(model: &Problem<'_>, mean: Parameters, information: &[Vec<f64>]) -> Option<Self> {
        let active = (0..PARAMETERS)
            .filter(|&i| model.lower[i] < model.upper[i])
            .collect::<Vec<_>>();
        Self::with_active(mean, active, model.scales, information)
    }

    fn with_active(
        mean: Parameters,
        active: Vec<usize>,
        scales: Parameters,
        information: &[Vec<f64>],
    ) -> Option<Self> {
        let n = active.len();
        if information.len() != n || information.iter().any(|r| r.len() != n) {
            return None;
        }
        let mut lower = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let v = information[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                if !v.is_finite() || (i == j && v <= 0.0) {
                    return None;
                }
                lower[i][j] = if i == j { v.sqrt() } else { v / lower[j][j] };
            }
        }
        let log_sqrt_determinant = (0..n).map(|i| lower[i][i].ln()).sum();
        Some(Self {
            mean,
            active,
            scales,
            information: information.to_vec(),
            lower,
            log_sqrt_determinant,
            conditional_lower: None,
            original_sampler: None,
            profile_transport: None,
        })
    }

    /// Marginal Student-t proposal: retain the covariance submatrix, not the
    /// information submatrix (which would condition on the omitted radii).
    fn marginal(self, omitted: &[usize], preserve_draws: bool) -> Option<Self> {
        assert!(self.profile_transport.is_none(), "marginalize before fitting a transport");
        if omitted.is_empty() {
            return Some(self);
        }
        let retained = (0..self.active.len())
            .filter(|&i| !omitted.contains(&self.active[i]))
            .collect::<Vec<_>>();
        if retained.len() == self.active.len() {
            return Some(self);
        }
        let covariance = uncertainty::inverse_information(&self.information)?;
        let covariance = retained
            .iter()
            .map(|&i| retained.iter().map(|&j| covariance[i][j]).collect())
            .collect::<Vec<Vec<f64>>>();
        let information = uncertainty::inverse_information(&covariance)?;
        let mut marginal = Self::with_active(
            self.mean,
            retained.iter().map(|&i| self.active[i]).collect(),
            self.scales,
            &information,
        )?;
        if preserve_draws {
            marginal.original_sampler = Some(Box::new(self));
        }
        Some(marginal)
    }

    fn draw(&self, rng: &mut Random) -> Parameters {
        if let Some(transport) = &self.profile_transport {
            return transport.apply(transport.base.draw(rng), 1.0);
        }
        if let Some(original) = &self.original_sampler {
            return original.draw(rng);
        }
        let n = self.active.len();
        let mut x = (0..n).map(|_| rng.normal()).collect::<Vec<_>>();
        if let Some(lower) = &self.conditional_lower {
            let t_scale =
                (T_DOF as f64 / (0..T_DOF).map(|_| rng.normal().powi(2)).sum::<f64>()).sqrt();
            let target_count = self
                .active
                .iter()
                .take_while(|&&i| i < TARGET_PARAMETERS)
                .count();
            for v in x.iter_mut().take(target_count) {
                *v *= t_scale;
            }
            let mut p = self.mean;
            for (a, &i) in self.active.iter().enumerate() {
                p[i] += self.scales[i] * (0..=a).map(|b| lower[a][b] * x[b]).sum::<f64>();
            }
            return p;
        }
        for i in (0..n).rev() {
            x[i] = (x[i] - (i + 1..n).map(|j| self.lower[j][i] * x[j]).sum::<f64>())
                / self.lower[i][i];
        }
        let t_scale = (T_DOF as f64 / (0..T_DOF).map(|_| rng.normal().powi(2)).sum::<f64>()).sqrt();
        let mut p = self.mean;
        for (j, &i) in self.active.iter().enumerate() {
            p[i] += x[j] * self.scales[i] * t_scale;
        }
        p
    }

    /// Constants depending only on dimension/DOF/parameter scales cancel in
    /// self-normalized weights. All mixture components share those constants.
    fn log_density(&self, p: &Parameters) -> f64 {
        if let Some(transport) = &self.profile_transport {
            return transport.base.log_density(&transport.apply(*p, -1.0));
        }
        let x = self
            .active
            .iter()
            .map(|&i| (p[i] - self.mean[i]) / self.scales[i])
            .collect::<Vec<_>>();
        if let Some(lower) = &self.conditional_lower {
            let n = x.len();
            let k = self
                .active
                .iter()
                .take_while(|&&i| i < TARGET_PARAMETERS)
                .count();
            let mut z = vec![0.0; n];
            for i in 0..n {
                z[i] = (x[i] - (0..i).map(|j| lower[i][j] * z[j]).sum::<f64>()) / lower[i][i];
            }
            let target = z[..k].iter().map(|v| v * v).sum::<f64>();
            let nuisance = z[k..].iter().map(|v| v * v).sum::<f64>();
            return self.log_sqrt_determinant + t_log_normalizer(k)
                - t_log_normalizer(n)
                - 0.5 * (n - k) as f64 * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * (T_DOF + k) as f64 * (target / T_DOF as f64).ln_1p()
                - 0.5 * nuisance;
        }
        let distance = (0..x.len())
            .map(|i| {
                x[i] * (0..x.len())
                    .map(|j| self.information[i][j] * x[j])
                    .sum::<f64>()
            })
            .sum::<f64>()
            .max(0.0);
        self.log_sqrt_determinant
            - 0.5 * (T_DOF + x.len()) as f64 * (distance / T_DOF as f64).ln_1p()
    }

    fn with_conditional_nuisance(&self) -> Option<Self> {
        assert!(self.profile_transport.is_none(), "set the base proposal before fitting a transport");
        let covariance = uncertainty::inverse_information(&self.information)?;
        let n = covariance.len();
        let mut lower = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let value =
                    covariance[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                if !value.is_finite() || (i == j && value <= 0.0) {
                    return None;
                }
                lower[i][j] = if i == j {
                    value.sqrt()
                } else {
                    value / lower[j][j]
                };
            }
        }
        let mut proposal = self.clone();
        proposal.conditional_lower = Some(lower);
        proposal.original_sampler = None;
        Some(proposal)
    }

    fn recentered(&self, mean: Parameters) -> Self {
        assert!(self.profile_transport.is_none(), "recenter before fitting a transport");
        let mut proposal = self.clone();
        proposal.mean = mean;
        // A retained full-dimensional sampler still has its OLD mean. The
        // new component instead draws its actual marginal with the new mean.
        proposal.original_sampler = None;
        proposal
    }

    fn with_profile_transport(&self, model: &Problem<'_>, quadratic: bool) -> Option<(Self, serde_json::Value)> {
        assert!(self.profile_transport.is_none());
        let lower = self.conditional_lower.as_ref()?;
        let k = self.active.iter().take_while(|&&i| i < TARGET_PARAMETERS).count();
        let cost = |p: &Parameters| model.conics(p)
            .and_then(|c| model.residuals(p, &model.select(&c))).map(|r| squared_norm(&r));
        let (center, center_cost, center_steps) = conditional_refined_center(model, self.mean)?;
        assert_eq!(&center[..TARGET_PARAMETERS], &self.mean[..TARGET_PARAMETERS]);
        let mut transport = ProfileTransport {
            base: Box::new(self.clone()),
            target_lower: lower[..k].iter().map(|row| row[..k].to_vec()).collect(),
            steps: vec![0.0; k], constant: [0.0; PARAMETERS],
            linear: vec![[0.0; PARAMETERS]; k], quadratic: vec![[0.0; PARAMETERS]; k],
            include_quadratic: quadratic,
        };
        for &i in self.active.iter().filter(|&&i| i >= TARGET_PARAMETERS) {
            transport.constant[i] = center[i] - self.mean[i];
        }
        let mut fits = vec![serde_json::json!({"axis":null,"sign":0,"target_parameters":self.mean[..TARGET_PARAMETERS],
            "cost_before":cost(&self.mean),"cost_after":center_cost,"refinement_steps":center_steps})];
        let mut fitted_axes = 0;
        for axis in 0..k {
            let mut step = 1.0_f64;
            // Probe along a whitened target axis, keeping both signs inside
            // the ORIGINAL target bounds. Bound-active axes may be skipped.
            for a in 0..k {
                let i = self.active[a];
                let delta = (self.scales[i] * lower[a][axis]).abs();
                if delta > 0.0 {
                    let margin = (self.mean[i] - model.lower[i]).min(model.upper[i] - self.mean[i]);
                    step = step.min(0.5 * margin / delta);
                }
            }
            if !step.is_finite() || step < 1e-3 { continue; }
            let mut displacements = Vec::new();
            for sign in [-1.0, 1.0] {
                // This is the base distribution's conditional nuisance center
                // at the displaced target. Projection only initializes fitting;
                // drift is measured against the unprojected base center.
                let mut predicted = self.mean;
                for (a, &i) in self.active.iter().enumerate() {
                    predicted[i] += self.scales[i] * lower[a][axis] * step * sign;
                }
                let Some(seed) = model.project_step(predicted) else { break; };
                if seed[..TARGET_PARAMETERS] != predicted[..TARGET_PARAMETERS] { break; }
                let Some((refined, fitted_cost, steps)) = conditional_refined_center(model, seed) else { break; };
                assert_eq!(&refined[..TARGET_PARAMETERS], &predicted[..TARGET_PARAMETERS]);
                fits.push(serde_json::json!({"axis":axis,"sign":sign,"target_parameters":predicted[..TARGET_PARAMETERS],
                    "cost_before":cost(&seed),"cost_after":fitted_cost,"refinement_steps":steps}));
                displacements.push(std::array::from_fn::<_, PARAMETERS, _>(|i| refined[i] - predicted[i]));
            }
            if displacements.len() != 2 { continue; }
            transport.steps[axis] = step;
            fitted_axes += 1;
            for &i in self.active.iter().filter(|&&i| i >= TARGET_PARAMETERS) {
                transport.linear[axis][i] = 0.5 * (displacements[1][i] - displacements[0][i]);
                transport.quadratic[axis][i] = 0.5 * (displacements[1][i] + displacements[0][i]) - transport.constant[i];
            }
        }
        let trace = serde_json::json!({"quadratic":quadratic,"fitted_axes":fitted_axes,"target_axes":k,
            "target_steps":transport.steps,"constant":transport.constant,
            "linear":transport.linear,"quadratic_coefficients":transport.quadratic,"fits":fits});
        let mut curved = self.clone();
        curved.profile_transport = Some(transport);
        Some((curved, trace))
    }

    fn adapted(&self, model: &Problem<'_>, samples: &[(Parameters, f64)]) -> Option<Self> {
        let logs = samples.iter().map(|s| s.1).collect::<Vec<_>>();
        let (weights, effective) = weights(&logs)?;
        if effective < 4.0 {
            return None;
        }
        let n = self.active.len();
        let original = uncertainty::inverse_information(&self.information)?;
        let mut mean = self.mean;
        for &i in &self.active {
            mean[i] = samples.iter().zip(&weights).map(|(s, w)| w * s.0[i]).sum();
        }
        // Regularize the PROPOSAL only. Every final draw is still weighted by
        // the unchanged robust target density; no extra precision enters it.
        let blend = (effective / (effective + 8.0)).min(0.8);
        let scatter = (0..n)
            .map(|a| {
                (0..n)
                    .map(|b| {
                        let (i, j) = (self.active[a], self.active[b]);
                        let covariance = samples
                            .iter()
                            .zip(&weights)
                            .map(|(s, w)| {
                                w * ((s.0[i] - mean[i]) / self.scales[i])
                                    * ((s.0[j] - mean[j]) / self.scales[j])
                            })
                            .sum::<f64>();
                        (1.0 - blend) * original[a][b]
                            + blend * covariance * (T_DOF - 2) as f64 / T_DOF as f64
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let information = uncertainty::inverse_information(&scatter)?;
        Self::with_active(mean, self.active.clone(), model.scales, &information)
    }

    fn global(model: &Problem<'_>) -> Option<Self> {
        // A proposal based on the frozen scene support, with no contour or
        // historical target factors. It can visit regions far from every MAP
        // mode. Its density is corrected in exactly the same mixture as the
        // local Student-t components; it is not an additional model prior.
        let prior_model = Problem {
            request: model.request,
            target_chart: model.target_chart,
            groups: Vec::new(),
            present: model.present,
            initial: model.initial,
            lower: model.lower,
            upper: model.upper,
            scales: model.scales,
            marginalize_arc_alternatives: model.marginalize_arc_alternatives,
        };
        let mut mean = model.initial;
        mean[0] = 0.0;
        mean[1] = 0.0;
        mean[2] = model.request.scene.fixation_axial_distance_mm.nominal.ln();
        for eye in 0..2 {
            if model.present[eye] {
                let prior = model.request.scene.eyes[eye]?;
                let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
                mean[k..k + 3].copy_from_slice(&prior.limbus_center.camera_mm);
                for boundary in 0..3 {
                    mean[k + 3 + boundary] = prior.radii_mm[boundary].nominal;
                }
                mean[k + 6] = 0.0;
                mean[k + 7] = 0.0;
                mean[k + 8] = prior.pupil_inward_depth_mm.nominal;
                if let Some(a) = prior.surface_axis_alignment {
                    mean[k + 9..k + 11].copy_from_slice(&a.nominal_radians);
                }
            }
        }
        let mean = prior_model.project_step(mean)?;
        let mut information = constrained_proposal_information(&prior_model, &mean)?;
        for i in 0..2 {
            information[i][i] +=
                3.0 * (model.scales[i] / model.request.scene.maximum_gaze_slope).powi(2);
        }
        Self::new(model, mean, &information)
    }
}

/// Exact one-dimensional integration of a radius with NO observation factor.
/// Its Gaussian prior and outer >= inner > pupil constraint still contribute
/// probability mass. The reconstructed midpoint is only for evaluating the
/// other conics; it is never a new fitted radius or an observation.
pub(super) struct IntegratedInner {
    pub(super) indices: Vec<usize>,
    priors: Vec<ScalarSupport>,
}

impl IntegratedInner {
    pub(super) fn new(model: &Problem<'_>) -> Self {
        let eyes = (0..2)
            .filter(|&eye| {
                model.present[eye]
                    && !model.groups.iter().any(|g| {
                        g.alternatives
                            .iter()
                            .any(|a| a.eye == eye && a.boundary == 1)
                    })
            })
            .collect::<Vec<_>>();
        Self {
            indices: eyes
                .iter()
                .map(|&eye| TARGET_PARAMETERS + eye * EYE_PARAMETERS + 4)
                .collect(),
            priors: eyes
                .iter()
                .map(|&eye| model.request.scene.eyes[eye].unwrap().radii_mm[1])
                .collect(),
        }
    }

    pub(super) fn condition(&self, p: &mut Parameters) -> Option<f64> {
        let mut correction = 0.0;
        for (&index, &prior) in self.indices.iter().zip(&self.priors) {
            let lower = prior.minimum.max(p[index + 1]);
            let upper = prior.maximum.min(p[index - 1]);
            let mass = normal_interval_mass(
                (lower - prior.nominal) / prior.sigma,
                (upper - prior.nominal) / prior.sigma,
            )?;
            p[index] = lower + 0.5 * (upper - lower);
            if p[index] <= p[index + 1] {
                return None;
            }
            correction += 0.5 * ((p[index] - prior.nominal) / prior.sigma).powi(2)
                + (prior.sigma * (2.0 * std::f64::consts::PI).sqrt() * mass).ln();
        }
        correction.is_finite().then_some(correction)
    }
}

fn normal_interval_mass(lower: f64, upper: f64) -> Option<f64> {
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return None;
    }
    let midpoint = lower + 0.5 * (upper - lower);
    // Avoid subtracting nearly equal CDF/tail values in a very narrow interval.
    let mass = if (upper - lower) * (1.0 + midpoint.abs()) < 1e-3 {
        let half = 0.5 * (upper - lower);
        let pair = |x: f64| {
            (-0.5 * (midpoint - half * x).powi(2)).exp()
                + (-0.5 * (midpoint + half * x).powi(2)).exp()
        };
        half * (0.6521451548625461 * pair(0.3399810435848563)
            + 0.3478548451374538 * pair(0.8611363115940526))
            / (2.0 * std::f64::consts::PI).sqrt()
    } else {
        let a = lower / std::f64::consts::SQRT_2;
        let b = upper / std::f64::consts::SQRT_2;
        if lower >= 0.0 {
            0.5 * (libm::erfc(a) - libm::erfc(b))
        } else if upper <= 0.0 {
            0.5 * (libm::erfc(-b) - libm::erfc(-a))
        } else {
            0.5 * (libm::erf(b) - libm::erf(a))
        }
    };
    (mass.is_finite() && mass > 0.0).then_some(mass)
}

fn t_log_normalizer(dimension: usize) -> f64 {
    let gamma_half = |twice: usize| {
        if twice % 2 == 0 {
            (1..twice / 2).map(|i| (i as f64).ln()).sum::<f64>()
        } else {
            0.5 * std::f64::consts::PI.ln()
                + (0..twice / 2).map(|i| (i as f64 + 0.5).ln()).sum::<f64>()
        }
    };
    gamma_half(T_DOF + dimension)
        - gamma_half(T_DOF)
        - 0.5 * dimension as f64 * (T_DOF as f64 * std::f64::consts::PI).ln()
}

// Deterministic common random numbers keep matched A/B integrations comparable.
// This is numerical quadrature randomness, never a security or detector seed.
struct Random(u64);
impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

fn log_mean_exp(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + (values.iter().map(|v| (v - max).exp()).sum::<f64>() / values.len() as f64).ln()
}

fn weights(log_weights: &[f64]) -> Option<(Vec<f64>, f64)> {
    let max = log_weights
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return None;
    }
    let mut weights = log_weights
        .iter()
        .map(|v| (v - max).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    for w in &mut weights {
        *w /= sum;
    }
    let effective = weights.iter().map(|w| w * w).sum::<f64>().recip();
    Some((weights, effective))
}

fn quantile(mut values: Vec<(f64, f64)>, probability: f64) -> Option<f64> {
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut mass = 0.0;
    for (value, weight) in &values {
        mass += weight;
        if mass >= probability {
            return Some(*value);
        }
    }
    values.last().map(|v| v.0)
}

#[cfg(test)]
pub(super) fn without_pilot_outliers<'a>(
    model: &Problem<'a>,
    p: &Parameters,
) -> Option<(Problem<'a>, Vec<usize>)> {
    with_relaxed_proposal_groups(model, p, &[])
}

pub(super) fn boundary_relaxations(model: &Problem<'_>) -> Vec<Vec<usize>> {
    let mut masks = Vec::new();
    // Explore pupil/outer competition first. Genuine inner-limbus evidence
    // also remains eligible when those boundaries are absent. Mixed-kind
    // alternatives stay together and are never removed by this shortcut.
    for kind in [BoundaryKind::PupillaryBoundary, BoundaryKind::OuterLimbus, BoundaryKind::InnerLimbus] {
        for eye in 0..2 {
            let groups = model.groups.iter().enumerate().filter_map(|(i,g)| {
                g.alternatives.iter().all(|a| a.eye == eye && a.kind == kind).then_some(i)
            }).collect::<Vec<_>>();
            if !groups.is_empty() { masks.push(groups); }
        }
    }
    masks
}

pub(super) fn with_relaxed_proposal_groups<'a>(
    model: &Problem<'a>,
    p: &Parameters,
    forced: &[usize],
) -> Option<(Problem<'a>, Vec<usize>)> {
    let conics = model.conics(p)?;
    let mut rejected = model.rejected_groups(&conics, &model.select(&conics));
    for &index in forced { rejected[index] = true; }
    let omitted = rejected.iter().enumerate().filter_map(|(i, &r)| r.then_some(i)).collect();
    // This temporary objective only constructs q. The final target retains
    // every group and reconsiders its outlier status at every fresh sample.
    let relaxed = Problem {
        request: model.request,
        target_chart: model.target_chart,
        groups: model.groups.iter().zip(rejected).filter(|(_, r)| !*r).map(|(g, _)| Group {
            alternatives: g.alternatives.clone(), weight: g.weight,
        }).collect(),
        present: model.present,
        initial: model.initial,
        lower: model.lower,
        upper: model.upper,
        scales: model.scales,
        marginalize_arc_alternatives: model.marginalize_arc_alternatives,
    };
    Some((relaxed, omitted))
}

pub(super) fn conditional_refined_center(
    model: &Problem<'_>,
    p: Parameters,
) -> Option<(Parameters, f64, usize)> {
    // Optimize nuisance geometry at the pilot's ONE shared fixation. These
    // temporary fixed coordinates only fit a proposal: the original model,
    // MAP, target density and full-dimensional curvature stay unchanged.
    let mut conditional = Problem {
        request: JointConicRequest {
            maximum_refinements: model.request.maximum_refinements.min(6), ..model.request
        },
        target_chart: model.target_chart,
        groups: model.groups.iter().map(|g| Group {
            alternatives: g.alternatives.clone(), weight: g.weight,
        }).collect(),
        present: model.present,
        initial: model.initial,
        lower: model.lower,
        upper: model.upper,
        scales: model.scales,
        marginalize_arc_alternatives: model.marginalize_arc_alternatives,
    };
    conditional.lower[..TARGET_PARAMETERS].copy_from_slice(&p[..TARGET_PARAMETERS]);
    conditional.upper[..TARGET_PARAMETERS].copy_from_slice(&p[..TARGET_PARAMETERS]);
    conditional.refine(p)
}

/// Cover every state for up to two mask families. For larger state spaces,
/// cover individual family changes and both extremes before bounded joint
/// combinations. The cap is work accounting, never a posterior truncation.
pub(super) fn mask_assignments(families: usize) -> Vec<Vec<usize>> {
    mask_assignments_for_counts(&vec![3;families])
}

pub(super) fn mask_assignments_for_counts(counts: &[usize]) -> Vec<Vec<usize>> {
    assert!(counts.len() <= 6 && counts.iter().all(|&n|n==3 || n==7));
    let families = counts.len();
    if families == 0 { return Vec::new(); }
    let mut assignments = Vec::new();
    let mut push = |value: Vec<usize>| {
        if assignments.len() < 16 && !assignments.contains(&value) {
            assignments.push(value);
        }
        assignments.len() < 16
    };
    push(vec![1; families]);
    for family in 0..families {
        for level in (0..counts[family]).filter(|&level|level!=1) {
            let mut value = vec![1; families];
            value[family] = level;
            push(value);
        }
    }
    push(vec![0; families]);
    push(counts.iter().map(|n|n-1).collect());
    for mut code in 0..counts.iter().product() {
        if !push(counts.iter().map(|&count| { let level = code % count; code /= count; level }).collect()) {break;}
    }
    assignments
}

/// Keep quadrature, weights, sigmas, priors and bounds from the actual packet.
/// In particular, moved sample coordinates must not create extra point mass.
/// The cloned request remains provenance only: refinement consumes these arcs,
/// without calling seeds() or constructing new completed-ellipse hints.
pub(super) fn conditioned_mask_model<'a>(model: &Problem<'a>,
    families: &[mask_levels::FamilyActivity], levels: &[usize]) -> Problem<'a>
{
    assert_eq!(families.len(), levels.len());
    let mut conditional = model.clone();
    for (family, &level) in families.iter().zip(levels) {
        assert!(level < family.states.len());
        for &index in &family.groups {
            for arc in &mut conditional.groups[index].alternatives {
                arc.points = (0..arc.points.len()).map(|i| arc.point_at_level(i, level)).collect();
                arc.level_sets.fill(None);
            }
        }
    }
    conditional
}

/// A displaced state can start beyond a group's robust cap, where its gradient
/// is zero. Two scalar radius steps initialize the proposal fit from the real
/// shifted samples. This is not a likelihood term, a new prior or a gaze vote.
pub(super) fn mask_radius_start(model: &Problem<'_>, mut p: Parameters,
    families: &[mask_levels::FamilyActivity]) -> Parameters
{
    for _ in 0..2 {
        let Some(conics) = model.conics(&p) else { break; };
        let selection = model.select(&conics);
        let mut next = p;
        for family in families {
            let eye = family.eye;
            let boundary = family.boundary;
            let radius = TARGET_PARAMETERS + eye * EYE_PARAMETERS + 3 + boundary;
            if model.lower[radius] == model.upper[radius] { continue; }
            let base = conics[eye][boundary].unwrap();
            let h = model.scales[radius] * 1e-4;
            let sample = |sign: f64| {
                let mut q = p;
                q[radius] += sign * h;
                if q[radius] < model.lower[radius] || q[radius] > model.upper[radius] { return None; }
                model.conics(&q).and_then(|c| c[eye][boundary])
            };
            let (a, b, delta) = match (sample(-1.0), sample(1.0)) {
                (Some(a), Some(b)) => (a, b, 2.0 * h),
                (Some(a), None) => (a, base, h),
                (None, Some(b)) => (base, b, h),
                _ => continue,
            };
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for &index in &family.groups {
                let arc = &model.groups[index].alternatives[selection[index]];
                for (&point, &weight) in arc.points.iter().zip(&arc.quadrature) {
                    let residual = base.residual_px(point) / arc.sigma;
                    let derivative = (b.residual_px(point) - a.residual_px(point)) / (delta * arc.sigma);
                    numerator += arc.weight * weight * residual * derivative;
                    denominator += arc.weight * weight * derivative.powi(2);
                }
            }
            if numerator.is_finite() && denominator.is_finite() && denominator > 1e-12 {
                next[radius] -= numerator / denominator;
            }
        }
        let Some(next) = model.project_step(next).filter(|q| model.conics(q).is_some()) else { break; };
        p = next;
    }
    p
}

#[derive(Default)]
pub(super) struct MaskStateRefinement {
    pub(super) fits: Vec<(Parameters, f64)>,
    pub(super) attempts: usize,
    pub(super) steps: usize,
}

/// Conditional masks only supply starting coordinates. Every returned cost
/// and fit comes from refinement of the unchanged full marginal objective.
/// No conditional cost is eligible to choose the final joint solution.
pub(super) fn refine_mask_state_initializations(model: &Problem<'_>, starts: &[Parameters],
    maximum_starts: usize) -> MaskStateRefinement
{
    let mut result = MaskStateRefinement::default();
    let Some(conics) = starts.first().and_then(|p| model.conics(p)) else { return result; };
    let selection = model.select(&conics);
    let families = &selection.families;
    let counts = families.iter().map(|f| f.states.len()).collect::<Vec<_>>();
    for states in mask_assignments_for_counts(&counts) {
        let conditional = conditioned_mask_model(model, families, &states);
        for &start in starts.iter().take(2) {
            if result.attempts >= maximum_starts { return result; }
            result.attempts += 1;
            let start = mask_radius_start(&conditional, start, families);
            let Some((initial, _, steps)) = conditional.refine(start) else { continue; };
            result.steps += steps;
            let Some((fitted, cost, steps)) = model.refine(initial) else { continue; };
            result.steps += steps;
            result.fits.push((fitted, cost));
        }
    }
    result
}

fn add_mask_state_proposals(model: &Problem<'_>, modes: &[(Parameters, JointConicSolution)],
    config: IntegrationConfig, inner: &IntegratedInner, proposals: &mut Vec<Proposal>,
    diagnostics: &mut Vec<serde_json::Value>)
{
    let Some(conics) = model.conics(&modes[0].0) else { return; };
    let selection = model.select(&conics);
    let families = &selection.families;
    let counts = families.iter().map(|f|f.states.len()).collect::<Vec<_>>();
    let assignments = mask_assignments_for_counts(&counts);
    for levels in &assignments {
        let conditional = conditioned_mask_model(model, families, levels);
        for (mode, (p, _)) in modes.iter().enumerate()
            .filter(|(_, (_, s))| s.modeled_eyes == model.present).take(2)
        {
            let mut trace = serde_json::json!({"mode":mode,
                "families":families.iter().map(|f|[f.eye,f.boundary]).collect::<Vec<_>>(),
                "levels":levels.iter().map(|&x|x as i8-1).collect::<Vec<_>>(),
                "state_space_size":counts.iter().product::<usize>(),
                "enumerated_states":assignments.len(),"status":"refinement-unavailable"});
            if counts.contains(&7) {
                trace.as_object_mut().unwrap().remove("levels");
                trace["states"]=serde_json::json!(levels);
                trace["family_state_counts"]=serde_json::json!(counts);
            }
            let start = mask_radius_start(&conditional, *p, families);
            if let Some((fitted, cost, steps)) = conditional.refine(start) {
                trace["conditional_cost"] = serde_json::json!(cost);
                trace["refinement_steps"] = serde_json::json!(steps);
                trace["target_camera_mm"] = serde_json::json!(model.target(&fitted));
                trace["marginal_cost"] = serde_json::json!(model.conics(&fitted)
                    .and_then(|c|model.residuals(&fitted,&model.select(&c))).map(|r|squared_norm(&r)));
                let proposal = constrained_proposal_information(&conditional, &fitted)
                    .and_then(|i| Proposal::new(model, fitted, &i))
                    .and_then(|p| p.marginal(&inner.indices, config.preserve_marginal_draws));
                if let Some(proposal) = proposal {
                    // Retain the original heavy-tailed components. Extra state
                    // components need not multiply all nuisance tail scales.
                    let proposal = if config.conditional_nuisance {
                        proposal.with_conditional_nuisance().unwrap_or(proposal)
                    } else { proposal };
                    proposals.push(proposal);
                    trace["status"] = serde_json::json!("added");
                } else {
                    trace["status"] = serde_json::json!("curvature-unavailable");
                }
            }
            diagnostics.push(trace);
        }
    }
}

/// At a hard bound, a one-sided derivative is useful to propose samples even
/// though it cannot certify a symmetric Gaussian uncertainty interval. The
/// sampler still rejects infeasible points and scores the exact undamped
/// objective. No ridge/damping is added to either the density or its priors.
fn constrained_proposal_information(model: &Problem<'_>, p: &Parameters) -> Option<Vec<Vec<f64>>> {
    let conics = model.conics(p)?;
    let selected = model.select(&conics);
    let rejected = model.rejected_groups(&conics, &selected);
    let base = model.residuals_with_rejection(p, &selected, Some(&rejected))?;
    let mut columns = Vec::new();
    for i in (0..PARAMETERS).filter(|&i| model.lower[i] < model.upper[i]) {
        let step = 1e-4 * model.scales[i];
        let sample = |sign: f64| {
            let mut q = *p;
            q[i] += sign * step;
            if q[i] < model.lower[i] || q[i] > model.upper[i] {
                return None;
            }
            model.residuals_with_rejection(&q, &selected, Some(&rejected))
        };
        let (a, b, denominator) = match (sample(-1.0), sample(1.0)) {
            (Some(a), Some(b)) => (a, b, 2.0 * step),
            (Some(a), None) => (a, base.clone(), step),
            (None, Some(b)) => (base.clone(), b, step),
            (None, None) => return None,
        };
        if a.len() != base.len() || b.len() != base.len() {
            return None;
        }
        columns.push(
            a.iter()
                .zip(&b)
                .map(|(a, b)| (b - a) * model.scales[i] / denominator)
                .collect::<Vec<_>>(),
        );
    }
    Some(
        (0..columns.len())
            .map(|i| {
                (0..columns.len())
                    .map(|j| columns[i].iter().zip(&columns[j]).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect(),
    )
}

pub(super) fn integrate(
    model: &Problem<'_>,
    modes: &[(Parameters, JointConicSolution)],
    config: IntegrationConfig,
) -> ModelPosterior {
    assert!((1..=4).contains(&config.replicas), "bounded replica count");
    assert!(!config.adaptive || config.tail_proposal == TailProposal::Off,
        "compare per-basin moment adaptation and tail refinement separately");
    let mut result = ModelPosterior {
        status: "proposal-unavailable",
        samples: 0,
        feasible_samples: 0,
        pilot_samples: 0,
        pilot_feasible_samples: 0,
        adapted_proposals: 0,
        global_proposals: 0,
        effective_samples: 0.0,
        maximum_sample_mass: None,
        modeled_eyes: model.present,
        modes: Vec::new(),
        target_mean_camera_mm: None,
        target_covariance_mm2: None,
        gaze_radius_90_degrees: [None; 2],
        direction_numerics: [None; 2],
        require_numerical_margin: config.numerical_admission,
        marginalized_inner_radii: 0,
        marginal_draw_policy: "unchanged-full",
        replicas: config.replicas,
        replicate_direction_numerics: [None, None],
        mask_state_proposals: Vec::new(),
        population_integration: None,
    };
    let integrated_inner = if config.marginalize_unobserved_inner {
        IntegratedInner::new(model)
    } else {
        IntegratedInner {
            indices: Vec::new(),
            priors: Vec::new(),
        }
    };
    {
        result.marginalized_inner_radii = integrated_inner.indices.len();
        if !integrated_inner.indices.is_empty() {
            result.marginal_draw_policy = if config.preserve_marginal_draws {
                "original-full-proposal"
            } else {
                "reparameterized-marginal-proposal"
            };
        }
    }
    let best = &modes[0].1;
    let mut proposals = Vec::new();
    for (p, s) in modes
        .iter()
        .filter(|(_, s)| s.modeled_eyes == model.present)
    {
        // A close competing optimum with no valid proposal is an unresolved
        // integration problem, never evidence that the winner has all mass.
        let information = s
            .local_uncertainty
            .as_ref()
            .and_then(|u| u.information.clone())
            .or_else(|| constrained_proposal_information(model, p));
        let proposal = information
            .as_deref()
            .and_then(|i| Proposal::new(model, *p, i));
        let proposal = proposal
            .and_then(|p| p.marginal(&integrated_inner.indices, config.preserve_marginal_draws));
        if let Some(proposal) = proposal {
            proposals.push(proposal);
            result.modes.push(PosteriorMode {
                target_camera_mm: s.target_camera_mm,
                model_mass: None,
            });
        } else if s.robust_cost - best.robust_cost < 12.0 {
            result.status = "constraint-or-rank-limited";
            return result;
        }
    }
    if proposals.is_empty() {
        return result;
    }
    let mut rng = Random(config.seed.max(1));
    let mut samples: Vec<PosteriorSample> = Vec::new();
    #[cfg(test)]
    let mut sample_parameters = Vec::new();
    let mut sample_replicas = Vec::new();
    let mut sample_proposals = Vec::new();
    let mut log_weights = Vec::new();
    let original_count = proposals.len();
    let closest_mode = |p: &Parameters, proposals: &[Proposal]| {
        (0..original_count)
            .min_by(|&a, &b| {
                (p[0] - proposals[a].mean[0])
                    .hypot(p[1] - proposals[a].mean[1])
                    .total_cmp(&(p[0] - proposals[b].mean[0]).hypot(p[1] - proposals[b].mean[1]))
            })
            .unwrap()
    };
    let log_target = |p: &mut Parameters| -> Option<f64> {
        let correction = integrated_inner.condition(p)?;
        if (0..PARAMETERS)
            .any(|i| !p[i].is_finite() || p[i] < model.lower[i] || p[i] > model.upper[i])
        {
            return None;
        }
        let conics = model.conics(p)?;
        let residuals = model.residuals(p, &model.select(&conics))?;
        // The axial-distance Gaussian prior has the same log-chart Jacobian
        // in pilot fitting and final integration.
        let value = -0.5 * residuals.iter().map(|r| r * r).sum::<f64>() + p[2];
        let value = if integrated_inner.indices.is_empty() {
            value
        } else {
            value + correction
        };
        value.is_finite().then_some(value)
    };
    if config.conditional_nuisance {
        for i in 0..original_count {
            if let Some(proposal) = proposals[i].with_conditional_nuisance() {
                proposals.push(proposal);
            }
        }
    }
    if config.mask_state_proposals {
        add_mask_state_proposals(model, modes, config, &integrated_inner,
            &mut proposals, &mut result.mask_state_proposals);
    }
    if config.global_proposal {
        if let Some(global) = Proposal::global(model)
            .and_then(|p| p.marginal(&integrated_inner.indices, config.preserve_marginal_draws))
        {
            // A broad target draw need not inflate every anatomical variable
            // by the same random t scale. Keep the original heavy-tailed
            // component as well; every draw uses the complete mixture density.
            if config.conditional_global {
                if let Some(conditional) = global.with_conditional_nuisance() {
                    proposals.push(conditional);
                    result.global_proposals += 1;
                }
            }
            proposals.push(global);
            result.global_proposals += 1;
        }
    }
    if (config.adaptive || config.tail_proposal != TailProposal::Off) && config.budget >= 1024 {
        let pilot_components = proposals.len();
        let pilot_count = (config.budget / 4).min(2048) / pilot_components * pilot_components;
        let mut pilot = vec![Vec::new(); original_count];
        let mut tail_pilot = Vec::new();
        let mut tail_refinements = Vec::new();
        for index in 0..pilot_count {
            result.samples += 1;
            result.pilot_samples += 1;
            let mut p = proposals[index % pilot_components].draw(&mut rng);
            let Some(target) = log_target(&mut p) else {
                continue;
            };
            let density = log_mean_exp(proposals.iter().map(|q| q.log_density(&p)));
            if !density.is_finite() {
                continue;
            }
            pilot[closest_mode(&p, &proposals)].push((p, target - density));
            if config.tail_proposal != TailProposal::Off {
                tail_pilot.push((p, target));
            }
            result.pilot_feasible_samples += 1;
        }
        if config.adaptive {
            for i in 0..original_count {
                let adapted = proposals[i].adapted(model, &pilot[i]);
                result.adapted_proposals += usize::from(adapted.is_some());
                // Equal original/adapted slots preserve the original mixture even
                // if a basin lacks sufficient pilot support to fit a covariance.
                proposals.push(adapted.unwrap_or_else(|| proposals[i].clone()));
            }
        } else if matches!(config.tail_proposal,
            TailProposal::Recenter | TailProposal::Refit | TailProposal::ConditionalRefit | TailProposal::OutlierRefit | TailProposal::BoundaryRefit | TailProposal::BoundaryDefensive | TailProposal::ProfileAffine | TailProposal::ProfileQuadratic) {
            let mut selected = vec![false; tail_pilot.len()];
            let boundary_masks = if config.tail_proposal.explicit_boundaries() {
                boundary_relaxations(model)
            } else { Vec::new() };
            // All selection and curvature work uses only discarded pilot
            // configurations. Up to four components augment the unchanged
            // original mixture; neither the MAP nor the model priors change.
            for attempt in 0..4 {
                let next = tail_pilot.iter().enumerate()
                    .filter(|(i, _)| !selected[*i])
                    .filter_map(|(i, (p, target))| {
                        let importance = target - log_mean_exp(proposals.iter().map(|q| q.log_density(p)));
                        importance.is_finite().then_some((i, importance))
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                let Some((index, _)) = next else { break; };
                selected[index] = true;
                let mut p = tail_pilot[index].0;
                let forced = if boundary_masks.is_empty() { &[][..] }
                    else { boundary_masks[attempt % boundary_masks.len()].as_slice() };
                let relaxed = if matches!(config.tail_proposal, TailProposal::OutlierRefit | TailProposal::BoundaryRefit | TailProposal::BoundaryDefensive) {
                    let Some(relaxed) = with_relaxed_proposal_groups(model, &p, forced) else { continue; };
                    Some(relaxed)
                } else { None };
                let shaping_model = relaxed.as_ref().map(|(m, _)| m).unwrap_or(model);
                if matches!(config.tail_proposal, TailProposal::ConditionalRefit | TailProposal::OutlierRefit | TailProposal::BoundaryRefit | TailProposal::BoundaryDefensive | TailProposal::ProfileAffine | TailProposal::ProfileQuadratic) {
                    let before = shaping_model.conics(&p).and_then(|c| shaping_model.residuals(&p, &shaping_model.select(&c)))
                        .map(|r| squared_norm(&r));
                    let Some((refined, cost, steps)) = conditional_refined_center(shaping_model, p) else { continue; };
                    assert_eq!(&refined[..TARGET_PARAMETERS], &p[..TARGET_PARAMETERS]);
                    let mut trace = serde_json::json!({
                        "shared_target_parameters":p[..TARGET_PARAMETERS],
                        "cost_before":before,"cost_after":cost,"refinement_steps":steps,
                    });
                    if let Some((_, omitted)) = &relaxed {
                        if config.tail_proposal.explicit_boundaries() {
                            trace["forced_proposal_groups"] = serde_json::json!(forced);
                        }
                        trace["omitted_proposal_groups"] = serde_json::json!(omitted.iter().map(|&i| {
                            let g = &model.groups[i];
                            serde_json::json!({"index":i,"eye":g.alternatives[0].eye,
                                "group":g.alternatives[0].group,"weight":g.weight,
                                "alternative_kinds":g.alternatives.iter().map(|a|format!("{:?}",a.kind)).collect::<Vec<_>>()})
                        }).collect::<Vec<_>>());
                        trace["complete_model_cost_after"] = serde_json::json!(model.conics(&refined)
                            .and_then(|c|model.residuals(&refined,&model.select(&c))).map(|r|squared_norm(&r)));
                    }
                    tail_refinements.push(trace);
                    p = refined;
                }
                let proposal = if config.tail_proposal == TailProposal::Recenter {
                    Some(proposals[closest_mode(&p, &proposals)].recentered(p))
                } else {
                    constrained_proposal_information(shaping_model, &p)
                        .and_then(|information| Proposal::new(shaping_model, p, &information))
                        .and_then(|q| q.marginal(&integrated_inner.indices, config.preserve_marginal_draws))
                };
                let proposal = proposal.and_then(|q| {
                    if config.conditional_nuisance { q.with_conditional_nuisance() } else { Some(q) }
                });
                if let Some(proposal) = proposal {
                    proposals.push(proposal);
                    result.adapted_proposals += 1;
                }
            }
        }
        // Fit every ordinary component first. Affine and quadratic controls
        // therefore use the same pilot, anchors, curvature and profiling probes;
        // their only difference is applying the fitted quadratic coefficients.
        // No final sample can influence this frozen unit-Jacobian transport.
        let mut profile_transports = Vec::new();
        if config.tail_proposal.profiled() {
            for (index, proposal) in proposals.iter_mut().enumerate().skip(pilot_components) {
                if let Some((transported, mut trace)) = proposal.with_profile_transport(
                    model, config.tail_proposal == TailProposal::ProfileQuadratic,
                ) {
                    trace["component"] = serde_json::json!(index);
                    trace["fitted"] = serde_json::json!(true);
                    profile_transports.push(trace);
                    *proposal = transported;
                } else {
                    // A failed profile retains the proper original proposal.
                    profile_transports.push(serde_json::json!({"component":index,"fitted":false}));
                }
            }
        }
        if config.tail_proposal == TailProposal::BoundaryDefensive {
            // Integer allocation weights, not additional observations or MAP
            // modes. Keep the same pilot and fitted alternatives, then triple
            // each original component's share in the complete final mixture.
            // Uniform stratification over these slots and their full density
            // gives exactly the corresponding 3:1 component weighting.
            for _ in 0..2 { proposals.extend_from_within(..pilot_components); }
        }
        if config.trace_tail && config.tail_proposal != TailProposal::Off {
            let mut trace = serde_json::json!({
                "recipe":format!("{:?}",config.tail_proposal),"seed":config.seed.to_string(),
                "pilot_draws":result.pilot_samples,"pilot_feasible":result.pilot_feasible_samples,
                "added_components":result.adapted_proposals,"original_components":pilot_components,
                "conditional_refinements":tail_refinements,
                "final_components":proposals.len()});
            if config.tail_proposal.profiled() {
                trace["profile_transports"] = serde_json::json!(profile_transports);
            }
            eprintln!("posterior-pilot {trace}");
        }
        // Pilot points never enter the final mean, covariance, ESS, mode mass
        // or direction radius. The frozen q below generated every such point.
    }
    #[cfg(test)]
    if let Some(reference) = config.annealed_reference {
        // Independent numerical control: never advances the production draw
        // stream or changes its geometry, weights, confidence or publication.
        let diagnostic = annealed::diagnose(model, best, &modes[0].0, &integrated_inner.indices,
            &proposals, &log_target, reference, config.seed);
        eprintln!("posterior-annealed {diagnostic}");
    }
    if let Some(populations) = config.populations {
        return populations::integrate(model, best, &proposals, original_count,
            &log_target, populations, config.seed, result);
    }
    // Stratify uniformly by proposal. Use equal counts so the density below
    // is the exact mixture used to draw this bounded sample population.
    let balanced_components = proposals.len();
    let balanced_components = balanced_components
        * if config.preserve_replica_draws { 1 } else { config.replicas };
    let mut replica_draw_counts = vec![0; config.replicas];
    let mut replica_rngs = (0..config.replicas)
        .map(|replica| {
            Random(if replica == 0 {
                rng.0
            } else {
                replica_seed(config.seed, replica)
            })
        })
        .collect::<Vec<_>>();
    let count = (config.budget - result.pilot_samples) / balanced_components * balanced_components;
    let batch = (512 / balanced_components).max(1) * balanced_components;
    for index in 0..count {
        // Easy observations stop early. Difficult constrained observations
        // receive more integration work within the same strict source-local
        // budget, never extra optimizer starts or borrowed temporal evidence.
        if config.early_stop && index > 0 && index % batch == 0 {
            if let Some((weights, effective)) = weights(&log_weights) {
                if effective >= 48.0 && weights.iter().all(|w| *w <= 0.1) {
                    #[allow(unused_mut)]
                    let mut settled = true;
                    if config.numerical_admission {
                        let precision = direction_numerics(
                            best,
                            &samples,
                            &weights,
                            &sample_proposals,
                            proposals.len(),
                            index,
                        );
                        let replicated = replicate_direction_numerics(
                            best,
                            &samples,
                            &weights,
                            &sample_replicas,
                            &replica_draw_counts,
                        );
                        settled = (0..2).filter(|&eye| model.present[eye]).all(|eye| {
                            precision[eye].is_some_and(|mut n| {
                                if config.replicas > 1 {
                                    let Some(r) = &replicated[eye] else {
                                        return false;
                                    };
                                    n.standard_error = n.standard_error.max(r.standard_error);
                                }
                                n.mass - 2.0 * n.standard_error >= 0.9
                                    || n.mass + 2.0 * n.standard_error < 0.9
                            })
                        });
                    }
                    if settled {
                        break;
                    }
                }
            }
        }
        result.samples += 1;
        let replica = (index / proposals.len()) % config.replicas;
        { replica_draw_counts[replica] += 1; }
        let random = &mut rng;
        let random = if config.replicas > 1 && !config.preserve_replica_draws {
            &mut replica_rngs[replica]
        } else {
            random
        };
        let mut p = proposals[index % proposals.len()].draw(random);
        let Some(log_target) = log_target(&mut p) else {
            continue;
        };
        let Some(target) = model.target(&p) else {
            continue;
        };
        let gazes = std::array::from_fn::<_, 2, _>(|eye| {
            if !model.present[eye] {
                return None;
            }
            let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
            normalized3(sub3(target, [p[k], p[k + 1], p[k + 2]]))
        });
        let log_proposal = log_mean_exp(proposals.iter().map(|q| q.log_density(&p)));
        if !log_target.is_finite() || !log_proposal.is_finite() {
            continue;
        }
        let mode = closest_mode(&p, &proposals);
        log_weights.push(log_target - log_proposal);
        samples.push((target, gazes, mode));
        #[cfg(test)]
        sample_parameters.push(p);
        sample_replicas.push(replica);
        sample_proposals.push(index % proposals.len());
    }
    result.feasible_samples = samples.len();
    let Some((weights, effective)) = weights(&log_weights) else {
        result.status = "no-feasible-samples";
        return result;
    };
    result.effective_samples = effective;
    result.maximum_sample_mass = weights.iter().copied().reduce(f64::max);
    result.direction_numerics = direction_numerics(
        best,
        &samples,
        &weights,
        &sample_proposals,
        proposals.len(),
        result.samples - result.pilot_samples,
    );
    if config.replicas > 1 {
        result.replicate_direction_numerics = replicate_direction_numerics(
            best,
            &samples,
            &weights,
            &sample_replicas,
            &replica_draw_counts,
        );
    }
    #[cfg(test)]
    if config.trace_tail && result.maximum_sample_mass.is_some_and(|w| w > 0.2) {
        let index = (0..weights.len())
            .max_by(|&a, &b| weights[a].total_cmp(&weights[b]))
            .unwrap();
        let p = sample_parameters[index];
        let conics = model.conics(&p).unwrap();
        let cost = squared_norm(&model.residuals(&p, &model.select(&conics)).unwrap());
        let fit = model.solution(&p, cost);
        eprintln!(
            "posterior-tail {}",
            serde_json::json!({"seed":config.seed.to_string(),"budget":config.budget,
            "adaptive":config.adaptive,"tail_proposal":format!("{:?}",config.tail_proposal),"global":config.global_proposal,"conditional_global":config.conditional_global,"conditional_nuisance":config.conditional_nuisance,"best_cost":best.robust_cost,
            "sample_cost":cost,"sample_mass":weights[index],"parameters":p,
            "sample_target":samples[index].0,"best_target":best.target_camera_mm,
            "arcs":fit.as_ref().map(|s|s.arcs.iter().map(|a|serde_json::json!({"eye":a.exposure.roi.0,
                "kind":format!("{:?}",a.kind),"used":a.used,"rms":a.rms_px,"sigma":a.sigma_px})).collect::<Vec<_>>())})
        );
    }
    if effective < 24.0 || result.maximum_sample_mass.is_some_and(|w| w > 0.2) {
        result.status = "insufficient-sampling";
        return result;
    }
    result.status = "estimated-conditional";
    let mean = std::array::from_fn(|i| {
        samples
            .iter()
            .zip(&weights)
            .map(|(s, w)| w * s.0[i])
            .sum::<f64>()
    });
    result.target_mean_camera_mm = Some(mean);
    result.target_covariance_mm2 = Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            samples
                .iter()
                .zip(&weights)
                .map(|(s, w)| w * (s.0[i] - mean[i]) * (s.0[j] - mean[j]))
                .sum()
        })
    }));
    for (i, mode) in result.modes.iter_mut().enumerate() {
        mode.model_mass = Some(
            samples
                .iter()
                .zip(&weights)
                .filter(|(s, _)| s.2 == i)
                .map(|(_, w)| w)
                .sum(),
        );
    }
    for eye in 0..2 {
        if let Some(axis) = best.eye_gaze_directions[eye] {
            result.gaze_radius_90_degrees[eye] = quantile(
                samples
                    .iter()
                    .zip(&weights)
                    .filter_map(|(s, w)| {
                        Some((
                            dot3(axis, s.1[eye]?).clamp(-1.0, 1.0).acos().to_degrees(),
                            *w,
                        ))
                    })
                    .collect(),
                0.9,
            );
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_transport_proposal_preserves_fixation_and_recovers_known_uniform_moments() {
        let mut origin = [0.0; PARAMETERS];
        origin[0] = -0.15;
        origin[3] = 0.4;
        let mut scales = [1.0; PARAMETERS];
        scales[0] = 0.75;
        scales[3] = 1.25;
        let original = Proposal::with_active(origin, vec![0, 1, 3], scales,
            &[vec![2.0, 0.3, 0.5], vec![0.3, 1.2, 0.1], vec![0.5, 0.1, 1.5]])
            .unwrap().marginal(&[1], true).unwrap();
        for conditional in [false, true] {
            for quadratic in [false, true] {
                let base = if conditional { original.with_conditional_nuisance().unwrap() }
                    else { original.clone() };
                assert_eq!(base.original_sampler.is_some(), !conditional);
                let target_scale = base.with_conditional_nuisance().unwrap()
                    .conditional_lower.unwrap()[0][0];
                let mut transport = ProfileTransport {
                    base: Box::new(base.clone()), target_lower: vec![vec![target_scale]],
                    steps: vec![1.0], constant: [0.0; PARAMETERS],
                    linear: vec![[0.0; PARAMETERS]], quadratic: vec![[0.0; PARAMETERS]],
                    include_quadratic: quadratic,
                };
                transport.constant[3] = 0.35;
                transport.linear[0][3] = 0.9;
                transport.quadratic[0][3] = 1.2;
                let mut shifted = base.clone();
                shifted.profile_transport = Some(transport.clone());
                // Include extrapolated targets beyond both feature clamps.
                // The omitted coordinate must also keep its delegated draw.
                for t in [-12.0, -1.0, 0.0, 1.0, 12.0] {
                    let mut p = origin;
                    p[0] = t;
                    p[1] = 17.0;
                    let moved = transport.apply(p, 1.0);
                    let recovered = transport.apply(moved, -1.0);
                    for i in 0..PARAMETERS {
                        if i == 3 { assert!((recovered[i] - p[i]).abs() < 1e-12); }
                        else { assert_eq!(moved[i], p[i]); assert_eq!(recovered[i], p[i]); }
                    }
                }
                let mut original_rng = Random(0x7174_14c0_a4b6_2391);
                let mut shifted_rng = Random(original_rng.0);
                for _ in 0..128 {
                    let p = base.draw(&mut original_rng);
                    let q = shifted.draw(&mut shifted_rng);
                    assert_eq!(original_rng.0, shifted_rng.0);
                    assert_eq!(q, transport.apply(p, 1.0));
                    assert_eq!(&q[..TARGET_PARAMETERS], &p[..TARGET_PARAMETERS]);
                }
                // The target is uniform in the FINAL square, so its means,
                // second moments and cross moment are known independently of
                // the proposal and transport. Use the complete mixture density.
                let proposals = [base, shifted];
                let mut rng = Random(0x91ae_9134_38bd_4a25);
                let mut samples = Vec::new();
                let mut logs = Vec::new();
                for i in 0..64000 {
                    let p = proposals[i % 2].draw(&mut rng);
                    if [0, 3].iter().any(|&j| !(-1.0..=1.0).contains(&p[j])) { continue; }
                    logs.push(-log_mean_exp(proposals.iter().map(|q| q.log_density(&p))));
                    samples.push(p);
                }
                let (weights, effective) = weights(&logs).unwrap();
                assert!(effective > 4000.0, "conditional={conditional} quadratic={quadratic}: {effective}");
                for axis in [0, 3] {
                    let mean = samples.iter().zip(&weights).map(|(p, w)| w * p[axis]).sum::<f64>();
                    let second = samples.iter().zip(&weights).map(|(p, w)| w * p[axis].powi(2)).sum::<f64>();
                    assert!(mean.abs() < 0.025, "conditional={conditional} quadratic={quadratic} axis={axis}: {mean}");
                    assert!((second - 1.0 / 3.0).abs() < 0.025,
                        "conditional={conditional} quadratic={quadratic} axis={axis}: {second}");
                }
                let cross = samples.iter().zip(&weights).map(|(p, w)| w * p[0] * p[3]).sum::<f64>();
                assert!(cross.abs() < 0.025, "conditional={conditional} quadratic={quadratic}: {cross}");
            }
        }
    }

    #[test]
    fn recentered_marginal_sampler_corrects_a_known_uniform_target() {
        let mut origin = [0.0; PARAMETERS];
        origin[0] = -0.5;
        origin[3] = 0.7;
        let full = Proposal::with_active(origin,vec![0,1,3],[1.0;PARAMETERS],
            &[vec![2.0,0.3,0.5],vec![0.3,1.2,0.1],vec![0.5,0.1,1.5]]).unwrap();
        let original = full.marginal(&[1],true).unwrap();
        assert!(original.original_sampler.is_some());
        let mut center = origin;
        center[0] = 1.5;
        center[3] = -1.0;
        let moved = original.recentered(center);
        for original_slots in [1, 3] {
        let mut proposals = vec![original.clone(); original_slots];
        proposals.push(moved.clone());
        let mut rng = Random(0x4875_abc3_4956_ade1);
        let mut samples=Vec::new();let mut logs=Vec::new();
        for i in 0..48000 {
            let p=proposals[i%proposals.len()].draw(&mut rng);
            if [0,3].iter().any(|&j| !(-1.0..=1.0).contains(&p[j])) {continue;}
            samples.push(p);
            logs.push(-log_mean_exp(proposals.iter().map(|q|q.log_density(&p))));
        }
        let (weights,effective)=weights(&logs).unwrap();
        assert!(effective>6000.0,"{effective}");
        for axis in [0,3] {
            let mean=samples.iter().zip(&weights).map(|(p,w)|p[axis]*w).sum::<f64>();
            let second=samples.iter().zip(&weights).map(|(p,w)|p[axis].powi(2)*w).sum::<f64>();
            assert!(mean.abs()<0.025,"axis={axis} mean={mean}");
            assert!((second-1.0/3.0).abs()<0.02,"axis={axis} second={second}");
        }
        }
    }

    #[test]
    fn replicate_ratio_preserves_unequal_normalizers_and_requires_every_replica() {
        let observations = [
            (0.4, true, 0),
            (0.2, false, 0),
            (0.4 / 3.0, true, 1),
            (0.4 / 3.0, true, 2),
            (0.4 / 3.0, true, 3),
        ];
        let n = replicate_indicator_numerics(&observations, &[2; 4]).unwrap();
        assert!((n.mass - 0.8).abs() < 1e-12);
        assert!(
            (n.masses.iter().sum::<f64>() / 4.0 - n.mass).abs() > 0.1,
            "averaging normalized replicas would hide the competing mass"
        );
        assert!((n.standard_error - 0.10666666666666667).abs() < 1e-12);
        assert!((n.normalizer_shares[0] - 0.6).abs() < 1e-12);
        assert!(replicate_indicator_numerics(&observations[..4], &[2; 4]).is_none());
        assert!(replicate_indicator_numerics(&observations, &[5]).is_none());
        assert!(replicate_indicator_numerics(&observations, &[2, 2, 2, 0]).is_none());
        let mut zero = observations.to_vec();
        zero.push((0.0, false, 3));
        assert_eq!(
            replicate_indicator_numerics(&zero, &[2; 4])
                .unwrap()
                .standard_error,
            n.standard_error
        );
    }

    #[test]
    fn between_replica_error_matches_independent_known_probability_trials() {
        let mut rng = Random(0x26d1_e743_819b_7451);
        for counts in [[200; 4], [50, 100, 200, 450]] {
            let mut squared_error = 0.0;
            let mut estimated_variance = 0.0;
            for _ in 0..1200 {
                let mut observations = Vec::new();
                for (replica, &count) in counts.iter().enumerate() {
                    for proposal in 0..2 {
                        let (weight, probability) = if proposal == 0 {
                            (0.8 / 400.0, 0.8)
                        } else {
                            (0.2 / 400.0, 0.05)
                        };
                        for _ in 0..count / 2 {
                            observations.push((weight, rng.uniform() < probability, replica));
                        }
                    }
                }
                let n = replicate_indicator_numerics(&observations, &counts).unwrap();
                squared_error += (n.mass - 0.65).powi(2);
                estimated_variance += n.standard_error.powi(2);
            }
            let expected = (0.8_f64.powi(2) * 0.8 * 0.2 + 0.2_f64.powi(2) * 0.05 * 0.95) / 400.0;
            assert!((squared_error / 1200.0 / expected - 1.0).abs() < 0.12, "{counts:?}");
            assert!((estimated_variance / 1200.0 / expected - 1.0).abs() < 0.10, "{counts:?}");
        }
    }

    #[test]
    fn integrated_normal_mass_matches_direct_quadrature_including_narrow_tails() {
        for (a, b) in [
            (-6.0, 7.0),
            (-1.0, 1.0),
            (4.0, 5.0),
            (9.0, 10.0),
            (10.0, 10.0000001),
            (0.0, 1e-10),
        ] {
            let n = 32768;
            let numerical = (0..n)
                .map(|i| {
                    let z = a + (b - a) * (i as f64 + 0.5) / n as f64;
                    (-0.5 * z * z).exp()
                })
                .sum::<f64>()
                * (b - a)
                / n as f64
                / (2.0 * std::f64::consts::PI).sqrt();
            let mass = normal_interval_mass(a, b).unwrap();
            assert!(
                (mass / numerical - 1.0).abs() < 1e-7,
                "{a}..{b}: {mass} vs {numerical}"
            );
            assert!((normal_interval_mass(-b, -a).unwrap() / mass - 1.0).abs() < 1e-13);
        }
        assert!((normal_interval_mass(-1.0, 1.0).unwrap() - 0.6826894921370859).abs() < 1e-14);
        assert!(normal_interval_mass(2.0, 2.0).is_none());
        assert!(normal_interval_mass(3.0, 2.0).is_none());
        assert!(normal_interval_mass(f64::NAN, 2.0).is_none());
    }

    #[test]
    fn marginal_student_proposal_integrates_correlations_instead_of_fixing_nuisance() {
        let original = Proposal::with_active(
            [0.0; PARAMETERS],
            vec![0, 1],
            [1.0; PARAMETERS],
            &[vec![1.0, 0.8], vec![0.8, 1.0]],
        )
        .unwrap();
        let marginal = original.clone().marginal(&[1], true).unwrap();
        assert!((marginal.information[0][0] - 0.36).abs() < 1e-12);
        let mut original_rng = Random(0x1234_5678_9abc_def0);
        let mut marginal_rng = Random(original_rng.0);
        for _ in 0..512 {
            let a = original.draw(&mut original_rng);
            let b = marginal.draw(&mut marginal_rng);
            assert_eq!(
                a[0], b[0],
                "retained candidates must match the full proposal exactly"
            );
            assert_eq!(
                original_rng.0, marginal_rng.0,
                "subsequent proposals must share the same random stream"
            );
        }
        for x in [-4.0, 0.0, 3.0] {
            let n = 32768;
            let integrated = (0..n)
                .map(|i| {
                    let angle = std::f64::consts::PI * ((i as f64 + 0.5) / n as f64 - 0.5);
                    let mut p = [0.0; PARAMETERS];
                    p[0] = x;
                    p[1] = angle.tan();
                    (original.log_density(&p) + t_log_normalizer(2)).exp() * std::f64::consts::PI
                        / angle.cos().powi(2)
                })
                .sum::<f64>()
                / n as f64;
            let mut p = [0.0; PARAMETERS];
            p[0] = x;
            let density = (marginal.log_density(&p) + t_log_normalizer(1)).exp();
            assert!(
                (density / integrated - 1.0).abs() < 1e-8,
                "{x}: {density} vs {integrated}"
            );
        }
    }

    #[test]
    fn stratified_mass_standard_error_matches_independent_known_probability_trials() {
        // A target with mass .8/.2 in two disjoint proposal strata. The event
        // probabilities within them are .8/.05. Its exact probability is .65;
        // balanced sampling has analytically known variance .00026075.
        let mut rng = Random(0x791b_692b_1a83_a46d);
        let n = 400;
        let trials = 800;
        let mut squared_errors = 0.0;
        let mut estimated_variances = 0.0;
        let mut means = 0.0;
        for _ in 0..trials {
            let observations = (0..2)
                .flat_map(|proposal| {
                    let probability = if proposal == 0 { 0.8 } else { 0.05 };
                    let weight = if proposal == 0 {
                        0.8 / n as f64
                    } else {
                        0.2 / n as f64
                    };
                    (0..n)
                        .map(|_| (weight, rng.uniform() < probability, proposal))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let result = indicator_numerics(&observations, 2, n).unwrap();
            means += result.mass;
            squared_errors += (result.mass - 0.65).powi(2);
            estimated_variances += result.standard_error.powi(2);
        }
        let expected = (0.8_f64.powi(2) * 0.8 * 0.2 + 0.2_f64.powi(2) * 0.05 * 0.95) / n as f64;
        assert!((means / trials as f64 - 0.65).abs() < 0.002);
        assert!((squared_errors / trials as f64 / expected - 1.0).abs() < 0.12);
        assert!((estimated_variances / trials as f64 / expected - 1.0).abs() < 0.03);
    }

    #[test]
    fn infeasible_draws_remain_zero_weight_members_of_their_proposal_strata() {
        let observed = vec![
            (0.4, true, 0),
            (0.2, false, 0),
            (0.3, true, 1),
            (0.1, false, 1),
        ];
        let mut explicit = observed.clone();
        explicit.extend([
            (0.0, false, 0),
            (0.0, true, 0),
            (0.0, false, 1),
            (0.0, true, 1),
        ]);
        let a = indicator_numerics(&observed, 2, 4).unwrap();
        let b = indicator_numerics(&explicit, 2, 4).unwrap();
        assert_eq!(a.mass, b.mass);
        assert_eq!(a.standard_error, b.standard_error);
        assert_eq!(a.mass, 0.7);
        assert!(
            a.standard_error > 0.15,
            "few feasible draws cannot certify a settled admission probability"
        );
        assert!(indicator_numerics(&observed, 2, 1).is_none());
    }
    #[test]
    fn fixed_target_coordinates_do_not_turn_nuisance_parameters_into_student_tails() {
        // Only target slope 1 is free; indices 3 and 4 remain Gaussian
        // nuisance innovations even though they are among the first 3 active
        // coordinates. This tests the actual draw law and density together.
        let information = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let proposal = Proposal::with_active(
            [0.0; PARAMETERS],
            vec![1, 3, 4],
            [1.0; PARAMETERS],
            &information,
        )
        .unwrap()
        .with_conditional_nuisance()
        .unwrap();
        let mut rng = Random(0x761a_58c9_944e_93e1);
        let mut second_moments = [0.0; 3];
        for _ in 0..50000 {
            let p = proposal.draw(&mut rng);
            for (j, i) in [1, 3, 4].into_iter().enumerate() {
                second_moments[j] += p[i] * p[i] / 50000.0;
            }
            assert_eq!(p[0], 0.0);
            assert_eq!(p[2], 0.0);
        }
        assert!(
            (second_moments[0] - 7.0 / 5.0).abs() < 0.06,
            "{second_moments:?}"
        );
        for variance in &second_moments[1..] {
            assert!((variance - 1.0).abs() < 0.04, "{second_moments:?}");
        }
        let origin = [0.0; PARAMETERS];
        for index in [3, 4] {
            let mut p = origin;
            p[index] = 2.0;
            assert!((proposal.log_density(&p) - proposal.log_density(&origin) + 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mixed_target_tails_and_gaussian_nuisance_keep_their_relative_normalization() {
        let first = Proposal {
            mean: [0.0; PARAMETERS],
            active: vec![0, 1, 2, 3],
            scales: [1.0; PARAMETERS],
            information: (0..4)
                .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect(),
            lower: (0..4)
                .map(|i| (0..4).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect(),
            log_sqrt_determinant: 0.0,
            conditional_lower: None,
            original_sampler: None,
            profile_transport: None,
        };
        let mut second = first.with_conditional_nuisance().unwrap();
        second.mean[0] = 0.75;
        second.mean[3] = -0.75;
        let proposals = [first, second];
        let mut rng = Random(0x48d3_22a1_1931_df5b);
        let mut samples = Vec::new();
        let mut logs = Vec::new();
        for i in 0..36000 {
            let p = proposals[i % 2].draw(&mut rng);
            if !p[..4].iter().all(|v| (-1.0..=1.0).contains(v)) {
                continue;
            }
            samples.push(p);
            logs.push(-log_mean_exp(proposals.iter().map(|q| q.log_density(&p))));
        }
        let (weights, effective) = weights(&logs).unwrap();
        assert!(effective > 1500.0, "{effective}");
        for axis in 0..4 {
            let mean = samples
                .iter()
                .zip(&weights)
                .map(|(s, w)| w * s[axis])
                .sum::<f64>();
            let variance = samples
                .iter()
                .zip(&weights)
                .map(|(s, w)| w * s[axis] * s[axis])
                .sum::<f64>();
            assert!(mean.abs() < 0.035, "axis {axis}: {mean}");
            assert!(
                (variance - 1.0 / 3.0).abs() < 0.035,
                "axis {axis}: {variance}"
            );
        }
    }
    #[test]
    fn correlated_mixture_density_recovers_bounded_uniform_target_moments() {
        let first = Proposal {
            mean: [0.0; PARAMETERS],
            active: vec![0, 1],
            scales: [1.0; PARAMETERS],
            information: vec![vec![4.0, -1.0], vec![-1.0, 1.0]],
            lower: vec![vec![2.0, 0.0], vec![-0.5, 0.75_f64.sqrt()]],
            log_sqrt_determinant: 0.5 * 3.0_f64.ln(),
            conditional_lower: None,
            original_sampler: None,
            profile_transport: None,
        };
        let mut second = first.clone();
        second.mean[0] = 1.5;
        second.mean[1] = -1.0;
        for row in &mut second.information {
            for v in row {
                *v /= 4.0;
            }
        }
        for row in &mut second.lower {
            for v in row {
                *v /= 2.0;
            }
        }
        second.log_sqrt_determinant -= 2.0 * 2.0_f64.ln();
        let proposals = [first, second];
        let mut rng = Random(0xb6df_9a17_51ab_d291);
        let mut samples = Vec::new();
        let mut logs = Vec::new();
        for i in 0..24000 {
            let p = proposals[i % 2].draw(&mut rng);
            if !(-1.0..=2.0).contains(&p[0]) || !(-2.0..=1.0).contains(&p[1]) {
                continue;
            }
            samples.push([p[0], p[1]]);
            logs.push(-log_mean_exp(proposals.iter().map(|q| q.log_density(&p))));
        }
        let (weights, effective) = weights(&logs).unwrap();
        assert!(effective > 3000.0, "{effective}");
        for (axis, expected) in [(0, 0.5), (1, -0.5)] {
            let mean = samples
                .iter()
                .zip(&weights)
                .map(|(s, w)| w * s[axis])
                .sum::<f64>();
            let variance = samples
                .iter()
                .zip(&weights)
                .map(|(s, w)| w * (s[axis] - expected).powi(2))
                .sum::<f64>();
            assert!((mean - expected).abs() < 0.04, "axis {axis}: mean {mean}");
            assert!(
                (variance - 0.75).abs() < 0.05,
                "axis {axis}: variance {variance}"
            );
        }
        let covariance = samples
            .iter()
            .zip(&weights)
            .map(|(s, w)| w * (s[0] - 0.5) * (s[1] + 0.5))
            .sum::<f64>();
        assert!(
            covariance.abs() < 0.04,
            "correlated proposals must not create target correlation: {covariance}"
        );
    }
    #[test]
    fn log_weight_normalization_survives_extreme_losses_and_reports_degeneracy() {
        let (w, n) = weights(&[-10000.0, -10000.0, f64::NEG_INFINITY]).unwrap();
        assert_eq!(w, [0.5, 0.5, 0.0]);
        assert_eq!(n, 2.0);
        let (w, n) = weights(&[-10000.0, 0.0]).unwrap();
        assert_eq!(w, [0.0, 1.0]);
        assert_eq!(n, 1.0);
        assert!(weights(&[f64::NEG_INFINITY]).is_none());
    }
    #[test]
    fn proposal_density_correction_recovers_gaussian_mass_from_a_heavy_tailed_sampler() {
        let mut rng = Random(0xd1b5_4a32_d192_ed03);
        let mut x = Vec::new();
        let mut logs = Vec::new();
        for _ in 0..12000 {
            let value = rng.normal()
                * (T_DOF as f64 / (0..T_DOF).map(|_| rng.normal().powi(2)).sum::<f64>()).sqrt();
            let log_q = -0.5 * (T_DOF + 1) as f64 * (value * value / T_DOF as f64).ln_1p();
            x.push(value);
            logs.push(-0.5 * value * value - log_q);
        }
        let (w, n) = weights(&logs).unwrap();
        assert!(n > 9000.0);
        let mean = x.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>();
        let variance = x.iter().zip(&w).map(|(x, w)| x * x * w).sum::<f64>();
        assert!(mean.abs() < 0.04, "{mean}");
        assert!((variance - 1.0).abs() < 0.05, "{variance}");
    }
}
