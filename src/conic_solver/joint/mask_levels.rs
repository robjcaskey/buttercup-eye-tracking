//! Exact finite-state marginalization of correlated mask-boundary sensitivity.
//! The sensitivity states have a declared uniform engineering prior. This is
//! not a calibrated segmentation posterior. All states evaluate the SAME joint
//! fixation, and their mass is shared across the mask's observed contour arcs.

use super::*;

#[derive(Clone, Debug)]
pub(super) struct Activity {
    pub(super) choice: usize,
    pub(super) outlier: bool,
    pub(super) cost: f64,
    pub(super) mixture: Option<ArcMixture>,
}

#[derive(Clone, Debug)]
pub(super) struct ArcComponent {
    pub(super) choice: usize,
    pub(super) outlier: bool,
    pub(super) cost: f64,
}

#[derive(Clone, Debug)]
pub(super) struct ArcMixture {
    pub(super) components: Vec<ArcComponent>,
    pub(super) responsibilities: Vec<f64>,
    pub(super) entropy_penalty: f64,
    pub(super) marginal_cost: f64,
}

#[derive(Clone, Debug)]
pub(super) struct FamilyActivity {
    pub(super) eye: usize,
    pub(super) boundary: usize,
    pub(super) groups: Vec<usize>,
    pub(super) states: Vec<Vec<Activity>>,
    pub(super) costs: Vec<f64>,
    pub(super) responsibilities: Vec<f64>,
    pub(super) entropy_penalty: f64,
    pub(super) marginal_cost: f64,
    pub(super) representative: usize,
}

#[derive(Clone, Debug)]
pub(super) struct Selection<'a> {
    pub(super) choices: Vec<usize>,
    pub(super) levels: Vec<usize>,
    pub(super) in_family: Vec<bool>,
    pub(super) families: Vec<FamilyActivity>,
    pub(super) group_mixtures: Vec<Option<ArcMixture>>,
    pub(super) cached_arcs: Vec<CachedArcResiduals<'a>>,
}

impl Selection<'_> {
    pub(super) fn has_group_mixture(&self, index: usize) -> bool {
        self.group_mixtures.get(index).is_some_and(Option::is_some)
    }
    pub(super) fn has_marginalization(&self) -> bool {
        !self.families.is_empty() || self.group_mixtures.iter().any(Option::is_some)
    }
}

impl std::ops::Deref for Selection<'_> {
    type Target = [usize];
    fn deref(&self) -> &Self::Target { &self.choices }
}
impl<'a, 'model> IntoIterator for &'a Selection<'model> {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;
    fn into_iter(self) -> Self::IntoIter { self.choices.iter() }
}
impl IntoIterator for Selection<'_> {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;
    fn into_iter(self) -> Self::IntoIter { self.choices.into_iter() }
}

pub(super) fn mixture(costs: impl AsRef<[f64]>) -> (Vec<f64>, f64, f64) {
    let costs = costs.as_ref();
    assert!(!costs.is_empty() && costs.len() <= 7);
    let count = costs.len() as f64;
    let minimum = costs.iter().copied().fold(f64::INFINITY, f64::min);
    let weights = costs.iter().map(|cost| (-0.5 * (cost-minimum)).exp()).collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    let responsibilities = weights.into_iter().map(|weight| weight/sum).collect::<Vec<_>>();
    let entropy = 2.0 * responsibilities.iter().copied().filter(|&q| q > 0.0)
        .map(|q| q*(count*q).ln()).sum::<f64>().max(0.0);
    (responsibilities, entropy, minimum - 2.0*(sum/count).ln())
}

fn same_observation(a: &SparseArc, b: &SparseArc) -> bool {
    a.kind == b.kind && a.points == b.points && a.sigma == b.sigma
        && a.outward_normals == b.outward_normals && a.level_sets == b.level_sets
        && a.weight == b.weight && a.quadrature == b.quadrature
}

/// Marginalize distinct alternatives within ONE correlated observation group.
/// Components keep the existing robust loss, shorter-support charge and cap.
/// Uniform weights are a declared engineering prior, not detector calibration.
fn activity(model: &Problem<'_>, conics: &[[Option<ProjectedCircle>;3];2],
    index: usize, level: usize) -> Activity
{
    let group = &model.groups[index];
    if !model.marginalize_arc_alternatives || group.alternatives.len()==1 {
        return group.alternatives.iter().enumerate().map(|(choice,arc)| {
            let mean=arc.mean_cost_at_level(conics[arc.eye][arc.boundary].unwrap(),level);
            let cost=arc.weight*mean.min(MAXIMUM_GROUP_COST)+(group.weight-arc.weight)*MAXIMUM_GROUP_COST;
            (Activity {choice,outlier:mean>=MAXIMUM_GROUP_COST,cost,mixture:None},mean)
        }).min_by(|a,b|a.0.cost.total_cmp(&b.0.cost).then(a.1.total_cmp(&b.1))).unwrap().0;
    }
    let mut components = Vec::<ArcComponent>::new();
    let mut means = Vec::new();
    for (choice, arc) in group.alternatives.iter().enumerate() {
        if model.marginalize_arc_alternatives && components.iter().any(|c|
            same_observation(arc, &group.alternatives[c.choice])) { continue; }
        let mean = arc.mean_cost_at_level(conics[arc.eye][arc.boundary].unwrap(),level);
        components.push(ArcComponent {choice,outlier:mean>=MAXIMUM_GROUP_COST,
            cost:arc.weight*mean.min(MAXIMUM_GROUP_COST)+(group.weight-arc.weight)*MAXIMUM_GROUP_COST});
        means.push(mean);
    }
    let representative = (0..components.len()).min_by(|&a,&b|
        components[a].cost.total_cmp(&components[b].cost).then(means[a].total_cmp(&means[b]))).unwrap();
    let chosen = &components[representative];
    let mut result = Activity {choice:chosen.choice,outlier:chosen.outlier,cost:chosen.cost,mixture:None};
    if model.marginalize_arc_alternatives && components.len()>1 {
        let (responsibilities,entropy_penalty,marginal_cost)=mixture(components.iter().map(|a|a.cost).collect::<Vec<_>>());
        result.cost=marginal_cost;
        result.mixture=Some(ArcMixture {components,responsibilities,entropy_penalty,marginal_cost});
    }
    result
}

pub(super) fn select<'a>(model: &'a Problem<'_>, conics: &[[Option<ProjectedCircle>; 3]; 2],
    choices: Vec<usize>) -> Selection<'a>
{
    let mut selected = Selection { levels: vec![1; choices.len()],
        in_family: vec![false; choices.len()],
        group_mixtures:if model.marginalize_arc_alternatives {vec![None;choices.len()]} else {Vec::new()},
        choices, families: Vec::new(), cached_arcs: Vec::new() };
    let mut memberships: [Vec<usize>; 6] = std::array::from_fn(|_| Vec::new());
    for (index, group) in model.groups.iter().enumerate() {
        if group.alternatives.iter().any(|arc| arc.level_sets.iter().flatten().any(|p| p.varies())) {
            let arc = &group.alternatives[0];
            memberships[arc.eye*3+arc.boundary].push(index);
        }
    }
    for (id, groups) in memberships.into_iter().enumerate().filter(|(_, groups)| !groups.is_empty()) {
        let state_count = if groups.iter().any(|&index|model.groups[index].alternatives.iter()
            .any(|a|a.level_sets.iter().flatten().any(|p|p.spatial_displacement_px.is_some()))) {7} else {3};
        let states = (0..state_count).map(|level| groups.iter().map(|&index|
            activity(model,conics,index,level)).collect::<Vec<_>>()).collect::<Vec<_>>();
        let costs = states.iter().map(|state| state.iter().map(|a| a.cost).sum()).collect::<Vec<_>>();
        let (responsibilities, entropy_penalty, marginal_cost) = mixture(&costs);
        // The representative is for per-arc diagnostics only. The objective
        // and every posterior evaluation continue to use all three states.
        let representative = (0..state_count).min_by(|&a,&b| costs[a].total_cmp(&costs[b])
            .then((a != 1).cmp(&(b != 1))).then(a.cmp(&b))).unwrap();
        for (&group, activity) in groups.iter().zip(&states[representative]) {
            selected.choices[group] = activity.choice;
            selected.levels[group] = representative;
            selected.in_family[group] = true;
        }
        selected.families.push(FamilyActivity { eye: id/3, boundary: id%3, groups,
            states, costs, responsibilities, entropy_penalty, marginal_cost, representative });
    }
    if model.marginalize_arc_alternatives {
        for index in 0..model.groups.len() {
            if selected.in_family[index] {continue;}
            let activity=activity(model,conics,index,1);
            selected.choices[index]=activity.choice;
            selected.group_mixtures[index]=activity.mixture;
        }
    }
    selected
}

fn append_component(group: &Group, conics: &[[Option<ProjectedCircle>;3];2],
    choice: usize, level: usize, outlier: bool, q: f64, residuals: &mut Vec<f64>)
{
    let arc=&group.alternatives[choice];
    let conic=conics[arc.eye][arc.boundary].unwrap();
    for (point,&weight) in arc.quadrature.iter().enumerate() {
        residuals.push(if outlier {(q*MAXIMUM_GROUP_COST*arc.weight*weight).sqrt()} else {
            robust_residual(conic.residual_px(arc.point_at_level(point,level))/arc.sigma)*(q*arc.weight*weight).sqrt()
        });
        if arc.outward_normals[point].is_some() {
            residuals.push(if outlier {0.0} else {
                arc.normal_residual_at_level(conic,point,level)*(q*arc.weight*weight).sqrt()
            });
        }
    }
    residuals.push((q*(group.weight-arc.weight)*MAXIMUM_GROUP_COST).sqrt());
}

fn append_arc_mixture(group: &Group, conics: &[[Option<ProjectedCircle>;3];2],
    mixture: &ArcMixture, level: usize, parent_q: f64, residuals: &mut Vec<f64>)
{
    for (component,&q) in mixture.components.iter().zip(&mixture.responsibilities) {
        append_component(group,conics,component.choice,level,component.outlier,parent_q*q,residuals);
    }
    residuals.push((parent_q*mixture.entropy_penalty).sqrt());
}

/// Variational identity: marginal cost = sum(q_s * cost_s) + 2 KL(q || prior).
/// With q/activity frozen this is an EM upper bound tangent at the base point.
/// Fresh evaluations recompute q, selections and outliers before entering here.
pub(super) fn append_residuals(model: &Problem<'_>, conics: &[[Option<ProjectedCircle>; 3]; 2],
    selection: &Selection, residuals: &mut Vec<f64>)
{
    for (index,mixture) in selection.group_mixtures.iter().enumerate() {
        if let Some(mixture)=mixture {
            append_arc_mixture(&model.groups[index],conics,mixture,1,1.0,residuals);
        }
    }
    for family in &selection.families {
        for level in 0..family.states.len() {
            let q = family.responsibilities[level];
            for (&index, activity) in family.groups.iter().zip(&family.states[level]) {
                let group = &model.groups[index];
                if let Some(mixture)=&activity.mixture {
                    append_arc_mixture(group,conics,mixture,level,q,residuals);
                } else {
                    append_component(group,conics,activity.choice,level,activity.outlier,q,residuals);
                }
            }
        }
        residuals.push(family.entropy_penalty.sqrt());
    }
}

pub(super) fn arc_diagnostics(model: &Problem<'_>, selection: &Selection) -> Vec<serde_json::Value> {
    let mut rows=Vec::new();
    let mut add=|index:usize,level:usize,mask_q:f64,mixture:&ArcMixture| {
        let group=&model.groups[index];let source=model.request.eyes[group.alternatives[0].eye].unwrap().exposure;
        rows.push(serde_json::json!({"roi_id":source.roi.0,"sequence":source.sequence.to_string(),
            "timestamp_ns":source.timestamp_ns.to_string(),"group":group.alternatives[0].group,
            "mask_state":level,"mask_state_responsibility":mask_q,
            "prior_weights":vec![1.0/mixture.components.len() as f64;mixture.components.len()],
            "conditional_weights_at_joint_fit":mixture.responsibilities,
            "marginal_cost":mixture.marginal_cost,"components":mixture.components.iter().map(|c| {
                let arc=&group.alternatives[c.choice];
                serde_json::json!({"arc":arc.index,"kind":format!("{:?}",arc.kind),"cost":c.cost,"outlier":c.outlier})
            }).collect::<Vec<_>>(),
            "contract":"Distinct alternatives of one correlated observation group, marginalized at the SAME joint fixation. Uniform engineering prior; duplicate observations do not add states. Representative support rows are diagnostics, not extra votes or calibrated probabilities."}));
    };
    for (index,mixture) in selection.group_mixtures.iter().enumerate() {
        if let Some(mixture)=mixture {add(index,1,1.0,mixture);}
    }
    for family in &selection.families {
        for (level,state) in family.states.iter().enumerate() {
            for (&index,activity) in family.groups.iter().zip(state) {
                if let Some(mixture)=&activity.mixture {add(index,level,family.responsibilities[level],mixture);}
            }
        }
    }
    rows
}

pub(super) fn diagnostics(model: &Problem<'_>, selection: &Selection) -> Vec<serde_json::Value> {
    selection.families.iter().map(|family| {
        let source = model.request.eyes[family.eye].unwrap().exposure;
        let mut diagnostic = serde_json::json!({"roi_id":source.roi.0,"sequence":source.sequence.to_string(),
            "timestamp_ns":source.timestamp_ns.to_string(),"boundary_index":family.boundary,
            "levels":[-1,0,1],"prior_weights":[1.0/3.0,1.0/3.0,1.0/3.0],
            "conditional_weights_at_joint_fit":family.responsibilities,
            "conditional_state_costs":family.costs,"marginal_cost":family.marginal_cost,
            "representative_level":family.representative as i8-1,
            "groups":family.groups.iter().map(|&g|model.groups[g].alternatives[0].group).collect::<Vec<_>>(),
            "contract":"One mask-level state shared by all listed arcs; exact finite-state likelihood at the SAME joint fixation. Uniform engineering weights, not calibrated mask probabilities."});
        if family.states.len() == 7 {
            let object=diagnostic.as_object_mut().unwrap();
            object.remove("levels");object.remove("representative_level");
            diagnostic["states"]=serde_json::json!(["uniform-minus","nominal","uniform-plus",
                "cos2-normal","minus-cos2-normal","sin2-normal","minus-sin2-normal"]);
            diagnostic["prior_weights"]=serde_json::json!(vec![1.0/7.0;7]);
            diagnostic["representative_state"]=serde_json::json!(family.representative);
            diagnostic["contract"]=serde_json::json!("One coherent spatial sensitivity state shared by every listed arc at the SAME joint fixation. Uniform engineering weights; shape fields use measured contour-normal harmonics within each original logit-level envelope. Not calibrated probabilities, extra observations or completed occluded boundaries.");
        }
        diagnostic
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_level_marginal_matches_direct_likelihood_and_variational_identity() {
        for costs in [[0.,0.,0.],[1.,2.,3.],[1000.,1003.,1200.],[0.,9.,36.]] {
            let (q, entropy, marginal) = mixture(costs);
            assert!((q.iter().sum::<f64>()-1.0).abs()<1e-14);
            let variational = q.iter().zip(costs).map(|(q,c)|q*c).sum::<f64>()+entropy;
            assert!((variational-marginal).abs()<1e-11);
            let centered = costs.into_iter().fold(f64::INFINITY,f64::min);
            let direct = centered-2.0*(costs.into_iter().map(|c|(-0.5*(c-centered)).exp()).sum::<f64>()/3.0).ln();
            assert!((direct-marginal).abs()<1e-12);
        }
    }

    #[test]
    fn mask_level_correlated_arcs_cannot_each_choose_a_different_mask() {
        let a=[0.,4.,16.];let b=[16.,4.,0.];
        let coherent=mixture(std::array::from_fn::<_,3,_>(|i|a[i]+b[i])).2;
        let independent=mixture(a).2+mixture(b).2;
        assert!(coherent>independent+3.0);
        assert_eq!(mixture([7.;3]).2,7.0,"duplicate states do not increase information");
        let q=mixture(std::array::from_fn::<_,3,_>(|i|a[i]+b[i])).0;
        assert!(q[1]>q[0] && q[1]>q[2]);
    }

    #[test]
    fn mask_level_frozen_responsibilities_are_a_tangent_upper_bound() {
        let costs=|x:f64|[(x+2.0).powi(2),(x-0.5).powi(2),(x-3.0).powi(2)];
        let at=0.7;let (q,entropy,base)=mixture(costs(at));
        let upper=|x:f64|q.iter().zip(costs(x)).map(|(q,c)|q*c).sum::<f64>()+entropy;
        assert!((upper(at)-base).abs()<1e-12);
        for x in [-4.,-1.,0.,0.7,2.,4.] {assert!(upper(x)+1e-12>=mixture(costs(x)).2);}
        let h=1e-5;
        let gradient=(mixture(costs(at+h)).2-mixture(costs(at-h)).2)/(2.0*h);
        assert!((gradient-(upper(at+h)-upper(at-h))/(2.0*h)).abs()<1e-8);
    }
}
