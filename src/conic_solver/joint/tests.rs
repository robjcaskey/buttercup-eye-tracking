use super::*;
use crate::roi_evidence::{BoundaryArcObservation, ConicObservation, RoiId, SourceClock};
use std::f64::consts::TAU;

fn attribute_arc_mixture(model:&Problem<'_>, index:usize, level:usize, mask_q:f64,
    mixture:&mask_levels::ArcMixture, take:&mut impl FnMut(serde_json::Value,usize)) {
    let group=&model.groups[index];
    for (component,&q) in mixture.components.iter().zip(&mixture.responsibilities) {
        let arc=&group.alternatives[component.choice];
        take(serde_json::json!({"family":"boundary_alternative","eye":arc.eye,
            "kind":format!("{:?}",arc.kind),"group":arc.group,"arc":arc.index,
            "mask_state":level,"responsibility":mask_q*q,"used":!component.outlier,
            "sigma_px":arc.sigma,"arc_weight":arc.weight}),
            arc.points.len()+arc.outward_normals.iter().filter(|n|n.is_some()).count()+1);
    }
    take(serde_json::json!({"family":"boundary_alternative_entropy","eye":group.alternatives[0].eye,
        "group":group.alternatives[0].group,"mask_state":level,"mask_state_responsibility":mask_q}),1);
}

impl Problem<'_> {
    /// Attribute the actual residual vector, instead of approximating its
    /// capped/quadrature-weighted objective from the exported unweighted RMS.
    /// This module and its diagnostic are absent from production builds.
    pub(super) fn factor_cost_diagnostic(&self, p: &Parameters) -> Option<serde_json::Value> {
        let conics = self.conics(p)?;
        let selected = self.select(&conics);
        let residuals = self.residuals(p, &selected)?;
        let mut offset = 0;
        let mut factors = Vec::new();
        let mut take = |mut identity: serde_json::Value, count: usize| {
            let values = &residuals[offset..offset + count];
            identity["squared_cost"] = serde_json::json!(squared_norm(values));
            identity["native_residuals"] = serde_json::json!(values);
            factors.push(identity);
            offset += count;
        };
        for (group_index,(group, &choice)) in self.groups.iter().zip(&selected).enumerate() {
            if selected.in_family[group_index] || selected.has_group_mixture(group_index) { continue; }
            let arc = &group.alternatives[choice];
            take(serde_json::json!({"family":"boundary", "eye":arc.eye,
                "kind":format!("{:?}", arc.kind), "group":arc.group, "arc":arc.index,
                "used":arc.mean_cost(conics[arc.eye][arc.boundary]?) < MAXIMUM_GROUP_COST,
                "sigma_px":arc.sigma, "arc_weight":arc.weight, "group_weight":group.weight}),
                arc.points.len() + arc.outward_normals.iter().filter(|n|n.is_some()).count() + 1);
        }
        for (index,mixture) in selected.group_mixtures.iter().enumerate() {
            if let Some(mixture)=mixture {attribute_arc_mixture(self,index,1,1.0,mixture,&mut take);}
        }
        for family in &selected.families {
            for level in 0..family.states.len() {
                for (&index,activity) in family.groups.iter().zip(&family.states[level]) {
                    if let Some(mixture)=&activity.mixture {
                        attribute_arc_mixture(self,index,level,family.responsibilities[level],mixture,&mut take);
                        continue;
                    }
                    let arc=&self.groups[index].alternatives[activity.choice];
                    let mut identity=serde_json::json!({"family":"mask_boundary_level","eye":family.eye,
                        "kind":format!("{:?}",arc.kind),"group":arc.group,"arc":arc.index,
                        "level":level as i8-1,"responsibility":family.responsibilities[level],
                        "used":!activity.outlier,"sigma_px":arc.sigma,"arc_weight":arc.weight});
                    if family.states.len()==7 {
                        identity.as_object_mut().unwrap().remove("level");
                        identity["state"]=serde_json::json!(level);
                        identity["family"]=serde_json::json!("mask_boundary_spatial_sensitivity");
                    }
                    take(identity,
                        arc.points.len()+arc.outward_normals.iter().filter(|n|n.is_some()).count()+1);
                }
            }
            take(serde_json::json!({"family":"mask_boundary_level_entropy","eye":family.eye,
                "boundary":family.boundary}),1);
        }
        take(serde_json::json!({"family":"prior", "kind":"fixation_axial_distance"}), 1);
        let mut eye_parameters = Vec::new();
        for eye in 0..2 {
            if !self.present[eye] { continue; }
            let prior = self.request.scene.eyes[eye]?;
            let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
            for (kind, count) in [("limbus_center", 3), ("outer_radius", 1),
                ("inner_radius", 1), ("pupil_radius", 1), ("pupil_decentration", 2),
                ("pupil_depth", 1)] {
                take(serde_json::json!({"family":"prior", "eye":eye, "kind":kind}), count);
            }
            if prior.surface_axis_alignment.is_some() {
                take(serde_json::json!({"family":"prior", "eye":eye, "kind":"surface_axis_alignment"}), 2);
            }
            if prior.effective_pivot.is_some() {
                take(serde_json::json!({"family":"prior", "eye":eye, "kind":"effective_pivot"}), 3);
            }
            eye_parameters.push(serde_json::json!({"eye":eye,
                "limbus_center_camera_mm":&p[k..k+3], "radii_mm":&p[k+3..k+6],
                "pupil_decentration_mm":&p[k+6..k+8], "pupil_inward_depth_mm":p[k+8],
                "surface_axis_alignment_radians":&p[k+9..k+11]}));
        }
        if self.present == [true, true] && self.request.scene.interocular_distance_mm.is_some() {
            take(serde_json::json!({"family":"prior", "kind":"interocular_distance"}), 1);
        }
        assert_eq!(offset, residuals.len(), "every native residual must be attributed exactly once");
        let sum: f64 = factors.iter().map(|v|v["squared_cost"].as_f64().unwrap()).sum();
        let native = squared_norm(&residuals);
        assert!((sum-native).abs() <= 1e-10*(1.0+native));
        Some(serde_json::json!({"schema":"buttercup-joint-native-factor-costs-v1",
            "contract":"Test-only exact native local residual costs; add unlocalized_eye_cost for the full association objective. Cost differences are conditional model preferences, not calibrated probabilities or measured accuracy.",
            "native_local_cost":native, "factors":factors, "eye_parameters":eye_parameters}))
    }
}

fn support(nominal: f64, half_width: f64, sigma: f64) -> ScalarSupport {
    ScalarSupport {
        nominal,
        minimum: nominal - half_width,
        maximum: nominal + half_width,
        sigma,
    }
}

fn scene() -> JointScenePrior {
    JointScenePrior {
        camera: PinholeCamera {
            focal_px: [3200.0, 3200.0],
            principal_px: [4000.0, 3000.0],
        },
        eyes: [-32.0, 32.0].map(|x| {
            Some(EyeScenePrior {
                limbus_center: PositionSupport {
                    camera_mm: [x, 0.0, -350.0],
                    sigma_mm: [2.0, 2.0, 20.0],
                    maximum_displacement_mm: [8.0, 8.0, 60.0],
                    transverse_frame: TransversePositionFrame::Cartesian,
                },
                radii_mm: [
                    support(6.0, 1.0, 0.6),
                    support(5.5, 1.0, 0.6),
                    support(2.4, 1.0, 0.6),
                ],
                pupil_inward_depth_mm: support(0.6, 0.2, 0.15),
                pupil_decentration_sigma_mm: 0.05,
                pupil_maximum_decentration_mm: 0.15,
                effective_pivot: None,
                limbus_to_pivot_mm: 9.0,
                surface_axis_alignment: None,
            })
        }),
        target_reference_camera_mm: [0.0, 0.0, -350.0],
        fixation_axial_distance_mm: support(600.0, 350.0, 300.0),
        target_seed_camera_mm: None,
        secondary_target_seed_camera_mm: None,
        maximum_gaze_slope: 1.5,
        interocular_distance_mm: Some(support(64.0, 12.0, 8.0)),
    }
}

fn exposure(eye: usize) -> ExposureKey {
    ExposureKey {
        roi: RoiId(eye as u32 + 1),
        clock: SourceClock {
            domain: 1,
            epoch: 5,
        },
        sequence: 500,
        timestamp_ns: 1_000_000_000,
    }
}

// Independent forward generator: construct actual 3D circle points, then
// perspective-project each point. It does NOT use the solver's conic matrix.
fn ring_points(
    camera: PinholeCamera,
    center: [f64; 3],
    normal: [f64; 3],
    radius: f64,
    origin: [u32; 2],
    begin: f64,
    end: f64,
    count: usize,
) -> Vec<(f64, f64)> {
    let u = normalized3(cross3([0.0, 1.0, 0.0], normal)).unwrap();
    let v = cross3(normal, u);
    (0..count)
        .map(|i| {
            let phase = begin + (end - begin) * i as f64 / (count - 1) as f64;
            let p = add3(
                center,
                add3(
                    scale3(u, radius * phase.cos()),
                    scale3(v, radius * phase.sin()),
                ),
            );
            let projected = camera.project(p).unwrap();
            (
                projected[0] - origin[0] as f64,
                projected[1] - origin[1] as f64,
            )
        })
        .collect()
}

#[derive(Clone)]
struct OwnedArc {
    group: u32,
    kind: BoundaryKind,
    points: Vec<(f64, f64)>,
    band: f64,
}
struct Fixture {
    scene: JointScenePrior,
    target: [f64; 3],
    origins: [[u32; 2]; 2],
    arcs: [Vec<OwnedArc>; 2],
    hints: [Vec<(BoundaryKind, Ellipse)>; 2],
    detail: [f64; 2],
    exposures: [ExposureKey; 2],
}

impl Fixture {
    fn new(target: [f64; 3]) -> Self {
        let scene = scene();
        let origins = [[3500, 2850], [4100, 2850]];
        let mut f = Self {
            scene,
            target,
            origins,
            arcs: [Vec::new(), Vec::new()],
            hints: [Vec::new(), Vec::new()],
            detail: [1.0; 2],
            exposures: [exposure(0), exposure(1)],
        };
        for eye in 0..2 {
            for kind in [
                BoundaryKind::OuterLimbus,
                BoundaryKind::InnerLimbus,
                BoundaryKind::PupillaryBoundary,
            ] {
                f.add_arc(
                    eye,
                    kind,
                    0.0,
                    TAU,
                    16,
                    boundary_index(kind).unwrap() as u32,
                );
            }
        }
        f
    }

    fn add_arc(
        &mut self,
        eye: usize,
        kind: BoundaryKind,
        begin: f64,
        end: f64,
        count: usize,
        group: u32,
    ) {
        let prior = self.scene.eyes[eye].unwrap();
        let center = prior.limbus_center.camera_mm;
        let gaze = normalized3(sub3(self.target, center)).unwrap();
        let u = normalized3(cross3([0.0, 1.0, 0.0], gaze)).unwrap();
        let v = cross3(gaze, u);
        let angles = prior
            .surface_axis_alignment
            .map(|a| a.nominal_radians)
            .unwrap_or([0.0; 2]);
        let normal = normalized3(add3(
            gaze,
            add3(scale3(u, angles[0].tan()), scale3(v, angles[1].tan())),
        ))
        .unwrap();
        let center = if kind == BoundaryKind::PupillaryBoundary {
            sub3(center, scale3(normal, prior.pupil_inward_depth_mm.nominal))
        } else {
            center
        };
        let radius = prior.radii_mm[boundary_index(kind).unwrap()].nominal;
        let ellipse =
            ProjectedCircle::project(self.scene.camera, center, normal, radius, self.origins[eye])
                .unwrap()
                .ellipse()
                .unwrap();
        self.hints[eye].push((kind, ellipse));
        self.arcs[eye].push(OwnedArc {
            group,
            kind,
            points: ring_points(
                self.scene.camera,
                center,
                normal,
                radius,
                self.origins[eye],
                begin,
                end,
                count,
            ),
            band: 0.0,
        });
    }

    fn solve(
        &self,
        enabled: [bool; 2],
        budget: usize,
    ) -> Result<JointConicSolution, JointConicUnavailable> {
        self.solve_with_distribution(enabled, budget, false)
    }

    fn solve_with_distribution(
        &self,
        enabled: [bool; 2],
        budget: usize,
        probabilistic: bool,
    ) -> Result<JointConicSolution, JointConicUnavailable> {
        self.solve_with_integration(
            enabled,
            budget,
            probabilistic.then(posterior::IntegrationConfig::default),
        )
    }

    fn solve_with_integration(
        &self,
        enabled: [bool; 2],
        budget: usize,
        integration: Option<posterior::IntegrationConfig>,
    ) -> Result<JointConicSolution, JointConicUnavailable> {
        self.with_request(enabled,budget,|request| {
            if let Some(config) = integration {
                solve_joint_conic_distribution_diagnostic(request, 1, config)
                    .map(|mut solutions| solutions.remove(0))
            } else {
                solve_joint_conics(request)
            }
        })
    }

    fn with_request<R>(&self, enabled: [bool;2], budget: usize, apply: impl FnOnce(JointConicRequest<'_>)->R) -> R {
        let arcs = self.arcs.each_ref().map(|arcs| {
            arcs.iter()
                .map(|a| BoundaryArcObservation { level_sets_roi: None,
                    evidence_group: a.group,
                    kind: a.kind,
                    points_roi_px: &a.points,
                    normal_band_half_width_px: Some(a.band),
                    detector_score: None,
                    outward_normals_roi: None,
                    localization_sigma_px: None,
                })
                .collect::<Vec<_>>()
        });
        let conics = self.hints.each_ref().map(|hints| {
            hints
                .iter()
                .map(|&(kind, ellipse_roi_px)| ConicObservation {
                    kind,
                    ellipse_roi_px,
                    supporting_arc_indices: &[],
                    residual_px: None,
                })
                .collect::<Vec<_>>()
        });
        let eyes = std::array::from_fn(|eye| {
            enabled[eye].then_some(RoiConicEvidence {
                exposure: self.exposures[eye],
                sensor_origin_px: self.origins[eye],
                dimensions_px: [420, 280],
                arcs: &arcs[eye],
                conics: &conics[eye],
                detail_reliability: Some(self.detail[eye]),
            })
        });
        let request = JointConicRequest {
            eyes,
            scene: &self.scene,
            maximum_hypotheses: budget,
            maximum_refinements: 16,
            maximum_source_skew_ns: 0,
            exposure_uncertainty_ns: 0,
            motion_bound_px_per_second: 0.0,
        };
        apply(request)
    }
}

#[test]
fn integrating_unobserved_inner_radius_preserves_original_nested_model_density() {
    let mut fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.arcs[0].retain(|a|a.kind!=BoundaryKind::InnerLimbus);
    // Keep its initializer deliberately: a conic hint cannot become evidence.
    fixture.with_request([true,true],24,|request| {
        let model=Problem::new(request).unwrap();
        let integrated=posterior::IntegratedInner::new(&model);
        assert_eq!(integrated.indices,vec![TARGET_PARAMETERS+4],"observed left inner boundary must remain sampled");
        for (slope,outer) in [(-0.25,5.0),(0.0,6.0),(0.2,6.9)] {
            let mut p=model.project_step(model.initial).unwrap();p[0]=slope;p[TARGET_PARAMETERS+3]=outer;
            let correction=integrated.condition(&mut p).unwrap();
            let loss=|p:&Parameters| {
                let c=model.conics(p).unwrap();
                squared_norm(&model.residuals(p,&model.select(&c)).unwrap())
            };
            let reference_loss=loss(&p);
            let prior=fixture.scene.eyes[0].unwrap().radii_mm[1];
            let lower=prior.minimum.max(p[TARGET_PARAMETERS+5]);
            let upper=prior.maximum.min(p[TARGET_PARAMETERS+3]);
            let n=8192;
            let direct=(0..n).map(|i| {
                let mut draw=p;draw[TARGET_PARAMETERS+4]=lower+(upper-lower)*(i as f64+0.5)/n as f64;
                (-0.5*(loss(&draw)-reference_loss)).exp()
            }).sum::<f64>()*(upper-lower)/n as f64;
            assert!((correction.exp()/direct-1.0).abs()<1e-7,"nested prior mass changed for outer radius {outer}");
        }
    });
}

#[test]
fn observed_inner_evidence_disables_analytic_elimination_without_changing_output() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    let config=posterior::IntegrationConfig {budget:2048,early_stop:false,..Default::default()};
    let ordinary=fixture.solve_with_integration([true,true],24,Some(config)).unwrap();
    let integrated=fixture.solve_with_integration([true,true],24,Some(posterior::IntegrationConfig {
        marginalize_unobserved_inner:true,..config
    })).unwrap();
    assert_eq!(ordinary.target_camera_mm,integrated.target_camera_mm);
    assert_eq!(ordinary.posterior.unwrap().json(),integrated.posterior.unwrap().json());
}

fn angular_error(solution: &JointConicSolution, fixture: &Fixture, eye: usize) -> f64 {
    let truth = normalized3(sub3(
        fixture.target,
        fixture.scene.eyes[eye].unwrap().limbus_center.camera_mm,
    ))
    .unwrap();
    dot3(solution.eye_normals[eye].unwrap(), truth)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

#[test]
fn local_uncertainty_accounts_for_optical_noise_and_partial_coverage() {
    let mut fixture = Fixture::new([90.0, -140.0, 250.0]);
    let sharp = fixture.solve([true, true], 24).unwrap();
    let sharp_u = sharp.local_uncertainty.as_ref().unwrap();
    assert_eq!(sharp_u.status, "local-conditional", "{sharp_u:?}");
    fixture.detail = [0.05; 2];
    let blurred = fixture.solve([true, true], 24).unwrap();
    let blurred_u = blurred.local_uncertainty.as_ref().unwrap();
    assert_eq!(blurred_u.status, "local-conditional", "{blurred_u:?}");
    for eye in 0..2 {
        assert!(
            blurred_u.worst_axis_sigma_degrees(eye).unwrap()
                > sharp_u.worst_axis_sigma_degrees(eye).unwrap(),
            "blur must widen joint angular uncertainty: {sharp_u:?} -> {blurred_u:?}"
        );
    }
    fixture.detail = [1.0; 2];
    fixture.arcs = [vec![], vec![]];
    fixture.hints = [vec![], vec![]];
    for eye in 0..2 {
        fixture.add_arc(eye, BoundaryKind::OuterLimbus, 0.15, 1.0, 16, 0);
        fixture.add_arc(eye, BoundaryKind::PupillaryBoundary, 0.15, 1.0, 16, 2);
    }
    let partial = fixture.solve([true, true], 24).unwrap();
    let partial_u = partial.local_uncertainty.as_ref().unwrap();
    assert_eq!(partial_u.status, "local-conditional", "{partial_u:?}");
    for eye in 0..2 {
        assert!(
            partial_u.worst_axis_sigma_degrees(eye).unwrap()
                > sharp_u.worst_axis_sigma_degrees(eye).unwrap(),
            "short same-side arcs must not claim all-sided precision: {sharp_u:?} -> {partial_u:?}"
        );
    }
}

#[test]
fn local_uncertainty_is_source_coordinate_invariant_and_keeps_missing_eye_missing() {
    let mut fixture = Fixture::new([90.0, -140.0, 250.0]);
    let base = fixture.solve([true, false], 24).unwrap();
    let before = base.local_uncertainty.as_ref().unwrap();
    assert_eq!(before.status, "local-conditional", "{before:?}");
    assert!(before.gaze_tangent_covariance_rad2[1].is_none());
    fixture.origins[0][0] += 24;
    for arc in &mut fixture.arcs[0] {
        for p in &mut arc.points {
            p.0 -= 24.0;
        }
    }
    for (_, e) in &mut fixture.hints[0] {
        e.center.0 -= 24.0;
    }
    let shifted = fixture.solve([true, false], 24).unwrap();
    let after = shifted.local_uncertainty.as_ref().unwrap();
    assert_eq!(after.status, "local-conditional", "{after:?}");
    for axis in 0..3 {
        let a = before.target_covariance_mm2.unwrap()[axis][axis];
        let b = after.target_covariance_mm2.unwrap()[axis][axis];
        assert!(
            (a - b).abs() < 1e-3 * a.max(1.0),
            "crop translation changed the physical uncertainty: {a} -> {b}"
        );
    }
}

#[test]
fn exact_perspective_circle_and_conic_agree_off_axis_with_native_crop() {
    let camera = scene().camera;
    for normal in [
        [0.3, -0.4, 0.866025403784],
        [-0.2, 0.5, 0.842614977],
        [0.0, 0.0, 1.0],
    ] {
        let normal = normalized3(normal).unwrap();
        for center in [[-35.0, -20.0, -300.0], [40.0, 15.0, -550.0]] {
            let origin = [3600, 2700];
            let conic = ProjectedCircle::project(camera, center, normal, 6.0, origin).unwrap();
            let ellipse = conic.ellipse().unwrap();
            for p in ring_points(camera, center, normal, 6.0, origin, 0.0, TAU, 97) {
                assert!(conic.residual_px(p).abs() < 1.0e-7, "{p:?} {:?}", conic);
                assert!(crate::conic_solver::ellipse_residual(p, ellipse) < 1.0e-7);
            }
        }
    }
}

#[test]
fn camera_facing_is_a_hard_projection_constraint() {
    let camera = scene().camera;
    assert!(
        ProjectedCircle::project(camera, [0.0, 0.0, -300.0], [0.0, 0.0, -1.0], 6.0, [0, 0])
            .is_none()
    );
    assert!(
        ProjectedCircle::project(camera, [0.0, 0.0, 300.0], [0.0, 0.0, 1.0], 6.0, [0, 0]).is_none()
    );
}

#[test]
fn nested_radius_projection_moves_coupled_radii_without_relaxing_frozen_bounds() {
    let q =
        project_nested_radii([4.1, 5.6, 2.4], [4.0, 3.5, 0.6], [8.0, 7.5, 4.5], [1.0; 3]).unwrap();
    assert!((q[0] - 4.85).abs() < 1.0e-10);
    assert_eq!(q[0], q[1]);
    assert!((q[2] - 2.4).abs() < 1.0e-10);
    let q =
        project_nested_radii([4.1, 5.6, 6.0], [4.0, 3.5, 0.6], [4.4, 7.5, 4.5], [1.0; 3]).unwrap();
    assert!(q[0] <= 4.4 && q[1] <= q[0] && q[2] < q[1]);
    assert!(
        project_nested_radii([4.0, 5.6, 2.4], [3.0, 5.0, 0.6], [4.0, 7.5, 4.5], [1.0; 3]).is_none()
    );
}

#[test]
fn a_large_observed_limbus_can_initialize_inside_a_broad_range_prior() {
    let mut fixture = Fixture::new([40.0, -120.0, 260.0]);
    fixture.scene.camera.focal_px = [4000.0; 2];
    fixture.origins[0] = [2560, 960];
    fixture.scene.eyes[0]
        .as_mut()
        .unwrap()
        .limbus_center
        .camera_mm = [-58.0, -89.0, -190.0];
    fixture.arcs = [Vec::new(), Vec::new()];
    fixture.hints = [Vec::new(), Vec::new()];
    for (group, (begin, end)) in [(0.1, 0.7), (1.1, 1.7), (3.2, 3.8), (4.1, 4.7)]
        .into_iter()
        .enumerate()
    {
        fixture.add_arc(0, BoundaryKind::OuterLimbus, begin, end, 12, group as u32);
    }
    let truth = fixture.hints[0][0].1;
    let prior = fixture.scene.eyes[0].as_mut().unwrap();
    prior.limbus_center = PositionSupport {
        camera_mm: scale3(prior.limbus_center.camera_mm, 350.0 / 190.0),
        sigma_mm: [2.5, 2.5, 122.5],
        maximum_displacement_mm: [10.0, 10.0, 245.0],
        transverse_frame: TransversePositionFrame::AtNominalDepth,
    };
    prior.radii_mm = [
        support(6.0, 2.0, 1.0),
        support(5.6, 1.9, 1.0),
        support(2.4, 1.8, 1.2),
    ];
    fixture.scene.target_reference_camera_mm = prior.limbus_center.camera_mm;
    fixture.scene.interocular_distance_mm = None;
    let solution = fixture.solve([true, false], 16).unwrap();
    let ellipse = solution.ellipses_roi_px[0][0].unwrap();
    let error = truth
        .dense_points(64)
        .into_iter()
        .map(|p| crate::conic_solver::ellipse_residual(p, ellipse).powi(2))
        .sum::<f64>();
    assert!(
        (error / 64.0).sqrt() < 1.0,
        "an arbitrary range start must not censor good limbus arcs: {ellipse:?}"
    );
    assert!(solution.arcs.iter().all(|a| a.used));
}

#[test]
fn joint_fixation_recovers_both_vertical_signs_from_mixed_boundary_samples() {
    for y in [-180.0, 180.0] {
        let fixture = Fixture::new([95.0, y, 250.0]);
        let solution = fixture.solve([true, true], 24).unwrap();
        eprintln!(
            "target={:?} fit={:?} errors={:?} cost={} margin={:?}",
            fixture.target,
            solution.target_camera_mm,
            [
                angular_error(&solution, &fixture, 0),
                angular_error(&solution, &fixture, 1)
            ],
            solution.robust_cost,
            solution.alternative_cost_margin
        );
        assert_eq!(solution.contributing_eyes, [true, true]);
        assert!(angular_error(&solution, &fixture, 0) < 1.0);
        assert!(angular_error(&solution, &fixture, 1) < 1.0);
        assert!(norm3(sub3(solution.target_camera_mm, fixture.target)) < 25.0);
        for eye in 0..2 {
            let n = normalized3(sub3(
                solution.target_camera_mm,
                solution.eye_centers_camera_mm[eye].unwrap(),
            ))
            .unwrap();
            assert!(norm3(sub3(n, solution.eye_normals[eye].unwrap())) < 1.0e-12);
        }
    }
}

#[test]
fn off_axis_visible_eye_does_not_hit_an_optical_axis_gaze_limit() {
    for sign in [-1.0, 1.0] {
        let mut fixture = Fixture::new([0.0, -1000.0 * sign, 300.0]);
        fixture.scene.target_reference_camera_mm = [0.0, 130.0 * sign, -300.0];
        // The constructed target is about 1.28 m away, with 1.00 m axial
        // distance along this off-axis observer ray (not 600 mm optical Z).
        fixture.scene.fixation_axial_distance_mm = support(1000.0, 700.0, 500.0);
        fixture.arcs = [Vec::new(), Vec::new()];
        fixture.hints = [Vec::new(), Vec::new()];
        for eye in 0..2 {
            let center = [if eye == 0 { -32.0 } else { 32.0 }, 130.0 * sign, -300.0];
            fixture.scene.eyes[eye]
                .as_mut()
                .unwrap()
                .limbus_center
                .camera_mm = center;
            let sensor = fixture.scene.camera.project(center).unwrap();
            fixture.origins[eye] = [
                (sensor[0] - 210.0).round() as u32,
                (sensor[1] - 140.0).round() as u32,
            ];
            for kind in [
                BoundaryKind::OuterLimbus,
                BoundaryKind::InnerLimbus,
                BoundaryKind::PupillaryBoundary,
            ] {
                fixture.add_arc(
                    eye,
                    kind,
                    0.0,
                    TAU,
                    16,
                    boundary_index(kind).unwrap() as u32,
                );
            }
            let normal = normalized3(sub3(fixture.target, center)).unwrap();
            assert!(
                normal[1].abs() / normal[2] > fixture.scene.maximum_gaze_slope,
                "the historical optical-axis bound must exclude this construction"
            );
            assert!(
                dot3(normal, normalized3(scale3(center, -1.0)).unwrap()) > 0.7,
                "this is a visible convex eye, not a forbidden back-facing disk"
            );
        }
        for enabled in [[true, true], [true, false], [false, true]] {
            let solution = fixture.solve(enabled, 24).unwrap();
            for eye in 0..2 {
                if enabled[eye] {
                    assert!(angular_error(&solution,&fixture,eye)<0.5,
                    "sign={sign} eye={eye} enabled={enabled:?} target={:?} truth={:?} error={} cost={}",solution.target_camera_mm,
                    fixture.target,angular_error(&solution,&fixture,eye),solution.robust_cost);
                    assert!(solution
                        .arcs
                        .iter()
                        .filter(|a| a.exposure.roi == RoiId(eye as u32 + 1))
                        .all(|a| a.used && a.rms_px < 0.2));
                }
            }
        }
    }
}

#[test]
fn viewpoint_ray_coordinates_round_trip_at_the_specified_axial_distance() {
    for origin in [
        [0.0, 0.0, -300.0],
        [-110.0, 140.0, -260.0],
        [110.0, -140.0, -260.0],
    ] {
        let chart = ViewpointRayChart::new(origin).unwrap();
        for x in [-1.2, 0.0, 1.2] {
            for y in [-1.2, 0.0, 1.2] {
                let coordinates = [x, y, 600.0_f64.ln()];
                let Some(target) = chart.target(coordinates) else {
                    continue;
                };
                assert!((dot3(sub3(target, origin), chart.toward_camera) - 600.0).abs() < 1.0e-9);
                assert!(
                    (norm3(sub3(target, origin)) - 600.0 * (1.0 + x * x + y * y).sqrt()).abs()
                        < 1.0e-9
                );
                let recovered = chart.coordinates(target).unwrap();
                assert!(coordinates
                    .into_iter()
                    .zip(recovered)
                    .all(|(a, b)| (a - b).abs() < 1.0e-12));
                assert!(dot3(sub3(target, origin), chart.toward_camera) > 0.0);
                if origin[0] == 0.0 && origin[1] == 0.0 {
                    assert!(
                        norm3(sub3(target, [600.0 * x, 600.0 * y, origin[2] + 600.0])) < 1.0e-9,
                        "on-axis coordinates retain the historical exact parameterization"
                    );
                }
            }
        }
    }
}

#[test]
fn a_near_optical_horizon_does_not_turn_bounded_viewpoint_depth_into_infinite_range() {
    let chart = ViewpointRayChart::new([270.0, 160.0, -350.0]).unwrap();
    for optical_forward in [1.0e-3, 1.0e-4, 1.0e-5] {
        let direction = normalized3([-0.66, -0.75, optical_forward]).unwrap();
        let forward = dot3(direction, chart.toward_camera);
        assert!(forward > 0.5, "a camera-facing ray, not a hidden backside");
        let coordinates = [
            dot3(direction, chart.right) / forward,
            dot3(direction, chart.down) / forward,
            600.0_f64.ln(),
        ];
        assert!(coordinates[..2].iter().all(|v| v.abs() < 1.5));
        let target = chart.target(coordinates).unwrap();
        let offset = sub3(target, chart.origin_camera_mm);
        assert!((dot3(offset, chart.toward_camera) - 600.0).abs() < 1.0e-9);
        assert!(
            norm3(offset) < 600.0 * (1.0 + 2.0 * 1.5_f64.powi(2)).sqrt(),
            "the same finite slope/depth envelope must bound metric range near the optical horizon"
        );
        assert!(
            dot3(normalized3(offset).unwrap(), direction) > 1.0 - 1.0e-12,
            "bounding the range must not silently reverse or clip the ray direction"
        );
    }
}

#[test]
fn viewpoint_ray_chart_rejects_invalid_or_backward_targets_instead_of_flipping_them() {
    assert!(ViewpointRayChart::new([0.0, 0.0, 0.0]).is_none());
    assert!(ViewpointRayChart::new([0.0, 0.0, 300.0]).is_none());
    assert!(ViewpointRayChart::new([f64::NAN, 0.0, -300.0]).is_none());
    let chart = ViewpointRayChart::new([0.0, 130.0, -300.0]).unwrap();
    assert!(
        chart.target([0.0, -10.0, 600.0_f64.ln()]).is_none(),
        "dividing by a negative camera-Z component would manufacture its antipode"
    );
    assert!(chart.coordinates([0.0, 130.0, -400.0]).is_none());
    assert!(chart.coordinates([0.0, 1.0e6, 1.0]).is_none());
    assert!(chart.target([f64::NAN, 0.0, 1.0]).is_none());
}

#[test]
fn correlated_alternatives_choose_compatible_segments_without_double_voting() {
    let mut fixture = Fixture::new([70.0, -150.0, 250.0]);
    let baseline = fixture.solve([true, true], 24).unwrap();
    let original = &fixture.arcs[1][2];
    let bad = OwnedArc {
        group: original.group,
        kind: original.kind,
        points: original
            .points
            .iter()
            .map(|&(x, y)| (x + 17.0, y + 32.0))
            .collect(),
        band: 0.0,
    };
    fixture.arcs[1].insert(0, bad);
    let solution = fixture.solve([true, true], 24).unwrap();
    assert!(norm3(sub3(solution.target_camera_mm, baseline.target_camera_mm)) < 1.0e-8);
    assert_eq!(solution.arcs.len(), 6);
    assert!(solution
        .arcs
        .iter()
        .filter(|a| a.exposure.roi == RoiId(2))
        .all(|a| a.arc_index != 0));
}

#[test]
fn clocks_missing_eyes_and_budgets_have_explicit_semantics() {
    let mut fixture = Fixture::new([50.0, -150.0, 250.0]);
    assert!(matches!(
        fixture.solve([false, false], 24),
        Err(JointConicUnavailable::NoBoundaryEvidence)
    ));
    assert!(matches!(
        fixture.solve([true, true], 0),
        Err(JointConicUnavailable::InvalidRequest)
    ));
    fixture.exposures[1].clock.epoch += 1;
    assert!(matches!(
        fixture.solve([true, true], 24),
        Err(JointConicUnavailable::IncompatibleClocks)
    ));
    let mono = fixture.solve([true, false], 4).unwrap();
    assert_eq!(mono.contributing_eyes, [true, false]);
    assert!(mono.eye_normals[1].is_none());
    assert!(mono.hypotheses_evaluated <= 4);
    assert!(mono.refinement_steps <= 4 * 16 * 4);
    fixture.exposures[1].clock = fixture.exposures[0].clock;
    fixture.exposures[1].timestamp_ns += 1;
    assert!(matches!(
        fixture.solve([true, true], 24),
        Err(JointConicUnavailable::ExcessiveSourceSkew)
    ));
}

#[test]
fn roi_reframe_is_coordinate_change_not_an_eye_motion_measurement() {
    let mut fixture = Fixture::new([80.0, -130.0, 250.0]);
    let before = fixture.solve([true, true], 24).unwrap();
    for eye in 0..2 {
        let shift = if eye == 0 { [48, 26] } else { [18, 64] };
        for axis in 0..2 {
            fixture.origins[eye][axis] += shift[axis];
        }
        for arc in &mut fixture.arcs[eye] {
            for p in &mut arc.points {
                p.0 -= shift[0] as f64;
                p.1 -= shift[1] as f64;
            }
        }
        for (_, ellipse) in &mut fixture.hints[eye] {
            ellipse.center.0 -= shift[0] as f64;
            ellipse.center.1 -= shift[1] as f64;
        }
    }
    let after = fixture.solve([true, true], 24).unwrap();
    assert!(
        norm3(sub3(before.target_camera_mm, after.target_camera_mm)) < 0.01,
        "before={:?} after={:?}",
        before.target_camera_mm,
        after.target_camera_mm
    );
}

#[test]
fn complementary_partial_arcs_solve_one_target_not_an_average_of_monocular_targets() {
    let mut fixture = Fixture::new([80.0, -170.0, 250.0]);
    fixture.arcs = [Vec::new(), Vec::new()];
    fixture.hints = [Vec::new(), Vec::new()];
    // One side has only a short outer-limbus fragment and a weak pupil arc;
    // the other has a complementary inner limbus and a sharper pupil arc.
    fixture.add_arc(0, BoundaryKind::OuterLimbus, -0.5, 0.5, 11, 0);
    fixture.add_arc(0, BoundaryKind::PupillaryBoundary, 2.4, 3.8, 11, 1);
    fixture.arcs[0][1].band = 5.0;
    fixture.add_arc(1, BoundaryKind::InnerLimbus, 1.2, 2.4, 11, 0);
    fixture.add_arc(1, BoundaryKind::PupillaryBoundary, 3.0, 5.0, 11, 1);
    for (i, p) in fixture.arcs[0][0].points.iter_mut().enumerate() {
        p.1 += 1.4 * (i as f64 * 0.6).cos();
    }
    for (i, p) in fixture.arcs[1][0].points.iter_mut().enumerate() {
        p.0 += 0.4 * (i as f64 * 0.7).sin();
    }
    let left = fixture.solve([true, false], 24).unwrap();
    let right = fixture.solve([false, true], 24).unwrap();
    let joint = fixture.solve([true, true], 24).unwrap();
    let prohibited_average = scale3(add3(left.target_camera_mm, right.target_camera_mm), 0.5);
    let joint_error = norm3(sub3(joint.target_camera_mm, fixture.target));
    let average_error = norm3(sub3(prohibited_average, fixture.target));
    eprintln!(
        "partial arcs mono={:?}/{:?} joint={:?} avg={:?} errors={joint_error}/{average_error}",
        left.target_camera_mm, right.target_camera_mm, joint.target_camera_mm, prohibited_average
    );
    assert_eq!(joint.contributing_eyes, [true, true]);
    assert!(norm3(sub3(joint.target_camera_mm, prohibited_average)) > 1.0);
    assert!(joint_error < average_error);
}

#[test]
fn defocus_weakens_conflicting_pupil_localization_without_erasing_the_other_eye() {
    let mut fixture = Fixture::new([90.0, -140.0, 250.0]);
    // Shift the second eye's pupil section, not its whole observed iris.
    // This simulates a bad edge estimate; it isn't treated as a new gaze point.
    for p in &mut fixture.arcs[1][2].points {
        p.1 += 2.0;
    }
    let sharp = fixture.solve([true, true], 24).unwrap();
    fixture.detail[1] = 0.05;
    let blurred = fixture.solve([true, true], 24).unwrap();
    eprintln!(
        "defocus errors {} -> {}",
        angular_error(&sharp, &fixture, 0),
        angular_error(&blurred, &fixture, 0)
    );
    assert!(angular_error(&blurred, &fixture, 0) < angular_error(&sharp, &fixture, 0));
    assert!(blurred
        .arcs
        .iter()
        .filter(|a| a.exposure.roi == RoiId(2))
        .all(|a| a.sigma_px > 3.0));
    assert!(blurred.contributing_eyes[0]);
}

#[test]
fn strongest_signed_pupil_section_resolves_a_weak_opposing_cue_in_the_other_roi() {
    let mut fixture = Fixture::new([0.0, -160.0, 250.0]);
    let opposite = Fixture::new([0.0, 160.0, 250.0]);
    fixture.arcs[0][2].points = opposite.arcs[0][2].points.clone();
    fixture.arcs[0][2].band = 7.0;
    let weak = fixture.solve([true, false], 24).unwrap();
    let joint = fixture.solve([true, true], 24).unwrap();
    eprintln!(
        "opposing pupil weak {:?} joint {:?}",
        weak.target_camera_mm, joint.target_camera_mm
    );
    assert!(weak.target_camera_mm[1] > 0.0);
    assert!(joint.target_camera_mm[1] < 0.0);
    assert!(angular_error(&joint, &fixture, 0) < 2.0);
    assert_eq!(joint.contributing_eyes, [true, true]);
}

#[test]
fn uncertain_range_preserves_off_axis_viewing_ray_instead_of_pinning_metric_xy() {
    let mut prior = PositionSupport {
        camera_mm: [40.0, -80.0, -300.0],
        sigma_mm: [2.0, 2.0, 80.0],
        maximum_displacement_mm: [8.0, 8.0, 160.0],
        transverse_frame: TransversePositionFrame::AtNominalDepth,
    };
    let moved = [60.0, -120.0, -450.0];
    assert_eq!(prior.displacement(moved), [0.0, 0.0, -150.0]);
    prior.transverse_frame = TransversePositionFrame::Cartesian;
    assert_eq!(prior.displacement(moved), [20.0, -40.0, -150.0]);
}

#[test]
fn shared_fixation_ray_and_surface_normal_remain_distinct_with_explicit_axis_alignment() {
    let mut fixture = Fixture::new([70.0, -120.0, 250.0]);
    fixture.arcs = [Vec::new(), Vec::new()];
    fixture.hints = [Vec::new(), Vec::new()];
    for eye in 0..2 {
        fixture.scene.eyes[eye]
            .as_mut()
            .unwrap()
            .surface_axis_alignment = Some(SurfaceAxisAlignment {
            nominal_radians: if eye == 0 {
                [-0.04, 0.02]
            } else {
                [0.04, -0.01]
            },
            sigma_radians: [0.01; 2],
            maximum_deviation_radians: [0.0; 2],
        });
        for kind in [
            BoundaryKind::OuterLimbus,
            BoundaryKind::InnerLimbus,
            BoundaryKind::PupillaryBoundary,
        ] {
            fixture.add_arc(
                eye,
                kind,
                0.0,
                TAU,
                16,
                boundary_index(kind).unwrap() as u32,
            );
        }
    }
    let solution = fixture.solve([true, true], 24).unwrap();
    assert!(norm3(sub3(solution.target_camera_mm, fixture.target)) < 2.0);
    for eye in 0..2 {
        let expected = normalized3(sub3(
            fixture.target,
            fixture.scene.eyes[eye].unwrap().limbus_center.camera_mm,
        ))
        .unwrap();
        let gaze = solution.eye_gaze_directions[eye].unwrap();
        let surface = solution.eye_normals[eye].unwrap();
        assert!(norm3(sub3(expected, gaze)) < 0.002);
        assert!(norm3(sub3(surface, gaze)) > 0.025);
    }
}

#[test]
fn incompatible_arc_groups_cannot_drag_a_well_supported_partner() {
    let mut fixture = Fixture::new([70.0, -120.0, 250.0]);
    for arc in &mut fixture.arcs[0] {
        for p in &mut arc.points {
            p.0 += 65.0;
            p.1 += 50.0;
        }
    }
    let solution = fixture.solve([true, true], 24).unwrap();
    assert_eq!(solution.contributing_eyes, [false, true]);
    assert!(solution
        .arcs
        .iter()
        .filter(|a| a.exposure.roi == RoiId(1))
        .all(|a| !a.used));
    assert!(angular_error(&solution, &fixture, 1) < 0.1);
}

#[test]
fn perspective_circle_normal_decomposition_recovers_off_axis_planes_and_both_mirrors() {
    for focal in [[3200.0, 3200.0], [3200.0, 4100.0]] {
        let camera = PinholeCamera {
            focal_px: focal,
            principal_px: [4000.0, 3000.0],
        };
        for center in [
            [0.0, 0.0, -300.0],
            [90.0, -40.0, -218.0],
            [-70.0, 80.0, -410.0],
        ] {
            for normal in [[0.2, 0.45, 0.87], [-0.3, -0.55, 0.78], [0.0, 0.0, 1.0]] {
                let normal = normalized3(normal).unwrap();
                let origin = [3000, 2300];
                let ellipse = ProjectedCircle::project(camera, center, normal, 6.0, origin)
                    .unwrap()
                    .ellipse()
                    .unwrap();
                let normals = circle_normal_hypotheses(camera, ellipse, origin).unwrap();
                let error = normals
                    .into_iter()
                    .map(|n| norm3(sub3(n, normal)))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    error < 1.0e-7,
                    "center={center:?} normal={normal:?} solutions={normals:?} error={error}"
                );
                assert!(normals.into_iter().all(|n| n[2] > 0.0));
                let poses = circle_pose_hypotheses(camera, ellipse, origin).unwrap();
                let closest = poses
                    .into_iter()
                    .min_by(|a, b| {
                        norm3(sub3(a.normal, normal)).total_cmp(&norm3(sub3(b.normal, normal)))
                    })
                    .unwrap();
                assert!(norm3(sub3(scale3(closest.center_per_radius, 6.0), center)) < 1.0e-6);
                for pose in poses {
                    let regenerated = ProjectedCircle::project(
                        camera,
                        scale3(pose.center_per_radius, 6.0),
                        pose.normal,
                        6.0,
                        origin,
                    )
                    .unwrap();
                    assert!(ellipse
                        .dense_points(32)
                        .into_iter()
                        .all(|p| regenerated.residual_px(p).abs() < 1.0e-7));
                }
            }
        }
    }
}

#[test]
fn polyline_information_is_geometric_not_the_number_of_fragments_or_points() {
    let points = [(0.0, 0.0), (10.0, 0.0), (30.0, 0.0), (30.0, 40.0)];
    let (weights, length) = polyline_quadrature(&points).unwrap();
    assert_eq!(length, 70.0);
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    let repeated = points
        .iter()
        .flat_map(|p| std::iter::repeat_n(*p, 4))
        .collect::<Vec<_>>();
    assert_eq!(polyline_quadrature(&repeated).unwrap().1, length);
    let split = polyline_quadrature(&points[..=2]).unwrap().1
        + polyline_quadrature(&points[2..]).unwrap().1;
    assert_eq!(split, length);
    let resampled = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (20.0, 0.0),
        (30.0, 0.0),
        (30.0, 20.0),
        (30.0, 40.0),
    ];
    assert_eq!(polyline_quadrature(&resampled).unwrap().1, length);
    assert!(polyline_quadrature(&[(2.0, 3.0); 10]).is_none());
}

#[test]
fn matching_points_do_not_override_opposite_measured_boundary_directions() {
    let points = [-0.2f64, 0.0, 0.2]
        .map(|t| (10.0 * t.cos(), 10.0 * t.sin()))
        .to_vec();
    let (quadrature, length_px) = polyline_quadrature(&points).unwrap();
    let normals = points
        .iter()
        .map(|&(x, y)| {
            Some(BoundaryNormalObservation {
                unit_outward_roi: [x / 10.0, y / 10.0],
                angular_sigma_radians: 0.2,
            })
        })
        .collect();
    let mut arc = SparseArc {
        eye: 0,
        index: 0,
        kind: BoundaryKind::OuterLimbus,
        boundary: 0,
        group: 0,
        level_sets: vec![None; points.len()],
        points,
        outward_normals: normals,
        sigma: 1.0,
        quadrature,
        length_px,
        weight: 1.0,
    };
    let conic = ProjectedCircle([1.0, 0.0, 1.0, 0.0, 0.0, -100.0]);
    assert!(arc.mean_cost(conic) < 1.0e-20);
    for normal in arc.outward_normals.iter_mut().flatten() {
        normal.unit_outward_roi = normal.unit_outward_roi.map(|v| -v);
    }
    assert!(arc.mean_cost(conic)>MAXIMUM_GROUP_COST,
        "identical point positions do not make an inward or opposite-polarity boundary a compatible outer rim");
    for normal in arc.outward_normals.iter_mut().flatten() {
        normal.angular_sigma_radians = 2.0;
    }
    assert!(
        arc.mean_cost(conic) < MAXIMUM_GROUP_COST,
        "uncertain direction must have weaker influence"
    );
    arc.outward_normals.fill(None);
    assert!(
        arc.mean_cost(conic) < 1.0e-20,
        "missing direction is not fabricated from the candidate conic"
    );
}

#[test]
fn uncertain_contour_directions_are_a_compatibility_band_not_a_second_precise_fit() {
    let points = [-0.2f64, 0.0, 0.2]
        .map(|t| (10.0 * t.cos(), 10.0 * t.sin()))
        .to_vec();
    let (quadrature, length_px) = polyline_quadrature(&points).unwrap();
    let (s, c) = 0.3f64.sin_cos();
    let normals = points
        .iter()
        .map(|&(x, y)| {
            Some(BoundaryNormalObservation {
                unit_outward_roi: [(c * x - s * y) / 10.0, (s * x + c * y) / 10.0],
                angular_sigma_radians: 0.2,
            })
        })
        .collect();
    let arc = SparseArc {
        eye: 0,
        index: 0,
        kind: BoundaryKind::OuterLimbus,
        boundary: 0,
        group: 0,
        level_sets: vec![None; points.len()],
        points,
        outward_normals: normals,
        sigma: 1.0,
        quadrature,
        length_px,
        weight: 1.0,
    };
    assert!(arc.mean_cost(ProjectedCircle([1.0,0.0,1.0,0.0,0.0,-100.0]))<1.0e-20,
        "contour position and tangent share pixels: do not chase a noisy direction within its two-sigma engineering allowance");
}

#[test]
fn numerical_linearization_cannot_turn_a_capped_arc_into_a_force() {
    let scene = scene();
    let center = scene.eyes[0].unwrap().limbus_center.camera_mm;
    let target = add3(
        scene.target_reference_camera_mm,
        [0.0, 0.0, scene.fixation_axial_distance_mm.nominal],
    );
    let points = ring_points(
        scene.camera,
        center,
        normalized3(sub3(target, center)).unwrap(),
        6.0,
        [3500, 2850],
        0.1,
        0.5,
        12,
    );
    let arcs = [BoundaryArcObservation { level_sets_roi: None,
        evidence_group: 0,
        kind: BoundaryKind::OuterLimbus,
        points_roi_px: &points,
        outward_normals_roi: None,
        localization_sigma_px: None,
        normal_band_half_width_px: Some(0.0),
        detector_score: None,
    }];
    let evidence = RoiConicEvidence {
        exposure: exposure(0),
        sensor_origin_px: [3500, 2850],
        dimensions_px: [420, 280],
        arcs: &arcs,
        conics: &[],
        detail_reliability: Some(1.0),
    };
    let mut problem = Problem::new(JointConicRequest {
        eyes: [Some(evidence), None],
        scene: &scene,
        maximum_hypotheses: 16,
        maximum_refinements: 12,
        maximum_source_skew_ns: 0,
        exposure_uncertainty_ns: 0,
        motion_bound_px_per_second: 0.0,
    })
    .unwrap();
    let base = problem.initial;
    let conic = problem.conics(&base).unwrap()[0][0].unwrap();
    let [a, b, c, d, e, _] = conic.0;
    let (s, co) = 0.6f64.sin_cos();
    // The arc sits just outside the existing capped compatibility cost. A
    // finite-difference perturbation crosses that gate while its position and
    // direction residual components are entirely different vectors.
    problem.groups[0].alternatives[0].outward_normals = points
        .iter()
        .map(|&(x, y)| {
            let u = 2.0 * a * x + b * y + d;
            let v = b * x + 2.0 * c * y + e;
            let length = u.hypot(v);
            Some(BoundaryNormalObservation {
                unit_outward_roi: [(co * u - s * v) / length, (s * u + co * v) / length],
                angular_sigma_radians: 0.6 / (BOUNDARY_DIRECTION_ALLOWANCE_SIGMAS + 3.0 + 1.0e-8),
            })
        })
        .collect();
    let arc = &problem.groups[0].alternatives[0];
    assert!(arc.mean_cost(conic) > MAXIMUM_GROUP_COST);
    let trial = (0..PARAMETERS)
        .flat_map(|i| [-1.0, 1.0].map(move |direction| (i, direction)))
        .find_map(|(i, direction)| {
            let mut q = base;
            q[i] += direction * 1.0e-4 * problem.scales[i];
            if q[i] < problem.lower[i] || q[i] > problem.upper[i] {
                return None;
            }
            let trial_conic = problem.conics(&q)?[0][0]?;
            (arc.mean_cost(trial_conic) < MAXIMUM_GROUP_COST).then_some(q)
        })
        .expect("exercise a real finite-difference crossing, not merely a far rejected arc");
    let selected = problem.select(&problem.conics(&base).unwrap());
    let rejected = problem.rejected_groups(&problem.conics(&base).unwrap(), &selected);
    assert_eq!(rejected, vec![true]);
    let before = problem.residuals(&base, &selected).unwrap();
    let actual = problem.residuals(&trial, &selected).unwrap();
    let linearized = problem
        .residuals_with_rejection(&trial, &selected, Some(&rejected))
        .unwrap();
    let sample_terms = points.len() * 2;
    assert!(
        before[..sample_terms]
            .iter()
            .zip(&actual)
            .any(|(a, b)| (a - b).abs() > 0.1),
        "a real objective evaluation must still reconsider previously rejected evidence"
    );
    assert_eq!(&before[..sample_terms],&linearized[..sample_terms],
        "a rejected arc has constant cost and zero force during this local derivative; a new trial rechecks admission separately");
    let readmitted = problem.rejected_groups(&problem.conics(&trial).unwrap(), &selected);
    assert_eq!(
        readmitted,
        vec![false],
        "this is not persistent exclusion memory"
    );
    let reverse = problem
        .residuals_with_rejection(&base, &selected, Some(&readmitted))
        .unwrap();
    assert!(reverse[..sample_terms].iter().skip(1).step_by(2).all(|&r|r.abs()>0.1),
        "an admitted arc retains its actual direction derivatives even if a numerical step crosses the cap");
}

#[test]
fn capped_negative_position_residuals_cannot_pull_ordinary_live_evidence() {
    let mut scene = scene();
    scene.eyes[0].as_mut().unwrap().limbus_center.camera_mm[0] = 0.0;
    let center = scene.eyes[0].unwrap().limbus_center.camera_mm;
    let image_radius = 6.0 * scene.camera.focal_px[0] / (-center[2]);
    let gap = 3.0 * 0.75 * (1.0 + 1.0e-8);
    // For an observed radius r inside circle R, Sampson residual is
    // (r²-R²)/(2r). Choose r so the existing rejection cost is just exceeded.
    let observed_radius = (image_radius.hypot(gap) - gap) * (-center[2]) / scene.camera.focal_px[0];
    let points = ring_points(
        scene.camera,
        center,
        [0.0, 0.0, 1.0],
        observed_radius,
        [3800, 2850],
        0.1,
        0.5,
        12,
    );
    let arcs = [BoundaryArcObservation { level_sets_roi: None,
        evidence_group: 0,
        kind: BoundaryKind::OuterLimbus,
        points_roi_px: &points,
        outward_normals_roi: None,
        localization_sigma_px: None,
        normal_band_half_width_px: Some(0.0),
        detector_score: None,
    }];
    let evidence = RoiConicEvidence {
        exposure: exposure(0),
        sensor_origin_px: [3800, 2850],
        dimensions_px: [420, 280],
        arcs: &arcs,
        conics: &[],
        detail_reliability: Some(1.0),
    };
    let problem = Problem::new(JointConicRequest {
        eyes: [Some(evidence), None],
        scene: &scene,
        maximum_hypotheses: 16,
        maximum_refinements: 12,
        maximum_source_skew_ns: 0,
        exposure_uncertainty_ns: 0,
        motion_bound_px_per_second: 0.0,
    })
    .unwrap();
    let base = problem.initial;
    let selected = problem.select(&problem.conics(&base).unwrap());
    let rejected = problem.rejected_groups(&problem.conics(&base).unwrap(), &selected);
    assert_eq!(rejected, vec![true]);
    let mut trial = base;
    trial[6] -= 1.0e-4 * problem.scales[6];
    assert_eq!(
        problem.rejected_groups(&problem.conics(&trial).unwrap(), &selected),
        vec![false]
    );
    let before = problem.residuals(&base, &selected).unwrap();
    let actual = problem.residuals(&trial, &selected).unwrap();
    assert!(before[..points.len()].iter().all(|&r| r > 0.0));
    assert!(actual[..points.len()].iter().all(|&r|r<0.0),
        "the unfixed derivative crosses a residual sign discontinuity even without measured directions");
    let linearized = problem
        .residuals_with_rejection(&trial, &selected, Some(&rejected))
        .unwrap();
    assert_eq!(&before[..points.len()], &linearized[..points.len()]);
}

#[test]
fn joint_selection_uses_measured_direction_not_just_equal_point_alternatives() {
    use crate::outline_conic_segments::sparse_evidence::{
        OwnedBoundaryArc, OwnedConicHint, OwnedRoiEvidence,
    };
    let fixture = Fixture::new([70.0, -130.0, 250.0]);
    let mut packets = std::array::from_fn::<_, 2, _>(|eye| OwnedRoiEvidence {
        exposure: fixture.exposures[eye],
        sensor_origin_px: fixture.origins[eye],
        dimensions_px: [420, 280],
        detail_reliability: Some(1.0),
        arcs: fixture.arcs[eye]
            .iter()
            .map(|a| OwnedBoundaryArc { level_sets_roi: None,
                evidence_group: a.group,
                kind: a.kind,
                points_roi_px: a.points.clone(),
                outward_normals_roi: None,
                localization_sigma_px: None,
                normal_band_half_width_px: 0.0,
                detector_score: None,
            })
            .collect(),
        conics: fixture.hints[eye]
            .iter()
            .map(|&(kind, ellipse_roi_px)| OwnedConicHint {
                kind,
                ellipse_roi_px,
                supporting_arc_indices: vec![],
            })
            .collect(),
    });
    let e = fixture.hints[0][0].1;
    let (s, c) = e.angle.sin_cos();
    packets[0].arcs[0].outward_normals_roi = Some(
        packets[0].arcs[0]
            .points_roi_px
            .iter()
            .map(|&(x, y)| {
                let u = (c * (x - e.center.0) + s * (y - e.center.1)) / e.major_radius.powi(2);
                let v = (-s * (x - e.center.0) + c * (y - e.center.1)) / e.minor_radius.powi(2);
                let magnitude = u.hypot(v);
                Some(BoundaryNormalObservation {
                    unit_outward_roi: [(c * u - s * v) / magnitude, (s * u + c * v) / magnitude],
                    angular_sigma_radians: 0.2,
                })
            })
            .collect(),
    );
    let mut wrong = packets[0].arcs[0].clone();
    for n in wrong
        .outward_normals_roi
        .as_mut()
        .unwrap()
        .iter_mut()
        .flatten()
    {
        n.unit_outward_roi = n.unit_outward_roi.map(|v| -v);
    }
    packets[0].arcs.insert(0, wrong);
    let prepared = packets.each_ref().map(OwnedRoiEvidence::prepare);
    let result = solve_joint_conics(JointConicRequest {
        eyes: prepared.each_ref().map(|p| Some(p.evidence())),
        scene: &fixture.scene,
        maximum_hypotheses: 16,
        maximum_refinements: 16,
        maximum_source_skew_ns: 0,
        exposure_uncertainty_ns: 0,
        motion_bound_px_per_second: 0.0,
    })
    .unwrap();
    let selected = result
        .arcs
        .iter()
        .find(|a| a.exposure.roi == RoiId(1) && a.evidence_group == 0)
        .unwrap();
    assert_eq!(
        selected.arc_index, 1,
        "the first alternative has the same coordinates but contradicts measured outward direction"
    );
    assert!(selected.used);
    assert_eq!(result.contributing_eyes, [true, true]);
    for eye in 0..2 {
        assert!(angular_error(&result, &fixture, eye) < 1.0);
    }
}

#[test]
fn malformed_or_misaligned_boundary_directions_cannot_be_silently_ignored() {
    let fixture = Fixture::new([70.0, -130.0, 250.0]);
    let points = &fixture.arcs[0][0].points;
    let good = BoundaryNormalObservation {
        unit_outward_roi: [1.0, 0.0],
        angular_sigma_radians: 0.2,
    };
    for normals in [
        vec![Some(good); points.len() - 1],
        vec![
            Some(BoundaryNormalObservation {
                unit_outward_roi: [0.0, 0.0],
                ..good
            });
            points.len()
        ],
        vec![
            Some(BoundaryNormalObservation {
                angular_sigma_radians: 0.0,
                ..good
            });
            points.len()
        ],
        vec![
            Some(BoundaryNormalObservation {
                angular_sigma_radians: f64::NAN,
                ..good
            });
            points.len()
        ],
    ] {
        let arcs = [BoundaryArcObservation { level_sets_roi: None,
            evidence_group: 0,
            kind: BoundaryKind::OuterLimbus,
            points_roi_px: points,
            outward_normals_roi: Some(&normals),
            localization_sigma_px: None,
            normal_band_half_width_px: None,
            detector_score: None,
        }];
        let eye = RoiConicEvidence {
            exposure: fixture.exposures[0],
            sensor_origin_px: fixture.origins[0],
            dimensions_px: [420, 280],
            arcs: &arcs,
            conics: &[],
            detail_reliability: None,
        };
        assert!(matches!(
            solve_joint_conics(JointConicRequest {
                eyes: [Some(eye), None],
                scene: &fixture.scene,
                maximum_hypotheses: 16,
                maximum_refinements: 16,
                maximum_source_skew_ns: 0,
                exposure_uncertainty_ns: 0,
                motion_bound_px_per_second: 0.0
            }),
            Err(JointConicUnavailable::InvalidRequest)
        ));
    }
}

#[test]
fn image_boundary_normals_do_not_fabricate_a_three_dimensional_mirror_sign() {
    let fixture = Fixture::new([70.0, -130.0, 250.0]);
    let e = fixture.hints[0][0].1;
    let points = e.dense_points(16);
    let (s, c) = e.angle.sin_cos();
    let normals = points
        .iter()
        .map(|&(x, y)| {
            let u = (c * (x - e.center.0) + s * (y - e.center.1)) / e.major_radius.powi(2);
            let v = (-s * (x - e.center.0) + c * (y - e.center.1)) / e.minor_radius.powi(2);
            Some(BoundaryNormalObservation {
                unit_outward_roi: [(c * u - s * v) / u.hypot(v), (s * u + c * v) / u.hypot(v)],
                angular_sigma_radians: 0.2,
            })
        })
        .collect();
    let (quadrature, length_px) = polyline_quadrature(&points).unwrap();
    let arc = SparseArc {
        eye: 0,
        index: 0,
        kind: BoundaryKind::OuterLimbus,
        boundary: 0,
        group: 0,
        level_sets: vec![None; points.len()],
        points,
        outward_normals: normals,
        sigma: 1.0,
        quadrature,
        length_px,
        weight: 1.0,
    };
    let poses = circle_pose_hypotheses(fixture.scene.camera, e, fixture.origins[0]).unwrap();
    assert!(
        norm3(sub3(poses[0].normal, poses[1].normal)) > 0.1,
        "exercise distinct 3D mirror branches"
    );
    for pose in poses {
        let conic = ProjectedCircle::project(
            fixture.scene.camera,
            scale3(pose.center_per_radius, 6.0),
            pose.normal,
            6.0,
            fixture.origins[0],
        )
        .unwrap();
        assert!(
            arc.mean_cost(conic) < 1.0e-12,
            "the identical projected boundary cannot distinguish these 3D branches"
        );
    }
}

#[test]
fn many_short_pupil_fragments_cannot_outvote_a_long_well_supported_limbus() {
    let mut fixture = Fixture::new([70.0, -130.0, 250.0]);
    fixture.arcs = [Vec::new(), Vec::new()];
    fixture.hints = [Vec::new(), Vec::new()];
    fixture.add_arc(0, BoundaryKind::OuterLimbus, 0.0, TAU, 64, 0);
    let outer = fixture.hints[0][0].1;
    for sector in 0..8 {
        fixture.add_arc(
            0,
            BoundaryKind::PupillaryBoundary,
            sector as f64 * TAU / 8.0,
            (sector + 1) as f64 * TAU / 8.0,
            8,
            100 + sector,
        );
        // A displaced shadow/glint boundary is not a second iris-center vote.
        for point in &mut fixture.arcs[0].last_mut().unwrap().points {
            point.0 += 16.0;
            point.1 += 10.0;
        }
    }
    let solution = fixture.solve([true, false], 16).unwrap();
    let fitted = solution.ellipses_roi_px[0][0].unwrap();
    assert!(outer
        .dense_points(48)
        .into_iter()
        .all(|p| crate::conic_solver::ellipse_residual(p, fitted) < 1.0));
    assert!(solution
        .arcs
        .iter()
        .any(|a| a.kind == BoundaryKind::PupillaryBoundary && !a.used));
}

#[test]
fn independently_scaled_seed_radii_do_not_make_every_joint_start_infeasible() {
    let mut fixture = Fixture::new([70.0, -130.0, 250.0]);
    // Bad scale hypotheses on one defocused eye, with broad range support.
    // The other eye remains useful. The frozen center/IPD bounds do not move.
    for prior in fixture.scene.eyes.iter_mut().flatten() {
        prior.limbus_center.transverse_frame = TransversePositionFrame::AtNominalDepth;
        prior.limbus_center.sigma_mm[2] = 122.5;
        prior.limbus_center.maximum_displacement_mm[2] = 245.0;
    }
    for (_, hint) in &mut fixture.hints[1] {
        hint.major_radius *= 1.7;
        hint.minor_radius *= 1.7;
    }
    fixture.detail[1] = 0.2;
    let solution = fixture.solve([true, true], 24).unwrap();
    assert!(solution.contributing_eyes[0]);
    assert!(angular_error(&solution, &fixture, 0) < 1.0);
    let distance = norm3(sub3(
        solution.eye_centers_camera_mm[0].unwrap(),
        solution.eye_centers_camera_mm[1].unwrap(),
    ));
    let support = fixture.scene.interocular_distance_mm.unwrap();
    assert!(distance >= support.minimum && distance <= support.maximum);
}

#[test]
fn raw_verified_partial_outline_without_a_fitted_ellipse_contributes_to_one_shared_target() {
    use crate::outline_conic_segments::partial_outline::{
        append_unfitted_outline_arcs, OutlineCandidate,
    };
    use crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence;
    let mut fixture = Fixture::new([70.0, -130.0, 250.0]);
    let ellipse = fixture.hints[0][0].1;
    let raw = (0..280)
        .flat_map(|y| {
            (0..420).map(move |x| {
                let (s, c) = ellipse.angle.sin_cos();
                let dx = x as f64 - ellipse.center.0;
                let dy = y as f64 - ellipse.center.1;
                if ((c * dx + s * dy) / ellipse.major_radius)
                    .hypot((-s * dx + c * dy) / ellipse.minor_radius)
                    <= 1.0
                {
                    150
                } else {
                    550
                }
            })
        })
        .collect::<Vec<_>>();
    let contour = fixture.arcs[0][0].points.clone();
    let dense = ring_points(
        fixture.scene.camera,
        fixture.scene.eyes[0].unwrap().limbus_center.camera_mm,
        normalized3(sub3(
            fixture.target,
            fixture.scene.eyes[0].unwrap().limbus_center.camera_mm,
        ))
        .unwrap(),
        6.0,
        fixture.origins[0],
        0.0,
        TAU,
        128,
    );
    let clipped = dense
        .iter()
        .map(|&(x, y)| (x.max(ellipse.center.0), y))
        .collect::<Vec<_>>();
    let mut packet = OwnedRoiEvidence {
        exposure: fixture.exposures[0],
        sensor_origin_px: fixture.origins[0],
        dimensions_px: [420, 280],
        arcs: Vec::new(),
        conics: Vec::new(),
        detail_reliability: None,
    };
    append_unfitted_outline_arcs(
        &mut packet,
        &raw,
        &[OutlineCandidate {
            points_roi_px: &clipped,
            detector_score: None,
        }],
        ellipse.center,
        20,
    );
    assert!(packet.arcs.len() >= 2);
    // Deliberately discard every fitted-conic seed for this eye. Its only
    // contribution is the current RAW-supported incomplete boundary.
    fixture.hints[0].clear();
    fixture.arcs[0] = packet
        .arcs
        .iter()
        .map(|a| OwnedArc {
            group: a.evidence_group,
            kind: a.kind,
            points: a.points_roi_px.clone(),
            band: a.normal_band_half_width_px,
        })
        .collect();
    let joint = fixture.solve([true, true], 24).unwrap();
    assert_eq!(joint.contributing_eyes, [true, true]);
    assert!(angular_error(&joint, &fixture, 1) < 2.0);
    let recovered = joint.ellipses_roi_px[0][0].unwrap();
    let error = contour
        .iter()
        .map(|&p| crate::conic_solver::ellipse_residual(p, recovered).powi(2))
        .sum::<f64>();
    assert!((error / contour.len() as f64).sqrt() < 3.0);
    for eye in 0..2 {
        let ray = normalized3(sub3(
            joint.target_camera_mm,
            joint.eye_centers_camera_mm[eye].unwrap(),
        ))
        .unwrap();
        assert!(norm3(sub3(ray, joint.eye_gaze_directions[eye].unwrap())) < 1.0e-10);
    }
}

#[test]
fn an_unlocalized_roi_cannot_force_a_good_eye_into_its_false_interocular_geometry() {
    let mut fixture = Fixture::new([70.0, -130.0, 250.0]);
    // A skin/foreground detection is not the second eye. Its tight but wrong
    // ROI association would make every coupled initialization violate IPD.
    let prior = fixture.scene.eyes[1].as_mut().unwrap();
    prior.limbus_center.camera_mm[0] = 160.0;
    prior.limbus_center.maximum_displacement_mm = [0.1; 3];
    fixture.hints[1].clear();
    fixture.arcs[1] = vec![OwnedArc {
        group: 40,
        kind: BoundaryKind::OuterLimbus,
        points: vec![(10.0, 10.0), (12.0, 10.2), (14.0, 10.3)],
        band: 8.0,
    }];
    let result = fixture.solve([true, true], 16).unwrap();
    assert_eq!(result.modeled_eyes, [true, false]);
    assert_eq!(result.contributing_eyes, [true, false]);
    assert!(result.eye_centers_camera_mm[1].is_none() && result.ellipses_roi_px[1][0].is_none());
    assert!(result.unlocalized_eye_cost[1] > 0.0);
    assert!(angular_error(&result, &fixture, 0) < 1.0);
    assert!(result.hypotheses_evaluated <= 16);
    assert!(result.refinement_steps <= 16 * 16 * 4);
    // Correlated detector alternatives cannot multiply the rejection penalty.
    for _ in 0..8 {
        fixture.arcs[1].push(OwnedArc {
            group: 40,
            kind: BoundaryKind::OuterLimbus,
            points: vec![(10.0, 10.0), (12.0, 10.2), (14.0, 10.3)],
            band: 8.0,
        });
    }
    let repeated = fixture.solve([true, true], 16).unwrap();
    assert_eq!(repeated.unlocalized_eye_cost, result.unlocalized_eye_cost);
    assert!(norm3(sub3(repeated.target_camera_mm, result.target_camera_mm)) < 1.0e-8);
}

#[test]
fn useful_two_eye_evidence_is_not_replaced_by_a_free_single_eye_hypothesis() {
    let fixture = Fixture::new([70.0, -130.0, 250.0]);
    for budget in [8, 16, 24] {
        let result = fixture.solve([true, true], budget).unwrap();
        assert_eq!(result.modeled_eyes, [true, true]);
        assert_eq!(result.contributing_eyes, [true, true]);
        assert_eq!(result.unlocalized_eye_cost, [0.0; 2]);
        assert!(result.hypotheses_evaluated <= budget);
    }
}

#[test]
fn dominated_unlocalized_models_return_their_starts_to_the_joint_search() {
    let fixture = Fixture::new([70.0, -130.0, 250.0]);
    let result = fixture.solve([true, true], 16).unwrap();
    assert!(
        result.robust_cost < 0.01,
        "current evidence already supports a near-zero joint cost"
    );
    assert_eq!(result.hypotheses_by_association,[16,0,0],
        "omitting either strongly supported ROI has an irreducible cost above the current fit; do not halve the useful joint search");
    assert_eq!(result.hypotheses_evaluated, 16);
}

#[test]
fn misleading_first_conic_does_not_replace_strong_two_eye_boundary_support() {
    for eye in 0..2 {
        let mut fixture = Fixture::new([70.0, -130.0, 250.0]);
        let truth = fixture.hints[eye][0].1;
        let prior = fixture.scene.eyes[eye].as_mut().unwrap();
        prior.limbus_center.sigma_mm = [6.0, 6.0, 90.0];
        prior.limbus_center.maximum_displacement_mm = [16.0, 16.0, 200.0];
        let mut bad = truth;
        bad.center.0 += 80.0;
        bad.center.1 += 30.0;
        bad.major_radius *= 0.6;
        bad.minor_radius *= 0.6;
        fixture.hints[eye].insert(0, (BoundaryKind::OuterLimbus, bad));
        let result = fixture.solve([true, true], 24).unwrap();
        assert_eq!(
            result.contributing_eyes,
            [true, true],
            "a bad first seed must not erase directly observed good arcs"
        );
        let fit = result.ellipses_roi_px[eye][0].unwrap();
        let rms = (truth
            .dense_points(64)
            .iter()
            .map(|&p| crate::conic_solver::ellipse_residual(p, fit).powi(2))
            .sum::<f64>()
            / 64.0)
            .sqrt();
        assert!(
            rms < 1.0,
            "eye={eye} actual boundary error={rms} result={fit:?}"
        );
    }
}

#[test]
fn a_secondary_circle_seed_carries_its_own_center_and_metric_radius() {
    let fixture = Fixture::new([70.0, -130.0, 250.0]);
    let truth = fixture.hints[0][0].1;
    let mut wrong = truth;
    wrong.major_radius *= 1.2;
    wrong.minor_radius *= 1.2;
    wrong.center.0 += 15.0;
    let hints = [wrong, truth].map(|ellipse_roi_px| ConicObservation {
        kind: BoundaryKind::OuterLimbus,
        ellipse_roi_px,
        supporting_arc_indices: &[0],
        residual_px: None,
    });
    let arcs = [BoundaryArcObservation { level_sets_roi: None,
        evidence_group: 0,
        kind: BoundaryKind::OuterLimbus,
        points_roi_px: &fixture.arcs[0][0].points,
        outward_normals_roi: None,
        localization_sigma_px: None,
        normal_band_half_width_px: Some(0.0),
        detector_score: None,
    }];
    let evidence = RoiConicEvidence {
        exposure: fixture.exposures[0],
        sensor_origin_px: fixture.origins[0],
        dimensions_px: [420, 280],
        arcs: &arcs,
        conics: &hints,
        detail_reliability: Some(1.0),
    };
    let problem = Problem::new(JointConicRequest {
        eyes: [Some(evidence), None],
        scene: &fixture.scene,
        maximum_hypotheses: 24,
        maximum_refinements: 16,
        maximum_source_skew_ns: 0,
        exposure_uncertainty_ns: 0,
        motion_bound_px_per_second: 0.0,
    })
    .unwrap();
    let k = TARGET_PARAMETERS;
    assert!(
        problem
            .seeds()
            .iter()
            .any(|p| (p[k + 2] + 350.0).abs() < 1.0e-5 && (p[k + 3] - 6.0).abs() < 1.0e-5),
        "normal-only starts remain stuck in the first conic's center/range geometry"
    );
}

#[test]
fn previous_targets_are_competing_initializations_not_averaged_points_or_extra_residuals() {
    let mut fixture = Fixture::new([70.0, -130.0, 250.0]);
    let targets = [[-100.0, 80.0, 250.0], [150.0, -120.0, 250.0]];
    fixture.scene.target_seed_camera_mm = Some(targets[0]);
    fixture.scene.secondary_target_seed_camera_mm = Some(targets[1]);
    let arcs = [BoundaryArcObservation { level_sets_roi: None,
        evidence_group: 0,
        kind: BoundaryKind::OuterLimbus,
        points_roi_px: &fixture.arcs[0][0].points,
        outward_normals_roi: None,
        localization_sigma_px: None,
        normal_band_half_width_px: Some(0.0),
        detector_score: None,
    }];
    let evidence = RoiConicEvidence {
        exposure: fixture.exposures[0],
        sensor_origin_px: fixture.origins[0],
        dimensions_px: [420, 280],
        arcs: &arcs,
        conics: &[],
        detail_reliability: Some(1.0),
    };
    let request = JointConicRequest {
        eyes: [Some(evidence), None],
        scene: &fixture.scene,
        maximum_hypotheses: 8,
        maximum_refinements: 12,
        maximum_source_skew_ns: 0,
        exposure_uncertainty_ns: 0,
        motion_bound_px_per_second: 0.0,
    };
    let problem = Problem::new(request).unwrap();
    let seeds = problem.seeds();
    assert!(seeds.len() <= 8);
    for i in 0..2 {
        assert!(
            norm3(sub3(problem.target(&seeds[i]).unwrap(), targets[i])) < 1e-9,
            "a previous read supplies its own start, not an average with a different read/eye"
        );
    }
    let mut unseeded = fixture.scene;
    unseeded.target_seed_camera_mm = None;
    unseeded.secondary_target_seed_camera_mm = None;
    let control = Problem::new(JointConicRequest {
        scene: &unseeded,
        ..request
    })
    .unwrap();
    let selected = problem.select(&problem.conics(&problem.initial).unwrap());
    assert_eq!(problem.initial, control.initial);
    assert_eq!(problem.residuals(&problem.initial,&selected),control.residuals(&control.initial,&selected),
        "historical initializations cannot become temporal observations, extra weights or new pixels");
    let solved = solve_joint_conics(request).unwrap();
    assert!(solved.hypotheses_evaluated <= 8);
    assert_eq!(
        solved.hypotheses_by_association.iter().sum::<usize>(),
        solved.hypotheses_evaluated
    );
    let mut single = fixture.scene;
    single.secondary_target_seed_camera_mm = None;
    let expected = Problem::new(JointConicRequest {
        scene: &single,
        ..request
    })
    .unwrap()
    .seeds();
    for secondary in [targets[0], [f64::NAN; 3]] {
        let mut duplicate = single;
        duplicate.secondary_target_seed_camera_mm = Some(secondary);
        assert_eq!(
            Problem::new(JointConicRequest {
                scene: &duplicate,
                ..request
            })
            .unwrap()
            .seeds(),
            expected,
            "duplicate/nonfinite starts do not crowd out useful hypotheses"
        );
    }
}

#[test]
fn posterior_integrates_shared_geometry_without_changing_the_map_solution() {
    for enabled in [[true, false], [true, true]] {
        let fixture = Fixture::new([70.0, -130.0, 250.0]);
        let baseline = fixture.solve(enabled, 24).unwrap();
        // Exercise the production entry point. Offline controls deliberately
        // retain the original integration configuration for matched replays.
        let candidate = fixture.with_request(enabled, 24, |request|
            solve_joint_conic_distribution(request, 1).unwrap().remove(0));
        assert_eq!(baseline.target_camera_mm, candidate.target_camera_mm);
        assert_eq!(baseline.contributing_eyes, candidate.contributing_eyes);
        assert_eq!(baseline.robust_cost, candidate.robust_cost);
        assert_eq!(
            baseline.hypotheses_evaluated,
            candidate.hypotheses_evaluated
        );
        let posterior = candidate.posterior.as_ref().unwrap();
        assert_eq!(posterior.status, "estimated-conditional");
        eprintln!(
            "posterior fixture {enabled:?}: {} ESS {:.1} feasible {} radius {:?}",
            posterior.status,
            posterior.effective_samples,
            posterior.feasible_samples,
            posterior.gaze_radius_90_degrees
        );
        assert!(posterior.samples <= 8192);
        assert_eq!(posterior.modeled_eyes, enabled);
        assert_eq!(
            posterior.gaze_radius_90_degrees[1].is_some(),
            enabled[1] && posterior.status == "estimated-conditional"
        );
        if posterior.status == "estimated-conditional" {
            assert!(posterior.effective_samples >= 24.0);
            assert!(
                (posterior
                    .modes
                    .iter()
                    .map(|m| m.model_mass.unwrap())
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1e-10
            );
            assert!(posterior.target_covariance_mm2.unwrap()[0][0] > 0.0);
            assert!(posterior.gaze_radius_90_degrees[0].unwrap() > 0.0);
        } else {
            assert!(posterior.target_covariance_mm2.is_none());
            assert!(posterior.modes.iter().all(|m| m.model_mass.is_none()));
        }
    }
}

#[test]
fn same_eye_pupil_support_resolves_the_outer_only_mirror_distribution() {
    let mut fixture = Fixture::new([90.0, -170.0, 250.0]);
    fixture.arcs[0].retain(|a| a.kind != BoundaryKind::InnerLimbus);
    fixture.hints[0].retain(|a| a.0 != BoundaryKind::InnerLimbus);
    let supported = fixture.with_request([true, false], 24, |request|
        solve_joint_conic_distribution(request, 1).unwrap().remove(0));
    assert!(angular_error(&supported, &fixture, 0) < 0.1);
    fixture.arcs[0].retain(|a| a.kind == BoundaryKind::OuterLimbus);
    fixture.hints[0].retain(|a| a.0 == BoundaryKind::OuterLimbus);
    let outer_only = fixture.with_request([true, false], 24, |request|
        solve_joint_conic_distribution(request, 1).unwrap().remove(0));
    let good = supported.posterior.as_ref().unwrap();
    let ambiguous = outer_only.posterior.as_ref().unwrap();
    eprintln!(
        "pupil distribution: {} {:?}; outer-only: {} {:?} masses {:?}",
        good.status,
        good.gaze_radius_90_degrees,
        ambiguous.status,
        ambiguous.gaze_radius_90_degrees,
        ambiguous
            .modes
            .iter()
            .map(|m| m.model_mass)
            .collect::<Vec<_>>()
    );
    assert_eq!(good.status, "estimated-conditional");
    assert_eq!(ambiguous.status, "estimated-conditional");
    assert!(
        good.supports_direction(0),
        "strong same-eye support can authorize a direction"
    );
    assert!(
        !ambiguous.supports_direction(0),
        "the mirrored outer-only pair cannot authorize its arbitrary winner"
    );
    assert!(
        ambiguous
            .modes
            .iter()
            .filter(|m| m.model_mass.is_some_and(|p| p > 0.05))
            .count()
            >= 2
    );
    assert!(
        ambiguous.gaze_radius_90_degrees[0].unwrap()
            > 2.0 * good.gaze_radius_90_degrees[0].unwrap()
    );
}

#[test]
fn a_sharp_pupil_with_sparse_same_eye_outer_support_remains_solvable() {
    let mut fixture = Fixture::new([70.0, -130.0, 250.0]);
    fixture.arcs[0].clear();
    fixture.hints[0].clear();
    fixture.add_arc(0, BoundaryKind::OuterLimbus, 0.0, TAU * 0.25, 12, 0);
    fixture.add_arc(0, BoundaryKind::PupillaryBoundary, 0.0, TAU, 32, 1);
    fixture.hints[0].retain(|(kind, _)| *kind == BoundaryKind::PupillaryBoundary);
    let result = fixture.with_request([true, false], 24, |request|
        solve_joint_conic_distribution(request, 1).unwrap().remove(0));
    let error = angular_error(&result, &fixture, 0);
    let posterior = result.posterior.as_ref().unwrap();
    eprintln!(
        "sparse outer + pupil error {error:.2} posterior {} {:?}",
        posterior.status, posterior.gaze_radius_90_degrees
    );
    assert!(
        error < 1.0,
        "sharp pupil and a measured outer arc should preserve gaze: {error}"
    );
    assert_eq!(result.contributing_eyes, [true, false]);
    assert!(posterior.gaze_radius_90_degrees[1].is_none());
    // Missing sampling support must remain unknown instead of manufacturing a
    // precise radius from an underconstrained same-eye configuration.
    assert_eq!(
        posterior.gaze_radius_90_degrees[0].is_some(),
        posterior.status == "estimated-conditional"
    );
    assert!(
        !posterior.supports_direction(0),
        "an exact synthetic optimum is not enough when the posterior remains broad"
    );
}

#[test]
fn separated_same_eye_outer_arcs_can_anchor_a_sharp_pupil() {
    for (phase, expected) in [(0.0, false), (0.1875, true)] {
        let mut fixture = Fixture::new([90.0, -170.0, 250.0]);
        fixture.arcs[0].clear();
        fixture.hints[0].clear();
        // Equal quarter-perimeter coverage and noise, in two separated runs.
        // Support near the projected pupil offset constrains its direction
        // better than the same amount of support at the other two sides.
        fixture.add_arc(
            0,
            BoundaryKind::OuterLimbus,
            TAU * phase,
            TAU * (phase + 0.125),
            8,
            0,
        );
        fixture.add_arc(
            0,
            BoundaryKind::OuterLimbus,
            TAU * (phase + 0.5),
            TAU * (phase + 0.625),
            8,
            1,
        );
        fixture.add_arc(0, BoundaryKind::PupillaryBoundary, 0.0, TAU, 32, 2);
        fixture.hints[0].retain(|(kind, _)| *kind == BoundaryKind::PupillaryBoundary);
        let result = fixture
            .solve_with_distribution([true, false], 24, true)
            .unwrap();
        let p = result.posterior.as_ref().unwrap();
        eprintln!(
            "separated same-eye arcs phase {phase}: {} {:?}",
            p.status, p.gaze_radius_90_degrees
        );
        assert!(angular_error(&result, &fixture, 0) < 1.0);
        assert_eq!(
            p.supports_direction(0),
            expected,
            "support placement controls identifiability: {} {:?}",
            p.status,
            p.gaze_radius_90_degrees
        );
        assert!(!p.supports_direction(1));
    }
}

#[test]
#[ignore = "large independent-seed reference audit; prints numerical support failures rather than treating a good MAP as proof of posterior convergence"]
fn same_eye_and_complementary_stereo_posterior_reference_diagnostic() {
    support_case_posterior_reference(false,false,false,None);
}

fn add_test_mask_levels(model:&mut Problem<'_>,at:&Parameters,width:f64) {
    let conics=model.conics(at).unwrap();
    for group in &mut model.groups {for arc in &mut group.alternatives {
        if arc.kind!=BoundaryKind::OuterLimbus {continue;}
        let [a,b,c,d,e,_]=conics[arc.eye][0].unwrap().0;
        arc.level_sets=arc.points.iter().map(|&(x,y)| {
            let gradient=[2.0*a*x+b*y+d,b*x+2.0*c*y+e];
            let length=gradient[0].hypot(gradient[1]);
            Some(BoundaryLevelSetObservation {unit_normal_roi:gradient.map(|g|g/length),
                displacement_px:[-width,0.0,width],spatial_displacement_px:None})
        }).collect();
    }}
}

fn add_competing_test_arcs(model:&mut Problem<'_>) {
    for group in &mut model.groups {
        let mut alternate=group.alternatives[0].clone();
        alternate.index+=64;
        for p in &mut alternate.points {p.0+=1.5;p.1-=0.7;}
        group.alternatives.push(alternate);
    }
}

#[test]
fn arc_alternative_marginal_matches_enumerated_likelihood_and_keeps_group_mass() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        let mut model=Problem::new(request).unwrap();
        let mut p=model.initial;p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        add_competing_test_arcs(&mut model);
        let conics=model.conics(&p).unwrap();
        let before=model.select(&conics);
        let baseline=squared_norm(&model.residuals(&p,&before).unwrap());
        let old_cost=model.groups.iter().zip(&before).map(|(g,&choice)| {
            let a=&g.alternatives[choice];a.weight*a.mean_cost(conics[a.eye][a.boundary].unwrap()).min(MAXIMUM_GROUP_COST)
                +(g.weight-a.weight)*MAXIMUM_GROUP_COST
        }).sum::<f64>();
        let mass=model.groups.iter().map(|g|g.weight).collect::<Vec<_>>();
        model.marginalize_arc_alternatives=true;
        let selected=model.select(&conics);
        let expected=model.groups.iter().map(|g| {
            let costs=g.alternatives.iter().map(|a|a.weight*a.mean_cost(conics[a.eye][a.boundary].unwrap()).min(MAXIMUM_GROUP_COST)
                +(g.weight-a.weight)*MAXIMUM_GROUP_COST).collect::<Vec<_>>();
            -2.0*(costs.iter().map(|c|(-0.5*c).exp()).sum::<f64>()/costs.len() as f64).ln()
        }).sum::<f64>();
        let actual=squared_norm(&model.residuals(&p,&selected).unwrap());
        assert!((actual-(baseline-old_cost+expected)).abs()<1e-10);
        assert!(selected.group_mixtures.iter().all(Option::is_some));
        assert_eq!(mass,model.groups.iter().map(|g|g.weight).collect::<Vec<_>>());
        assert_eq!(model.local_uncertainty(&p).status,"arc-alternative-mixture-requires-distribution");
        assert!((model.factor_cost_diagnostic(&p).unwrap()["native_local_cost"].as_f64().unwrap()-actual).abs()<1e-12);
    });
}

#[test]
fn duplicate_arc_alternatives_do_not_change_the_marginal_or_its_geometry_gradient() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        let mut model=Problem::new(request).unwrap().with_arc_marginalization(true);
        add_competing_test_arcs(&mut model);
        let mut repeated=model.clone();
        for g in &mut repeated.groups {g.alternatives.push(g.alternatives[0].clone());}
        let mut at=model.initial;at[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        for dx in [-0.02,0.0,0.02] {
            let mut p=at;p[0]+=dx;
            let a=model.residuals(&p,&model.select(&model.conics(&p).unwrap())).unwrap();
            let b=repeated.residuals(&p,&repeated.select(&repeated.conics(&p).unwrap())).unwrap();
            assert_eq!(a,b,"copying one physical alternative cannot change its prior weight");
        }
    });
}

#[test]
fn nested_arc_and_mask_marginals_have_tangent_em_derivatives_and_exact_attribution() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        for profiles in [false,true] {
            let mut model=Problem::new(request).unwrap().with_arc_marginalization(true);
            let mut p=model.initial;p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
            p[0]+=0.01;
            add_competing_test_arcs(&mut model);
            if profiles {add_test_mask_levels(&mut model,&p,1.5);}
            let conics=model.conics(&p).unwrap();let selected=model.select(&conics);
            let rejected=model.rejected_groups(&conics,&selected);
            let marginal=|p:&Parameters|squared_norm(&model.residuals(p,&selected).unwrap());
            let upper=|p:&Parameters|squared_norm(&model.residuals_with_rejection(p,&selected,Some(&rejected)).unwrap());
            assert!((marginal(&p)-upper(&p)).abs()<1e-10);
            assert!((model.factor_cost_diagnostic(&p).unwrap()["native_local_cost"].as_f64().unwrap()-marginal(&p)).abs()<1e-12);
            for i in 0..PARAMETERS {
                if model.lower[i]==model.upper[i] {continue;}
                let h=1e-6*model.scales[i];let mut a=p;let mut b=p;a[i]-=h;b[i]+=h;
                if model.conics(&a).is_none() || model.conics(&b).is_none() {continue;}
                let gradient=(marginal(&b)-marginal(&a))/(2.0*h);
                let em=(upper(&b)-upper(&a))/(2.0*h);
                assert!((gradient-em).abs()<8e-5*(1.0+gradient.abs()),"profiles {profiles} parameter {i}: {gradient} != {em}");
                for v in [a,b] {assert!(upper(&v)+1e-9>=marginal(&v));}
            }
            let mut trial=p;trial[0]+=0.04;
            let refreshed=model.select(&model.conics(&trial).unwrap());
            assert!((marginal(&trial)-squared_norm(&model.residuals(&trial,&refreshed).unwrap())).abs()<1e-10);
            assert!(mask_levels::arc_diagnostics(&model,&selected).len()>=model.groups.len());
        }
    });
}

#[test]
fn enabling_arc_marginalization_without_alternatives_preserves_exact_native_model() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        for profiles in [false,true] {
            let mut model=Problem::new(request).unwrap();
            let mut p=model.initial;p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
            if profiles {add_test_mask_levels(&mut model,&p,1.5);}
            let before=model.residuals(&p,&model.select(&model.conics(&p).unwrap())).unwrap();
            model.marginalize_arc_alternatives=true;
            let selected=model.select(&model.conics(&p).unwrap());
            assert_eq!(before,model.residuals(&p,&selected).unwrap());
            assert!(mask_levels::arc_diagnostics(&model,&selected).is_empty());
        }
    });
    assert!(!posterior::IntegrationConfig::live().marginalize_arc_alternatives);
}

#[test]
fn zero_width_mask_states_preserve_the_exact_original_residual_vector() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        let mut model=Problem::new(request).unwrap();
        let mut p=model.initial;
        p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        let before=model.residuals(&p,&model.select(&model.conics(&p).unwrap())).unwrap();
        add_test_mask_levels(&mut model,&p,0.0);
        let selected=model.select(&model.conics(&p).unwrap());
        assert!(selected.families.is_empty());
        let after=model.residuals(&p,&selected).unwrap();
        assert_eq!(before.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
            after.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
    });
}

#[test]
fn native_mask_family_residuals_equal_the_correlated_marginal_and_keep_point_mass() {
    let mut fixture=Fixture::new([90.0,-170.0,250.0]);
    for eye in 0..2 {
        fixture.arcs[eye].retain(|a|a.kind!=BoundaryKind::OuterLimbus);
        fixture.add_arc(eye,BoundaryKind::OuterLimbus,0.0,TAU*0.25,12,10);
        fixture.add_arc(eye,BoundaryKind::OuterLimbus,TAU*0.5,TAU*0.75,12,11);
    }
    fixture.with_request([true,true],24,|request| {
        let mut model=Problem::new(request).unwrap();
        let mut p=model.initial;
        p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        p[0]+=0.02;
        let conics=model.conics(&p).unwrap();
        let before=model.select(&conics);
        let baseline=squared_norm(&model.residuals(&p,&before).unwrap());
        let old_outer_cost=model.groups.iter().zip(&before).filter(|(g,_)|g.alternatives[0].boundary==0)
            .map(|(g,&i)|{let a=&g.alternatives[i];a.weight*a.mean_cost(conics[a.eye][0].unwrap()).min(MAXIMUM_GROUP_COST)
                +(g.weight-a.weight)*MAXIMUM_GROUP_COST}).sum::<f64>();
        let mass=model.groups.iter().map(|g|g.weight).collect::<Vec<_>>();
        let lengths=model.groups.iter().map(|g|g.alternatives[0].length_px).collect::<Vec<_>>();
        add_test_mask_levels(&mut model,&p,1.5);
        let selected=model.select(&conics);
        assert_eq!(selected.families.len(),2);
        assert!(selected.families.iter().all(|f|f.groups.len()==2));
        assert_eq!(mass,model.groups.iter().map(|g|g.weight).collect::<Vec<_>>());
        assert_eq!(lengths,model.groups.iter().map(|g|g.alternatives[0].length_px).collect::<Vec<_>>());
        let expected=baseline-old_outer_cost+selected.families.iter().map(|f|f.marginal_cost).sum::<f64>();
        let actual=squared_norm(&model.residuals(&p,&selected).unwrap());
        assert!((actual-expected).abs()<1e-9);
        let attribution=model.factor_cost_diagnostic(&p).unwrap();
        assert!((attribution["native_local_cost"].as_f64().unwrap()-actual).abs()<1e-12);
        let local=model.local_uncertainty(&p);
        assert_eq!(local.status,"mask-level-mixture-requires-distribution");
        assert!(local.target_covariance_mm2.is_none());
    });
}

#[test]
fn native_mask_em_derivatives_are_tangent_to_the_actual_trial_likelihood() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        let mut model=Problem::new(request).unwrap();
        let mut p=model.initial;
        p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        p[0]+=0.015;
        add_test_mask_levels(&mut model,&p,1.5);
        let conics=model.conics(&p).unwrap();
        let selected=model.select(&conics);
        let rejected=model.rejected_groups(&conics,&selected);
        let marginal=|p:&Parameters|squared_norm(&model.residuals(p,&selected).unwrap());
        let upper=|p:&Parameters|squared_norm(&model.residuals_with_rejection(p,&selected,Some(&rejected)).unwrap());
        assert!((marginal(&p)-upper(&p)).abs()<1e-10);
        for i in 0..PARAMETERS {
            if model.lower[i]==model.upper[i] {continue;}
            let h=1e-6*model.scales[i];let mut a=p;let mut b=p;a[i]-=h;b[i]+=h;
            if model.conics(&a).is_none() || model.conics(&b).is_none() {continue;}
            let gradient=(marginal(&b)-marginal(&a))/(2.0*h);
            let em_gradient=(upper(&b)-upper(&a))/(2.0*h);
            assert!((gradient-em_gradient).abs()<5e-5*(1.0+gradient.abs()),"parameter {i}: {gradient} versus {em_gradient}");
            for value in [a,b] {assert!(upper(&value)+1e-10>=marginal(&value));}
        }
        let mut trial=p;trial[0]+=0.08;
        let refreshed=model.select(&model.conics(&trial).unwrap());
        assert_ne!(selected.families[0].responsibilities,refreshed.families[0].responsibilities);
        assert!((marginal(&trial)-squared_norm(&model.residuals(&trial,&refreshed).unwrap())).abs()<1e-12);
    });
}

#[test]
fn mask_state_proposal_conditioning_preserves_the_exact_joint_state_likelihood() {
    let fixture = Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        let mut model = Problem::new(request).unwrap();
        let mut p = model.initial;
        p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        p[0] += 0.015;
        add_test_mask_levels(&mut model,&p,4.0);
        let conics = model.conics(&p).unwrap();
        let selected = model.select(&conics);
        let residuals = model.residuals(&p,&selected).unwrap();
        let common = squared_norm(&residuals) - selected.families.iter().map(|f|f.marginal_cost).sum::<f64>();
        let assignments = posterior::mask_assignments(selected.families.len());
        assert_eq!(assignments.len(),9,"all two-eye mask combinations are proposed");
        let mut likelihoods = Vec::new();
        for levels in assignments {
            let conditional = posterior::conditioned_mask_model(&model,&selected.families,&levels);
            let selection = conditional.select(&conics);
            assert!(selection.families.is_empty());
            let cost = squared_norm(&conditional.residuals(&p,&selection).unwrap());
            let expected = common + selected.families.iter().zip(&levels).map(|(f,&s)|f.costs[s]).sum::<f64>();
            assert!((cost-expected).abs()<1e-9,"conditional state {levels:?}");
            likelihoods.push((-0.5*cost).exp());
            assert_eq!(model.lower,conditional.lower);
            assert_eq!(model.upper,conditional.upper);
            assert_eq!(model.scales,conditional.scales);
            for (a,b) in model.groups.iter().zip(&conditional.groups) {
                assert_eq!(a.weight,b.weight);
                for (a,b) in a.alternatives.iter().zip(&b.alternatives) {
                    assert_eq!(a.sigma,b.sigma);
                    assert_eq!(a.quadrature,b.quadrature);
                    assert_eq!(a.length_px,b.length_px);
                    assert_eq!(a.weight,b.weight);
                    if a.boundary != 0 { assert_eq!(a.points,b.points,"independent pupil and inner samples unchanged"); }
                }
            }
            let initialized = posterior::mask_radius_start(&conditional,p,&selected.families);
            assert_eq!(&initialized[..TARGET_PARAMETERS],&p[..TARGET_PARAMETERS]);
            assert!(conditional.conics(&initialized).is_some());
        }
        let reconstructed = -2.0*(likelihoods.iter().sum::<f64>()/likelihoods.len() as f64).ln();
        assert!((reconstructed-squared_norm(&residuals)).abs()<1e-9);
        assert_eq!(model.residuals(&p,&selected).unwrap(),residuals,"proposal fitting cannot mutate the target");
        let bounded = posterior::mask_assignments(6);
        assert_eq!(bounded.len(),16);
        for family in 0..6 { for level in 0..3 {
            assert!(bounded.iter().any(|s|s[family]==level));
        }}
    });
}

#[test]
fn mask_state_proposals_without_profiles_preserve_exact_posterior_draws() {
    let fixture = Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        let config = posterior::IntegrationConfig {budget:1024,..posterior::IntegrationConfig::live()};
        let a = solve_joint_conic_distribution_diagnostic(request,1,config).unwrap().remove(0);
        let b = solve_joint_conic_distribution_diagnostic(request,1,
            posterior::IntegrationConfig {mask_state_proposals:true,mask_state_refinement:true,..config}).unwrap().remove(0);
        assert_eq!(a.target_camera_mm,b.target_camera_mm);
        assert_eq!(a.robust_cost,b.robust_cost);
        assert_eq!(a.hypotheses_evaluated,b.hypotheses_evaluated);
        assert_eq!(a.refinement_steps,b.refinement_steps);
        assert_eq!(a.posterior.unwrap().json(),b.posterior.unwrap().json());
    });
}

#[test]
fn mask_state_initializations_are_bounded_and_scored_on_the_full_marginal() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        for spatial in [false,true] {
            let mut model=Problem::new(request).unwrap();
            let mut p=model.initial;
            p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
            add_test_mask_levels(&mut model,&p,4.0);
            if spatial {for group in &mut model.groups {for arc in &mut group.alternatives {
                for level in arc.level_sets.iter_mut().flatten() {*level=level.with_spatial_sensitivity();}
            }}}
            let before=model.factor_cost_diagnostic(&p).unwrap();
            let expanded=posterior::refine_mask_state_initializations(&model,&[p,p],5);
            assert_eq!(expanded.attempts,5);
            assert!(!expanded.fits.is_empty());
            assert_eq!(before,model.factor_cost_diagnostic(&p).unwrap());
            for (fitted,cost) in expanded.fits {
                let conics=model.conics(&fitted).unwrap();
                let full=squared_norm(&model.residuals(&fitted,&model.select(&conics)).unwrap());
                assert!((cost-full).abs()<1e-9,"conditional costs must not select a fit");
                let solution=model.solution(&fitted,cost).unwrap();
                for eye in 0..2 {
                    let ray=normalized3(sub3(solution.target_camera_mm,solution.eye_centers_camera_mm[eye].unwrap())).unwrap();
                    assert!(norm3(sub3(ray,solution.eye_gaze_directions[eye].unwrap()))<1e-12);
                }
            }
            assert_eq!(posterior::refine_mask_state_initializations(&model,&[p],0).attempts,0);
        }
    });
}

#[test]
fn spatial_mask_states_preserve_native_envelopes_and_exact_shared_likelihood() {
    let fixture=Fixture::new([90.0,-170.0,250.0]);
    fixture.with_request([true,true],24,|request| {
        let mut model=Problem::new(request).unwrap();
        let mut p=model.initial;
        p[..3].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        add_test_mask_levels(&mut model,&p,4.0);
        let before=model.clone();
        for group in &mut model.groups {for arc in &mut group.alternatives {
            for level in arc.level_sets.iter_mut().flatten() {
                *level=level.with_spatial_sensitivity();assert!(level.valid());
            }
        }}
        let conics=model.conics(&p).unwrap();let selected=model.select(&conics);
        assert_eq!(selected.families.len(),2);
        assert!(selected.families.iter().all(|f|f.states.len()==7));
        for (old,new) in before.groups.iter().zip(&model.groups) {
            assert_eq!(old.weight,new.weight);
            for (a,b) in old.alternatives.iter().zip(&new.alternatives) {
                assert_eq!(a.points,b.points);assert_eq!(a.sigma,b.sigma);
                assert_eq!(a.weight,b.weight);assert_eq!(a.quadrature,b.quadrature);
                for (i,point) in b.points.iter().enumerate() {
                    for state in 0..7 {
                        let q=b.point_at_level(i,state);
                        assert!((q.0-point.0).hypot(q.1-point.1)<=4.0+1e-9);
                        if state<3 {assert_eq!(a.point_at_level(i,state),q);}
                        if b.boundary!=0 {assert_eq!(*point,q,"sharp inner/pupil evidence is untouched");}
                    }
                }
            }
        }
        let actual=squared_norm(&model.residuals(&p,&selected).unwrap());
        let common=actual-selected.families.iter().map(|f|f.marginal_cost).sum::<f64>();
        let mut costs=Vec::new();
        for a in 0..7 {for b in 0..7 {
            let conditional=posterior::conditioned_mask_model(&model,&selected.families,&[a,b]);
            let cost=squared_norm(&conditional.residuals(&p,&conditional.select(&conics)).unwrap());
            assert!((cost-(common+selected.families[0].costs[a]+selected.families[1].costs[b])).abs()<1e-9);
            costs.push(cost);
        }}
        let minimum=costs.iter().copied().fold(f64::INFINITY,f64::min);
        let mixture=minimum-2.0*(costs.iter().map(|c|(-0.5*(c-minimum)).exp()).sum::<f64>()/49.0).ln();
        assert!((actual-mixture).abs()<1e-9);
        assert!((model.factor_cost_diagnostic(&p).unwrap()["native_local_cost"].as_f64().unwrap()-actual).abs()<1e-10);
        assert_eq!(posterior::mask_assignments_for_counts(&[7,7]).len(),16,"proposal budget does not truncate likelihood states");
        let mut translated=model.clone();
        for group in &mut translated.groups {for arc in &mut group.alternatives {
            for point in &mut arc.points {point.0+=31.0;point.1-=17.0;}
        }}
        for (a,b) in model.groups.iter().zip(&translated.groups) {for (a,b) in a.alternatives.iter().zip(&b.alternatives) {
            for i in 0..a.points.len() {for state in 0..7 {
                let x=a.point_at_level(i,state);let y=b.point_at_level(i,state);
                assert!((x.0+31.0-y.0).abs()<1e-12 && (x.1-17.0-y.1).abs()<1e-12);
            }}
        }}
    });
}

#[test]
#[ignore = "matched synthetic RAW outer-spread candidate; reports complete/partial/complementary and true-inner limits"]
fn raw_outer_spread_support_cases_diagnostic() {
    raw_outer_support_cases_diagnostic(false);
}

#[test]
#[ignore = "matched RAW position-discrepancy candidate including displaced edges and independent inner evidence"]
fn raw_outer_position_support_cases_diagnostic() {
    raw_outer_support_cases_diagnostic(true);
}

#[test]
#[ignore = "matched mask-level likelihood audit across complete, partial, true-inner and complementary support"]
fn correlated_mask_level_support_cases_diagnostic() {
    mask_level_support_cases_diagnostic(false,None,false,None);
}

#[test]
#[ignore = "matched conditional-state proposal audit against the unchanged marginal-mask target and MAP"]
fn mask_state_proposal_support_cases_diagnostic() {
    mask_level_support_cases_diagnostic(true,None,false,None);
}

#[test]
#[ignore = "matched spatial sensitivity across complete, partial and complementary support, including anisotropic boundary bias"]
fn spatial_mask_support_cases_diagnostic() {
    for spatial in [false,true] {mask_level_support_cases_diagnostic(true,Some(spatial),false,None);}
}

#[test]
#[ignore = "matched bounded mask initialization and full marginal refinement across all support cases"]
fn mask_marginal_refinement_support_cases_diagnostic() {
    for spatial in [false,true] {for refine in [false,true] {
        mask_level_support_cases_diagnostic(true,Some(spatial),refine,None);
    }}
}

#[test]
#[ignore = "matched independent SMC population integration across complete, partial, true-inner and complementary support"]
fn population_mask_support_cases_diagnostic() {
    let config=posterior::populations::Config {
        particles:std::env::var("BUTTERCUP_POPULATION_PARTICLES").expect("set population particles").parse().unwrap(),
        steps:std::env::var("BUTTERCUP_POPULATION_STEPS").expect("set population steps").parse().unwrap(),
        populations:std::env::var("BUTTERCUP_POPULATION_COUNT").expect("set population count").parse().unwrap()};
    for spatial in [false,true] {for populations in [None,Some(config)] {
        mask_level_support_cases_diagnostic(true,Some(spatial),true,populations);
    }}
}

fn mask_level_support_cases_diagnostic(mask_state_proposals: bool, spatial_arm: Option<bool>, mask_state_refinement: bool,
    populations: Option<posterior::populations::Config>) {
    use crate::outline_conic_segments::sparse_evidence::{OwnedBoundaryArc,OwnedConicHint,OwnedRoiEvidence};
    let tag=match (spatial_arm,mask_state_refinement) {
        (None,_)=>"mask-level-support", (Some(false),false)=>"spatial-mask-control",
        (Some(true),false)=>"spatial-mask-candidate", (Some(false),true)=>"mask-refinement-uniform",
        (Some(true),true)=>"mask-refinement-spatial"};
    let tag=if populations.is_some() {if spatial_arm==Some(true) {"population-spatial"} else {"population-uniform"}} else {tag};
    for case in ["complete-pupil","complete-inner","partial-pupil","partial-inner","complementary","outer-only"] {
        let (mut fixture,enabled)=mask_support_case_fixture(case);
        let render=Fixture::new(fixture.target);
        // All variants receive the same observed inner/pupil search hints.
        // No exact outer hint leaks the synthetic truth into displaced cases.
        for hints in &mut fixture.hints {hints.retain(|a|a.0!=BoundaryKind::OuterLimbus);}
        let mut variants=vec![("control",0.0,None),("sharp-levels",0.0,Some(0.5)),
            ("broad-levels",0.0,Some(4.0)),("displaced-control",4.0,None),("displaced-levels",4.0,Some(4.0))];
        if spatial_arm.is_some() {variants.extend([("shape-displaced-control",4.0,None),("shape-displaced-levels",4.0,Some(4.0))]);}
        for (variant,offset,width) in variants {
            let packets=std::array::from_fn::<_,2,_>(|eye| {
                let e=render.hints[eye].iter().find(|a|a.0==BoundaryKind::OuterLimbus).unwrap().1;
                let (sin,cos)=e.angle.sin_cos();
                let normal=|point:(f64,f64)| {
                    let x=point.0-e.center.0;let y=point.1-e.center.1;
                    let u=(cos*x+sin*y)/e.major_radius.powi(2);
                    let v=(-sin*x+cos*y)/e.minor_radius.powi(2);
                    let n=[cos*u-sin*v,sin*u+cos*v];let length=n[0].hypot(n[1]);n.map(|x|x/length)
                };
                OwnedRoiEvidence {exposure:fixture.exposures[eye],sensor_origin_px:fixture.origins[eye],
                    dimensions_px:[420,280],detail_reliability:Some(fixture.detail[eye]),
                    arcs:fixture.arcs[eye].iter().map(|a| {
                        let outer=a.kind==BoundaryKind::OuterLimbus;
                        OwnedBoundaryArc {evidence_group:a.group,kind:a.kind,normal_band_half_width_px:a.band,
                            localization_sigma_px:None,outward_normals_roi:None,detector_score:None,
                            points_roi_px:a.points.iter().map(|&p| {
                                if !outer || offset==0.0 {p} else {
                                    let n=normal(p);
                                    let offset=if variant.starts_with("shape-") {offset*(n[0]*n[0]-n[1]*n[1])} else {offset};
                                    (p.0+offset*n[0],p.1+offset*n[1])
                                }
                            }).collect(),
                            level_sets_roi:width.filter(|_|outer).map(|width|a.points.iter().map(|&p|
                                {let level=BoundaryLevelSetObservation {unit_normal_roi:normal(p),displacement_px:[-width,0.0,width],spatial_displacement_px:None};
                                    Some(if spatial_arm==Some(true) {level.with_spatial_sensitivity()} else {level})}).collect())}
                    }).collect(),
                    conics:fixture.hints[eye].iter().map(|&(kind,ellipse_roi_px)|OwnedConicHint {
                        kind,ellipse_roi_px,supporting_arc_indices:vec![]}).collect()}
            });
            let prepared=packets.each_ref().map(|p|p.prepare());
            let request=JointConicRequest {eyes:std::array::from_fn(|eye|enabled[eye].then(||prepared[eye].evidence())),
                scene:&fixture.scene,maximum_hypotheses:24,maximum_refinements:16,maximum_source_skew_ns:0,
                exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
            for seed in [0xd1b5_4a32_d192_ed03,0x94c5_09a1_814f_753d,0x419b_79df_a2c7_5301] {
                let result=match solve_joint_conic_distribution_diagnostic(request,1,
                    posterior::IntegrationConfig {seed,mask_state_proposals,mask_state_refinement,populations,..posterior::IntegrationConfig::live()}) {
                    Ok(mut results)=>results.remove(0),
                    Err(reason)=> {
                        eprintln!("{tag} {}",serde_json::json!({"case":case,"variant":variant,
                            "seed":seed.to_string(),"available":false,"reason":format!("{reason:?}")}));
                        continue;
                    }
                };
                assert!(result.hypotheses_evaluated<=if mask_state_refinement {48} else {24});
                for eye in 0..2 {
                    if !enabled[eye] {assert!(result.eye_gaze_directions[eye].is_none());continue;}
                    if let Some(center)=result.eye_centers_camera_mm[eye] {
                        let ray=normalized3(sub3(result.target_camera_mm,center)).unwrap();
                        assert!(norm3(sub3(ray,result.eye_gaze_directions[eye].unwrap()))<1e-12);
                    }
                }
                if case=="outer-only" {assert!(!result.posterior.as_ref().unwrap().supports_direction(0));}
                eprintln!("{tag} {}",serde_json::json!({"case":case,"variant":variant,"seed":seed.to_string(),
                    "available":true,
                    "gaze_error_degrees":std::array::from_fn::<_,2,_>(|eye|(enabled[eye] && result.eye_normals[eye].is_some()).then(||angular_error(&result,&fixture,eye))),
                    "cost":result.robust_cost,"contributing_eyes":result.contributing_eyes,
                    "mask_level_families":result.mask_level_families,"posterior":result.posterior.as_ref().unwrap().json(),
                    "contract":"Synthetic projected 3D circles with one shared fixation; coherent outer-boundary displacement and level alternatives, unchanged independent pupil/true-inner points and scene support. Normals are known synthetic geometry, not native confidence calibration."}));
            }
        }
    }
}

fn mask_support_case_fixture(case:&str)->(Fixture,[bool;2]) {
    let mut fixture=Fixture::new([90.0,-170.0,250.0]);
    let enabled=[true,case=="complementary"];
    let inner=if case.ends_with("inner") {BoundaryKind::InnerLimbus} else {BoundaryKind::PupillaryBoundary};
    for eye in 0..2 {
        fixture.arcs[eye].retain(|a|a.kind==BoundaryKind::OuterLimbus || a.kind==inner);
        fixture.hints[eye].retain(|a|a.0==BoundaryKind::OuterLimbus || a.0==inner);
    }
    if case.starts_with("partial") || case=="complementary" {
        for eye in 0..if enabled[1] {2} else {1} {
            fixture.arcs[eye].clear();fixture.hints[eye].clear();
            let phase=if eye==0 {0.1875} else {0.0};
            fixture.add_arc(eye,BoundaryKind::OuterLimbus,TAU*phase,TAU*(phase+0.125),8,0);
            fixture.add_arc(eye,BoundaryKind::OuterLimbus,TAU*(phase+0.5),TAU*(phase+0.625),8,1);
            fixture.add_arc(eye,inner,0.0,TAU,32,2);
            fixture.hints[eye].retain(|a|a.0==inner);
            if enabled[1] {
                fixture.arcs[eye].remove(if eye==0 {0} else {1});
                fixture.arcs[eye].last_mut().unwrap().points.truncate(16);
            }
        }
    } else if case=="outer-only" {
        fixture.arcs[0].retain(|a|a.kind==BoundaryKind::OuterLimbus);
        fixture.hints[0].retain(|a|a.0==BoundaryKind::OuterLimbus);
    }
    (fixture,enabled)
}

#[test]
#[ignore = "matched synthetic complete/partial/complementary support with correlated competing edge alternatives"]
fn arc_alternative_support_cases_diagnostic() {
    for case in ["complete-pupil","complete-inner","partial-pupil","partial-inner","complementary","outer-only"] {
        for variant in ["clean","competing-pupil","competing-outer","duplicated-pupil"] {
            let (mut fixture,enabled)=mask_support_case_fixture(case);
            if variant!="clean" {
                for eye in 0..2 {
                    if !enabled[eye] {continue;}
                    let original=fixture.arcs[eye].clone();
                    for mut arc in original {
                        let outer=arc.kind==BoundaryKind::OuterLimbus;
                        if outer!=(variant=="competing-outer") {continue;}
                        // A second measured-edge hypothesis in the SAME group,
                        // with unchanged geometry/scale priors and search hints.
                        // This synthetic adjacent edge is not a SAM inference.
                        for point in &mut arc.points {
                            point.0+=if eye==0 {3.0} else {-3.0};point.1-=1.0;
                        }
                        fixture.arcs[eye].push(arc.clone());
                        if variant=="duplicated-pupil" {fixture.arcs[eye].push(arc);}
                    }
                }
            }
            for seed in [0xd1b5_4a32_d192_ed03,0x94c5_09a1_814f_753d,0x419b_79df_a2c7_5301] {
                let mut control:Option<JointConicSolution>=None;
                for marginalize_arc_alternatives in [false,true] {
                    let config=posterior::IntegrationConfig {seed,marginalize_arc_alternatives,..posterior::IntegrationConfig::live()};
                    let result=match fixture.solve_with_integration(enabled,24,Some(config)) {
                        Ok(result)=>result,
                        Err(reason)=>{
                            eprintln!("arc-mixture-support {}",serde_json::json!({"case":case,"variant":variant,
                                "seed":seed.to_string(),"candidate":marginalize_arc_alternatives,"available":false,
                                "reason":format!("{reason:?}")}));continue;
                        }
                    };
                    if !marginalize_arc_alternatives {control=Some(result.clone());}
                    if variant=="clean" && marginalize_arc_alternatives {
                        let before=control.as_ref().unwrap();
                        assert_eq!(result.target_camera_mm,before.target_camera_mm);
                        assert_eq!(result.robust_cost,before.robust_cost);
                        assert_eq!(result.posterior.as_ref().unwrap().json(),before.posterior.as_ref().unwrap().json());
                    }
                    for eye in 0..2 {
                        if !enabled[eye] {assert!(result.eye_gaze_directions[eye].is_none());}
                        if let Some(center)=result.eye_centers_camera_mm[eye] {
                            let ray=normalized3(sub3(result.target_camera_mm,center)).unwrap();
                            assert!(norm3(sub3(ray,result.eye_gaze_directions[eye].unwrap()))<1e-12);
                        }
                    }
                    if case=="outer-only" {assert!(!result.posterior.as_ref().unwrap().supports_direction(0));}
                    eprintln!("arc-mixture-support {}",serde_json::json!({"case":case,"variant":variant,
                        "seed":seed.to_string(),"candidate":marginalize_arc_alternatives,"available":true,
                        "target":result.target_camera_mm,"cost":result.robust_cost,
                        "gaze_error_degrees":std::array::from_fn::<_,2,_>(|eye|result.eye_gaze_directions[eye]
                            .map(|_|angular_error(&result,&fixture,eye))),
                        "arc_alternative_marginals":result.arc_alternative_marginals,
                        "posterior":result.posterior.as_ref().unwrap().json(),
                        "contract":"Independently forward-projected 3D circles with correlated adjacent-edge hypotheses. Both arms have the same observations, original conic search hints, priors and shared fixation. Duplicate alternatives are not additional evidence; no native anatomical or calibrated probability claim."}));
                }
            }
        }
    }
}

fn raw_outer_support_cases_diagnostic(position:bool) {
    use crate::outline_conic_segments::sparse_evidence::{OwnedBoundaryArc,OwnedConicHint,OwnedRoiEvidence};
    use crate::outline_conic_segments::sparse_evidence::uncertainty::{measure_outer_spread,measure_outer_position};
    for case in ["complete-pupil","complete-inner","partial-pupil","partial-inner","complementary","outer-only"] {
        let (fixture,enabled)=mask_support_case_fixture(case);
        let render=Fixture::new(fixture.target);
        let packets=std::array::from_fn::<_,2,_>(|eye|OwnedRoiEvidence {
            exposure:fixture.exposures[eye],sensor_origin_px:fixture.origins[eye],dimensions_px:[420,280],
            detail_reliability:Some(fixture.detail[eye]),
            arcs:fixture.arcs[eye].iter().map(|a|OwnedBoundaryArc { level_sets_roi: None,evidence_group:a.group,kind:a.kind,
                points_roi_px:a.points.clone(),normal_band_half_width_px:a.band,outward_normals_roi:None,
                localization_sigma_px:None,detector_score:None}).collect(),
            conics:fixture.hints[eye].iter().map(|&(kind,ellipse_roi_px)|OwnedConicHint {
                kind,ellipse_roi_px,supporting_arc_indices:vec![]}).collect()});
        let mut variants=vec![("control",0.5,400.0,0.0),("sharp-bright",0.5,400.0,0.0),
            ("sharp-dim",0.5,30.0,0.0),("outer-blurred",8.0,400.0,0.0)];
        if position {variants.push(("outer-displaced",0.5,400.0,8.0));}
        for (variant,blur,amplitude,offset) in variants {
            let mut current=packets.clone();let mut receipts:[Vec<serde_json::Value>;2]=[Vec::new(),Vec::new()];
            if variant!="control" {
                for eye in 0..2 {
                    if !enabled[eye] {continue;}
                    let e=render.hints[eye].iter().find(|a|a.0==BoundaryKind::OuterLimbus).unwrap().1;
                    // An independent signed-distance edge around the known
                    // projected disk; blur changes the optical transition,
                    // while measured boundary positions stay fixed. This is
                    // not a SAM inference, real occluder or anatomical image.
                    let (sin,cos)=e.angle.sin_cos();
                    let raw=(0..420*280).map(|i| {
                        let x=(i%420) as f64-e.center.0;let y=(i/420) as f64-e.center.1;
                        let u=cos*x+sin*y;let v=-sin*x+cos*y;
                        let q=(u/e.major_radius).powi(2)+(v/e.minor_radius).powi(2);
                        let distance=crate::conic_solver::ellipse_residual(((i%420) as f64,(i/420) as f64),e)
                            *if q<1.0 {-1.0} else {1.0};
                        (150.0+amplitude*(1.0+((distance-offset)/blur).tanh())*0.5).round() as u16
                    }).collect::<Vec<_>>();
                    let before=&packets[eye];
                    receipts[eye]=if position {
                        measure_outer_position(&mut current[eye],&raw).into_iter().map(|r|serde_json::json!(r)).collect()
                    } else {
                        measure_outer_spread(&mut current[eye],&raw).into_iter().map(|r|serde_json::json!(r)).collect()
                    };
                    assert_eq!(format!("{:?}",current[eye].conics),format!("{:?}",before.conics));
                    assert_eq!(current[eye].exposure,before.exposure);
                    for (a,b) in current[eye].arcs.iter().zip(&before.arcs) {
                        assert_eq!(a.points_roi_px,b.points_roi_px);assert_eq!(a.evidence_group,b.evidence_group);
                        assert_eq!(a.normal_band_half_width_px,b.normal_band_half_width_px);
                        if a.kind!=BoundaryKind::OuterLimbus {assert_eq!(format!("{a:?}"),format!("{b:?}"));}
                    }
                }
            }
            let prepared=current.each_ref().map(|p|p.prepare());
            let request=JointConicRequest {eyes:std::array::from_fn(|eye|enabled[eye].then(||prepared[eye].evidence())),
                scene:&fixture.scene,maximum_hypotheses:24,maximum_refinements:16,
                maximum_source_skew_ns:0,exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
            for seed in [0xd1b5_4a32_d192_ed03,0x94c5_09a1_814f_753d,0x419b_79df_a2c7_5301] {
                let result=solve_joint_conic_distribution_diagnostic(request,1,
                    posterior::IntegrationConfig {seed,..posterior::IntegrationConfig::live()}).unwrap().remove(0);
                for eye in 0..2 {
                    if !enabled[eye] {assert!(result.eye_gaze_directions[eye].is_none());continue;}
                    let ray=normalized3(sub3(result.target_camera_mm,result.eye_centers_camera_mm[eye].unwrap())).unwrap();
                    assert!(norm3(sub3(ray,result.eye_gaze_directions[eye].unwrap()))<1e-12);
                }
                if case=="outer-only" {assert!(!result.posterior.as_ref().unwrap().supports_direction(0));}
                if position && case=="complete-pupil" && variant!="outer-displaced" {
                    assert!(result.posterior.as_ref().unwrap().supports_direction(0),
                        "a centered optical transition must preserve the ideal complete-pupil direction: {variant} seed {seed}");
                }
                eprintln!("{} {}",if position {"outer-position-support"} else {"outer-spread-support"},serde_json::json!({"case":case,"variant":variant,"seed":seed.to_string(),
                    "gaze_error_degrees":std::array::from_fn::<_,2,_>(|eye|enabled[eye].then(||angular_error(&result,&fixture,eye))),
                    "posterior":result.posterior.as_ref().unwrap().json(),"outer_measurements":receipts,
                    "contract":"Synthetic known 3D circles; fixed contour points with varied RAW outer transition width/brightness/offset, separately preserved inner/pupil factors. No native accuracy claim."}));
            }
        }
    }
}

#[test]
#[ignore = "compare test-only numerical admission on complete, partial, complementary and ambiguous synthetic support"]
fn same_eye_and_complementary_stereo_admission_diagnostic() {
    support_case_posterior_reference(true,false,false,None);
}

#[test]
#[ignore = "independent-seed comparison of analytic unobserved-inner integration on complete, partial, complementary and ambiguous geometry"]
fn same_eye_and_complementary_stereo_integrated_inner_diagnostic() {
    support_case_posterior_reference(false,true,false,None);
}

#[test]
#[ignore = "independent-seed global scene proposals on complete, partial, complementary and ambiguous geometry"]
fn same_eye_and_complementary_stereo_conditional_scene_diagnostic() {
    support_case_posterior_reference(false,true,true,None);
}

#[test]
#[ignore = "matched independent-replica integration and admission across complete/partial/complementary/mirror fixtures"]
fn same_eye_and_complementary_stereo_replicated_diagnostic() {
    for replicas in [1,4] {
        for precision in [false,true] {
            support_case_posterior_reference(precision,true,false,Some(replicas));
        }
    }
}

#[test]
#[ignore = "matched broader scene proposal and numerical precision on complete/partial/complementary/mirror fixtures"]
fn same_eye_and_complementary_stereo_scene_replicated_diagnostic() {
    for replicas in [1,4] {
        for precision in [false,true] {
            support_case_posterior_reference(precision,true,true,Some(replicas));
        }
    }
}

#[test]
#[ignore = "discarded-pilot tail components with paired numerical admission at live and larger diagnostic budgets"]
fn same_eye_and_complementary_stereo_tail_precision_diagnostic() {
    assert!(matches!(std::env::var("BUTTERCUP_POSTERIOR_TAIL_PROPOSAL").ok().as_deref(),
        Some("pilot" | "recenter" | "refit" | "conditional-refit" | "outlier-refit" | "boundary-refit" | "boundary-defensive" | "profile-affine" | "profile-quadratic")), "select a tail proposal experiment");
    assert_eq!(std::env::var("BUTTERCUP_POSTERIOR_REPLICA_ORIGINAL_DRAWS").ok().as_deref(),Some("1"));
    support_case_posterior_reference(true,true,false,Some(4));
}

#[test]
#[ignore = "independent annealed reference paths for complete, partial, complementary and mirror support"]
fn same_eye_and_complementary_stereo_annealed_reference() {
    assert!(std::env::var("BUTTERCUP_ANNEALED_STEPS").is_ok());
    assert_eq!(std::env::var("BUTTERCUP_POSTERIOR_TAIL_PROPOSAL").ok().as_deref(),Some("conditional-refit"));
    assert_eq!(std::env::var("BUTTERCUP_POSTERIOR_REPLICA_ORIGINAL_DRAWS").ok().as_deref(),Some("1"));
    support_case_posterior_reference(true,true,false,Some(4));
}

#[test]
fn explicit_boundary_proposals_keep_the_opposite_eye_and_original_density() {
    let fixture = Fixture::new([90.0, -170.0, 250.0]);
    fixture.with_request([true, true], 24, |request| {
        let model = Problem::new(request).unwrap();
        let mut p = model.initial;
        p[..TARGET_PARAMETERS].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
        let masks = posterior::boundary_relaxations(&model);
        assert_eq!(masks.len(), 6);
        let score = |m: &Problem<'_>| squared_norm(&m.residuals(&p, &m.select(&m.conics(&p).unwrap())).unwrap());
        let before = score(&model);
        let kinds = [BoundaryKind::PupillaryBoundary, BoundaryKind::OuterLimbus, BoundaryKind::InnerLimbus];
        for (index, mask) in masks.iter().enumerate() {
            assert_eq!(mask.len(), 1);
            let eye = index % 2;
            let kind = kinds[index / 2];
            assert!(model.groups[mask[0]].alternatives.iter().all(|a| a.eye == eye && a.kind == kind));
            let (relaxed, omitted) = posterior::with_relaxed_proposal_groups(&model, &p, mask).unwrap();
            assert_eq!(&omitted, mask);
            assert_eq!(relaxed.present, [true, true]);
            assert_eq!(relaxed.lower, model.lower);
            assert_eq!(relaxed.upper, model.upper);
            assert_eq!(relaxed.groups.iter().filter(|g|g.alternatives[0].eye != eye).count(), 3);
            assert_eq!(model.groups.len(), 6);
            assert_eq!(score(&model), before);
        }
    });
}

#[test]
fn outlier_proposal_keeps_original_pupil_penalty_and_scene_support() {
    for conflicting_pupil in [false, true] {
        let mut fixture = Fixture::new([90.0, -170.0, 250.0]);
        if conflicting_pupil {
            for arc in &mut fixture.arcs[0] {
                if arc.kind == BoundaryKind::PupillaryBoundary {
                    for p in &mut arc.points { p.0 += 80.0; }
                }
            }
        }
        fixture.with_request([true, true], 24, |request| {
            let model = Problem::new(request).unwrap();
            let mut p = model.initial;
            p[..TARGET_PARAMETERS].copy_from_slice(&model.target_chart.coordinates(fixture.target).unwrap());
            let original_cost = squared_norm(&model.residuals(&p, &model.select(&model.conics(&p).unwrap())).unwrap());
            let original_group_count = model.groups.len();
            let (relaxed, omitted) = posterior::without_pilot_outliers(&model, &p).unwrap();
            assert_eq!(omitted.len(), usize::from(conflicting_pupil));
            for &i in &omitted {
                assert!(model.groups[i].alternatives.iter().all(|a| a.eye == 0 && a.kind == BoundaryKind::PupillaryBoundary));
            }
            assert_eq!(relaxed.present, model.present);
            assert_eq!(relaxed.lower, model.lower);
            assert_eq!(relaxed.upper, model.upper);
            assert_eq!(relaxed.scales, model.scales);
            assert_eq!(model.groups.len(), original_group_count);
            assert_eq!(relaxed.groups.len() + omitted.len(), original_group_count);
            let relaxed_cost = squared_norm(&relaxed.residuals(&p, &relaxed.select(&relaxed.conics(&p).unwrap())).unwrap());
            let outlier_charge = omitted.iter().map(|&i| model.groups[i].weight * MAXIMUM_GROUP_COST).sum::<f64>();
            assert!((original_cost - relaxed_cost - outlier_charge).abs() < 1e-8);
            // The source objective still charges the complete pupil evidence.
            assert_eq!(original_cost, squared_norm(&model.residuals(&p, &model.select(&model.conics(&p).unwrap())).unwrap()));
            assert_eq!(relaxed.target(&p), model.target(&p));
        });
    }
}

#[test]
fn conditional_tail_refinement_keeps_one_fixation_and_original_model_support() {
    let fixture = Fixture::new([90.0, -170.0, 250.0]);
    for enabled in [[true, false], [true, true]] {
        fixture.with_request(enabled, 24, |request| {
            let model = Problem::new(request).unwrap();
            let (mut pilot, _, _) = model.seeds().into_iter()
                .filter_map(|p| model.refine(p))
                .min_by(|a,b| a.1.total_cmp(&b.1)).unwrap();
            // Perturb observed eye geometry without changing the common target.
            for eye in 0..2 {
                if enabled[eye] {
                    let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
                    pilot[k] += 0.3 * model.scales[k];
                    pilot[k + 3] += 0.2 * model.scales[k + 3];
                }
            }
            let pilot = model.project_step(pilot).unwrap();
            let bounds = (model.lower, model.upper);
            let cost_before = squared_norm(&model.residuals(&pilot, &model.select(&model.conics(&pilot).unwrap())).unwrap());
            let (refined, cost_after, steps) = posterior::conditional_refined_center(&model, pilot).unwrap();
            assert_eq!(&refined[..TARGET_PARAMETERS], &pilot[..TARGET_PARAMETERS]);
            assert!(cost_after < cost_before - 1e-5, "{enabled:?}: {cost_before} -> {cost_after}");
            assert!(steps > 0 && steps <= 6 * 4);
            assert_eq!((model.lower, model.upper), bounds);
            assert!((0..TARGET_PARAMETERS).all(|i| model.upper[i] > model.lower[i]));
            assert!(model.conics(&refined).is_some());
        });
    }
}

#[test]
fn posterior_batch_diagnostic_preserves_original_samples_and_stopping() {
    let mut fixture = Fixture::new([90.0, -170.0, 250.0]);
    fixture.arcs[0].retain(|a| a.kind != BoundaryKind::InnerLimbus);
    fixture.hints[0].retain(|a| a.0 != BoundaryKind::InnerLimbus);
    for early_stop in [false, true] {
        let mut outputs = Vec::new();
        for replicas in [1, 4] {
            let solution = fixture.solve_with_integration([true, false], 24,
                Some(posterior::IntegrationConfig {
                    budget: if early_stop { 8192 } else { 1031 },
                    early_stop, replicas, preserve_replica_draws: true,
                    marginalize_unobserved_inner: true,
                    global_proposal: true, conditional_nuisance: true,
                    ..Default::default()
                })).unwrap();
            let mut json = solution.posterior.as_ref().unwrap().json();
            json.as_object_mut().unwrap().remove("replicas");
            json.as_object_mut().unwrap().remove("replicate_direction_numerics");
            outputs.push(json);
        }
        assert_eq!(outputs[0], outputs[1], "same draws, weights, fitted mass and stopping; early_stop={early_stop}");
    }
}

fn support_case_posterior_reference(numerical_admission:bool,marginalize_unobserved_inner:bool,conditional_scene:bool,replicated_global:Option<usize>) {
    let annealed_reference=std::env::var("BUTTERCUP_ANNEALED_STEPS").ok().map(|steps|posterior::annealed::Config {
        steps:steps.parse().unwrap(),paths:std::env::var("BUTTERCUP_ANNEALED_PATHS").expect("set path count").parse().unwrap(),
    });
    let preserve_marginal_draws=std::env::var("BUTTERCUP_POSTERIOR_PAIRED_DRAWS").ok().as_deref()==Some("1");
    let preserve_replica_draws=std::env::var("BUTTERCUP_POSTERIOR_REPLICA_ORIGINAL_DRAWS").ok().as_deref()==Some("1");
    let tail_proposal=match std::env::var("BUTTERCUP_POSTERIOR_TAIL_PROPOSAL").ok().as_deref() {
        None=>posterior::TailProposal::Off,
        Some("pilot")=>posterior::TailProposal::PilotOnly,
        Some("recenter")=>posterior::TailProposal::Recenter,
        Some("refit")=>posterior::TailProposal::Refit,
        Some("conditional-refit")=>posterior::TailProposal::ConditionalRefit,
        Some("outlier-refit")=>posterior::TailProposal::OutlierRefit,
        Some("boundary-refit")=>posterior::TailProposal::BoundaryRefit,
        Some("boundary-defensive")=>posterior::TailProposal::BoundaryDefensive,
        Some("profile-affine")=>posterior::TailProposal::ProfileAffine,
        Some("profile-quadratic")=>posterior::TailProposal::ProfileQuadratic,
        _=>panic!("unknown tail proposal recipe"),
    };
    let mut cases = vec![
        "complete-same-eye",
        "separated-outer-pupil",
        "complementary-stereo",
        "outer-only-mirror",
    ];
    if annealed_reference.is_some() {
        cases.extend(["complete-outer-inner", "separated-outer-inner"]);
    }
    for case in cases {
        let mut fixture = Fixture::new([90.0, -170.0, 250.0]);
        let enabled = if case == "complementary-stereo" {
            [true, true]
        } else {
            [true, false]
        };
        for eye in 0..2 {
            let omitted = if case.ends_with("-inner") { BoundaryKind::PupillaryBoundary }
                else { BoundaryKind::InnerLimbus };
            fixture.arcs[eye].retain(|a| a.kind != omitted);
            fixture.hints[eye].retain(|a| a.0 != omitted);
        }
        if matches!(case, "separated-outer-pupil" | "separated-outer-inner" | "complementary-stereo") {
            for eye in 0..if enabled[1] { 2 } else { 1 } {
                fixture.arcs[eye].clear();
                fixture.hints[eye].clear();
                let phase = if eye == 0 { 0.1875 } else { 0.0 };
                fixture.add_arc(
                    eye,
                    BoundaryKind::OuterLimbus,
                    TAU * phase,
                    TAU * (phase + 0.125),
                    8,
                    0,
                );
                fixture.add_arc(
                    eye,
                    BoundaryKind::OuterLimbus,
                    TAU * (phase + 0.5),
                    TAU * (phase + 0.625),
                    8,
                    1,
                );
                let inner = if case.ends_with("-inner") { BoundaryKind::InnerLimbus }
                    else { BoundaryKind::PupillaryBoundary };
                fixture.add_arc(eye, inner, 0.0, TAU, 32, 2);
                fixture.hints[eye].retain(|(kind, _)| *kind == inner);
                if enabled[1] {
                    // Complementary observed coverage, without an artificial
                    // third semantic boundary unavailable to SAM/Student.
                    fixture.arcs[eye].remove(if eye == 0 { 0 } else { 1 });
                    fixture.arcs[eye].last_mut().unwrap().points.truncate(16);
                }
            }
        } else if case == "outer-only-mirror" {
            fixture.arcs[0].retain(|a| a.kind == BoundaryKind::OuterLimbus);
            fixture.hints[0].retain(|a| a.0 == BoundaryKind::OuterLimbus);
        }
        let baseline = fixture.solve(enabled, 24).unwrap();
        let truth_rays = std::array::from_fn::<_, 2, _>(|eye| {
            enabled[eye].then(|| normalized3(sub3(fixture.target,
                fixture.scene.eyes[eye].unwrap().limbus_center.camera_mm)).unwrap())
        });
        let gaze_error_degrees = std::array::from_fn::<_, 2, _>(|eye| {
            Some(dot3(truth_rays[eye]?, baseline.eye_gaze_directions[eye]?)
                .clamp(-1.0, 1.0).acos().to_degrees())
        });
        eprintln!("support-case-fit {}", serde_json::json!({"case":case,
            "truth_target_camera_mm":fixture.target,"selected_target_camera_mm":baseline.target_camera_mm,
            "truth_gaze_directions":truth_rays,"selected_gaze_directions":baseline.eye_gaze_directions,
            "gaze_error_degrees":gaze_error_degrees,"modeled_eyes":enabled,
            "contract":"Independent noiseless 3D circle forward fixture; conditional synthetic geometry, not native gaze accuracy."}));
        if std::env::var("BUTTERCUP_POSTERIOR_REFERENCE_FITS_ONLY").ok().as_deref()==Some("1") {
            continue;
        }
        for seed in [
            0xd1b5_4a32_d192_ed03,
            0x94c5_09a1_814f_753d,
            0x419b_79df_a2c7_5301,
        ] {
            let budgets=if annealed_reference.is_some() {vec![8192]}
                else if numerical_admission {vec![8192,65536]} else {vec![8192,65536,1048576]};
            for budget in budgets {
                let solution = fixture
                    .solve_with_integration(
                        enabled,
                        24,
                        Some(posterior::IntegrationConfig {
                            budget,
                            seed,
                            early_stop: budget == 8192,
                            trace_tail: true,
                            numerical_admission,
                            tail_proposal,
                            marginalize_unobserved_inner,
                            preserve_marginal_draws,
                            global_proposal: conditional_scene || replicated_global.is_some(),
                            conditional_global: conditional_scene,
                            conditional_nuisance: conditional_scene || replicated_global.is_some(),
                            replicas: replicated_global.unwrap_or(1),
                            preserve_replica_draws,
                            annealed_reference,
                            ..Default::default()
                        }),
                    )
                    .unwrap();
                assert_eq!(solution.target_camera_mm, baseline.target_camera_mm);
                assert_eq!(solution.ellipses_roi_px, baseline.ellipses_roi_px);
                let p = solution.posterior.as_ref().unwrap();
                eprintln!(
                    "posterior-case {case} {seed} {budget}: {} ESS {:.1} {:?}",
                    p.status, p.effective_samples, p.gaze_radius_90_degrees
                );
                eprintln!("support-case-numerics {}",serde_json::json!({"case":case,"seed":seed.to_string(),
                    "budget":budget,"numerical_admission":numerical_admission,"conditional_scene":conditional_scene,"marginalize_unobserved_inner":marginalize_unobserved_inner,"preserve_marginal_draws":preserve_marginal_draws,"preserve_replica_draws":preserve_replica_draws,"tail_proposal":format!("{tail_proposal:?}"),"posterior":p.json()}));
                if tail_proposal==posterior::TailProposal::Off {assert_eq!(p.pilot_samples,0);}
                else {
                    assert!(p.pilot_samples>0 && p.pilot_samples<=2048 && p.pilot_samples<=budget/4);
                    assert!(p.feasible_samples<=p.samples-p.pilot_samples);
                    assert!(p.adapted_proposals<=4);
                    if tail_proposal==posterior::TailProposal::PilotOnly {assert_eq!(p.adapted_proposals,0);}
                }
                assert_eq!(p.modeled_eyes,enabled);
                if !enabled[1] {assert!(!p.supports_direction(1));}
                if case=="complete-same-eye" {assert!(p.supports_direction(0));}
                if case=="outer-only-mirror" {assert!(!p.supports_direction(0));}
            }
        }
    }
}
