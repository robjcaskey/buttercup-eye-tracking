use super::*;
use crate::roi_evidence::{BoundaryArcObservation, ConicObservation, RoiId, SourceClock};
use std::f64::consts::TAU;

fn support(nominal: f64, half_width: f64, sigma: f64) -> ScalarSupport {
    ScalarSupport { nominal, minimum: nominal-half_width, maximum: nominal+half_width, sigma }
}

fn scene() -> JointScenePrior {
    JointScenePrior {
        camera: PinholeCamera { focal_px: [3200.0,3200.0], principal_px: [4000.0,3000.0] },
        eyes: [-32.0,32.0].map(|x| Some(EyeScenePrior {
            limbus_center: PositionSupport { camera_mm: [x,0.0,-350.0],
                sigma_mm: [2.0,2.0,20.0], maximum_displacement_mm: [8.0,8.0,60.0],
                transverse_frame:TransversePositionFrame::Cartesian },
            radii_mm: [support(6.0,1.0,0.6),support(5.5,1.0,0.6),support(2.4,1.0,0.6)],
            pupil_inward_depth_mm: support(0.6,0.2,0.15),
            pupil_decentration_sigma_mm: 0.05, pupil_maximum_decentration_mm: 0.15,
            effective_pivot: None, limbus_to_pivot_mm: 9.0,
            surface_axis_alignment:None,
        })),
        target_reference_camera_mm: [0.0,0.0,-350.0],
        fixation_forward_mm: support(600.0,350.0,300.0),
        target_seed_camera_mm: None, maximum_gaze_slope: 1.5,
        interocular_distance_mm: Some(support(64.0,12.0,8.0)),
    }
}

fn exposure(eye: usize) -> ExposureKey {
    ExposureKey { roi: RoiId(eye as u32+1), clock: SourceClock {domain:1,epoch:5},
        sequence: 500, timestamp_ns: 1_000_000_000 }
}

// Independent forward generator: construct actual 3D circle points, then
// perspective-project each point. It does NOT use the solver's conic matrix.
fn ring_points(camera: PinholeCamera, center: [f64;3], normal: [f64;3], radius: f64,
               origin: [u32;2], begin: f64, end: f64, count: usize) -> Vec<(f64,f64)> {
    let u = normalized3(cross3([0.0,1.0,0.0],normal)).unwrap();
    let v = cross3(normal,u);
    (0..count).map(|i| {
        let phase = begin+(end-begin)*i as f64/(count-1) as f64;
        let p = add3(center,add3(scale3(u,radius*phase.cos()),scale3(v,radius*phase.sin())));
        let projected = camera.project(p).unwrap();
        (projected[0]-origin[0] as f64, projected[1]-origin[1] as f64)
    }).collect()
}

struct OwnedArc { group:u32, kind:BoundaryKind, points:Vec<(f64,f64)>, band:f64 }
struct Fixture {
    scene: JointScenePrior, target:[f64;3], origins:[[u32;2];2],
    arcs:[Vec<OwnedArc>;2], hints:[Vec<(BoundaryKind,Ellipse)>;2],
    detail:[f64;2], exposures:[ExposureKey;2],
}

impl Fixture {
    fn new(target: [f64;3]) -> Self {
        let scene = scene();
        let origins = [[3500,2850],[4100,2850]];
        let mut f = Self { scene,target,origins,arcs:[Vec::new(),Vec::new()],hints:[Vec::new(),Vec::new()],
            detail:[1.0;2],exposures:[exposure(0),exposure(1)] };
        for eye in 0..2 {
            for kind in [BoundaryKind::OuterLimbus,BoundaryKind::InnerLimbus,BoundaryKind::PupillaryBoundary] {
                f.add_arc(eye,kind,0.0,TAU,16,boundary_index(kind).unwrap() as u32);
            }
        }
        f
    }

    fn add_arc(&mut self, eye:usize, kind:BoundaryKind, begin:f64,end:f64,count:usize,group:u32) {
        let prior = self.scene.eyes[eye].unwrap();
        let center = prior.limbus_center.camera_mm;
        let gaze = normalized3(sub3(self.target,center)).unwrap();
        let u=normalized3(cross3([0.0,1.0,0.0],gaze)).unwrap();
        let v=cross3(gaze,u);
        let angles=prior.surface_axis_alignment.map(|a|a.nominal_radians).unwrap_or([0.0;2]);
        let normal = normalized3(add3(gaze,add3(scale3(u,angles[0].tan()),scale3(v,angles[1].tan())))).unwrap();
        let center = if kind == BoundaryKind::PupillaryBoundary { sub3(center,scale3(normal,prior.pupil_inward_depth_mm.nominal)) } else { center };
        let radius = prior.radii_mm[boundary_index(kind).unwrap()].nominal;
        let ellipse = ProjectedCircle::project(self.scene.camera,center,normal,radius,self.origins[eye]).unwrap().ellipse().unwrap();
        self.hints[eye].push((kind,ellipse));
        self.arcs[eye].push(OwnedArc { group,kind,points:ring_points(self.scene.camera,center,normal,radius,self.origins[eye],begin,end,count),band:0.0 });
    }

    fn solve(&self, enabled:[bool;2], budget:usize) -> Result<JointConicSolution,JointConicUnavailable> {
        let arcs = self.arcs.each_ref().map(|arcs| arcs.iter().map(|a| BoundaryArcObservation {
            evidence_group:a.group,kind:a.kind,points_roi_px:&a.points,normal_band_half_width_px:Some(a.band),detector_score:None,
        }).collect::<Vec<_>>());
        let conics = self.hints.each_ref().map(|hints| hints.iter().map(|&(kind,ellipse_roi_px)| ConicObservation {
            kind,ellipse_roi_px,supporting_arc_indices:&[],residual_px:None,
        }).collect::<Vec<_>>());
        let eyes = std::array::from_fn(|eye| enabled[eye].then_some(RoiConicEvidence {
            exposure:self.exposures[eye],sensor_origin_px:self.origins[eye],dimensions_px:[420,280],
            arcs:&arcs[eye],conics:&conics[eye],detail_reliability:Some(self.detail[eye]),
        }));
        solve_joint_conics(JointConicRequest { eyes,scene:&self.scene,maximum_hypotheses:budget,maximum_refinements:16,
            maximum_source_skew_ns:0,exposure_uncertainty_ns:0,motion_bound_px_per_second:0.0 })
    }
}

fn angular_error(solution: &JointConicSolution, fixture: &Fixture, eye:usize) -> f64 {
    let truth = normalized3(sub3(fixture.target,fixture.scene.eyes[eye].unwrap().limbus_center.camera_mm)).unwrap();
    dot3(solution.eye_normals[eye].unwrap(),truth).clamp(-1.0,1.0).acos().to_degrees()
}

#[test]
fn exact_perspective_circle_and_conic_agree_off_axis_with_native_crop() {
    let camera = scene().camera;
    for normal in [[0.3,-0.4,0.866025403784],[-0.2,0.5,0.842614977],[0.0,0.0,1.0]] {
        let normal = normalized3(normal).unwrap();
        for center in [[-35.0,-20.0,-300.0],[40.0,15.0,-550.0]] {
            let origin = [3600,2700];
            let conic = ProjectedCircle::project(camera,center,normal,6.0,origin).unwrap();
            let ellipse = conic.ellipse().unwrap();
            for p in ring_points(camera,center,normal,6.0,origin,0.0,TAU,97) {
                assert!(conic.residual_px(p).abs()<1.0e-7,"{p:?} {:?}",conic);
                assert!(crate::conic_solver::ellipse_residual(p,ellipse)<1.0e-7);
            }
        }
    }
}

#[test]
fn camera_facing_is_a_hard_projection_constraint() {
    let camera = scene().camera;
    assert!(ProjectedCircle::project(camera,[0.0,0.0,-300.0],[0.0,0.0,-1.0],6.0,[0,0]).is_none());
    assert!(ProjectedCircle::project(camera,[0.0,0.0,300.0],[0.0,0.0,1.0],6.0,[0,0]).is_none());
}

#[test]
fn nested_radius_projection_moves_coupled_radii_without_relaxing_frozen_bounds() {
    let q=project_nested_radii([4.1,5.6,2.4],[4.0,3.5,0.6],[8.0,7.5,4.5],[1.0;3]).unwrap();
    assert!((q[0]-4.85).abs()<1.0e-10);
    assert_eq!(q[0],q[1]);
    assert!((q[2]-2.4).abs()<1.0e-10);
    let q=project_nested_radii([4.1,5.6,6.0],[4.0,3.5,0.6],[4.4,7.5,4.5],[1.0;3]).unwrap();
    assert!(q[0]<=4.4&&q[1]<=q[0]&&q[2]<q[1]);
    assert!(project_nested_radii([4.0,5.6,2.4],[3.0,5.0,0.6],[4.0,7.5,4.5],[1.0;3]).is_none());
}

#[test]
fn joint_fixation_recovers_both_vertical_signs_from_mixed_boundary_samples() {
    for y in [-180.0,180.0] {
        let fixture = Fixture::new([95.0,y,250.0]);
        let solution = fixture.solve([true,true],24).unwrap();
        eprintln!("target={:?} fit={:?} errors={:?} cost={} margin={:?}",fixture.target,solution.target_camera_mm,
            [angular_error(&solution,&fixture,0),angular_error(&solution,&fixture,1)],solution.robust_cost,solution.alternative_cost_margin);
        assert_eq!(solution.contributing_eyes,[true,true]);
        assert!(angular_error(&solution,&fixture,0)<1.0);
        assert!(angular_error(&solution,&fixture,1)<1.0);
        assert!(norm3(sub3(solution.target_camera_mm,fixture.target))<25.0);
        for eye in 0..2 {
            let n = normalized3(sub3(solution.target_camera_mm,solution.eye_centers_camera_mm[eye].unwrap())).unwrap();
            assert!(norm3(sub3(n,solution.eye_normals[eye].unwrap()))<1.0e-12);
        }
    }
}

#[test]
fn correlated_alternatives_choose_compatible_segments_without_double_voting() {
    let mut fixture = Fixture::new([70.0,-150.0,250.0]);
    let baseline = fixture.solve([true,true],24).unwrap();
    let original = &fixture.arcs[1][2];
    let bad = OwnedArc { group:original.group,kind:original.kind,
        points:original.points.iter().map(|&(x,y)| (x+17.0,y+32.0)).collect(),band:0.0 };
    fixture.arcs[1].insert(0,bad);
    let solution = fixture.solve([true,true],24).unwrap();
    assert!(norm3(sub3(solution.target_camera_mm,baseline.target_camera_mm))<1.0e-8);
    assert_eq!(solution.arcs.len(),6);
    assert!(solution.arcs.iter().filter(|a| a.exposure.roi==RoiId(2)).all(|a| a.arc_index!=0));
}

#[test]
fn clocks_missing_eyes_and_budgets_have_explicit_semantics() {
    let mut fixture = Fixture::new([50.0,-150.0,250.0]);
    assert!(matches!(fixture.solve([false,false],24),Err(JointConicUnavailable::NoBoundaryEvidence)));
    assert!(matches!(fixture.solve([true,true],0),Err(JointConicUnavailable::InvalidRequest)));
    fixture.exposures[1].clock.epoch += 1;
    assert!(matches!(fixture.solve([true,true],24),Err(JointConicUnavailable::IncompatibleClocks)));
    let mono = fixture.solve([true,false],4).unwrap();
    assert_eq!(mono.contributing_eyes,[true,false]);
    assert!(mono.eye_normals[1].is_none());
    assert!(mono.hypotheses_evaluated<=4);
    assert!(mono.refinement_steps<=4*16*4);
    fixture.exposures[1].clock = fixture.exposures[0].clock;
    fixture.exposures[1].timestamp_ns += 1;
    assert!(matches!(fixture.solve([true,true],24),Err(JointConicUnavailable::ExcessiveSourceSkew)));
}

#[test]
fn roi_reframe_is_coordinate_change_not_an_eye_motion_measurement() {
    let mut fixture = Fixture::new([80.0,-130.0,250.0]);
    let before = fixture.solve([true,true],24).unwrap();
    for eye in 0..2 {
        let shift = if eye==0 { [48,26] } else { [18,64] };
        for axis in 0..2 { fixture.origins[eye][axis] += shift[axis]; }
        for arc in &mut fixture.arcs[eye] { for p in &mut arc.points { p.0 -= shift[0] as f64; p.1 -= shift[1] as f64; } }
        for (_,ellipse) in &mut fixture.hints[eye] { ellipse.center.0 -= shift[0] as f64; ellipse.center.1 -= shift[1] as f64; }
    }
    let after = fixture.solve([true,true],24).unwrap();
    assert!(norm3(sub3(before.target_camera_mm,after.target_camera_mm))<0.01,
        "before={:?} after={:?}",before.target_camera_mm,after.target_camera_mm);
}

#[test]
fn complementary_partial_arcs_solve_one_target_not_an_average_of_monocular_targets() {
    let mut fixture = Fixture::new([80.0,-170.0,250.0]);
    fixture.arcs = [Vec::new(),Vec::new()];
    fixture.hints = [Vec::new(),Vec::new()];
    // One side has only a short outer-limbus fragment and a weak pupil arc;
    // the other has a complementary inner limbus and a sharper pupil arc.
    fixture.add_arc(0,BoundaryKind::OuterLimbus,-0.5,0.5,11,0);
    fixture.add_arc(0,BoundaryKind::PupillaryBoundary,2.4,3.8,11,1);
    fixture.arcs[0][1].band = 5.0;
    fixture.add_arc(1,BoundaryKind::InnerLimbus,1.2,2.4,11,0);
    fixture.add_arc(1,BoundaryKind::PupillaryBoundary,3.0,5.0,11,1);
    for (i,p) in fixture.arcs[0][0].points.iter_mut().enumerate() { p.1 += 1.4*(i as f64*0.6).cos(); }
    for (i,p) in fixture.arcs[1][0].points.iter_mut().enumerate() { p.0 += 0.4*(i as f64*0.7).sin(); }
    let left = fixture.solve([true,false],24).unwrap();
    let right = fixture.solve([false,true],24).unwrap();
    let joint = fixture.solve([true,true],24).unwrap();
    let prohibited_average = scale3(add3(left.target_camera_mm,right.target_camera_mm),0.5);
    let joint_error = norm3(sub3(joint.target_camera_mm,fixture.target));
    let average_error = norm3(sub3(prohibited_average,fixture.target));
    eprintln!("partial arcs mono={:?}/{:?} joint={:?} avg={:?} errors={joint_error}/{average_error}",
        left.target_camera_mm,right.target_camera_mm,joint.target_camera_mm,prohibited_average);
    assert_eq!(joint.contributing_eyes,[true,true]);
    assert!(norm3(sub3(joint.target_camera_mm,prohibited_average))>1.0);
    assert!(joint_error<average_error);
}

#[test]
fn defocus_weakens_conflicting_pupil_localization_without_erasing_the_other_eye() {
    let mut fixture = Fixture::new([90.0,-140.0,250.0]);
    // Shift the second eye's pupil section, not its whole observed iris.
    // This simulates a bad edge estimate; it isn't treated as a new gaze point.
    for p in &mut fixture.arcs[1][2].points { p.1 += 2.0; }
    let sharp = fixture.solve([true,true],24).unwrap();
    fixture.detail[1] = 0.05;
    let blurred = fixture.solve([true,true],24).unwrap();
    eprintln!("defocus errors {} -> {}",angular_error(&sharp,&fixture,0),angular_error(&blurred,&fixture,0));
    assert!(angular_error(&blurred,&fixture,0)<angular_error(&sharp,&fixture,0));
    assert!(blurred.arcs.iter().filter(|a| a.exposure.roi==RoiId(2)).all(|a| a.sigma_px>3.0));
    assert!(blurred.contributing_eyes[0]);
}

#[test]
fn strongest_signed_pupil_section_resolves_a_weak_opposing_cue_in_the_other_roi() {
    let mut fixture = Fixture::new([0.0,-160.0,250.0]);
    let opposite = Fixture::new([0.0,160.0,250.0]);
    fixture.arcs[0][2].points = opposite.arcs[0][2].points.clone();
    fixture.arcs[0][2].band = 7.0;
    let weak = fixture.solve([true,false],24).unwrap();
    let joint = fixture.solve([true,true],24).unwrap();
    eprintln!("opposing pupil weak {:?} joint {:?}",weak.target_camera_mm,joint.target_camera_mm);
    assert!(weak.target_camera_mm[1]>0.0);
    assert!(joint.target_camera_mm[1]<0.0);
    assert!(angular_error(&joint,&fixture,0)<2.0);
    assert_eq!(joint.contributing_eyes,[true,true]);
}

#[test]
fn uncertain_range_preserves_off_axis_viewing_ray_instead_of_pinning_metric_xy() {
    let mut prior=PositionSupport {camera_mm:[40.0,-80.0,-300.0],sigma_mm:[2.0,2.0,80.0],
        maximum_displacement_mm:[8.0,8.0,160.0],transverse_frame:TransversePositionFrame::AtNominalDepth};
    let moved=[60.0,-120.0,-450.0];
    assert_eq!(prior.displacement(moved),[0.0,0.0,-150.0]);
    prior.transverse_frame=TransversePositionFrame::Cartesian;
    assert_eq!(prior.displacement(moved),[20.0,-40.0,-150.0]);
}

#[test]
fn shared_fixation_ray_and_surface_normal_remain_distinct_with_explicit_axis_alignment() {
    let mut fixture=Fixture::new([70.0,-120.0,250.0]);
    fixture.arcs=[Vec::new(),Vec::new()];fixture.hints=[Vec::new(),Vec::new()];
    for eye in 0..2 {
        fixture.scene.eyes[eye].as_mut().unwrap().surface_axis_alignment=Some(SurfaceAxisAlignment {
            nominal_radians:if eye==0 {[-0.04,0.02]} else {[0.04,-0.01]},
            sigma_radians:[0.01;2],maximum_deviation_radians:[0.0;2],
        });
        for kind in [BoundaryKind::OuterLimbus,BoundaryKind::InnerLimbus,BoundaryKind::PupillaryBoundary] {
            fixture.add_arc(eye,kind,0.0,TAU,16,boundary_index(kind).unwrap() as u32);
        }
    }
    let solution=fixture.solve([true,true],24).unwrap();
    assert!(norm3(sub3(solution.target_camera_mm,fixture.target))<2.0);
    for eye in 0..2 {
        let expected=normalized3(sub3(fixture.target,fixture.scene.eyes[eye].unwrap().limbus_center.camera_mm)).unwrap();
        let gaze=solution.eye_gaze_directions[eye].unwrap();
        let surface=solution.eye_normals[eye].unwrap();
        assert!(norm3(sub3(expected,gaze))<0.002);
        assert!(norm3(sub3(surface,gaze))>0.025);
    }
}

#[test]
fn incompatible_arc_groups_cannot_drag_a_well_supported_partner() {
    let mut fixture=Fixture::new([70.0,-120.0,250.0]);
    for arc in &mut fixture.arcs[0] {for p in &mut arc.points {p.0+=65.0;p.1+=50.0;}}
    let solution=fixture.solve([true,true],24).unwrap();
    assert_eq!(solution.contributing_eyes,[false,true]);
    assert!(solution.arcs.iter().filter(|a|a.exposure.roi==RoiId(1)).all(|a|!a.used));
    assert!(angular_error(&solution,&fixture,1)<0.1);
}

#[test]
fn perspective_circle_normal_decomposition_recovers_off_axis_planes_and_both_mirrors() {
    for focal in [[3200.0,3200.0],[3200.0,4100.0]] {
        let camera=PinholeCamera {focal_px:focal,principal_px:[4000.0,3000.0]};
        for center in [[0.0,0.0,-300.0],[90.0,-40.0,-218.0],[-70.0,80.0,-410.0]] {
            for normal in [[0.2,0.45,0.87],[-0.3,-0.55,0.78],[0.0,0.0,1.0]] {
                let normal=normalized3(normal).unwrap();
                let origin=[3000,2300];
                let ellipse=ProjectedCircle::project(camera,center,normal,6.0,origin).unwrap().ellipse().unwrap();
                let normals=circle_normal_hypotheses(camera,ellipse,origin).unwrap();
                let error=normals.into_iter().map(|n|norm3(sub3(n,normal))).fold(f64::INFINITY,f64::min);
                assert!(error<1.0e-7,"center={center:?} normal={normal:?} solutions={normals:?} error={error}");
                assert!(normals.into_iter().all(|n|n[2]>0.0));
            }
        }
    }
}
