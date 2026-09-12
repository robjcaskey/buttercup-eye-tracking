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
        fixation_axial_distance_mm: support(600.0,350.0,300.0),
        target_seed_camera_mm: None, secondary_target_seed_camera_mm: None, maximum_gaze_slope: 1.5,
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
            outward_normals_roi:None,
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
fn a_large_observed_limbus_can_initialize_inside_a_broad_range_prior() {
    let mut fixture=Fixture::new([40.0,-120.0,260.0]);
    fixture.scene.camera.focal_px=[4000.0;2];
    fixture.origins[0]=[2560,960];
    fixture.scene.eyes[0].as_mut().unwrap().limbus_center.camera_mm=[-58.0,-89.0,-190.0];
    fixture.arcs=[Vec::new(),Vec::new()];fixture.hints=[Vec::new(),Vec::new()];
    for (group,(begin,end)) in [(0.1,0.7),(1.1,1.7),(3.2,3.8),(4.1,4.7)].into_iter().enumerate() {
        fixture.add_arc(0,BoundaryKind::OuterLimbus,begin,end,12,group as u32);
    }
    let truth=fixture.hints[0][0].1;
    let prior=fixture.scene.eyes[0].as_mut().unwrap();
    prior.limbus_center=PositionSupport {camera_mm:scale3(prior.limbus_center.camera_mm,350.0/190.0),
        sigma_mm:[2.5,2.5,122.5],maximum_displacement_mm:[10.0,10.0,245.0],
        transverse_frame:TransversePositionFrame::AtNominalDepth};
    prior.radii_mm=[support(6.0,2.0,1.0),support(5.6,1.9,1.0),support(2.4,1.8,1.2)];
    fixture.scene.target_reference_camera_mm=prior.limbus_center.camera_mm;
    fixture.scene.interocular_distance_mm=None;
    let solution=fixture.solve([true,false],16).unwrap();
    let ellipse=solution.ellipses_roi_px[0][0].unwrap();
    let error=truth.dense_points(64).into_iter().map(|p|crate::conic_solver::ellipse_residual(p,ellipse).powi(2)).sum::<f64>();
    assert!((error/64.0).sqrt()<1.0,"an arbitrary range start must not censor good limbus arcs: {ellipse:?}");
    assert!(solution.arcs.iter().all(|a|a.used));
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
fn off_axis_visible_eye_does_not_hit_an_optical_axis_gaze_limit() {
    for sign in [-1.0,1.0] {
        let mut fixture=Fixture::new([0.0,-1000.0*sign,300.0]);
        fixture.scene.target_reference_camera_mm=[0.0,130.0*sign,-300.0];
        // The constructed target is about 1.28 m away, with 1.00 m axial
        // distance along this off-axis observer ray (not 600 mm optical Z).
        fixture.scene.fixation_axial_distance_mm=support(1000.0,700.0,500.0);
        fixture.arcs=[Vec::new(),Vec::new()];fixture.hints=[Vec::new(),Vec::new()];
        for eye in 0..2 {
            let center=[if eye==0 {-32.0} else {32.0},130.0*sign,-300.0];
            fixture.scene.eyes[eye].as_mut().unwrap().limbus_center.camera_mm=center;
            let sensor=fixture.scene.camera.project(center).unwrap();
            fixture.origins[eye]=[(sensor[0]-210.0).round() as u32,(sensor[1]-140.0).round() as u32];
            for kind in [BoundaryKind::OuterLimbus,BoundaryKind::InnerLimbus,BoundaryKind::PupillaryBoundary] {
                fixture.add_arc(eye,kind,0.0,TAU,16,boundary_index(kind).unwrap() as u32);
            }
            let normal=normalized3(sub3(fixture.target,center)).unwrap();
            assert!(normal[1].abs()/normal[2]>fixture.scene.maximum_gaze_slope,
                "the historical optical-axis bound must exclude this construction");
            assert!(dot3(normal,normalized3(scale3(center,-1.0)).unwrap())>0.7,
                "this is a visible convex eye, not a forbidden back-facing disk");
        }
        for enabled in [[true,true],[true,false],[false,true]] {
            let solution=fixture.solve(enabled,24).unwrap();
            for eye in 0..2 {if enabled[eye] {
                assert!(angular_error(&solution,&fixture,eye)<0.5,
                    "sign={sign} eye={eye} enabled={enabled:?} target={:?} truth={:?} error={} cost={}",solution.target_camera_mm,
                    fixture.target,angular_error(&solution,&fixture,eye),solution.robust_cost);
                assert!(solution.arcs.iter().filter(|a|a.exposure.roi==RoiId(eye as u32+1)).all(|a|a.used&&a.rms_px<0.2));
            }}
        }
    }
}

#[test]
fn viewpoint_ray_coordinates_round_trip_at_the_specified_axial_distance() {
    for origin in [[0.0,0.0,-300.0],[-110.0,140.0,-260.0],[110.0,-140.0,-260.0]] {
        let chart=ViewpointRayChart::new(origin).unwrap();
        for x in [-1.2,0.0,1.2] {for y in [-1.2,0.0,1.2] {
            let coordinates=[x,y,600.0_f64.ln()];
            let Some(target)=chart.target(coordinates) else {continue;};
            assert!((dot3(sub3(target,origin),chart.toward_camera)-600.0).abs()<1.0e-9);
            assert!((norm3(sub3(target,origin))-600.0*(1.0+x*x+y*y).sqrt()).abs()<1.0e-9);
            let recovered=chart.coordinates(target).unwrap();
            assert!(coordinates.into_iter().zip(recovered).all(|(a,b)|(a-b).abs()<1.0e-12));
            assert!(dot3(sub3(target,origin),chart.toward_camera)>0.0);
            if origin[0]==0.0&&origin[1]==0.0 {
                assert!(norm3(sub3(target,[600.0*x,600.0*y,origin[2]+600.0]))<1.0e-9,
                    "on-axis coordinates retain the historical exact parameterization");
            }
        }}
    }
}

#[test]
fn a_near_optical_horizon_does_not_turn_bounded_viewpoint_depth_into_infinite_range() {
    let chart=ViewpointRayChart::new([270.0,160.0,-350.0]).unwrap();
    for optical_forward in [1.0e-3,1.0e-4,1.0e-5] {
        let direction=normalized3([-0.66,-0.75,optical_forward]).unwrap();
        let forward=dot3(direction,chart.toward_camera);
        assert!(forward>0.5,"a camera-facing ray, not a hidden backside");
        let coordinates=[dot3(direction,chart.right)/forward,
            dot3(direction,chart.down)/forward,600.0_f64.ln()];
        assert!(coordinates[..2].iter().all(|v|v.abs()<1.5));
        let target=chart.target(coordinates).unwrap();
        let offset=sub3(target,chart.origin_camera_mm);
        assert!((dot3(offset,chart.toward_camera)-600.0).abs()<1.0e-9);
        assert!(norm3(offset)<600.0*(1.0+2.0*1.5_f64.powi(2)).sqrt(),
            "the same finite slope/depth envelope must bound metric range near the optical horizon");
        assert!(dot3(normalized3(offset).unwrap(),direction)>1.0-1.0e-12,
            "bounding the range must not silently reverse or clip the ray direction");
    }
}

#[test]
fn viewpoint_ray_chart_rejects_invalid_or_backward_targets_instead_of_flipping_them() {
    assert!(ViewpointRayChart::new([0.0,0.0,0.0]).is_none());
    assert!(ViewpointRayChart::new([0.0,0.0,300.0]).is_none());
    assert!(ViewpointRayChart::new([f64::NAN,0.0,-300.0]).is_none());
    let chart=ViewpointRayChart::new([0.0,130.0,-300.0]).unwrap();
    assert!(chart.target([0.0,-10.0,600.0_f64.ln()]).is_none(),
        "dividing by a negative camera-Z component would manufacture its antipode");
    assert!(chart.coordinates([0.0,130.0,-400.0]).is_none());
    assert!(chart.coordinates([0.0,1.0e6,1.0]).is_none());
    assert!(chart.target([f64::NAN,0.0,1.0]).is_none());
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
                let poses=circle_pose_hypotheses(camera,ellipse,origin).unwrap();
                let closest=poses.into_iter().min_by(|a,b|norm3(sub3(a.normal,normal)).total_cmp(&norm3(sub3(b.normal,normal)))).unwrap();
                assert!(norm3(sub3(scale3(closest.center_per_radius,6.0),center))<1.0e-6);
                for pose in poses {
                    let regenerated=ProjectedCircle::project(camera,scale3(pose.center_per_radius,6.0),pose.normal,6.0,origin).unwrap();
                    assert!(ellipse.dense_points(32).into_iter().all(|p|regenerated.residual_px(p).abs()<1.0e-7));
                }
            }
        }
    }
}

#[test]
fn polyline_information_is_geometric_not_the_number_of_fragments_or_points() {
    let points=[(0.0,0.0),(10.0,0.0),(30.0,0.0),(30.0,40.0)];
    let (weights,length)=polyline_quadrature(&points).unwrap();
    assert_eq!(length,70.0);
    assert!((weights.iter().sum::<f64>()-1.0).abs()<1.0e-12);
    let repeated=points.iter().flat_map(|p|std::iter::repeat_n(*p,4)).collect::<Vec<_>>();
    assert_eq!(polyline_quadrature(&repeated).unwrap().1,length);
    let split=polyline_quadrature(&points[..=2]).unwrap().1+polyline_quadrature(&points[2..]).unwrap().1;
    assert_eq!(split,length);
    let resampled=[(0.0,0.0),(1.0,0.0),(2.0,0.0),(20.0,0.0),(30.0,0.0),(30.0,20.0),(30.0,40.0)];
    assert_eq!(polyline_quadrature(&resampled).unwrap().1,length);
    assert!(polyline_quadrature(&[(2.0,3.0);10]).is_none());
}

#[test]
fn matching_points_do_not_override_opposite_measured_boundary_directions() {
    let points=[-0.2f64,0.0,0.2].map(|t|(10.0*t.cos(),10.0*t.sin())).to_vec();
    let (quadrature,length_px)=polyline_quadrature(&points).unwrap();
    let normals=points.iter().map(|&(x,y)|Some(BoundaryNormalObservation {
        unit_outward_roi:[x/10.0,y/10.0],angular_sigma_radians:0.2})).collect();
    let mut arc=SparseArc {eye:0,index:0,kind:BoundaryKind::OuterLimbus,boundary:0,group:0,
        points,outward_normals:normals,sigma:1.0,quadrature,length_px,weight:1.0};
    let conic=ProjectedCircle([1.0,0.0,1.0,0.0,0.0,-100.0]);
    assert!(arc.mean_cost(conic)<1.0e-20);
    for normal in arc.outward_normals.iter_mut().flatten() {normal.unit_outward_roi=normal.unit_outward_roi.map(|v|-v);}
    assert!(arc.mean_cost(conic)>MAXIMUM_GROUP_COST,
        "identical point positions do not make an inward or opposite-polarity boundary a compatible outer rim");
    for normal in arc.outward_normals.iter_mut().flatten() {normal.angular_sigma_radians=2.0;}
    assert!(arc.mean_cost(conic)<MAXIMUM_GROUP_COST,"uncertain direction must have weaker influence");
    arc.outward_normals.fill(None);
    assert!(arc.mean_cost(conic)<1.0e-20,"missing direction is not fabricated from the candidate conic");
}

#[test]
fn uncertain_contour_directions_are_a_compatibility_band_not_a_second_precise_fit() {
    let points=[-0.2f64,0.0,0.2].map(|t|(10.0*t.cos(),10.0*t.sin())).to_vec();
    let (quadrature,length_px)=polyline_quadrature(&points).unwrap();
    let (s,c)=0.3f64.sin_cos();
    let normals=points.iter().map(|&(x,y)|Some(BoundaryNormalObservation {
        unit_outward_roi:[(c*x-s*y)/10.0,(s*x+c*y)/10.0],angular_sigma_radians:0.2})).collect();
    let arc=SparseArc {eye:0,index:0,kind:BoundaryKind::OuterLimbus,boundary:0,group:0,
        points,outward_normals:normals,sigma:1.0,quadrature,length_px,weight:1.0};
    assert!(arc.mean_cost(ProjectedCircle([1.0,0.0,1.0,0.0,0.0,-100.0]))<1.0e-20,
        "contour position and tangent share pixels: do not chase a noisy direction within its two-sigma engineering allowance");
}

#[test]
fn numerical_linearization_cannot_turn_a_capped_arc_into_a_force() {
    let scene=scene();
    let center=scene.eyes[0].unwrap().limbus_center.camera_mm;
    let target=add3(scene.target_reference_camera_mm,[0.0,0.0,scene.fixation_axial_distance_mm.nominal]);
    let points=ring_points(scene.camera,center,normalized3(sub3(target,center)).unwrap(),6.0,[3500,2850],0.1,0.5,12);
    let arcs=[BoundaryArcObservation {evidence_group:0,kind:BoundaryKind::OuterLimbus,points_roi_px:&points,
        outward_normals_roi:None,normal_band_half_width_px:Some(0.0),detector_score:None}];
    let evidence=RoiConicEvidence {exposure:exposure(0),sensor_origin_px:[3500,2850],dimensions_px:[420,280],
        arcs:&arcs,conics:&[],detail_reliability:Some(1.0)};
    let mut problem=Problem::new(JointConicRequest {eyes:[Some(evidence),None],scene:&scene,
        maximum_hypotheses:16,maximum_refinements:12,maximum_source_skew_ns:0,
        exposure_uncertainty_ns:0,motion_bound_px_per_second:0.0}).unwrap();
    let base=problem.initial;
    let conic=problem.conics(&base).unwrap()[0][0].unwrap();
    let [a,b,c,d,e,_]=conic.0;
    let (s,co)=0.6f64.sin_cos();
    // The arc sits just outside the existing capped compatibility cost. A
    // finite-difference perturbation crosses that gate while its position and
    // direction residual components are entirely different vectors.
    problem.groups[0].alternatives[0].outward_normals=points.iter().map(|&(x,y)| {
        let u=2.0*a*x+b*y+d;let v=b*x+2.0*c*y+e;let length=u.hypot(v);
        Some(BoundaryNormalObservation {unit_outward_roi:[(co*u-s*v)/length,(s*u+co*v)/length],
            angular_sigma_radians:0.6/(BOUNDARY_DIRECTION_ALLOWANCE_SIGMAS+3.0+1.0e-8)})
    }).collect();
    let arc=&problem.groups[0].alternatives[0];
    assert!(arc.mean_cost(conic)>MAXIMUM_GROUP_COST);
    let trial=(0..PARAMETERS).flat_map(|i|[-1.0,1.0].map(move|direction|(i,direction))).find_map(|(i,direction)| {
        let mut q=base;q[i]+=direction*1.0e-4*problem.scales[i];
        if q[i]<problem.lower[i]||q[i]>problem.upper[i] {return None;}
        let trial_conic=problem.conics(&q)?[0][0]?;
        (arc.mean_cost(trial_conic)<MAXIMUM_GROUP_COST).then_some(q)
    }).expect("exercise a real finite-difference crossing, not merely a far rejected arc");
    let selected=problem.select(&problem.conics(&base).unwrap());
    let rejected=problem.rejected_groups(&problem.conics(&base).unwrap(),&selected);
    assert_eq!(rejected,vec![true]);
    let before=problem.residuals(&base,&selected).unwrap();
    let actual=problem.residuals(&trial,&selected).unwrap();
    let linearized=problem.residuals_with_rejection(&trial,&selected,Some(&rejected)).unwrap();
    let sample_terms=points.len()*2;
    assert!(before[..sample_terms].iter().zip(&actual).any(|(a,b)|(a-b).abs()>0.1),
        "a real objective evaluation must still reconsider previously rejected evidence");
    assert_eq!(&before[..sample_terms],&linearized[..sample_terms],
        "a rejected arc has constant cost and zero force during this local derivative; a new trial rechecks admission separately");
    let readmitted=problem.rejected_groups(&problem.conics(&trial).unwrap(),&selected);
    assert_eq!(readmitted,vec![false],"this is not persistent exclusion memory");
    let reverse=problem.residuals_with_rejection(&base,&selected,Some(&readmitted)).unwrap();
    assert!(reverse[..sample_terms].iter().skip(1).step_by(2).all(|&r|r.abs()>0.1),
        "an admitted arc retains its actual direction derivatives even if a numerical step crosses the cap");
}

#[test]
fn capped_negative_position_residuals_cannot_pull_ordinary_live_evidence() {
    let mut scene=scene();scene.eyes[0].as_mut().unwrap().limbus_center.camera_mm[0]=0.0;
    let center=scene.eyes[0].unwrap().limbus_center.camera_mm;
    let image_radius=6.0*scene.camera.focal_px[0]/(-center[2]);
    let gap=3.0*0.75*(1.0+1.0e-8);
    // For an observed radius r inside circle R, Sampson residual is
    // (r²-R²)/(2r). Choose r so the existing rejection cost is just exceeded.
    let observed_radius=(image_radius.hypot(gap)-gap)*(-center[2])/scene.camera.focal_px[0];
    let points=ring_points(scene.camera,center,[0.0,0.0,1.0],observed_radius,[3800,2850],0.1,0.5,12);
    let arcs=[BoundaryArcObservation {evidence_group:0,kind:BoundaryKind::OuterLimbus,points_roi_px:&points,
        outward_normals_roi:None,normal_band_half_width_px:Some(0.0),detector_score:None}];
    let evidence=RoiConicEvidence {exposure:exposure(0),sensor_origin_px:[3800,2850],dimensions_px:[420,280],
        arcs:&arcs,conics:&[],detail_reliability:Some(1.0)};
    let problem=Problem::new(JointConicRequest {eyes:[Some(evidence),None],scene:&scene,
        maximum_hypotheses:16,maximum_refinements:12,maximum_source_skew_ns:0,
        exposure_uncertainty_ns:0,motion_bound_px_per_second:0.0}).unwrap();
    let base=problem.initial;let selected=vec![0];
    let rejected=problem.rejected_groups(&problem.conics(&base).unwrap(),&selected);
    assert_eq!(rejected,vec![true]);
    let mut trial=base;trial[6]-=1.0e-4*problem.scales[6];
    assert_eq!(problem.rejected_groups(&problem.conics(&trial).unwrap(),&selected),vec![false]);
    let before=problem.residuals(&base,&selected).unwrap();
    let actual=problem.residuals(&trial,&selected).unwrap();
    assert!(before[..points.len()].iter().all(|&r|r>0.0));
    assert!(actual[..points.len()].iter().all(|&r|r<0.0),
        "the unfixed derivative crosses a residual sign discontinuity even without measured directions");
    let linearized=problem.residuals_with_rejection(&trial,&selected,Some(&rejected)).unwrap();
    assert_eq!(&before[..points.len()],&linearized[..points.len()]);
}

#[test]
fn joint_selection_uses_measured_direction_not_just_equal_point_alternatives() {
    use crate::outline_conic_segments::sparse_evidence::{OwnedBoundaryArc,OwnedConicHint,OwnedRoiEvidence};
    let fixture=Fixture::new([70.0,-130.0,250.0]);
    let mut packets=std::array::from_fn::<_,2,_>(|eye|OwnedRoiEvidence {
        exposure:fixture.exposures[eye],sensor_origin_px:fixture.origins[eye],dimensions_px:[420,280],detail_reliability:Some(1.0),
        arcs:fixture.arcs[eye].iter().map(|a|OwnedBoundaryArc {evidence_group:a.group,kind:a.kind,
            points_roi_px:a.points.clone(),outward_normals_roi:None,normal_band_half_width_px:0.0,detector_score:None}).collect(),
        conics:fixture.hints[eye].iter().map(|&(kind,ellipse_roi_px)|OwnedConicHint {kind,ellipse_roi_px,supporting_arc_indices:vec![]}).collect()});
    let e=fixture.hints[0][0].1;
    let (s,c)=e.angle.sin_cos();
    packets[0].arcs[0].outward_normals_roi=Some(packets[0].arcs[0].points_roi_px.iter().map(|&(x,y)| {
        let u=(c*(x-e.center.0)+s*(y-e.center.1))/e.major_radius.powi(2);
        let v=(-s*(x-e.center.0)+c*(y-e.center.1))/e.minor_radius.powi(2);
        let magnitude=u.hypot(v);
        Some(BoundaryNormalObservation {unit_outward_roi:[(c*u-s*v)/magnitude,(s*u+c*v)/magnitude],angular_sigma_radians:0.2})
    }).collect());
    let mut wrong=packets[0].arcs[0].clone();
    for n in wrong.outward_normals_roi.as_mut().unwrap().iter_mut().flatten() {n.unit_outward_roi=n.unit_outward_roi.map(|v|-v);}
    packets[0].arcs.insert(0,wrong);
    let prepared=packets.each_ref().map(OwnedRoiEvidence::prepare);
    let result=solve_joint_conics(JointConicRequest {eyes:prepared.each_ref().map(|p|Some(p.evidence())),scene:&fixture.scene,
        maximum_hypotheses:16,maximum_refinements:16,maximum_source_skew_ns:0,
        exposure_uncertainty_ns:0,motion_bound_px_per_second:0.0}).unwrap();
    let selected=result.arcs.iter().find(|a|a.exposure.roi==RoiId(1)&&a.evidence_group==0).unwrap();
    assert_eq!(selected.arc_index,1,"the first alternative has the same coordinates but contradicts measured outward direction");
    assert!(selected.used);assert_eq!(result.contributing_eyes,[true,true]);
    for eye in 0..2 {assert!(angular_error(&result,&fixture,eye)<1.0);}
}

#[test]
fn malformed_or_misaligned_boundary_directions_cannot_be_silently_ignored() {
    let fixture=Fixture::new([70.0,-130.0,250.0]);let points=&fixture.arcs[0][0].points;
    let good=BoundaryNormalObservation {unit_outward_roi:[1.0,0.0],angular_sigma_radians:0.2};
    for normals in [vec![Some(good);points.len()-1],
        vec![Some(BoundaryNormalObservation {unit_outward_roi:[0.0,0.0],..good});points.len()],
        vec![Some(BoundaryNormalObservation {angular_sigma_radians:0.0,..good});points.len()],
        vec![Some(BoundaryNormalObservation {angular_sigma_radians:f64::NAN,..good});points.len()]] {
        let arcs=[BoundaryArcObservation {evidence_group:0,kind:BoundaryKind::OuterLimbus,points_roi_px:points,
            outward_normals_roi:Some(&normals),normal_band_half_width_px:None,detector_score:None}];
        let eye=RoiConicEvidence {exposure:fixture.exposures[0],sensor_origin_px:fixture.origins[0],
            dimensions_px:[420,280],arcs:&arcs,conics:&[],detail_reliability:None};
        assert!(matches!(solve_joint_conics(JointConicRequest {eyes:[Some(eye),None],scene:&fixture.scene,
            maximum_hypotheses:16,maximum_refinements:16,maximum_source_skew_ns:0,
            exposure_uncertainty_ns:0,motion_bound_px_per_second:0.0}),Err(JointConicUnavailable::InvalidRequest)));
    }
}

#[test]
fn image_boundary_normals_do_not_fabricate_a_three_dimensional_mirror_sign() {
    let fixture=Fixture::new([70.0,-130.0,250.0]);let e=fixture.hints[0][0].1;
    let points=e.dense_points(16);let (s,c)=e.angle.sin_cos();
    let normals=points.iter().map(|&(x,y)| {
        let u=(c*(x-e.center.0)+s*(y-e.center.1))/e.major_radius.powi(2);
        let v=(-s*(x-e.center.0)+c*(y-e.center.1))/e.minor_radius.powi(2);
        Some(BoundaryNormalObservation {unit_outward_roi:[(c*u-s*v)/u.hypot(v),(s*u+c*v)/u.hypot(v)],angular_sigma_radians:0.2})
    }).collect();
    let (quadrature,length_px)=polyline_quadrature(&points).unwrap();
    let arc=SparseArc {eye:0,index:0,kind:BoundaryKind::OuterLimbus,boundary:0,group:0,
        points,outward_normals:normals,sigma:1.0,quadrature,length_px,weight:1.0};
    let poses=circle_pose_hypotheses(fixture.scene.camera,e,fixture.origins[0]).unwrap();
    assert!(norm3(sub3(poses[0].normal,poses[1].normal))>0.1,"exercise distinct 3D mirror branches");
    for pose in poses {
        let conic=ProjectedCircle::project(fixture.scene.camera,scale3(pose.center_per_radius,6.0),pose.normal,6.0,fixture.origins[0]).unwrap();
        assert!(arc.mean_cost(conic)<1.0e-12,"the identical projected boundary cannot distinguish these 3D branches");
    }
}

#[test]
fn many_short_pupil_fragments_cannot_outvote_a_long_well_supported_limbus() {
    let mut fixture=Fixture::new([70.0,-130.0,250.0]);
    fixture.arcs=[Vec::new(),Vec::new()];fixture.hints=[Vec::new(),Vec::new()];
    fixture.add_arc(0,BoundaryKind::OuterLimbus,0.0,TAU,64,0);
    let outer=fixture.hints[0][0].1;
    for sector in 0..8 {
        fixture.add_arc(0,BoundaryKind::PupillaryBoundary,sector as f64*TAU/8.0,(sector+1) as f64*TAU/8.0,8,100+sector);
        // A displaced shadow/glint boundary is not a second iris-center vote.
        for point in &mut fixture.arcs[0].last_mut().unwrap().points {point.0+=16.0;point.1+=10.0;}
    }
    let solution=fixture.solve([true,false],16).unwrap();
    let fitted=solution.ellipses_roi_px[0][0].unwrap();
    assert!(outer.dense_points(48).into_iter().all(|p|crate::conic_solver::ellipse_residual(p,fitted)<1.0));
    assert!(solution.arcs.iter().any(|a|a.kind==BoundaryKind::PupillaryBoundary&&!a.used));
}

#[test]
fn independently_scaled_seed_radii_do_not_make_every_joint_start_infeasible() {
    let mut fixture=Fixture::new([70.0,-130.0,250.0]);
    // Bad scale hypotheses on one defocused eye, with broad range support.
    // The other eye remains useful. The frozen center/IPD bounds do not move.
    for prior in fixture.scene.eyes.iter_mut().flatten() {
        prior.limbus_center.transverse_frame=TransversePositionFrame::AtNominalDepth;
        prior.limbus_center.sigma_mm[2]=122.5;
        prior.limbus_center.maximum_displacement_mm[2]=245.0;
    }
    for (_,hint) in &mut fixture.hints[1] {hint.major_radius*=1.7;hint.minor_radius*=1.7;}
    fixture.detail[1]=0.2;
    let solution=fixture.solve([true,true],24).unwrap();
    assert!(solution.contributing_eyes[0]);
    assert!(angular_error(&solution,&fixture,0)<1.0);
    let distance=norm3(sub3(solution.eye_centers_camera_mm[0].unwrap(),solution.eye_centers_camera_mm[1].unwrap()));
    let support=fixture.scene.interocular_distance_mm.unwrap();
    assert!(distance>=support.minimum&&distance<=support.maximum);
}

#[test]
fn raw_verified_partial_outline_without_a_fitted_ellipse_contributes_to_one_shared_target() {
    use crate::outline_conic_segments::partial_outline::{append_unfitted_outline_arcs,OutlineCandidate};
    use crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence;
    let mut fixture=Fixture::new([70.0,-130.0,250.0]);
    let ellipse=fixture.hints[0][0].1;
    let raw=(0..280).flat_map(|y|(0..420).map(move|x| {
        let (s,c)=ellipse.angle.sin_cos();
        let dx=x as f64-ellipse.center.0;let dy=y as f64-ellipse.center.1;
        if ((c*dx+s*dy)/ellipse.major_radius).hypot((-s*dx+c*dy)/ellipse.minor_radius)<=1.0 {150} else {550}
    })).collect::<Vec<_>>();
    let contour=fixture.arcs[0][0].points.clone();
    let dense=ring_points(fixture.scene.camera,fixture.scene.eyes[0].unwrap().limbus_center.camera_mm,
        normalized3(sub3(fixture.target,fixture.scene.eyes[0].unwrap().limbus_center.camera_mm)).unwrap(),6.0,
        fixture.origins[0],0.0,TAU,128);
    let clipped=dense.iter().map(|&(x,y)|(x.max(ellipse.center.0),y)).collect::<Vec<_>>();
    let mut packet=OwnedRoiEvidence {exposure:fixture.exposures[0],sensor_origin_px:fixture.origins[0],
        dimensions_px:[420,280],arcs:Vec::new(),conics:Vec::new(),detail_reliability:None};
    append_unfitted_outline_arcs(&mut packet,&raw,&[OutlineCandidate {points_roi_px:&clipped,detector_score:None}],ellipse.center,20);
    assert!(packet.arcs.len()>=2);
    // Deliberately discard every fitted-conic seed for this eye. Its only
    // contribution is the current RAW-supported incomplete boundary.
    fixture.hints[0].clear();
    fixture.arcs[0]=packet.arcs.iter().map(|a|OwnedArc {group:a.evidence_group,kind:a.kind,
        points:a.points_roi_px.clone(),band:a.normal_band_half_width_px}).collect();
    let joint=fixture.solve([true,true],24).unwrap();
    assert_eq!(joint.contributing_eyes,[true,true]);
    assert!(angular_error(&joint,&fixture,1)<2.0);
    let recovered=joint.ellipses_roi_px[0][0].unwrap();
    let error=contour.iter().map(|&p|crate::conic_solver::ellipse_residual(p,recovered).powi(2)).sum::<f64>();
    assert!((error/contour.len() as f64).sqrt()<3.0);
    for eye in 0..2 {
        let ray=normalized3(sub3(joint.target_camera_mm,joint.eye_centers_camera_mm[eye].unwrap())).unwrap();
        assert!(norm3(sub3(ray,joint.eye_gaze_directions[eye].unwrap()))<1.0e-10);
    }
}

#[test]
fn an_unlocalized_roi_cannot_force_a_good_eye_into_its_false_interocular_geometry() {
    let mut fixture=Fixture::new([70.0,-130.0,250.0]);
    // A skin/foreground detection is not the second eye. Its tight but wrong
    // ROI association would make every coupled initialization violate IPD.
    let prior=fixture.scene.eyes[1].as_mut().unwrap();
    prior.limbus_center.camera_mm[0]=160.0;
    prior.limbus_center.maximum_displacement_mm=[0.1;3];
    fixture.hints[1].clear();
    fixture.arcs[1]=vec![OwnedArc {group:40,kind:BoundaryKind::OuterLimbus,
        points:vec![(10.0,10.0),(12.0,10.2),(14.0,10.3)],band:8.0}];
    let result=fixture.solve([true,true],16).unwrap();
    assert_eq!(result.modeled_eyes,[true,false]);
    assert_eq!(result.contributing_eyes,[true,false]);
    assert!(result.eye_centers_camera_mm[1].is_none()&&result.ellipses_roi_px[1][0].is_none());
    assert!(result.unlocalized_eye_cost[1]>0.0);
    assert!(angular_error(&result,&fixture,0)<1.0);
    assert!(result.hypotheses_evaluated<=16);
    assert!(result.refinement_steps<=16*16*4);
    // Correlated detector alternatives cannot multiply the rejection penalty.
    for _ in 0..8 {fixture.arcs[1].push(OwnedArc {group:40,kind:BoundaryKind::OuterLimbus,
        points:vec![(10.0,10.0),(12.0,10.2),(14.0,10.3)],band:8.0});}
    let repeated=fixture.solve([true,true],16).unwrap();
    assert_eq!(repeated.unlocalized_eye_cost,result.unlocalized_eye_cost);
    assert!(norm3(sub3(repeated.target_camera_mm,result.target_camera_mm))<1.0e-8);
}

#[test]
fn useful_two_eye_evidence_is_not_replaced_by_a_free_single_eye_hypothesis() {
    let fixture=Fixture::new([70.0,-130.0,250.0]);
    for budget in [8,16,24] {
        let result=fixture.solve([true,true],budget).unwrap();
        assert_eq!(result.modeled_eyes,[true,true]);
        assert_eq!(result.contributing_eyes,[true,true]);
        assert_eq!(result.unlocalized_eye_cost,[0.0;2]);
        assert!(result.hypotheses_evaluated<=budget);
    }
}

#[test]
fn dominated_unlocalized_models_return_their_starts_to_the_joint_search() {
    let fixture=Fixture::new([70.0,-130.0,250.0]);
    let result=fixture.solve([true,true],16).unwrap();
    assert!(result.robust_cost<0.01,"current evidence already supports a near-zero joint cost");
    assert_eq!(result.hypotheses_by_association,[16,0,0],
        "omitting either strongly supported ROI has an irreducible cost above the current fit; do not halve the useful joint search");
    assert_eq!(result.hypotheses_evaluated,16);
}

#[test]
fn misleading_first_conic_does_not_replace_strong_two_eye_boundary_support() {
    for eye in 0..2 {
        let mut fixture=Fixture::new([70.0,-130.0,250.0]);
        let truth=fixture.hints[eye][0].1;
        let prior=fixture.scene.eyes[eye].as_mut().unwrap();
        prior.limbus_center.sigma_mm=[6.0,6.0,90.0];
        prior.limbus_center.maximum_displacement_mm=[16.0,16.0,200.0];
        let mut bad=truth;
        bad.center.0+=80.0;bad.center.1+=30.0;
        bad.major_radius*=0.6;bad.minor_radius*=0.6;
        fixture.hints[eye].insert(0,(BoundaryKind::OuterLimbus,bad));
        let result=fixture.solve([true,true],24).unwrap();
        assert_eq!(result.contributing_eyes,[true,true],"a bad first seed must not erase directly observed good arcs");
        let fit=result.ellipses_roi_px[eye][0].unwrap();
        let rms=(truth.dense_points(64).iter().map(|&p|crate::conic_solver::ellipse_residual(p,fit).powi(2)).sum::<f64>()/64.0).sqrt();
        assert!(rms<1.0,"eye={eye} actual boundary error={rms} result={fit:?}");
    }
}

#[test]
fn a_secondary_circle_seed_carries_its_own_center_and_metric_radius() {
    let fixture=Fixture::new([70.0,-130.0,250.0]);
    let truth=fixture.hints[0][0].1;
    let mut wrong=truth;wrong.major_radius*=1.2;wrong.minor_radius*=1.2;wrong.center.0+=15.0;
    let hints=[wrong,truth].map(|ellipse_roi_px|ConicObservation {kind:BoundaryKind::OuterLimbus,
        ellipse_roi_px,supporting_arc_indices:&[0],residual_px:None});
    let arcs=[BoundaryArcObservation {evidence_group:0,kind:BoundaryKind::OuterLimbus,
        points_roi_px:&fixture.arcs[0][0].points,outward_normals_roi:None,normal_band_half_width_px:Some(0.0),detector_score:None}];
    let evidence=RoiConicEvidence {exposure:fixture.exposures[0],sensor_origin_px:fixture.origins[0],
        dimensions_px:[420,280],arcs:&arcs,conics:&hints,detail_reliability:Some(1.0)};
    let problem=Problem::new(JointConicRequest {eyes:[Some(evidence),None],scene:&fixture.scene,
        maximum_hypotheses:24,maximum_refinements:16,maximum_source_skew_ns:0,
        exposure_uncertainty_ns:0,motion_bound_px_per_second:0.0}).unwrap();
    let k=TARGET_PARAMETERS;
    assert!(problem.seeds().iter().any(|p|(p[k+2]+350.0).abs()<1.0e-5&&(p[k+3]-6.0).abs()<1.0e-5),
        "normal-only starts remain stuck in the first conic's center/range geometry");
}

#[test]
fn previous_targets_are_competing_initializations_not_averaged_points_or_extra_residuals() {
    let mut fixture=Fixture::new([70.0,-130.0,250.0]);
    let targets=[[-100.0,80.0,250.0],[150.0,-120.0,250.0]];
    fixture.scene.target_seed_camera_mm=Some(targets[0]);
    fixture.scene.secondary_target_seed_camera_mm=Some(targets[1]);
    let arcs=[BoundaryArcObservation {evidence_group:0,kind:BoundaryKind::OuterLimbus,
        points_roi_px:&fixture.arcs[0][0].points,outward_normals_roi:None,normal_band_half_width_px:Some(0.0),detector_score:None}];
    let evidence=RoiConicEvidence {exposure:fixture.exposures[0],sensor_origin_px:fixture.origins[0],
        dimensions_px:[420,280],arcs:&arcs,conics:&[],detail_reliability:Some(1.0)};
    let request=JointConicRequest {eyes:[Some(evidence),None],scene:&fixture.scene,
        maximum_hypotheses:8,maximum_refinements:12,maximum_source_skew_ns:0,
        exposure_uncertainty_ns:0,motion_bound_px_per_second:0.0};
    let problem=Problem::new(request).unwrap();
    let seeds=problem.seeds();
    assert!(seeds.len()<=8);
    for i in 0..2 {assert!(norm3(sub3(problem.target(&seeds[i]).unwrap(),targets[i]))<1e-9,
        "a previous read supplies its own start, not an average with a different read/eye");}
    let mut unseeded=fixture.scene;unseeded.target_seed_camera_mm=None;unseeded.secondary_target_seed_camera_mm=None;
    let control=Problem::new(JointConicRequest {scene:&unseeded,..request}).unwrap();
    let selected=problem.select(&problem.conics(&problem.initial).unwrap());
    assert_eq!(problem.initial,control.initial);
    assert_eq!(problem.residuals(&problem.initial,&selected),control.residuals(&control.initial,&selected),
        "historical initializations cannot become temporal observations, extra weights or new pixels");
    let solved=solve_joint_conics(request).unwrap();
    assert!(solved.hypotheses_evaluated<=8);
    assert_eq!(solved.hypotheses_by_association.iter().sum::<usize>(),solved.hypotheses_evaluated);
    let mut single=fixture.scene;single.secondary_target_seed_camera_mm=None;
    let expected=Problem::new(JointConicRequest {scene:&single,..request}).unwrap().seeds();
    for secondary in [targets[0],[f64::NAN;3]] {
        let mut duplicate=single;duplicate.secondary_target_seed_camera_mm=Some(secondary);
        assert_eq!(Problem::new(JointConicRequest {scene:&duplicate,..request}).unwrap().seeds(),expected,
            "duplicate/nonfinite starts do not crowd out useful hypotheses");
    }
}
