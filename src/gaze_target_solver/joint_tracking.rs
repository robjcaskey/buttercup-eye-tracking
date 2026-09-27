//! Source-keyed real-time orchestration of the joint segment objective.
//! No independently solved gaze point enters this module. A previous joint
//! target is only an optimization start, never a new observation or smoothing.
//! Explicit direction priors are separately applied to both MAP and posterior.

use crate::binocular_coordinator::source_pairing::{PairingUnavailable, SourcePairer};
use crate::conic_solver::joint::{solve_joint_conics_with_mount, JointConicRequest, JointConicSolution, JointConicUnavailable, PinholeCamera};
use crate::eye_scene_model::binocular_pose::{approximate_scene, CoarseBinocularScene, EyePoseInput};
use crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence;
use crate::roi_evidence::{ExposureKey, SourceClock};
use std::sync::Arc;

const MAX_SOURCE_SEEDS: usize = 32;
const MAX_SEED_AGE_NS: u64 = 500_000_000;

/// Offline hypothesis only: a local diagonal approximation to a radial lens.
/// It matches the reference ray and diagonal projection derivatives, but omits
/// cross-axis shear and curvature across the ROI. Never a calibrated lens model.
#[cfg(test)]
fn local_radial_camera(camera:PinholeCamera,k1:f64,reference:[f64;2])->PinholeCamera {
    if k1==0.0 {return camera;}
    let distorted:[f64;2]=std::array::from_fn(|i|(reference[i]-camera.principal_px[i])/camera.focal_px[i]);
    let rd=distorted[0].hypot(distorted[1]);
    if rd<1e-12 {return camera;}
    let mut radius=rd;
    for _ in 0..16 {
        radius-=(radius*(1.0+k1*radius*radius)-rd)/(1.0+3.0*k1*radius*radius);
    }
    let undistorted=distorted.map(|v|v*radius/rd);
    let radial=1.0+k1*radius*radius;
    let focal_px=std::array::from_fn(|i|camera.focal_px[i]*(radial+2.0*k1*undistorted[i]*undistorted[i]));
    let principal_px=std::array::from_fn(|i|reference[i]-focal_px[i]*undistorted[i]);
    PinholeCamera {focal_px,principal_px}
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TargetSeed {
    timestamp_ns: u64,
    target_camera_mm: [f64;3],
}

impl TargetSeed {
    fn precedes(self, timestamp_ns:u64) -> bool {
        self.timestamp_ns < timestamp_ns && timestamp_ns-self.timestamp_ns <= MAX_SEED_AGE_NS
    }
}

/// One source-clock/configuration lineage, sorted by sensor time rather than
/// detector completion. These are starts, not residuals, evidence or displays.
#[derive(Default)]
struct SourceSeedHistory(Vec<TargetSeed>);

impl SourceSeedHistory {
    fn insert(&mut self, seed:TargetSeed) {
        if !seed.target_camera_mm.into_iter().all(f64::is_finite) {return;}
        let index=self.0.partition_point(|previous|previous.timestamp_ns<seed.timestamp_ns);
        if self.0.get(index).is_some_and(|previous|previous.timestamp_ns==seed.timestamp_ns) {return;}
        self.0.insert(index,seed);
        if self.0.len()>MAX_SOURCE_SEEDS {self.0.remove(0);}
    }

    fn preceding(&self, timestamp_ns:u64) -> Option<TargetSeed> {
        self.0.iter().rev().copied().find(|seed|seed.precedes(timestamp_ns))
    }

    fn preceding_secondary(&self,timestamp_ns:u64) -> Option<TargetSeed> {
        self.0.iter().rev().copied().filter(|seed|seed.precedes(timestamp_ns)).nth(1)
    }

    /// A newly admitted partner supersedes the provisional same-read solve.
    /// If the combined solve fails, the first publication cannot remain a seed
    /// that silently ignores the conflicting/missing second-eye constraints.
    fn supersede(&mut self,timestamp_ns:u64) {
        self.0.retain(|seed|seed.timestamp_ns!=timestamp_ns);
    }
}

pub(crate) struct FrameEvidence {
    pub(crate) packet: OwnedRoiEvidence,
    pub(crate) pose: EyePoseInput,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic_solver::joint::ProjectedCircle;
    use crate::geometry::{normalized3,sub3};
    use crate::outline_conic_segments::sparse_evidence::{OwnedBoundaryArc,OwnedConicHint};
    use crate::roi_evidence::{BoundaryKind,RoiId};

    fn camera()->PinholeCamera {PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]}}
    #[test]
    fn radial_camera_diagnostic_preserves_the_reference_ray_and_radial_derivative() {
        let c=camera();let k=0.4;let u=0.3;
        let reference=[4000.0,3000.0+4000.0*u*(1.0+k*u*u)];
        let local=local_radial_camera(c,k,reference);
        assert!((local.focal_px[1]-4000.0*(1.0+3.0*k*u*u)).abs()<1e-9);
        assert!((local.focal_px[0]-4000.0*(1.0+k*u*u)).abs()<1e-9);
        assert!((local.principal_px[1]+local.focal_px[1]*u-reference[1]).abs()<1e-9);
        let identity=local_radial_camera(c,0.0,reference);
        assert_eq!(identity.focal_px,c.focal_px);assert_eq!(identity.principal_px,c.principal_px);
        let centered=local_radial_camera(c,k,c.principal_px);
        assert_eq!(centered.focal_px,c.focal_px);assert_eq!(centered.principal_px,c.principal_px);
    }
    fn frame(eye:usize,time:u64)->FrameEvidence {
        frame_with_target(eye,time,[70.0,-130.0,250.0])
    }
    fn frame_with_target(eye:usize,time:u64,target:[f64;3])->FrameEvidence {
        let center=[if eye==0 {-32.0} else {32.0},0.0,-350.0];
        let origin=if eye==0 {[3420,2860]} else {[4100,2860]};
        let normal=normalized3(sub3(target,center)).unwrap();
        let e=ProjectedCircle::project(camera(),center,normal,6.0,origin).unwrap().ellipse().unwrap();
        FrameEvidence {packet:OwnedRoiEvidence {
            exposure:ExposureKey {roi:RoiId(eye as u32+1),clock:SourceClock {domain:1,epoch:5},
                sequence:time/100_000_000+eye as u64*1000,timestamp_ns:time},
            sensor_origin_px:origin,dimensions_px:[420,280],detail_reliability:Some(1.0),
            arcs:vec![OwnedBoundaryArc { support_length_cap_px: None, sampling_support_px: None, level_sets_roi: None,evidence_group:0,kind:BoundaryKind::OuterLimbus,
                points_roi_px:e.dense_points(32),outward_normals_roi:None,localization_sigma_px:None,normal_band_half_width_px:1.0,detector_score:None}],
            conics:vec![OwnedConicHint {kind:BoundaryKind::OuterLimbus,ellipse_roi_px:e,supporting_arc_indices:vec![0]}]},
            pose:EyePoseInput {limbus_center_sensor_px:camera().project(center).unwrap(),pixels_per_10mm:Some([4000.0/35.0,100.0,130.0])}}
    }

    #[test]
    fn native_screen_orientation_then_offscreen_continuation_retains_conditional_posterior() {
        use crate::eye_scene_model::CameraMount;
        let mut tracker=JointTracker {camera_mount:CameraMount::BelowEyes,
            legacy_mount_filter:false,..Default::default()};
        tracker.set_probabilistic(true);
        tracker.set_orientation_active(true);
        tracker.begin(frame(0,1_000_000_000).packet.exposure.clock,1);
        let mut last=None;
        for i in 0..16 {
            let time=1_000_000_000+i*500_000_000;
            for eye in 0..2 {
                last=tracker.observe_with_pair_wait(frame_with_target(eye,time,[70.,-400.,50.]),camera(),true).unwrap().or(last);
            }
        }
        let initialized=last.unwrap();
        assert_eq!(initialized.orientation_ready,Some([true,true]),"{:?}",initialized.solution.posterior.as_ref().map(|p|p.json()));
        tracker.set_orientation_active(false);
        for (i,y) in [-250.,-100.,0.,100.,200.].into_iter().enumerate() {
            let time=9_000_000_000+i as u64*500_000_000;
            for eye in 0..2 {
                if let Some(p)=tracker.observe_with_pair_wait(frame_with_target(eye,time,[70.,y,50.]),camera(),true).unwrap() {
                    assert_eq!(p.orientation_ready,Some([true,true]));
                    let posterior=p.solution.posterior.as_ref().unwrap();
                    assert_eq!(posterior.direction_prior.label(),"temporal-continuation-conditional");
                    assert_eq!(posterior.camera_mount,CameraMount::Flexible);
                    if y==200. {assert!(p.solution.eye_normals.iter().flatten().all(|n|n[1]>0.));}
                }
            }
        }
    }

    #[test]
    fn camera_mount_filters_whole_optimized_stereo_hypotheses() {
        use crate::eye_scene_model::CameraMount;
        let time=1_000_000_000;
        for mode in [CameraMount::Flexible,CameraMount::BelowEyes,CameraMount::AboveEyes] {
            let mut tracker=JointTracker::default();tracker.camera_mount=mode;tracker.legacy_mount_filter=true;
            tracker.begin(frame(0,time).packet.exposure.clock,1);
            for eye in 0..2 {
                match tracker.observe(frame(eye,time),camera()) {
                    Ok(Some(publication))=> {
                        assert!(publication.solution.eye_normals.iter().flatten().all(|n|mode.supports(n[1])));
                        assert_eq!(publication.exposures[eye].unwrap().timestamp_ns,time);
                    }
                    Err(TrackingUnavailable::MountingAssumptionConflict)=> {
                        assert_ne!(mode,CameraMount::Flexible);
                        assert!(tracker.latest(eye,frame(eye,time).packet.exposure.clock,time,1).is_none());
                    }
                    other=>panic!("unexpected mounting solve: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn mounted_probabilistic_publications_keep_the_selected_modes_confidence() {
        use crate::eye_scene_model::CameraMount;
        let time=1_000_000_000;
        let mut publications=0;
        for mode in [CameraMount::BelowEyes,CameraMount::AboveEyes] {
            let mut tracker=JointTracker::default();
            tracker.camera_mount=mode;
            tracker.legacy_mount_filter=true;
            tracker.set_probabilistic(true);
            tracker.set_posterior_diagnostic(crate::conic_solver::joint::posterior::IntegrationConfig {
                budget:512,early_stop:false,..Default::default()
            });
            tracker.begin(frame(0,time).packet.exposure.clock,1);
            for eye in 0..2 {
                match tracker.observe(frame(eye,time),camera()) {
                    Ok(Some(publication))=> {
                        let posterior=publication.solution.posterior.as_ref().expect("mounted alternative lost its posterior");
                        assert_eq!(posterior.camera_mount,mode);
                        assert_eq!(posterior.modeled_eyes,publication.solution.modeled_eyes);
                        assert!(publication.solution.eye_normals.iter().flatten().all(|n|mode.supports(n[1])));
                        // An unavailable/under-sampled integration is not a
                        // license to publish a resolved direction.
                        if posterior.status!="estimated-conditional" {
                            assert!(!posterior.supports_direction(0));
                            assert!(!posterior.supports_direction(1));
                        }
                        publications+=1;
                    }
                    Err(TrackingUnavailable::MountingAssumptionConflict)=>{},
                    other=>panic!("unexpected mounted posterior result: {other:?}"),
                }
            }
        }
        assert!(publications>=2,"fixture must exercise actual mounted publications");
    }

    #[test]
    fn diagnostic_hypotheses_preserve_the_default_winner_and_optimizer_budget() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut baseline=JointTracker::default();baseline.begin(clock,1);
        let mut diagnostic=JointTracker::default();diagnostic.begin(clock,1);
        diagnostic.retain_diagnostic_hypotheses(true);
        diagnostic.set_continuous_gaze_diagnostic(true);
        for eye in 0..2 {
            let a=baseline.observe(frame(eye,time),camera()).unwrap().unwrap();
            let b=diagnostic.observe(frame(eye,time),camera()).unwrap().unwrap();
            assert!(a.diagnostic_hypotheses.is_empty());
            assert!(a.continuous_gaze_sign.is_none());
            let trajectory=b.continuous_gaze_sign.as_ref().unwrap();
            assert_eq!(trajectory.candidate_scope,"unconditioned");
            assert_eq!(trajectory.preferred_candidate,None,"one exposure cannot establish trajectory support");
            assert!((1..=4).contains(&b.diagnostic_hypotheses.len()));
            assert_eq!(a.solution.target_camera_mm,b.solution.target_camera_mm);
            assert_eq!(a.solution.robust_cost,b.solution.robust_cost);
            assert_eq!(a.solution.alternative_cost_margin,b.solution.alternative_cost_margin);
            assert_eq!(a.solution.eye_normals,b.solution.eye_normals);
            assert_eq!(a.solution.hypotheses_evaluated,b.solution.hypotheses_evaluated);
            assert_eq!(a.solution.refinement_steps,b.solution.refinement_steps);
            assert_eq!(b.diagnostic_hypotheses[0].target_camera_mm,b.solution.target_camera_mm);
            assert!(b.diagnostic_hypotheses.windows(2).all(|pair|pair[0].robust_cost<=pair[1].robust_cost));
        }
    }

    #[test]
    fn late_unique_partner_cannot_resurrect_geometry_cleared_by_a_newer_empty_source() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        assert!(tracker.observe(frame(0,time),camera()).unwrap().is_some());
        let mut missing=frame(0,time+100_000_000);missing.packet.arcs.clear();missing.packet.conics.clear();
        assert!(tracker.observe(missing,camera()).is_err());
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none());
        let late=tracker.observe(frame(1,time),camera()).unwrap().unwrap();
        assert_eq!(late.exposures[0].unwrap().timestamp_ns,time,"the old pair is legitimate historical output");
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none(),
            "historical completion must not restore a newer rejected eye's live geometry");
        assert!(tracker.latest(1,clock,time+100_000_000,500_000_000).is_some());
        tracker.observe(frame(0,time+200_000_000),camera()).unwrap();
        assert_eq!(tracker.latest(0,clock,time+200_000_000,500_000_000).unwrap().exposures[0].unwrap().timestamp_ns,time+200_000_000);
    }

    #[test]
    fn fresh_invalid_scene_support_clears_previous_geometry_as_a_conic_failure_does() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        tracker.observe(frame(0,time),camera()).unwrap();
        let mut invalid=frame(0,time+100_000_000);invalid.pose.limbus_center_sensor_px=[f64::NAN;2];
        assert!(matches!(tracker.observe(invalid,camera()),Err(TrackingUnavailable::NoSceneSupport)));
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none());
    }

    #[test]
    fn paired_initialization_cannot_depend_on_which_eye_arrived_first() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut a=JointTracker::default();a.begin(clock,1);
        let mut b=JointTracker::default();b.begin(clock,1);
        // Both schedules know the same earlier single-eye observation. A
        // newer first arrival must not hide or expose that hint asymmetrically
        // when the identical current pair is finally solved.
        for tracker in [&mut a,&mut b] {tracker.observe(frame(0,time),camera()).unwrap();}
        a.observe(frame(0,time+100_000_000),camera()).unwrap();
        let right_first=a.observe(frame(1,time+100_000_000),camera()).unwrap().unwrap();
        b.observe(frame(1,time+100_000_000),camera()).unwrap();
        let left_first=b.observe(frame(0,time+100_000_000),camera()).unwrap().unwrap();
        assert_eq!(right_first.scene.prior.target_seed_camera_mm,left_first.scene.prior.target_seed_camera_mm,
            "a final paired solve must use source-keyed joint history, not an arrival-specific surviving display slot");
        assert_eq!(right_first.solution.target_camera_mm,left_first.solution.target_camera_mm);
    }

    #[test]
    fn a_joint_initialization_survives_display_invalidation_without_resurrecting_display_geometry() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        tracker.observe(frame(0,time),camera()).unwrap();
        let paired=tracker.observe(frame(1,time),camera()).unwrap().unwrap();
        let mut rejected=frame(0,time+100_000_000);rejected.packet.arcs.clear();rejected.packet.conics.clear();
        assert!(tracker.observe(rejected,camera()).is_err());
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none());
        tracker.observe(frame(1,time+200_000_000),camera()).unwrap();
        let historical=tracker.observe(frame(1,time+100_000_000),camera()).unwrap().unwrap();
        assert_eq!(historical.scene.prior.target_seed_camera_mm,Some(paired.solution.target_camera_mm),
            "a bounded past joint seed is not erased by rejected/newer display slots and is not a fresh observation");
        assert!(!historical.solution.contributing_eyes[0],"the empty current eye must not regain old evidence");
        assert_eq!(tracker.latest(1,clock,time+200_000_000,500_000_000).unwrap().exposures[1].unwrap().timestamp_ns,
            time+200_000_000,"a historical pair cannot rewind the newer presentation");
    }

    #[test]
    fn source_history_is_bounded_source_sorted_strictly_past_and_never_refreshed_by_duplicates() {
        let mut history=SourceSeedHistory::default();
        let seed=|timestamp_ns|TargetSeed {timestamp_ns,target_camera_mm:[timestamp_ns as f64,0.0,200.0]};
        history.insert(seed(200));history.insert(seed(100));history.insert(seed(300));
        assert_eq!(history.preceding(200),Some(seed(100)));
        assert_eq!(history.preceding(301),Some(seed(300)));
        assert_eq!(history.preceding_secondary(301),Some(seed(200)));
        assert!(history.preceding(100).is_none(),"equal/future sensor times cannot initialize the current solve");
        history.insert(TargetSeed {timestamp_ns:300,target_camera_mm:[999.0;3]});
        assert_eq!(history.preceding(301),Some(seed(300)));
        assert!(history.preceding(300+MAX_SEED_AGE_NS).is_some());
        assert!(history.preceding_secondary(300+MAX_SEED_AGE_NS).is_none(),
            "secondary seed age is measured from the current source, not from the newer historical seed");
        assert!(history.preceding(301+MAX_SEED_AGE_NS).is_none());
        for time in 400..500 {history.insert(seed(time));}
        assert_eq!(history.0.len(),MAX_SOURCE_SEEDS);
        history.insert(seed(1));
        assert_eq!(history.0.len(),MAX_SOURCE_SEEDS);
        assert_eq!(history.0[0].timestamp_ns,500-MAX_SOURCE_SEEDS as u64);
        assert_eq!(history.preceding(500),Some(seed(499)),"late insertion cannot rewind the newest seed");
        history.insert(TargetSeed {timestamp_ns:500,target_camera_mm:[f64::NAN;3]});
        assert_eq!(history.preceding(501),Some(seed(499)));
    }

    #[test]
    fn monocular_initialization_keeps_recent_single_eye_progress_but_never_future_or_expired_seeds() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        let first=tracker.observe(frame(0,time),camera()).unwrap().unwrap();
        let second=tracker.observe(frame(0,time+100_000_000),camera()).unwrap().unwrap();
        assert_eq!(second.scene.prior.target_seed_camera_mm,Some(first.solution.target_camera_mm));
        assert_eq!(tracker.source_seeds.0.len(),2,"valid missing-partner solves retain source-keyed initialization");
        let historical=tracker.observe(frame(0,time-100_000_000),camera()).unwrap().unwrap();
        assert!(historical.scene.prior.target_seed_camera_mm.is_none());
        let expired=tracker.observe(frame(0,time+100_000_001+MAX_SEED_AGE_NS),camera()).unwrap().unwrap();
        assert!(expired.scene.prior.target_seed_camera_mm.is_none());
    }

    #[test]
    fn source_history_resets_with_clock_or_configuration_not_crop_or_repeated_begin() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        tracker.observe(frame(0,time),camera()).unwrap();
        let paired=tracker.observe(frame(1,time),camera()).unwrap().unwrap();
        tracker.begin(clock,1);
        assert_eq!(tracker.source_seeds.preceding(time+1).unwrap().target_camera_mm,paired.solution.target_camera_mm);
        let mut reframed=frame(0,time+100_000_000);
        reframed.packet.sensor_origin_px[0]+=8;
        for arc in &mut reframed.packet.arcs {for p in &mut arc.points_roi_px {p.0-=8.0;}}
        for conic in &mut reframed.packet.conics {conic.ellipse_roi_px.center.0-=8.0;}
        let newer=tracker.observe(reframed,camera()).unwrap().unwrap();
        assert_eq!(newer.scene.prior.target_seed_camera_mm,Some(paired.solution.target_camera_mm));
        let newest=tracker.observe(frame(0,time+200_000_000),camera()).unwrap().unwrap();
        assert_eq!(newest.scene.prior.target_seed_camera_mm,Some(newer.solution.target_camera_mm),
            "a paired seed is not an excuse to discard newer usable single-eye initialization");
        tracker.begin(clock,2);
        assert!(tracker.source_seeds.0.is_empty());
        tracker.source_seeds.insert(TargetSeed {timestamp_ns:time,target_camera_mm:paired.solution.target_camera_mm});
        tracker.begin(SourceClock {epoch:clock.epoch+1,..clock},2);
        assert!(tracker.source_seeds.0.is_empty());
    }

    #[test]
    fn a_missing_partner_read_remains_a_useful_seed_after_newer_display_rejection() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        let previous=tracker.observe(frame(0,time),camera()).unwrap().unwrap();
        let mut empty=frame(0,time+100_000_000);empty.packet.arcs.clear();empty.packet.conics.clear();
        assert!(tracker.observe(empty,camera()).is_err());
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none());
        let next=tracker.observe(frame(1,time+200_000_000),camera()).unwrap().unwrap();
        assert_eq!(next.scene.prior.target_seed_camera_mm,Some(previous.solution.target_camera_mm),
            "discarding valid single-ROI initialization is not a solution to presentation-dependent history");
        assert!(next.exposures[0].is_none(),"a historical start must not resurrect the missing ROI or its pixels");
    }

    #[test]
    fn two_past_reads_remain_separate_initializations_when_the_partner_is_missing() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        let a=tracker.observe(frame(0,time),camera()).unwrap().unwrap();
        let b=tracker.observe(frame(1,time+100_000_000),camera()).unwrap().unwrap();
        assert_ne!(a.solution.target_camera_mm,b.solution.target_camera_mm);
        let current=tracker.observe(frame(0,time+200_000_000),camera()).unwrap().unwrap();
        assert_eq!(current.scene.prior.target_seed_camera_mm,Some(b.solution.target_camera_mm));
        assert_eq!(current.scene.prior.secondary_target_seed_camera_mm,Some(a.solution.target_camera_mm));
        assert!(current.exposures[1].is_none(),"using a start is not presenting an old eye as current evidence");
    }

    #[test]
    fn a_completed_pair_supersedes_its_provisional_seed_without_averaging_or_double_counting() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        let first=tracker.observe(frame(0,time),camera()).unwrap().unwrap();
        assert_eq!(tracker.source_seeds.0.len(),1);
        assert_eq!(tracker.source_seeds.preceding(time+1).unwrap().target_camera_mm,first.solution.target_camera_mm);
        let paired=tracker.observe(frame(1,time),camera()).unwrap().unwrap();
        assert_eq!(tracker.source_seeds.0.len(),1,"one physical read, not two independent history votes");
        assert_eq!(tracker.source_seeds.preceding(time+1).unwrap().target_camera_mm,paired.solution.target_camera_mm);
        assert!(paired.scene.prior.target_seed_camera_mm.is_none(),"the first same-time solve is not past evidence");
        assert!(tracker.observe(frame(1,time),camera()).unwrap().is_none());
        assert_eq!(tracker.source_seeds.0.len(),1);
        assert_eq!(tracker.source_seeds.preceding(time+1).unwrap().target_camera_mm,paired.solution.target_camera_mm,
            "a duplicate cannot invalidate or refresh successful history");
    }

    #[test]
    fn explicit_pair_wait_preserves_complete_solves_without_publishing_held_sources() {
        let time=1_000_000_000;let clock=frame(0,time).packet.exposure.clock;
        let mut baseline=JointTracker::default();let mut deferred=JointTracker::default();
        for tracker in [&mut baseline,&mut deferred] {tracker.begin(clock,1);tracker.set_probabilistic(true);}
        for (step,first) in [(0,0),(1,1)] {
            let at=time+step*100_000_000;
            baseline.observe(frame(first,at),camera()).unwrap();
            assert!(deferred.observe_with_pair_wait(frame(first,at),camera(),true).unwrap().is_none());
            assert!(deferred.latest(first,clock,at,500_000_000).is_none(),"held geometry cannot be the new exposure");
            let a=baseline.observe(frame(1-first,at),camera()).unwrap().unwrap();
            let b=deferred.observe_with_pair_wait(frame(1-first,at),camera(),true).unwrap().unwrap();
            assert_eq!(format!("{:?}",a.solution),format!("{:?}",b.solution),"paired solve and posterior changed");
            assert_eq!(a.exposures,b.exposures);
        }
        assert!(deferred.observe(frame(0,time+200_000_000),camera()).unwrap().is_some(),
            "independent single-ROI reads must keep their normal path");
    }

    #[test]
    fn a_failed_completed_pair_retires_only_its_own_provisional_seed() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        for invalid_scene in [false,true] {
            let mut tracker=JointTracker::default();tracker.begin(clock,1);
            let old=tracker.observe(frame(0,time),camera()).unwrap().unwrap();
            tracker.observe(frame(0,time+100_000_000),camera()).unwrap();
            assert_eq!(tracker.source_seeds.0.len(),2);
            let mut partner=frame(1,time+100_000_000);
            let mut intrinsics=camera();
            if invalid_scene {intrinsics.focal_px=[f64::NAN;2];}
            else {partner.packet.arcs[0].outward_normals_roi=Some(vec![]);}
            let failure=tracker.observe(partner,intrinsics);
            assert!(if invalid_scene {matches!(failure,Err(TrackingUnavailable::NoSceneSupport))}
                else {matches!(failure,Err(TrackingUnavailable::Conic(JointConicUnavailable::InvalidRequest)))});
            assert_eq!(tracker.source_seeds.0.len(),1);
            assert_eq!(tracker.source_seeds.preceding(time+200_000_000).unwrap().target_camera_mm,old.solution.target_camera_mm,
                "failed completion cannot retain a provisional seed that ignored the second ROI");
            assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none());
        }
    }

    fn moving_meridian_frame(eye:usize,step:usize,time:u64,angle:f64,pupil_band:f64)->(FrameEvidence,[f64;3]) {
        use crate::geometry::{add3,cross3,scale3};
        let phase=step as f64*0.3;
        let pivot=[(if eye==0 {-35.0} else {35.0})+phase.sin()*2.0,
            90.0+phase.cos()*1.5,-310.0+phase.sin()*1.0];
        let observer=normalized3(scale3(pivot,-1.0)).unwrap();
        let right=normalized3(cross3([0.0,1.0,0.0],observer)).unwrap();
        let down=cross3(observer,right);
        let normal=normalized3(add3(scale3(observer,angle.cos()),scale3(down,angle.sin()))).unwrap();
        let center=add3(pivot,scale3(normal,9.0));
        let pixel=camera().project(center).unwrap();
        let mut current=frame(eye,time);
        current.packet.sensor_origin_px=[(pixel[0]-210.0).round() as u32,(pixel[1]-140.0).round() as u32];
        current.packet.arcs.clear();current.packet.conics.clear();
        let scale=40000.0/-center[2];
        current.pose=EyePoseInput {limbus_center_sensor_px:pixel,pixels_per_10mm:Some([scale,scale*0.8,scale*1.2])};
        let u=normalized3(cross3([0.0,1.0,0.0],normal)).unwrap();let v=cross3(normal,u);
        for (group,kind,radius,depth) in [(0,BoundaryKind::OuterLimbus,6.0,0.0),
            (1,BoundaryKind::PupillaryBoundary,2.4,0.6)] {
            let center=sub3(center,scale3(normal,depth));
            let points=(0..32).map(|i| {
                let phase=i as f64*std::f64::consts::TAU/32.0;
                let point=add3(center,add3(scale3(u,radius*phase.cos()),scale3(v,radius*phase.sin())));
                let p=camera().project(point).unwrap();
                (p[0]-current.packet.sensor_origin_px[0] as f64,p[1]-current.packet.sensor_origin_px[1] as f64)
            }).collect();
            let ellipse=ProjectedCircle::project(camera(),center,normal,radius,current.packet.sensor_origin_px).unwrap().ellipse().unwrap();
            current.packet.arcs.push(OwnedBoundaryArc { support_length_cap_px: None, sampling_support_px: None, level_sets_roi: None,evidence_group:group,kind,points_roi_px:points,
                outward_normals_roi:None,localization_sigma_px:None,normal_band_half_width_px:if group==1 {pupil_band} else {0.0},detector_score:None});
            current.packet.conics.push(OwnedConicHint {kind,ellipse_roi_px:ellipse,supporting_arc_indices:vec![group as usize]});
        }
        (current,normal)
    }

    #[test]
    fn monocular_camera_relative_meridian_crossing_survives_head_motion_and_roi_reframes() {
        use crate::geometry::dot3;
        // The visible eye is off the optical axis. Its two circle-pose
        // branches coalesce when facing its OWN observer ray, not when the
        // camera-global y component happens to be zero.
        for eye in 0..2 {
            let first=1_000_000_000;
            let mut tracker=JointTracker::default();tracker.begin(frame(eye,first).packet.exposure.clock,1);
            let mut previous_origin=None;let mut reframes=0;
            for (step,angle) in [-0.45_f64,-0.3,-0.14,-0.04,0.04,0.14,0.3,0.45,0.25,0.02,-0.12,-0.35].into_iter().enumerate() {
                let time=first+step as u64*100_000_000;
                let (current,normal)=moving_meridian_frame(eye,step,time,angle,0.0);
                if previous_origin.is_some_and(|origin|origin!=current.packet.sensor_origin_px) {reframes+=1;}
                previous_origin=Some(current.packet.sensor_origin_px);
                let result=tracker.observe(current,camera()).unwrap().unwrap();
                assert_eq!(result.solution.contributing_eyes,[eye==0,eye==1]);
                let found=result.solution.eye_normals[eye].unwrap();
                let error=dot3(found,normal).clamp(-1.0,1.0).acos().to_degrees();
                assert!(error<4.0,"eye={eye} step={step} tilt={angle}: a real crossing/turnaround must not bounce at observer-normal; error={error}");
                let center=result.solution.eye_centers_camera_mm[eye].unwrap();
                assert!(dot3(found,center)<0.0,"convex camera-facing solutions only");
                assert!(result.solution.hypotheses_evaluated<=16);
            }
            assert!(reframes>=8,"exercise native crop translation throughout the crossing");
        }
    }

    #[test]
    #[ignore = "synthetic weak-boundary diagnostic; redirect SYNTHETIC_MERIDIAN records under outputs, not a claim of accuracy"]
    fn monocular_meridian_uncertainty_diagnostic() {
        // Truth is used only in the emitted post-fit diagnostic. It never
        // chooses a hypothesis or enters the source-keyed target history.
        let trajectories=[
            ("crossing",[-0.45,-0.3,-0.14,-0.04,0.04,0.14,0.3,0.45,0.25,0.02,-0.12,-0.35]),
            ("turnaround",[-0.45,-0.3,-0.14,-0.04,-0.02,-0.04,-0.14,-0.3,-0.45,-0.3,-0.14,-0.04]),
            ("step",[-0.4,-0.4,-0.4,0.4,0.4,0.4,-0.4,-0.4,-0.4,0.4,0.4,0.4]),
        ];
        let mut index=0;
        for eye in 0..2 {for (trajectory,angles) in trajectories {for pupil_band in [0.0,1.0,3.0,6.0] {
            for cadence_ns in [40_000_000,100_000_000,300_000_000] {
                let first=1_000_000_000;
                let mut tracker=JointTracker::default();tracker.begin(frame(eye,first).packet.exposure.clock,1);
                tracker.retain_diagnostic_hypotheses(true);
                for (step,angle) in angles.into_iter().enumerate() {
                    let time=first+step as u64*cadence_ns;
                    let (current,truth)=moving_meridian_frame(eye,step,time,angle,pupil_band);
                    let origin=current.packet.sensor_origin_px;
                    let publication=tracker.observe(current,camera()).unwrap().unwrap();
                    let hypotheses=publication.diagnostic_hypotheses.iter().map(|h|serde_json::json!({
                        "available":true,"cost":h.robust_cost,"contributing_eyes":h.contributing_eyes,
                        "eye_normals":h.eye_normals,"eye_gaze_directions":h.eye_gaze_directions,
                        "eye_centers_camera_mm":h.eye_centers_camera_mm,"target_camera_mm":h.target_camera_mm,
                        "hypotheses":h.hypotheses_evaluated})).collect::<Vec<_>>();
                    eprintln!("SYNTHETIC_MERIDIAN {}",serde_json::json!({
                        "schema":"buttercup-synthetic-meridian-conics-v1","input":{"index":index},
                        "eye":eye,"trajectory":trajectory,"pupil_band_px":pupil_band,"cadence_ns":cadence_ns,
                        "source_ns":time.to_string(),"native_crop_origin_px":origin,
                        "expected_normal_postfit_only":truth,"current_source_hypotheses":hypotheses,
                        "contract":"Exact forward-projected points with uncertainty bands, not simulated blur/noise or real sensor evidence."}));
                    index+=1;
                }
            }
        }}}
    }

    #[test]
    fn source_history_does_not_freeze_gaze_or_prevent_vertical_sign_crossings() {
        use crate::geometry::{add3,cross3,dot3,scale3};
        let observed=|eye:usize,time:u64,target:[f64;3]| {
            let mut frame=frame(eye,time);
            frame.packet.arcs.clear();frame.packet.conics.clear();
            let center=[if eye==0 {-32.0} else {32.0},0.0,-350.0];
            let normal=normalized3(sub3(target,center)).unwrap();
            let u=normalized3(cross3([0.0,1.0,0.0],normal)).unwrap();
            let v=cross3(normal,u);
            for (group,kind,radius,depth) in [(0,BoundaryKind::OuterLimbus,6.0,0.0),
                (1,BoundaryKind::PupillaryBoundary,2.4,0.6)] {
                let center=sub3(center,scale3(normal,depth));
                // Independent 3D points -> pinhole pixels, not samples from
                // the inverse solver's reconstructed conic matrix.
                let points=(0..32).map(|i| {
                    let phase=i as f64*std::f64::consts::TAU/32.0;
                    let point=add3(center,add3(scale3(u,radius*phase.cos()),scale3(v,radius*phase.sin())));
                    let pixel=camera().project(point).unwrap();
                    (pixel[0]-frame.packet.sensor_origin_px[0] as f64,pixel[1]-frame.packet.sensor_origin_px[1] as f64)
                }).collect();
                let ellipse=ProjectedCircle::project(camera(),center,normal,radius,frame.packet.sensor_origin_px).unwrap().ellipse().unwrap();
                frame.packet.arcs.push(OwnedBoundaryArc { support_length_cap_px: None, sampling_support_px: None, level_sets_roi: None,evidence_group:group,kind,points_roi_px:points,
                    outward_normals_roi:None,localization_sigma_px:None,normal_band_half_width_px:0.0,detector_score:None});
                frame.packet.conics.push(OwnedConicHint {kind,ellipse_roi_px:ellipse,supporting_arc_indices:vec![group as usize]});
            }
            frame
        };
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut right_first=JointTracker::default();right_first.begin(clock,1);
        let mut left_first=JointTracker::default();left_first.begin(clock,1);
        for (step,y) in [-130.0,-80.0,-30.0,0.0,30.0,80.0,130.0,60.0,0.0,-60.0,-130.0].into_iter().enumerate() {
            let time=time+step as u64*100_000_000;
            let target=[70.0,y,250.0];
            right_first.observe(observed(0,time,target),camera()).unwrap();
            let a=right_first.observe(observed(1,time,target),camera()).unwrap().unwrap();
            left_first.observe(observed(1,time,target),camera()).unwrap();
            let b=left_first.observe(observed(0,time,target),camera()).unwrap().unwrap();
            assert_eq!(a.solution.target_camera_mm,b.solution.target_camera_mm);
            assert_eq!(a.solution.contributing_eyes,[true,true]);
            for eye in 0..2 {
                let center=[if eye==0 {-32.0} else {32.0},0.0,-350.0];
                let truth=normalized3(sub3(target,center)).unwrap();
                let error=dot3(a.solution.eye_gaze_directions[eye].unwrap(),truth).clamp(-1.0,1.0).acos().to_degrees();
                assert!(error<4.0,"step {step}, eye {eye}: current mixed boundary samples must move the gaze through zero vertical gaze; error={error}");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PublishedJoint {
    pub(crate) continuous_gaze_sign: Option<crate::conic_solver::continuous_gaze_sign::Report>,
    /// None for legacy/unconstrained paths. Otherwise screen-reference readiness
    /// is separate from current-source posterior support and signal quality.
    pub(crate) orientation_ready: Option<[bool;2]>,
    pub(crate) exposures: [Option<ExposureKey>; 2],
    pub(crate) sensor_origins_px: [Option<[u32;2]>; 2],
    pub(crate) dimensions_px: [Option<[u32;2]>; 2],
    pub(crate) scene: CoarseBinocularScene,
    pub(crate) solution: JointConicSolution,
    /// Optional current-source optimizer alternatives, for replay diagnostics.
    /// They are neither extra observations nor independently averaged gazes.
    pub(crate) diagnostic_hypotheses: Vec<JointConicSolution>,
    pub(crate) source_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrackingUnavailable {
    Pairing(PairingUnavailable),
    NoSceneSupport,
    MountingAssumptionConflict,
    DirectionReferenceContradicted,
    Conic(JointConicUnavailable),
}

#[derive(Default)]
pub(crate) struct JointTracker {
    pub(crate) parallax_enabled: bool,
    pub(crate) parallax_normals: [Option<[[f64;3];2]>;2],
    continuous_gaze_enabled: bool,
    continuous_gaze: crate::conic_solver::continuous_gaze_sign::ContinuousGazeSign,
    pub(crate) camera_mount: crate::eye_scene_model::CameraMount,
    /// Explicit historical comparison only; live and ordinary trackers use the new phase policy.
    pub(crate) legacy_mount_filter: bool,
    direction_continuation: crate::conic_solver::camera_mount::DirectionContinuation,
    audit_source: Option<u64>,
    audit_sources: usize,
    lineage: Option<(SourceClock,u64)>,
    pairing: SourcePairer<FrameEvidence>,
    latest: [Option<Arc<PublishedJoint>>;2],
    source_seeds: SourceSeedHistory,
    /// Rejection is an observation too. Clearing an invalid fit must not erase
    /// this time floor and let a delayed historical pair resurrect old gaze.
    newest_observation_ns: [Option<u64>;2],
    retain_diagnostic_hypotheses: bool,
    probabilistic: bool,
    #[cfg(test)]
    posterior_diagnostic: Option<crate::conic_solver::joint::posterior::IntegrationConfig>,
    #[cfg(test)]
    camera_diagnostic: Option<PinholeCamera>,
    #[cfg(test)]
    radial_camera_diagnostic: Option<f64>,
}

impl JointTracker {
    pub(crate) fn set_continuous_gaze_diagnostic(&mut self, enabled:bool) {
        if enabled!=self.continuous_gaze_enabled {self.continuous_gaze=Default::default();}
        self.continuous_gaze_enabled=enabled;
    }
    pub(crate) fn anchor_orientation_source(&mut self, source:u64) {
        self.direction_continuation.anchor_source(source);
    }
    pub(crate) fn set_orientation_active(&mut self, active:bool) {
        if active!=self.direction_continuation.orientation_active {
            self.audit_source=None;self.audit_sources=0;
        }
        self.direction_continuation.set_orientation(active);
    }
    pub(crate) fn set_screen_fixation(&mut self, active:bool) {
        self.direction_continuation.set_screen_fixation(active);
    }
    pub(crate) fn set_routine_acquisition(&mut self, enabled:bool) {
        self.direction_continuation.set_routine_acquisition(enabled);
    }
    pub(crate) fn set_probabilistic(&mut self, enabled: bool) { self.probabilistic=enabled; }
    #[cfg(test)]
    pub(crate) fn set_posterior_diagnostic(&mut self, config:crate::conic_solver::joint::posterior::IntegrationConfig) {
        self.posterior_diagnostic=Some(config);
    }
    #[cfg(test)]
    pub(crate) fn set_camera_diagnostic(&mut self, camera:PinholeCamera) {
        assert!(self.lineage.is_none(), "set the camera trial before admitting any source");
        assert!(camera.focal_px.into_iter().all(|v|v.is_finite() && v>0.0));
        assert!(camera.principal_px.into_iter().all(f64::is_finite));
        self.camera_diagnostic=Some(camera);
    }
    #[cfg(test)]
    pub(crate) fn set_radial_camera_diagnostic(&mut self,k1:f64) {
        assert!(self.lineage.is_none(), "set the lens trial before admitting any source");
        assert!(k1.is_finite() && (0.0..=1.0).contains(&k1));
        self.radial_camera_diagnostic=Some(k1);
    }
    pub(crate) fn retain_diagnostic_hypotheses(&mut self, enabled: bool) {
        self.retain_diagnostic_hypotheses = enabled;
    }
    /// `generation` changes with provider/prompt configuration, not crop
    /// translation. The caller, not an arriving asynchronous proposal, owns it.
    pub(crate) fn begin(&mut self, clock:SourceClock, generation:u64) {
        if self.lineage == Some((clock,generation)) {return;}
        self.lineage=Some((clock,generation));self.pairing.begin(clock);self.latest=[None,None];
        self.newest_observation_ns=[None;2];
        self.source_seeds.0.clear();
        self.continuous_gaze=Default::default();
        let active=self.direction_continuation.orientation_active;
        let fixation=self.direction_continuation.screen_fixation;
        let routine=self.direction_continuation.routine_acquisition;
        self.direction_continuation=Default::default();
        self.direction_continuation.set_orientation(active);
        self.direction_continuation.set_screen_fixation(fixation);
        self.direction_continuation.set_routine_acquisition(routine);
        self.audit_source=None;self.audit_sources=0;
    }

    fn clear_current_sources(&mut self,sources:[Option<ExposureKey>;2]) {
        for eye in 0..2 {
            if sources[eye].is_some_and(|source|self.newest_observation_ns[eye]
                .is_none_or(|time|source.timestamp_ns>=time)) {self.latest[eye]=None;}
        }
    }

    pub(crate) fn observe(&mut self, frame:FrameEvidence, camera:PinholeCamera)
        ->Result<Option<Arc<PublishedJoint>>,TrackingUnavailable> {
        self.observe_with_pair_wait(frame,camera,false)
    }

    /// Explicit atomic two-ROI requests need no provisional one-eye solve.
    /// Keep the evidence and current-source floor while waiting; never count a
    /// held publication as this exposure, or use a same-time provisional seed.
    pub(crate) fn observe_with_pair_wait(&mut self, frame:FrameEvidence, camera:PinholeCamera, wait_for_pair:bool)
        ->Result<Option<Arc<PublishedJoint>>,TrackingUnavailable> {
        #[cfg(test)]
        let camera=self.camera_diagnostic.unwrap_or(camera);
        let source=frame.packet.exposure;
        let frames=match self.pairing.insert(source,frame) {
            Ok(frames)=>frames,
            Err(PairingUnavailable::DuplicateSource)=>return Ok(None),
            Err(e)=>return Err(TrackingUnavailable::Pairing(e)),
        };
        let seen=&mut self.newest_observation_ns[source.roi.0 as usize-1];
        *seen=Some(seen.unwrap_or(0).max(source.timestamp_ns));
        if wait_for_pair && frames.iter().any(Option::is_none) {return Ok(None);}
        let exposures=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.exposure));
        self.source_seeds.supersede(source.timestamp_ns);
        let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
        #[cfg(test)]
        let camera=self.radial_camera_diagnostic.map_or(camera,|k1| {
            let points=poses.iter().flatten().map(|p|p.limbus_center_sensor_px).collect::<Vec<_>>();
            let reference=std::array::from_fn(|i|points.iter().map(|p|p[i]).sum::<f64>()/points.len() as f64);
            local_radial_camera(camera,k1,reference)
        });
        let Some(mut scene)=approximate_scene(camera,poses) else {
            self.clear_current_sources(exposures);
            return Err(TrackingUnavailable::NoSceneSupport);
        };
        // The latest usable physical read may have only one ROI. Retain that
        // information independently of display-slot survival. A later partner
        // replaces its provisional same-read seed, never adds another vote.
        let seed=self.source_seeds.preceding(source.timestamp_ns);
        scene.prior.target_seed_camera_mm=seed.map(|seed|seed.target_camera_mm);
        scene.prior.secondary_target_seed_camera_mm=self.source_seeds.preceding_secondary(source.timestamp_ns)
            .map(|seed|seed.target_camera_mm);
        let prepared=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.prepare()));
        let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
        let request=JointConicRequest {eyes:evidence,scene:&scene.prior,
            maximum_hypotheses:16,maximum_refinements:12,maximum_source_skew_ns:0,
            // Engineering allowance, not a measured bound on rolling rows.
            exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
        let orientation_only=!self.parallax_enabled && !self.legacy_mount_filter
            && self.camera_mount==crate::eye_scene_model::CameraMount::BelowEyes;
        let effective_mount=if orientation_only || self.parallax_enabled {crate::eye_scene_model::CameraMount::Flexible} else {self.camera_mount};
        let direction_prior=if self.parallax_enabled {crate::conic_solver::camera_mount::DirectionPrior::Parallax(self.parallax_normals)} else if orientation_only {self.direction_continuation.prior(source.timestamp_ns)} else {Default::default()};
        let count=if self.camera_mount!=crate::eye_scene_model::CameraMount::Flexible {16}
            else if self.retain_diagnostic_hypotheses || self.continuous_gaze_enabled { 4 } else { 1 };
        let result=if self.probabilistic {
            #[cfg(test)]
            {if let Some(mut config)=self.posterior_diagnostic {
                config.camera_mount=effective_mount;
                config.direction_prior=direction_prior;
                crate::conic_solver::joint::solve_joint_conic_distribution_diagnostic(request,count,config)
            } else {crate::conic_solver::joint::solve_joint_conics_with_direction(request,count,effective_mount,true,direction_prior)}}
            #[cfg(not(test))]
            {crate::conic_solver::joint::solve_joint_conics_with_direction(request,count,effective_mount,true,direction_prior)}
        }
            else {crate::conic_solver::joint::solve_joint_conics_with_direction(request,count,effective_mount,false,direction_prior)};
        let mut hypotheses=match result {
            Ok(hypotheses)=>hypotheses,
            Err(error)=>{
                self.clear_current_sources(exposures);
                if error==JointConicUnavailable::MountingAssumptionConflict {
                    return Err(TrackingUnavailable::MountingAssumptionConflict);
                }
                return Err(TrackingUnavailable::Conic(error));
            }
        };
        let solution=hypotheses.remove(0);
        // Periodically challenge the continuation prior with the same raw
        // objective, without that prior. Three independent contradictory reads
        // release the reference; an isolated fit cannot cause a branch flip.
        // This audit is MAP-only and does not replace/copy a posterior.
        if orientation_only && (matches!(direction_prior,
            crate::conic_solver::camera_mount::DirectionPrior::Continue(_))
            // Screen fixation continues surviving references; audit them too.
            || (!self.direction_continuation.orientation_active && matches!(direction_prior,
                crate::conic_solver::camera_mount::DirectionPrior::ScreenAndContinue(n)
                | crate::conic_solver::camera_mount::DirectionPrior::AcquireAndContinue(n) if n.iter().any(Option::is_some))))
            && self.audit_source.is_none_or(|old|source.timestamp_ns>old) {
            self.audit_source=Some(source.timestamp_ns);self.audit_sources+=1;
            if self.audit_sources%8==0 || self.direction_continuation.contradiction_pending() {
                let alternative=solve_joint_conics_with_mount(request,1,effective_mount,false)
                    .ok().and_then(|mut fits|(!fits.is_empty()).then(||fits.remove(0)));
                let contradicts=alternative.as_ref().is_some_and(|other|
                    other.robust_cost+9.0<solution.robust_cost
                    && (0..2).any(|eye|other.eye_normals[eye].zip(other.eye_centers_camera_mm[eye])
                        .is_some_and(|(n,c)|!direction_prior.supports(eye,c,n))));
                if self.direction_continuation.observe_contradiction(source.timestamp_ns,contradicts) {
                    self.clear_current_sources(exposures);
                    return Err(TrackingUnavailable::DirectionReferenceContradicted);
                }
            }
        }
        assert!((0..2).all(|eye|solution.eye_normals[eye].is_none_or(|n|
            effective_mount.supports(n[1]) && solution.eye_centers_camera_mm[eye]
                .is_some_and(|c|direction_prior.supports(eye,c,n)))));
        let expects_posterior=self.probabilistic;
        #[cfg(test)]
        let expects_posterior=expects_posterior && self.posterior_diagnostic.is_none_or(|c|c.budget>0);
        assert!(!expects_posterior || solution.posterior.is_some(),
            "probabilistic publication must retain the selected mode's posterior");
        if orientation_only {
            let supported=std::array::from_fn(|eye|solution.contributing_eyes[eye]
                && solution.posterior.as_ref().is_some_and(|p|p.supports_direction(eye)));
            self.direction_continuation.observe(source.timestamp_ns,solution.eye_normals,supported,direction_prior);
            let continuing=std::array::from_fn(|eye|solution.contributing_eyes[eye]
                && solution.posterior.as_ref().is_some_and(|p|p.continues_direction(eye)));
            self.direction_continuation.refresh(source.timestamp_ns,solution.eye_normals,continuing,direction_prior);
        }
        let orientation_ready=if self.parallax_enabled {Some(self.parallax_normals.map(|n|n.is_some()))}
            else {orientation_only.then(||std::array::from_fn(|eye|self.direction_continuation.ready(eye)))};
        // A diagnostic preference never substitutes a MAP fit or borrows a
        // different candidate's integrated posterior. Its input scope is explicit.
        let continuous_gaze_sign=self.continuous_gaze_enabled.then(|| {
            use crate::conic_solver::continuous_gaze_sign::Candidate;
            let candidates=std::iter::once(&solution).chain(&hypotheses)
                .map(Candidate::from_solution).collect::<Option<Vec<_>>>().unwrap_or_default();
            self.continuous_gaze.observe(source.timestamp_ns,candidates,
                if effective_mount!=crate::eye_scene_model::CameraMount::Flexible {"legacy-mount-filtered"}
                else {direction_prior.label()})
        });
        let diagnostic_hypotheses = if self.retain_diagnostic_hypotheses {
            let mut all=vec![solution.clone()];all.extend(hypotheses);all
        } else {Vec::new()};
        let publication=Arc::new(PublishedJoint {solution,scene,diagnostic_hypotheses,orientation_ready,continuous_gaze_sign,
            exposures,
            sensor_origins_px:frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.sensor_origin_px)),
            dimensions_px:frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.dimensions_px)),
            source_generation:self.lineage.map_or(0,|(_,g)|g)});
        self.source_seeds.insert(TargetSeed {timestamp_ns:source.timestamp_ns,
            target_camera_mm:publication.solution.target_camera_mm});
        for eye in 0..2 {if publication.exposures[eye].is_some_and(|key|self.newest_observation_ns[eye]
            .is_none_or(|time|key.timestamp_ns>=time)) {
            self.latest[eye]=Some(Arc::clone(&publication));
        }}
        Ok(Some(publication))
    }

    pub(crate) fn latest(&self, eye:usize, clock:SourceClock, current_timestamp_ns:u64, maximum_age_ns:u64)
        ->Option<Arc<PublishedJoint>> {
        let result=self.latest.get(eye)?.as_ref()?;
        let source=result.exposures[eye]?;
        (source.clock==clock && source.timestamp_ns<=current_timestamp_ns
            && self.newest_observation_ns[eye].is_none_or(|time|source.timestamp_ns>=time)
            && current_timestamp_ns-source.timestamp_ns<=maximum_age_ns).then(||Arc::clone(result))
    }
}
