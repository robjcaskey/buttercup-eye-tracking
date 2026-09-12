//! Defeasible metric scene initialization for joint conic satisfaction.
//!
//! MediaPipe-derived scale is preferred. A nominal interocular span is only
//! an explicitly labeled engineering fallback, not a subject measurement.
//! Neither branch normalizes scale by the candidate's own fitted iris radius.

use crate::conic_solver::joint::{EyeScenePrior,JointScenePrior,PinholeCamera,PositionSupport,ScalarSupport,TransversePositionFrame,SurfaceAxisAlignment};

#[derive(Clone,Copy,Debug)]
pub(crate) struct EyePoseInput {
    pub(crate) limbus_center_sensor_px:[f64;2],
    /// [nominal,minimum,maximum], inherited from coarse acquisition and possibly
    /// held since an earlier semantic update. Candidate-independent is NOT a
    /// declaration of a fresh scale measurement or calibrated metric accuracy.
    pub(crate) pixels_per_10mm:Option<[f64;3]>,
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum ScaleProvenance { CoarseAcquisition,NominalInterocularSpan,NominalRange }

#[derive(Clone,Copy,Debug)]
pub(crate) struct CoarseBinocularScene {
    pub(crate) prior:JointScenePrior,
    pub(crate) scale_provenance:[Option<ScaleProvenance>;2],
    /// Only externally supplied coarse scale populates this diagnostic. Values
    /// may be held acquisition priors; no fresh measurement time is asserted.
    pub(crate) independent_pixels_per_mm:[Option<[f64;3]>;2],
}

/// These are deliberately broad, configurable-through-the-core engineering
/// priors, not anatomical population confidence intervals. The current model
/// does not include corneal refraction, kappa, torsion or lens accommodation.
pub(crate) fn approximate_scene(camera:PinholeCamera,eyes:[Option<EyePoseInput>;2]) -> Option<CoarseBinocularScene> {
    if !camera.focal_px.into_iter().all(|v|v.is_finite()&&v>0.0) {return None;}
    let span=eyes[0].zip(eyes[1]).map(|(a,b)| {
        let x=(a.limbus_center_sensor_px[0]-b.limbus_center_sensor_px[0])/camera.focal_px[0];
        let y=(a.limbus_center_sensor_px[1]-b.limbus_center_sensor_px[1])/camera.focal_px[1];
        x.hypot(y)
    }).filter(|v|v.is_finite()&&*v>0.01);
    let nominal_range=span.map(|s|64.0/s).unwrap_or(350.0).clamp(120.0,1200.0);
    let mut scale_provenance=[None;2];
    let mut independent=[None;2];
    let mut priors=[None;2];
    for eye in 0..2 {
        let Some(input)=eyes[eye] else {continue;};
        if !input.limbus_center_sensor_px.into_iter().all(f64::is_finite) {continue;}
        let measured=input.pixels_per_10mm.filter(|s| s.into_iter().all(|v|v.is_finite()&&*v>0.0)
            && s[1]<=s[0] && s[0]<=s[2]);
        let (range,range_sigma,provenance)=if let Some(s)=measured {
            independent[eye]=Some(s.map(|v|v/10.0));
            let focal=(camera.focal_px[0]*camera.focal_px[1]).sqrt();
            let d=focal*10.0/s[0];
            let sigma=((focal*10.0/s[1]-focal*10.0/s[2])*0.5).max(d*0.12);
            (d,sigma,ScaleProvenance::CoarseAcquisition)
        } else {
            (nominal_range,nominal_range*0.35,if span.is_some() {ScaleProvenance::NominalInterocularSpan} else {ScaleProvenance::NominalRange})
        };
        if !range.is_finite() || !(60.0..=3000.0).contains(&range) {continue;}
        scale_provenance[eye]=Some(provenance);
        let position=camera.unproject(input.limbus_center_sensor_px,range);
        let radius=|nominal,minimum,maximum,sigma|ScalarSupport {nominal,minimum,maximum,sigma};
        priors[eye]=Some(EyeScenePrior {
            limbus_center:PositionSupport {camera_mm:position,sigma_mm:[2.5,2.5,range_sigma],
                maximum_displacement_mm:[10.0,10.0,(range_sigma*2.0).min(range-40.0)],
                transverse_frame:TransversePositionFrame::AtNominalDepth},
            radii_mm:[radius(6.0,4.0,8.0,1.0),radius(5.6,3.5,7.5,1.0),radius(2.4,0.6,4.5,1.2)],
            pupil_inward_depth_mm:radius(0.6,0.0,1.5,0.5),
            pupil_decentration_sigma_mm:0.35,pupil_maximum_decentration_mm:0.9,
            effective_pivot:None,limbus_to_pivot_mm:9.0,
            surface_axis_alignment:Some(SurfaceAxisAlignment {nominal_radians:[0.0;2],
                sigma_radians:[0.045;2],maximum_deviation_radians:[0.14;2]}),
        });
    }
    let reference=priors[0].or(priors[1])?.limbus_center.camera_mm;
    let reference=if let [Some(a),Some(b)]=priors {
        std::array::from_fn(|i|(a.limbus_center.camera_mm[i]+b.limbus_center.camera_mm[i])*0.5)
    } else {reference};
    // Averaging TWO ORIGIN PRIORS merely chooses a coordinate reference. No
    // gaze direction or solved gaze point exists at this stage.
    Some(CoarseBinocularScene {prior:JointScenePrior {camera,eyes:priors,
        target_reference_camera_mm:reference,
        fixation_axial_distance_mm:ScalarSupport {nominal:609.6,minimum:150.0,maximum:2000.0,sigma:800.0},
        target_seed_camera_mm:None,secondary_target_seed_camera_mm:None,maximum_gaze_slope:1.5,
        interocular_distance_mm:Some(ScalarSupport {nominal:64.0,minimum:30.0,maximum:100.0,sigma:18.0}),
    },scale_provenance,independent_pixels_per_mm:independent})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acquired_scale_is_independent_of_fitted_radius_and_survives_roi_translation() {
        let camera=PinholeCamera {focal_px:[4000.0,4000.0],principal_px:[4000.0,3000.0]};
        let eyes=[Some(EyePoseInput {limbus_center_sensor_px:[3600.0,2100.0],pixels_per_10mm:Some([160.0,120.0,200.0])}),
            Some(EyePoseInput {limbus_center_sensor_px:[4600.0,2150.0],pixels_per_10mm:None})];
        let scene=approximate_scene(camera,eyes).unwrap();
        assert_eq!(scene.scale_provenance[0],Some(ScaleProvenance::CoarseAcquisition));
        assert_eq!(scene.independent_pixels_per_mm[0],Some([16.0,12.0,20.0]));
        assert_eq!(scene.prior.eyes[0].unwrap().limbus_center.camera_mm[2],-250.0);
        assert!(scene.independent_pixels_per_mm[1].is_none());
        assert!(scene.prior.eyes.iter().flatten().all(|e|e.effective_pivot.is_none()));
    }
}
