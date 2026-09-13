//! Source-time, CPU-only refinement of the shared SAM/Obelisk geometry product.
//! This is not a renderer or a second gaze solver. The pupil remains measured
//! from RAW/semantic evidence; signed pose is solved downstream as usual.
use super::*;
use crate::limbus_refiner::{self, Context, Field, Model};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Off,
    Experimental,
}

impl Mode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "off" => Ok(Self::Off),
            "experimental" => Ok(Self::Experimental),
            _ => Err("limbus refinement must be off or experimental (unverified checkpoint authority)".into()),
        }
    }

    pub fn label(self) -> &'static str {
        match self { Self::Off => "off", Self::Experimental => "experimental" }
    }
}

/// Immutable audit accompanying the exact proposal, including abstentions.
/// Original contour support is preserved for comparison, not counted twice.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub baseline: OuterMaskFitReview,
    pub field: Option<Arc<Field>>,
    pub applied: bool,
    pub status: String,
}

impl Attempt {
    pub fn diagnostic(&self) -> serde_json::Value {
        let ellipse = |e: Ellipse| serde_json::json!({
            "center":e.center,"major_radius":e.major_radius,
            "minor_radius":e.minor_radius,"angle":e.angle});
        serde_json::json!({"mode":"experimental","applied":self.applied,
            "status":self.status,"baseline":ellipse(self.baseline.ellipse),
            "candidate":self.field.as_ref().and_then(|f|f.candidate).map(ellipse),
            "cpu_ms":self.field.as_ref().map(|f|f.elapsed_ms),
            "corrected_samples":self.field.as_ref().map(|f|f.samples.iter().filter(|s|s.correction_px.is_some()).count()),
            "bootstrap_verified":false})
    }
}

// Startup-only selection. Merely visiting an F view cannot change this mode.
pub(super) fn apply(
    source: &RawFrame,
    image: &FloatImage,
    review: &mut OuterMaskFitReview,
    support: &mut RawRingSupport,
    pupil: Option<PupilVoidFitReview>,
) -> Option<Attempt> {
    static MODEL: OnceLock<Result<Option<Model>, String>> = OnceLock::new();
    let model = MODEL.get_or_init(|| {
        let mode = Mode::parse(&std::env::var("BUTTERCUP_LIMBUS_REFINEMENT").unwrap_or_else(|_| "off".into()))?;
        if mode == Mode::Off { return Ok(None); }
        eprintln!("LIMBUS_REFINEMENT experimental shared geometry authority; CPU only; bootstrap proof pending");
        Model::load(&limbus_refiner::default_model_path()).map(Some).map_err(|error| {
            eprintln!("LIMBUS_REFINEMENT unavailable; retaining baseline: {error}");
            error
        })
    });
    let model = match model {
        Ok(None) => return None,
        Ok(Some(model)) => model,
        Err(error) => return Some(Attempt { baseline:review.clone(),field:None,
            applied:false,status:format!("MODEL UNAVAILABLE: {error}") }),
    };
    // Neither a newer UI scale nor the candidate's radius is independent
    // source-time physical context. Missing observations remain missing.
    let field = Arc::new(limbus_refiner::refine(model, &source.pixels,
        source.width, source.height, review.ellipse, &review.retained_points, Context::default()));
    Some(apply_field(source, image, review, support, pupil, field))
}

pub(super) fn apply_field(
    source: &RawFrame,
    image: &FloatImage,
    review: &mut OuterMaskFitReview,
    support: &mut RawRingSupport,
    pupil: Option<PupilVoidFitReview>,
    field: Arc<Field>,
) -> Attempt {
    let mut attempt = Attempt { baseline:review.clone(), field:Some(Arc::clone(&field)),
        applied:false,status:field.status.into() };
    let Some(candidate) = field.candidate else { return attempt; };
    let rejection = if source.width.checked_mul(source.height) != Some(source.pixels.len())
        || (image.width,image.height) != (source.width,source.height)
        || field.baseline != review.ellipse
        || field.corrected_points.len() != review.retained_points.len()
        || !field.corrected_points.iter().zip(review.retained_points.iter())
            .all(|(a,b)|(a.0-b.0).hypot(a.1-b.1) <= limbus_refiner::MAX_CORRECTION_PX + 1e-9)
        || !field.corrected_points.iter().all(|p|p.0.is_finite() && p.1.is_finite()
            && p.0 >= 0.0 && p.1 >= 0.0 && p.0 < source.width as f64 && p.1 < source.height as f64) {
        Some("SOURCE OR CONTOUR MISMATCH")
    } else if !live_detector_raw_gate_passes(*support) {
        Some("BASELINE NOT RAW ADMITTED")
    } else if !limbus_refiner::bounded_candidate(review.ellipse, candidate) {
        Some("GLOBAL SHAPE BOUND")
    } else if pupil.is_some_and(|p|!pupil_ellipse_plausible(p.ellipse, candidate)) {
        // Do not discard or move a real pupil to make a learned rim succeed.
        Some("CONFLICTS WITH SOURCE PUPIL")
    } else { None };
    if let Some(reason) = rejection {
        attempt.status = format!("REFINEMENT REJECTED: {reason}");
        return attempt;
    }
    let refined_support = raw_ring_support(image, candidate);
    if !live_detector_raw_gate_passes(refined_support) {
        attempt.status = "REFINEMENT REJECTED: CURRENT SOURCE RAW SUPPORT".into();
        return attempt;
    }
    review.ellipse = candidate;
    // Indices/runs remain identical: excluded chords are never reintroduced.
    review.retained_points = Arc::new(field.corrected_points.clone());
    *support = refined_support;
    attempt.applied = true;
    attempt.status = "REFINED SHARED GEOMETRY / EXPERIMENTAL".into();
    attempt
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(crate) fn fixture() -> (RawFrame, FloatImage, OuterMaskFitReview, Arc<Field>) {
        let baseline = Ellipse { center:(192.0,128.0),major_radius:70.0,minor_radius:60.0,angle:0.1 };
        let candidate = Ellipse { center:(193.0,128.0), ..baseline };
        let pixels = (0..384*256).map(|i|if ellipse_coordinate(((i%384) as f64,(i/384) as f64),candidate)<1.0 {100} else {700}).collect();
        let source = RawFrame { eye_index:1,sequence:79,timestamp_ns:1_234_567_890,
            sensor_x:400,sensor_y:800,width:384,height:256,pixels:Arc::new(pixels),
            registration_anchor:None,pupil_component_seed:None };
        let image = raw_luma(&[Arc::new(source.clone())]).remove(0);
        let points = baseline.dense_points(96);
        let review = OuterMaskFitReview { ellipse:baseline, source_component_area_px:13000.0,
            retained_points:Arc::new(points.clone()),conic_segments:Arc::new(vec![(0..48).collect(),(48..96).collect()]),
            flat_tire_points:Arc::new(vec![(160.0,100.0)]),upper_flat_tire:true,lower_flat_tire:false };
        let field = Field {baseline,unmodified_refit:None,candidate:Some(candidate),
            samples:vec![],corrected_points:points.iter().map(|p|(p.0+1.0,p.1)).collect(),
            elapsed_ms:0.5,status:"SYNTHETIC TEST FIELD"};
        (source,image,review,Arc::new(field))
    }

    #[test]
    fn accepted_geometry_preserves_source_and_excluded_arcs() {
        let (source,image,mut review,field)=fixture();
        let original=review.clone();
        let mut support=raw_ring_support(&image,review.ellipse);
        let decision=apply_field(&source,&image,&mut review,&mut support,None,field.clone());
        assert!(decision.applied,"{}",decision.status);
        assert_eq!(review.ellipse,field.candidate.unwrap());
        assert_eq!(*review.retained_points,field.corrected_points);
        assert!(Arc::ptr_eq(&review.conic_segments,&original.conic_segments));
        assert!(Arc::ptr_eq(&review.flat_tire_points,&original.flat_tire_points));
        assert_eq!(decision.baseline.ellipse,original.ellipse);
        assert_eq!(source.timestamp_ns,1_234_567_890);
        assert_eq!(support,raw_ring_support(&image,review.ellipse));
    }

    #[test]
    fn rejection_keeps_exact_baseline_and_support() {
        for reason in 0..6 {
            let (source,mut image,mut review,mut field)=fixture();
            let baseline=review.clone();
            let mut support=raw_ring_support(&image,review.ellipse);
            let pupil=if reason==4 {Some(PupilVoidFitReview {
                ellipse:Ellipse {center:(10.0,10.0),major_radius:20.0,minor_radius:18.0,angle:0.0},
                raw_support:RawRingSupport::default()})} else {None};
            match reason {
                0=>Arc::make_mut(&mut field).candidate=None,
                1=>Arc::make_mut(&mut field).candidate.as_mut().unwrap().center.0+=20.0,
                2=>support.score=0.0,
                3=>image.data.fill([100.0;3]),
                4=>{},
                _=>Arc::make_mut(&mut field).corrected_points[0].0+=20.0,
            }
            let original_support=support;
            let decision=apply_field(&source,&image,&mut review,&mut support,pupil,field);
            assert!(!decision.applied,"reason {reason}");
            assert_eq!(review.ellipse,baseline.ellipse);
            assert!(Arc::ptr_eq(&review.retained_points,&baseline.retained_points));
            assert_eq!(support,original_support);
        }
    }

    #[test]
    fn mode_requires_explicit_experimental_opt_in() {
        assert_eq!(Mode::default(),Mode::Off);
        assert_eq!(Mode::parse("experimental").unwrap(),Mode::Experimental);
        for invalid in ["on","auto","true",""] { assert!(Mode::parse(invalid).is_err()); }
    }
}
