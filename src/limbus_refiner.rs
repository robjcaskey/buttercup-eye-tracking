//! Sparse, source-local optical limbus landmark refinement.
//!
//! The learned field is a normal displacement/visibility distribution, NOT a
//! measured anatomical height or a new gaze authority. The normal points from
//! iris to sclera. Six distinct label roles must not be silently collapsed.
use crate::geometry::Ellipse;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const PATCH_SIDE: usize = 16;
pub const CONTEXT_FEATURES: usize = 20;
pub const INPUTS: usize = PATCH_SIDE * PATCH_SIDE + CONTEXT_FEATURES;
pub const HIDDEN: usize = 64;
pub const BINS: usize = 17; // 16 normal positions, then explicitly not visible.
pub const ROLES: [&str; 6] = [
    "rim",
    "band_inner",
    "band_outer",
    "iris_onset",
    "surface_apex",
    "subsurface_limit",
];
pub const OUTPUTS: usize = ROLES.len() * BINS;
pub const ARCHITECTURE: &str = "buttercup-limbus-normal-field-16-v1";
pub const MAX_SAMPLES: usize = 96;
pub const MAX_CORRECTION_PX: f64 = 4.0;
pub const SAMPLE_STEP_PX: f64 = 2.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct Support {
    pub estimate: f64,
    pub half_width: f64,
}
impl Support {
    fn valid(self) -> bool {
        self.estimate.is_finite()
            && self.estimate > 0.0
            && self.half_width.is_finite()
            && self.half_width >= 0.0
    }
}

/// Optional physical context is measured outside this candidate. Missing
/// measurements stay missing; apparent ellipse size is separately identified.
#[derive(Clone, Copy, Debug, Default)]
pub struct Context {
    pub pixels_per_mm: Option<Support>,
    pub camera_distance_mm: Option<Support>,
    pub focus_position: Option<f64>,
}

impl Context {
    /// Same explicit uncalibrated pinhole prior as the live joint solver.
    /// Scale is externally supplied, not the candidate's own fitted radius.
    pub fn from_coarse_scale(scale: Option<Support>, focal_px: f64) -> Self {
        let pixels_per_mm = scale.filter(|s| s.valid() && s.half_width < s.estimate);
        let camera_distance_mm = pixels_per_mm.and_then(|s| {
            if !focal_px.is_finite() || focal_px <= 0.0 {
                return None;
            }
            let estimate = focal_px / s.estimate;
            let scale_span = (focal_px / (s.estimate - s.half_width)
                - focal_px / (s.estimate + s.half_width))
                * 0.5;
            (60.0..=3000.0).contains(&estimate).then_some(Support {
                estimate,
                half_width: scale_span + estimate * 0.25,
            })
        });
        Self {
            pixels_per_mm,
            camera_distance_mm,
            focus_position: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Patch {
    pub features: Vec<f32>,
    pub center: (f64, f64),
    pub normal: (f64, f64),
    pub step_px: f64,
    pub mean: f64,
    pub contrast: f64,
    pub focus_energy: f64,
    pub saturated_fraction: f64,
}

pub fn normal_at(ellipse: Ellipse, point: (f64, f64)) -> Option<(f64, f64)> {
    if ![
        ellipse.center.0,
        ellipse.center.1,
        ellipse.major_radius,
        ellipse.minor_radius,
        ellipse.angle,
        point.0,
        point.1,
    ]
    .iter()
    .all(|x| x.is_finite())
        || ellipse.minor_radius <= 0.0
        || ellipse.major_radius < ellipse.minor_radius
    {
        return None;
    }
    let (s, c) = ellipse.angle.sin_cos();
    let dx = point.0 - ellipse.center.0;
    let dy = point.1 - ellipse.center.1;
    let u = (c * dx + s * dy) / ellipse.major_radius.powi(2);
    let v = (-s * dx + c * dy) / ellipse.minor_radius.powi(2);
    let n = (c * u - s * v, s * u + c * v);
    let length = n.0.hypot(n.1);
    (length > 1e-9).then_some((n.0 / length, n.1 / length))
}

// Native linear RAW, deliberately independent of display color/lightbox.
// A 3x3 tent suppresses mosaic phase; this is not claimed to recover color.
fn sample(raw: &[u16], width: usize, height: usize, p: (f64, f64)) -> Option<f64> {
    if !p.0.is_finite()
        || !p.1.is_finite()
        || p.0 < 1.0
        || p.1 < 1.0
        || p.0 >= width.saturating_sub(2) as f64
        || p.1 >= height.saturating_sub(2) as f64
    {
        return None;
    }
    let x = p.0.floor() as usize;
    let y = p.1.floor() as usize;
    let filtered = |x: usize, y: usize| {
        let mut value = 0.0;
        for dy in 0..3 {
            for dx in 0..3 {
                let weight = [1.0, 2.0, 1.0][dx] * [1.0, 2.0, 1.0][dy];
                value += f64::from(raw[(y + dy - 1) * width + x + dx - 1]) * weight;
            }
        }
        value / (16.0 * 1023.0)
    };
    let (fx, fy) = (p.0 - x as f64, p.1 - y as f64);
    Some(
        (filtered(x, y) * (1.0 - fx) + filtered(x + 1, y) * fx) * (1.0 - fy)
            + (filtered(x, y + 1) * (1.0 - fx) + filtered(x + 1, y + 1) * fx) * fy,
    )
}

pub fn extract_patch(
    raw: &[u16],
    width: usize,
    height: usize,
    ellipse: Ellipse,
    center: (f64, f64),
    context: Context,
    step_px: f64,
) -> Option<Patch> {
    if raw.len() != width.checked_mul(height)? || !(0.5..=4.0).contains(&step_px) {
        return None;
    }
    let normal = normal_at(ellipse, center)?;
    let mut values = Vec::with_capacity(PATCH_SIDE * PATCH_SIDE);
    for y in 0..PATCH_SIDE {
        for x in 0..PATCH_SIDE {
            let u = (x as f64 - 7.5) * step_px;
            let v = (y as f64 - 7.5) * step_px;
            values.push(sample(
                raw,
                width,
                height,
                (
                    center.0 + normal.0 * u - normal.1 * v,
                    center.1 + normal.1 * u + normal.0 * v,
                ),
            )?);
        }
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let contrast =
        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    let mut energy = 0.0;
    for y in 1..15 {
        for x in 1..15 {
            let i = y * 16 + x;
            energy += (values[i - 1] + values[i + 1] + values[i - 16] + values[i + 16]
                - 4.0 * values[i])
                .abs();
        }
    }
    let focus_energy = energy / 196.0;
    let saturated_fraction = values.iter().filter(|&&x| x > 0.98).count() as f64 / 256.0;
    let phase = normal.1.atan2(normal.0) - ellipse.angle;
    let scale = context.pixels_per_mm.filter(|v| v.valid());
    let distance = context.camera_distance_mm.filter(|v| v.valid());
    let focus = context
        .focus_position
        .filter(|v| v.is_finite() && (0.0..=1023.0).contains(v));
    let features = values
        .iter()
        .map(|v| ((v - mean) / contrast.max(0.02)).clamp(-4.0, 4.0) as f32 / 4.0)
        .chain(
            [
                mean,
                contrast,
                focus_energy * 10.0,
                saturated_fraction,
                ellipse.minor_radius / ellipse.major_radius,
                (2.0 * phase).cos(),
                (2.0 * phase).sin(),
                (ellipse.major_radius / 100.0).ln().clamp(-3.0, 3.0),
                step_px / 2.0,
                scale.map_or(0.0, |s| s.estimate / 20.0),
                scale.map_or(0.0, |s| s.half_width / 20.0),
                f64::from(scale.is_some()),
                distance.map_or(0.0, |s| s.estimate / 1000.0),
                distance.map_or(0.0, |s| s.half_width / 1000.0),
                f64::from(distance.is_some()),
                focus.unwrap_or(0.0) / 1023.0,
                f64::from(focus.is_some()),
                values[8 * 16 + 13] - values[8 * 16 + 2],
                values.iter().filter(|&&x| x < 0.02).count() as f64 / 256.0,
                // The patch's raw-space sampling pitch is explicit, not SN-FEIDA.
                (ellipse.minor_radius / 100.0).ln().clamp(-3.0, 3.0),
            ]
            .map(|x| x as f32),
        )
        .collect();
    Some(Patch {
        features,
        center,
        normal,
        step_px,
        mean,
        contrast,
        focus_energy,
        saturated_fraction,
    })
}

#[derive(Clone, Debug)]
pub struct Landmark {
    pub offset_px: f64,
    pub spread_px: f64,
    pub visible_mass: f64,
}
impl Landmark {
    pub fn supported(&self) -> bool {
        self.visible_mass >= 0.85 && self.spread_px <= 4.0 && self.offset_px.is_finite()
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub first_weight: Vec<f32>,
    pub first_bias: Vec<f32>,
    pub second_weight: Vec<f32>,
    pub second_bias: Vec<f32>,
    /// Unknown during training is not permission to extrapolate at inference.
    pub trained_optional_context: [bool; 3],
    pub manifest: Value,
}

pub fn default_model_path() -> PathBuf {
    std::env::var_os("BUTTERCUP_LIMBUS_REFINER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/models/limbus_refiner_v1.json"))
}

impl Model {
    pub fn from_json(v: Value) -> Result<Self, String> {
        if v["architecture"] != ARCHITECTURE
            || v["roles"] != json!(ROLES)
            || v["inputs"] != json!(INPUTS)
            || v["hidden"] != json!(HIDDEN)
            || v["bins"] != json!(BINS)
            || v["preprocess"] != "native-linear-tent-normal-patch-v1"
        {
            return Err("limbus refiner architecture/role/preprocessing mismatch".into());
        }
        let array = |name: &str, n: usize| -> Result<Vec<f32>, String> {
            let items = v[name]
                .as_array()
                .filter(|a| a.len() == n)
                .ok_or_else(|| format!("invalid {name} shape"))?;
            items
                .iter()
                .map(|x| {
                    x.as_f64()
                        .filter(|x| x.is_finite() && x.abs() < 1e4)
                        .map(|x| x as f32)
                        .ok_or_else(|| format!("invalid {name} value"))
                })
                .collect()
        };
        let flags = v["trained_optional_context"]
            .as_array()
            .filter(|a| a.len() == 3)
            .ok_or("missing context provenance")?;
        let trained_optional_context = std::array::from_fn(|i| flags[i].as_bool().unwrap_or(false));
        Ok(Self {
            first_weight: array("first_weight", INPUTS * HIDDEN)?,
            first_bias: array("first_bias", HIDDEN)?,
            second_weight: array("second_weight", OUTPUTS * HIDDEN)?,
            second_bias: array("second_bias", OUTPUTS)?,
            trained_optional_context,
            manifest: v,
        })
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        if std::fs::metadata(path).map_err(|e| e.to_string())?.len() > 8 * 1024 * 1024 {
            return Err("oversized limbus refiner".into());
        }
        let bytes =
            std::fs::read(path).map_err(|e| format!("limbus refiner {}: {e}", path.display()))?;
        if bytes.len() > 8 * 1024 * 1024 {
            return Err("oversized limbus refiner".into());
        }
        Self::from_json(serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)
    }
    pub fn logits(&self, patch: &Patch) -> Option<Vec<f32>> {
        if patch.features.len() != INPUTS || patch.features.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut features = patch.features.clone();
        for (trained, range) in
            self.trained_optional_context
                .into_iter()
                .zip([9..12, 12..15, 15..17])
        {
            if !trained {
                for i in range {
                    features[256 + i] = 0.0;
                }
            }
        }
        let hidden: Vec<f32> = self
            .first_weight
            .chunks_exact(INPUTS)
            .zip(&self.first_bias)
            .map(|(w, b)| (w.iter().zip(&features).map(|(w, x)| w * x).sum::<f32>() + b).max(0.0))
            .collect();
        Some(
            self.second_weight
                .chunks_exact(HIDDEN)
                .zip(&self.second_bias)
                .map(|(w, b)| w.iter().zip(&hidden).map(|(w, x)| w * x).sum::<f32>() + b)
                .collect(),
        )
    }
    pub fn predict(&self, patch: &Patch) -> Option<[Landmark; 6]> {
        let logits = self.logits(patch)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(std::array::from_fn(|role| {
            let row = &logits[role * BINS..(role + 1) * BINS];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probability: Vec<f64> = row.iter().map(|v| f64::from(v - max).exp()).collect();
            let total = probability.iter().sum::<f64>();
            let visible = probability[..16].iter().sum::<f64>();
            let offset_px = probability[..16]
                .iter()
                .enumerate()
                .map(|(i, p)| p * (i as f64 - 7.5) * patch.step_px)
                .sum::<f64>()
                / visible.max(1e-12);
            let spread_px = (probability[..16]
                .iter()
                .enumerate()
                .map(|(i, p)| p * ((i as f64 - 7.5) * patch.step_px - offset_px).powi(2))
                .sum::<f64>()
                / visible.max(1e-12))
            .sqrt();
            Landmark {
                offset_px,
                spread_px,
                visible_mass: visible / total,
            }
        }))
    }
}

#[derive(Clone, Debug)]
pub struct FieldSample {
    pub origin: (f64, f64),
    pub normal: (f64, f64),
    pub landmarks: [Landmark; 6],
    pub correction_px: Option<f64>,
}
#[derive(Clone, Debug)]
pub struct Field {
    pub baseline: Ellipse,
    pub unmodified_refit: Option<Ellipse>,
    pub candidate: Option<Ellipse>,
    pub samples: Vec<FieldSample>,
    pub corrected_points: Vec<(f64, f64)>,
    pub elapsed_ms: f64,
    pub status: &'static str,
}

pub fn bounded_candidate(baseline: Ellipse, candidate: Ellipse) -> bool {
    // Compare shape in source pixels, not angle parameters (circular ellipses
    // have no meaningful major-axis direction). Check the whole inferred rim.
    if ![
        candidate.center.0,
        candidate.center.1,
        candidate.major_radius,
        candidate.minor_radius,
        candidate.angle,
    ]
    .iter()
    .all(|x| x.is_finite())
        || candidate.minor_radius <= 0.0
        || candidate.major_radius < candidate.minor_radius
        || (candidate.center.0 - baseline.center.0).hypot(candidate.center.1 - baseline.center.1)
            > MAX_CORRECTION_PX
        || (candidate.major_radius / baseline.major_radius - 1.0).abs() > 0.06
        || (candidate.minor_radius / baseline.minor_radius - 1.0).abs() > 0.06
    {
        return false;
    }
    baseline.dense_points(64).iter().all(|&p| {
        let (s, c) = candidate.angle.sin_cos();
        let x = p.0 - candidate.center.0;
        let y = p.1 - candidate.center.1;
        let q = ((c * x + s * y) / candidate.major_radius).powi(2)
            + ((-s * x + c * y) / candidate.minor_radius).powi(2);
        (q.sqrt() - 1.0).abs() * candidate.major_radius <= MAX_CORRECTION_PX
    })
}

/// Evaluate only observed, de-flat-tired samples; excluded/missing contour arcs
/// cannot become supervised evidence just because an ellipse crosses them.
pub fn refine(
    model: &Model,
    raw: &[u16],
    width: usize,
    height: usize,
    baseline: Ellipse,
    retained: &[(f64, f64)],
    context: Context,
) -> Field {
    let started = std::time::Instant::now();
    let unmodified_refit = crate::conic_solver::robust_contour_fit(retained, baseline)
        .filter(|&e| bounded_candidate(baseline, e));
    let mut field = Field {
        baseline,
        unmodified_refit,
        candidate: None,
        samples: vec![],
        corrected_points: retained.to_vec(),
        elapsed_ms: 0.0,
        status: "NO SUPPORTED LOCAL CORRECTIONS",
    };
    let count = retained.len().min(MAX_SAMPLES);
    let mut accepted = 0;
    for k in 0..count {
        let index = k * retained.len() / count;
        let point = retained[index];
        let Some(patch) =
            extract_patch(raw, width, height, baseline, point, context, SAMPLE_STEP_PX)
        else {
            continue;
        };
        let Some(landmarks) = model.predict(&patch) else {
            continue;
        };
        // Explicit apex supervision is preferred; legacy point/band midpoint
        // support remains a named fallback, never a claimed anatomical apex.
        let rim = if landmarks[4].supported() {
            &landmarks[4]
        } else {
            &landmarks[0]
        };
        let ordered = !(landmarks[3].supported()
            && landmarks[4].supported()
            && landmarks[3].offset_px > landmarks[4].offset_px + 1.0)
            && !(landmarks[4].supported()
                && landmarks[5].supported()
                && landmarks[4].offset_px > landmarks[5].offset_px + 1.0);
        let correction = (ordered
            && rim.supported()
            && patch.contrast >= 0.015
            && patch.saturated_fraction < 0.2
            && rim.offset_px.abs() <= MAX_CORRECTION_PX)
            .then_some(rim.offset_px);
        if let Some(shift) = correction {
            // Apply this sparse model sample to its own measured contour point,
            // not a filled angular arc or a hallucinated occluded quadrant.
            field.corrected_points[index] = (
                point.0 + patch.normal.0 * shift,
                point.1 + patch.normal.1 * shift,
            );
            accepted += 1;
        }
        field.samples.push(FieldSample {
            origin: point,
            normal: patch.normal,
            landmarks,
            correction_px: correction,
        });
    }
    if accepted >= 10 {
        if let Some(candidate) =
            crate::conic_solver::robust_contour_fit(&field.corrected_points, baseline)
        {
            if bounded_candidate(baseline, candidate) {
                field.candidate = Some(candidate);
                field.status = "EXPERIMENTAL LOCAL LIMBUS REFINEMENT";
            } else {
                field.status = "LOCAL FIELD REJECTED: GLOBAL SHAPE BOUND";
            }
        } else {
            field.status = "LOCAL FIELD REJECTED: CONIC FIT";
        }
    }
    field.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    field
}

#[cfg(feature = "sam31")]
#[path = "limbus_refiner_train.rs"]
pub mod training;

#[cfg(test)]
mod tests {
    use super::*;
    fn ellipse() -> Ellipse {
        Ellipse {
            center: (50.0, 50.0),
            major_radius: 30.0,
            minor_radius: 20.0,
            angle: 0.0,
        }
    }
    #[test]
    fn patch_normal_is_outward_and_pi_invariant() {
        assert_eq!(normal_at(ellipse(), (80.0, 50.0)), Some((1.0, 0.0)));
        let flipped = Ellipse {
            angle: std::f64::consts::PI,
            ..ellipse()
        };
        let n = normal_at(flipped, (80.0, 50.0)).unwrap();
        assert!((n.0 - 1.0).abs() < 1e-12 && n.1.abs() < 1e-12);
    }
    #[test]
    fn patch_cannot_clamp_missing_roi_pixels_into_fake_evidence() {
        let raw = vec![200; 10000];
        assert!(extract_patch(
            &raw,
            100,
            100,
            ellipse(),
            (95.0, 50.0),
            Context::default(),
            2.0
        )
        .is_none());
        let p = extract_patch(
            &raw,
            100,
            100,
            ellipse(),
            (80.0, 50.0),
            Context::default(),
            1.0,
        )
        .unwrap();
        assert_eq!(p.features.len(), INPUTS);
        assert!(p.features.iter().all(|x| x.is_finite()));
        assert_eq!(p.features[267], 0.0); // independent scale remains unavailable.
        assert_eq!(p.features[270], 0.0); // physical distance remains unavailable.
    }
    #[test]
    fn local_refinement_cannot_expand_into_a_different_ellipse() {
        assert!(bounded_candidate(
            ellipse(),
            Ellipse {
                center: (51.0, 50.0),
                ..ellipse()
            }
        ));
        assert!(!bounded_candidate(
            ellipse(),
            Ellipse {
                major_radius: 36.0,
                ..ellipse()
            }
        ));
        assert!(!bounded_candidate(
            ellipse(),
            Ellipse {
                minor_radius: -2.0,
                ..ellipse()
            }
        ));
    }
    #[test]
    fn uncertainty_is_not_visibility_and_flat_heatmaps_abstain() {
        let landmark = Landmark {
            offset_px: 0.0,
            spread_px: 9.0,
            visible_mass: 0.99,
        };
        assert!(!landmark.supported());
    }

    #[test]
    fn crop_translation_does_not_change_a_native_patch() {
        let raw: Vec<_> = (0..160 * 140)
            .map(|i| ((i * 73 + i / 160 * 81) % 1024) as u16)
            .collect();
        let mut crop = vec![];
        for y in 10..130 {
            crop.extend_from_slice(&raw[y * 160 + 20..y * 160 + 140]);
        }
        let e = Ellipse {
            center: (80.0, 70.0),
            major_radius: 35.0,
            minor_radius: 25.0,
            angle: 0.3,
        };
        let a = extract_patch(&raw, 160, 140, e, (112.0, 72.0), Context::default(), 1.0).unwrap();
        let b = extract_patch(
            &crop,
            120,
            120,
            Ellipse {
                center: (60.0, 60.0),
                ..e
            },
            (92.0, 62.0),
            Context::default(),
            1.0,
        )
        .unwrap();
        assert!(a
            .features
            .iter()
            .zip(&b.features)
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn coarse_distance_keeps_external_scale_and_focal_uncertainty() {
        let c = Context::from_coarse_scale(
            Some(Support {
                estimate: 20.0,
                half_width: 5.0,
            }),
            4000.0,
        );
        assert_eq!(c.camera_distance_mm.unwrap().estimate, 200.0);
        assert!(c.camera_distance_mm.unwrap().half_width > 50.0);
        assert!(Context::from_coarse_scale(None, 4000.0)
            .camera_distance_mm
            .is_none());
        assert!(Context::from_coarse_scale(
            Some(Support {
                estimate: 20.0,
                half_width: 25.0
            }),
            4000.0
        )
        .pixels_per_mm
        .is_none());
    }

    #[test]
    fn submerged_prediction_cannot_pull_the_surface_refit_outward() {
        let mut bias = vec![-20.0; OUTPUTS];
        for (role, bin) in [12, 7, 10, 7, 8, 12].into_iter().enumerate() {
            bias[role * BINS + bin] = 20.0;
        }
        let model=Model::from_json(json!({"architecture":ARCHITECTURE,"roles":ROLES,"inputs":INPUTS,
            "hidden":HIDDEN,"bins":BINS,"preprocess":"native-linear-tent-normal-patch-v1",
            "trained_optional_context":[false,false,false],"first_weight":vec![0.0;HIDDEN*INPUTS],
            "first_bias":vec![0.0;HIDDEN],"second_weight":vec![0.0;OUTPUTS*HIDDEN],"second_bias":bias})).unwrap();
        // Use the shared fitter's actual admitted size range, not a tiny
        // synthetic ellipse that it correctly refuses (minor radius <24px).
        let e = Ellipse {
            center: (128.0, 128.0),
            major_radius: 80.0,
            minor_radius: 60.0,
            angle: 0.2,
        };
        let raw: Vec<_> = (0..256 * 256)
            .map(|i| {
                if crate::geometry::ellipse_coordinate(((i % 256) as f64, (i / 256) as f64), e)
                    < 1.0
                {
                    200
                } else {
                    650
                }
            })
            .collect();
        let f = refine(
            &model,
            &raw,
            256,
            256,
            e,
            &e.dense_points(64),
            Context::default(),
        );
        let corrections: Vec<_> = f.samples.iter().filter_map(|p| p.correction_px).collect();
        assert!(corrections.len() >= 10);
        assert!(corrections.iter().all(|x| (x - 1.0).abs() < 1e-6)); // apex, never +9px submerged.
        assert!(f.candidate.is_some());
    }

    #[test]
    fn incompatible_model_cannot_silently_swap_optical_roles() {
        assert!(Model::from_json(json!({"architecture":ARCHITECTURE,
            "roles":["surface_apex","rim"],"inputs":INPUTS,"hidden":HIDDEN,"bins":BINS,
            "preprocess":"native-linear-tent-normal-patch-v1"}))
        .is_err());
    }
}
