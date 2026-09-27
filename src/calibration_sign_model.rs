//! Portable two-frame appearance model. Target-direction supervision is weak;
//! the optional branch head learns conditional RAW/target geometry labels.
//! The older target-only head needs an independently valid screen projection
//! for conic comparison. Neither path establishes measured 3D sign truth.
use serde::{Deserialize, Serialize};

pub const WIDTH: usize = 32;
pub const HEIGHT: usize = 24;
pub const PIXELS: usize = WIDTH * HEIGHT;
pub const INPUTS: usize = PIXELS * 2;
pub const HIDDEN: usize = 32;
pub const CLASSES: usize = 9;
pub const GRID: [[f32; 2]; 9] = [
    [0.1, 0.1],
    [0.5, 0.1],
    [0.9, 0.1],
    [0.1, 0.5],
    [0.5, 0.5],
    [0.9, 0.5],
    [0.1, 0.9],
    [0.5, 0.9],
    [0.9, 0.9],
];

/// Full native ROI, with no fitted ellipse, mask, target or clock as a feature.
/// Average complete CFA cells, area downsample, separable binomial blur, then
/// normalize by this frame's robust 5th/95th percentiles. No old model is read.
pub fn image(raw: &[u8], width: usize, height: usize, stride: usize) -> Result<Vec<f32>, String> {
    let raw = crate::raw10::try_unpack_raw10(raw, width, height, stride)?;
    if height % 4 != 0 || width < WIDTH || height < HEIGHT {
        return Err("unsupported RAW image dimensions".into());
    }
    let cw = width / 4;
    let ch = height / 4;
    let mut cells = vec![0f32; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let mut sum = 0u32;
            for yy in y * 4..y * 4 + 4 {
                for xx in x * 4..x * 4 + 4 {
                    sum += raw[yy * width + xx] as u32;
                }
            }
            cells[y * cw + x] = sum as f32 / 16.;
        }
    }
    let mut small = vec![0f32; PIXELS];
    // Fractional area integration avoids aliasing and includes every source cell.
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let x0 = x as f32 * cw as f32 / WIDTH as f32;
            let x1 = (x + 1) as f32 * cw as f32 / WIDTH as f32;
            let y0 = y as f32 * ch as f32 / HEIGHT as f32;
            let y1 = (y + 1) as f32 * ch as f32 / HEIGHT as f32;
            let mut sum = 0.;
            let mut area = 0.;
            for yy in y0.floor() as usize..(y1.ceil() as usize).min(ch) {
                for xx in x0.floor() as usize..(x1.ceil() as usize).min(cw) {
                    let a = ((xx + 1) as f32).min(x1).sub(x0.max(xx as f32))
                        * ((yy + 1) as f32).min(y1).sub(y0.max(yy as f32));
                    sum += a * cells[yy * cw + xx];
                    area += a;
                }
            }
            small[y * WIDTH + x] = sum / area.max(1e-6);
        }
    }
    let weights = [1., 4., 6., 4., 1.];
    let mut tmp = vec![0.; PIXELS];
    let mut blur = vec![0.; PIXELS];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for (k, w) in weights.iter().enumerate() {
                let xx = (x as isize + k as isize - 2).clamp(0, WIDTH as isize - 1) as usize;
                tmp[y * WIDTH + x] += small[y * WIDTH + xx] * w / 16.;
            }
        }
    }
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for (k, w) in weights.iter().enumerate() {
                let yy = (y as isize + k as isize - 2).clamp(0, HEIGHT as isize - 1) as usize;
                blur[y * WIDTH + x] += tmp[yy * WIDTH + x] * w / 16.;
            }
        }
    }
    let mut sorted = blur.clone();
    sorted.sort_by(f32::total_cmp);
    let lo = sorted[PIXELS / 20];
    let hi = sorted[PIXELS * 19 / 20];
    if hi - lo < 3. {
        return Err("RAW has insufficient measurable contrast".into());
    }
    for v in &mut blur {
        *v = 2. * ((*v - lo) / (hi - lo)).clamp(0., 1.) - 1.;
    }
    Ok(blur)
}
use std::ops::Sub;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub schema: String,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    #[serde(default)]
    pub branch: Option<BranchHead>,
    pub provenance: serde_json::Value,
}
/// Low/high camera-X ordering of the two camera-facing circle normals.
/// Supervision is an explicit conditional RAW/target geometry estimate.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchHead {
    pub weights: Vec<f32>,
    pub biases: [f32; 2],
}
impl BranchHead {
    pub fn zero() -> Self {
        Self {
            weights: vec![0.; 2 * HIDDEN],
            biases: [0.; 2],
        }
    }
    pub fn forward(&self, h: &[f32; HIDDEN]) -> [f32; 2] {
        let logits = [
            dot(&self.weights[..HIDDEN], h) + self.biases[0],
            dot(&self.weights[HIDDEN..], h) + self.biases[1],
        ];
        let a = (1. + (logits[1] - logits[0]).clamp(-80., 80.).exp()).recip();
        [a, 1. - a]
    }
}
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
pub fn class(uv: [f32; 2]) -> usize {
    let c = |u: f32| {
        if u < 0.3 {
            0
        } else if u > 0.7 {
            2
        } else {
            1
        }
    };
    c(uv[1]) * 3 + c(uv[0])
}
impl Model {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != "buttercup-two-frame-target-sign-v1"
            || self.w1.len() != INPUTS * HIDDEN
            || self.b1.len() != HIDDEN
            || self.w2.len() != HIDDEN * CLASSES
            || self.b2.len() != CLASSES
            || self.branch.as_ref().is_some_and(|b| {
                b.weights.len() != 2 * HIDDEN
                    || b.weights.iter().chain(&b.biases).any(|v| !v.is_finite())
            })
            || self
                .w1
                .iter()
                .chain(&self.b1)
                .chain(&self.w2)
                .chain(&self.b2)
                .any(|v| !v.is_finite())
        {
            return Err("invalid sign model".into());
        }
        Ok(())
    }
    pub fn forward(&self, x: &[f32], hidden: &mut [f32; HIDDEN]) -> [f32; CLASSES] {
        assert_eq!(x.len(), INPUTS);
        for (j, h) in hidden.iter_mut().enumerate() {
            *h = (dot(&self.w1[j * INPUTS..(j + 1) * INPUTS], x) + self.b1[j]).max(0.);
        }
        let mut p = [0.; CLASSES];
        for (j, p) in p.iter_mut().enumerate() {
            *p = dot(&self.w2[j * HIDDEN..(j + 1) * HIDDEN], hidden) + self.b2[j];
        }
        let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.;
        for p in &mut p {
            *p = (*p - max).exp();
            z += *p;
        }
        for p in &mut p {
            *p /= z;
        }
        p
    }
    pub fn predict(&self, previous: &[f32], current: &[f32]) -> Result<Prediction, String> {
        self.validate()?;
        if previous.len() != PIXELS
            || current.len() != PIXELS
            || previous.iter().chain(current).any(|v| !v.is_finite())
        {
            return Err("two finite preprocessed ROI frames required".into());
        }
        let x: Vec<_> = previous.iter().chain(current).copied().collect();
        let mut h = [0.; HIDDEN];
        let p = self.forward(&x, &mut h);
        let mut uv = [0.; 2];
        let mut horizontal = [0.; 3];
        let mut vertical = [0.; 3];
        for (i, &p) in p.iter().enumerate() {
            uv[0] += p * GRID[i][0];
            uv[1] += p * GRID[i][1];
            horizontal[i % 3] += p;
            vertical[i / 3] += p;
        }
        Ok(Prediction {
            target_scores: p,
            uv,
            horizontal,
            vertical,
            conditional_branch_scores: self.branch.as_ref().map(|b| b.forward(&h)),
        })
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub target_scores: [f32; 9],
    pub uv: [f32; 2],
    pub horizontal: [f32; 3],
    pub vertical: [f32; 3],
    #[serde(default)]
    pub conditional_branch_scores: Option<[f32; 2]>,
}
#[derive(Serialize)]
pub struct BranchPreference {
    pub selected: Option<usize>,
    pub votes: [f32; 2],
    pub reason: &'static str,
}
impl Prediction {
    /// Returns indices in the supplied candidate order. The learned head needs
    /// no screen projection at inference; its training assumptions still apply.
    pub fn choose_native_branches(&self, normals: Option<[[f64; 3]; 2]>) -> BranchPreference {
        let unavailable = |reason| BranchPreference {
            selected: None,
            votes: [0.; 2],
            reason,
        };
        let Some(scores) = self.conditional_branch_scores else {
            return unavailable("no-trained-conic-sign-head");
        };
        let Some(n) = normals else {
            return unavailable("no-current-conic");
        };
        if n.iter().flatten().any(|v| !v.is_finite())
            || n.iter()
                .any(|v| v[2] <= 0. || (v.iter().map(|x| x * x).sum::<f64>() - 1.).abs() > 0.01)
        {
            return unavailable("invalid-camera-facing-normals");
        }
        if (n[0][0] - n[1][0]).abs() < 0.08 {
            return unavailable("camera-x-branch-order-uncertain");
        }
        let lower = usize::from(n[1][0] < n[0][0]);
        let mut votes = [0.; 2];
        votes[lower] = scores[0];
        votes[1 - lower] = scores[1];
        let best = usize::from(votes[1] > votes[0]);
        BranchPreference {
            selected: (votes[best] >= 0.8).then_some(best),
            votes,
            reason: if votes[best] >= 0.8 {
                "conditional-learned-conic-sign"
            } else {
                "image-sign-ambiguous"
            },
        }
    }
    /// The caller must project both physical conic candidates with the SAME
    /// independently valid camera/display mapping. Missing mapping => abstain.
    /// Scores are uncalibrated target-model support, never mathematical proof.
    pub fn choose_projected_branches(&self, projected: Option<[[f32; 2]; 2]>) -> BranchPreference {
        let Some(p) = projected else {
            return BranchPreference {
                selected: None,
                votes: [0.; 2],
                reason: "screen-projection-unavailable",
            };
        };
        let dist = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        if p.iter().flatten().any(|v| !v.is_finite()) || dist(p[0], p[1]) < 0.15 {
            return BranchPreference {
                selected: None,
                votes: [0.; 2],
                reason: "invalid-or-converged-candidates",
            };
        }
        let mut votes = [0.; 2];
        for (i, g) in GRID.iter().enumerate() {
            let a = dist(*g, p[0]);
            let b = dist(*g, p[1]);
            if (a - b).abs() > 0.05 {
                votes[usize::from(b < a)] += self.target_scores[i];
            }
        }
        let selected = if votes[0] >= 0.8 && dist(self.uv, p[0]) < 0.5 {
            Some(0)
        } else if votes[1] >= 0.8 && dist(self.uv, p[1]) < 0.5 {
            Some(1)
        } else {
            None
        };
        BranchPreference {
            selected,
            votes,
            reason: if selected.is_some() {
                "conditional-target-direction-support"
            } else {
                "target-model-ambiguous"
            },
        }
    }
}
