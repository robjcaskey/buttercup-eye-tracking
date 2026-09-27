//! Iterative binary subdivision of rotation/translation hypotheses. Coarse
//! masked Gaussian images propose branches; the unchanged native score and
//! patch checks make the final decision. This is a bounded beam, not a proof
//! that every photometric minimum has been searched.
use super::geometry::{score, Affine, Features, Region, Samples, Score};
use super::prior::{Fit, Search};
use super::{color, distance, Image, Input, P};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub beam_width: usize,
    pub cycles: usize,
    pub blurred_pyramid: bool,
    pub coarse_factor: usize,
    pub coarse_post_blur_sigma_px: f64,
    pub relative_color: bool,
    pub pivot_rounds: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            beam_width: 32,
            cycles: 6,
            blurred_pyramid: true,
            coarse_factor: 4,
            coarse_post_blur_sigma_px: 0.,
            relative_color: false,
            pivot_rounds: 4,
        }
    }
}
impl Config {
    pub fn valid(&self) -> bool {
        (1..=64).contains(&self.beam_width)
            && (3..=8).contains(&self.cycles)
            && [4, 8].contains(&self.coarse_factor)
            && (0. ..=1.5).contains(&self.coarse_post_blur_sigma_px)
            && (1..=5).contains(&self.pivot_rounds)
    }
}
pub struct Level {
    pub factor: usize,
    pub samples: Samples,
    pub frames: Vec<Features>,
    pub relative: Option<color::Level>,
}
pub struct Pyramid {
    pub config: Config,
    pub levels: Vec<Level>,
    pub native_relative: Option<color::Level>,
}

// Integer-coordinate convention: coarse pixel (x,y) is native (factor*x,
// factor*y). Blur precedes decimation; no half-pixel offset is introduced.
pub fn scaled_warp(m: Affine, factor: usize) -> Affine {
    Affine {
        a: m.a,
        t: m.t.map(|x| x / factor as f64),
    }
}
fn gaussian_axis(v: &[f64], w: usize, h: usize, kernel: &[f64], horizontal: bool) -> Vec<f64> {
    let r = kernel.len() as isize / 2;
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            kernel
                .iter()
                .enumerate()
                .filter_map(|(j, &k)| {
                    let d = j as isize - r;
                    let (a, b) = if horizontal { (x + d, y) } else { (x, y + d) };
                    (a >= 0 && b >= 0 && a < w as isize && b < h as isize)
                        .then(|| k * v[b as usize * w + a as usize])
                })
                .sum()
        })
        .collect()
}
pub fn downsample_masked(im: &Image, mask: &[bool], factor: usize) -> Features {
    gaussian_masked(im, mask, factor, factor as f64 / 2.)
}
fn gaussian_masked(im: &Image, mask: &[bool], factor: usize, sigma: f64) -> Features {
    let radius = (3. * sigma).ceil() as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|x| (-0.5 * (x as f64 / sigma).powi(2)).exp())
        .collect();
    let sum = kernel.iter().sum::<f64>();
    kernel.iter_mut().for_each(|x| *x /= sum);
    let mut weight: Vec<_> = mask.iter().map(|&b| f64::from(b)).collect();
    let mut value: Vec<_> =
        im.v.iter()
            .zip(mask)
            .map(|(&v, &b)| if b { v } else { 0. })
            .collect();
    for horizontal in [true, false] {
        weight = gaussian_axis(&weight, im.w, im.h, &kernel, horizontal);
        value = gaussian_axis(&value, im.w, im.h, &kernel, horizontal);
    }
    let (w, h) = ((im.w - 1) / factor + 1, (im.h - 1) / factor + 1);
    let mut v = Vec::new();
    let mut valid = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let k = y * factor * im.w + x * factor;
            v.push(value[k] / weight[k].max(1e-12));
            valid.push(weight[k] >= 0.90);
        }
    }
    Features {
        image: Image { w, h, v },
        valid,
        center_hint: None,
    }
}
impl Pyramid {
    pub fn new(
        config: Config,
        inputs: &[Input],
        features: &[Features],
        train: &Samples,
        roi: Region,
        smoothing: usize,
    ) -> Self {
        let mut levels = Vec::new();
        let colors = config.relative_color.then(|| {
            inputs
                .iter()
                .map(|i| {
                    let channels = i
                        .rgb
                        .clone()
                        .unwrap_or_else(|| std::array::from_fn(|_| i.image.clone()));
                    // Native inputs still have unsmoothed RGB. Synthetic grayscale
                    // fallbacks already share the prepared image's smoothing.
                    channels.map(|im| {
                        if i.rgb.is_some() {
                            super::geometry::smooth(&im, smoothing)
                        } else {
                            im
                        }
                    })
                })
                .collect::<Vec<_>>()
        });
        let masks: Vec<Vec<bool>> = inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let mut current = roi;
                if i > 0 {
                    current.center = features[i].center_hint.unwrap_or(roi.center);
                }
                features[i]
                    .valid
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        v && current.iris_context(
                            [(k % input.image.w) as f64, (k / input.image.w) as f64],
                            smoothing as f64,
                        )
                    })
                    .collect()
            })
            .collect();
        let color_level = |factor| {
            colors.as_ref().map(|colors| {
                let frames = colors
                    .iter()
                    .enumerate()
                    .map(|(i, channels)| {
                        if factor == 1 {
                            return color::Frame {
                                channels: channels.clone(),
                                valid: masks[i].clone(),
                            };
                        }
                        let blurred: [Features; 3] = std::array::from_fn(|c| {
                            let f = downsample_masked(&channels[c], &masks[i], factor);
                            if factor == config.coarse_factor
                                && config.coarse_post_blur_sigma_px > 0.
                            {
                                gaussian_masked(
                                    &f.image,
                                    &f.valid,
                                    1,
                                    config.coarse_post_blur_sigma_px,
                                )
                            } else {
                                f
                            }
                        });
                        color::Frame {
                            valid: blurred[0].valid.clone(),
                            channels: blurred.map(|f| f.image),
                        }
                    })
                    .collect();
                color::Level::new(frames, &train.points, factor)
            })
        };
        let native_relative = color_level(1);
        if config.blurred_pyramid {
            for factor in [config.coarse_factor, 2] {
                let frames: Vec<_> = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, input)| {
                        let mut current = roi;
                        if i > 0 {
                            current.center = features[i].center_hint.unwrap_or(roi.center);
                        }
                        let mask: Vec<_> = features[i]
                            .valid
                            .iter()
                            .enumerate()
                            .map(|(k, &v)| {
                                v && current.iris_context(
                                    [(k % input.image.w) as f64, (k / input.image.w) as f64],
                                    smoothing as f64,
                                )
                            })
                            .collect();
                        let coarse = downsample_masked(&input.image, &mask, factor);
                        if factor == config.coarse_factor && config.coarse_post_blur_sigma_px > 0. {
                            // Blur the already downsampled image, keeping its
                            // invalid pixels out of both numerator and weight.
                            gaussian_masked(
                                &coarse.image,
                                &coarse.valid,
                                1,
                                config.coarse_post_blur_sigma_px,
                            )
                        } else {
                            coarse
                        }
                    })
                    .collect();
                let mut samples = Samples {
                    points: Vec::new(),
                    values: Vec::new(),
                    center: train.center.map(|x| x / factor as f64),
                };
                for &p in &train.points {
                    let q = p.map(|x| x / factor as f64);
                    if let Some(v) = frames[0].sample(q) {
                        samples.points.push(q);
                        samples.values.push(v);
                    }
                }
                levels.push(Level {
                    factor,
                    samples,
                    frames,
                    relative: color_level(factor),
                });
            }
        }
        Self {
            config,
            levels,
            native_relative,
        }
    }
    fn level(&self, cycle: usize) -> Option<&Level> {
        if !self.config.blurred_pyramid || cycle * 3 >= self.config.cycles * 2 {
            return None;
        }
        let i = if cycle * 3 < self.config.cycles { 0 } else { 1 };
        self.levels.get(i).filter(|l| {
            l.relative
                .as_ref()
                .map_or(l.samples.points.len(), color::Level::support)
                >= 12
        })
    }
    pub fn fit(
        &self,
        frame: usize,
        search: &Search<'_>,
        train: &Samples,
        target: &Features,
        pivot: [f64; 3],
        delta: P,
        start: Option<([f64; 3], P)>,
    ) -> (Fit, Vec<Step>, usize) {
        let limits = search.prior.angle_limit_deg.map(f64::to_radians);
        let bound = search.prior.translation_radii * search.radius();
        let dimensions = if bound > 0. { 5 } else { 3 };
        let mut rough = [0.; 5];
        if let Some(hint) = target.center_hint {
            rough[0] =
                ((hint[1] - train.center[1] - delta[1]) / pivot[2]).clamp(-limits[0], limits[0]);
            rough[1] =
                (-(hint[0] - train.center[0] - delta[0]) / pivot[2]).clamp(-limits[1], limits[1]);
            let q = search
                .warp(pivot, [rough[0], rough[1], 0.], delta, [0.; 2])
                .map(train.center);
            let t = [hint[0] - q[0], hint[1] - q[1]];
            let scale = (bound / t[0].hypot(t[1]).max(1e-12)).min(1.);
            rough[3] = t[0] * scale;
            rough[4] = t[1] * scale;
        }
        let mut cells = vec![Cell {
            center: [0.; 5],
            probe: [0.; 5],
            half: [limits[0], limits[1], limits[2], bound, bound],
            objective: 0.,
            ncc: 0.,
        }];
        let mut trace = Vec::new();
        let mut evaluated = 0;
        for cycle in 0..self.config.cycles {
            let level = self.level(cycle);
            for axis in 0..dimensions {
                let mut children = Vec::new();
                for cell in &cells {
                    for sign in [-1., 1.] {
                        let mut q = *cell;
                        q.half[axis] *= 0.5;
                        q.center[axis] += sign * q.half[axis];
                        q.objective = f64::INFINITY;
                        let guided: [f64; 5] = std::array::from_fn(|k| {
                            rough[k].clamp(q.center[k] - q.half[k], q.center[k] + q.half[k])
                        });
                        // A bad midpoint is not evidence that its whole interval
                        // misses the target. Probe the existing coarse-location
                        // estimate inside each child as well as its midpoint.
                        for probe in [q.center, guided] {
                            let w = [probe[0], probe[1], probe[2]];
                            let t = [probe[3], probe[4]];
                            if !search.feasible(w, t) {
                                continue;
                            }
                            let warp = search.warp(pivot, w, delta, t);
                            let sc = if target
                                .center_hint
                                .is_some_and(|c| distance(warp.map(train.center), c) > 16.)
                            {
                                Score {
                                    loss: 5.,
                                    ncc: -1.,
                                    coverage: 0.,
                                }
                            } else if let Some(l) = level {
                                if let Some(relative) = &l.relative {
                                    relative.score(frame, scaled_warp(warp, l.factor))
                                } else {
                                    score(&l.samples, &l.frames[frame], scaled_warp(warp, l.factor))
                                }
                            } else if let Some(relative) = &self.native_relative {
                                relative.score(frame, warp)
                            } else {
                                score(train, target, warp)
                            };
                            let objective = sc.loss + search.motion_penalty(w, t);
                            if objective < q.objective {
                                q.objective = objective;
                                q.ncc = sc.ncc;
                                q.probe = probe;
                            }
                            evaluated += 1;
                        }
                        if q.objective.is_finite() {
                            children.push(q);
                        }
                    }
                }
                children.sort_by(|a, b| a.objective.total_cmp(&b.objective));
                let scored = children.len();
                children.truncate(self.config.beam_width);
                // The root domain contains zero translation, so at least one
                // child remains inside the hard translation ball each round.
                assert!(!children.is_empty(), "binary search lost feasible domain");
                let best = children[0];
                trace.push(Step {
                    cycle,
                    axis,
                    factor: level.map_or(1, |l| l.factor),
                    scored,
                    retained: children.len(),
                    fitting_samples: level.map_or_else(
                        || {
                            self.native_relative
                                .as_ref()
                                .map_or(train.points.len(), color::Level::support)
                        },
                        |l| {
                            l.relative
                                .as_ref()
                                .map_or(l.samples.points.len(), color::Level::support)
                        },
                    ),
                    center: best.probe,
                    interval_center: best.center,
                    half_width: best.half,
                    objective: best.objective,
                    ncc: best.ncc,
                    runner_up: children.get(1).map(|q| q.objective),
                });
                cells = children;
            }
        }
        let mut starts: Vec<_> = cells
            .iter()
            .take(8)
            .map(|c| {
                (
                    [c.probe[0], c.probe[1], c.probe[2]],
                    [c.probe[3], c.probe[4]],
                )
            })
            .collect();
        starts.push(([0.; 3], [0.; 2]));
        if let Some(p) = start {
            starts.push(p);
        }
        let fit = starts
            .into_iter()
            .map(|(w, t)| {
                if let Some(relative) = &self.native_relative {
                    search.refine_scored(pivot, delta, w, t, |warp| {
                        if target
                            .center_hint
                            .is_some_and(|c| distance(warp.map(train.center), c) > 16.)
                        {
                            Score {
                                loss: 5.,
                                ncc: -1.,
                                coverage: 0.,
                            }
                        } else {
                            relative.score(frame, warp)
                        }
                    })
                } else {
                    search.refine(train, target, pivot, delta, w, t)
                }
            })
            .min_by(|a, b| a.objective.total_cmp(&b.objective))
            .unwrap();
        (fit, trace, evaluated)
    }
}
#[derive(Clone, Copy)]
struct Cell {
    center: [f64; 5],
    probe: [f64; 5],
    half: [f64; 5],
    objective: f64,
    ncc: f64,
}
#[derive(Clone, Serialize)]
pub struct Step {
    pub cycle: usize,
    pub axis: usize,
    pub factor: usize,
    pub scored: usize,
    pub retained: usize,
    pub fitting_samples: usize,
    pub center: [f64; 5],
    pub interval_center: [f64; 5],
    pub half_width: [f64; 5],
    pub objective: f64,
    pub ncc: f64,
    pub runner_up: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decimated_affine_keeps_crop_rotation_and_translation_in_native_units() {
        let m = Affine {
            a: [[0.97, 0.04], [-0.03, 1.02]],
            t: [24.5, -11.2],
        };
        let p = [150.25, 104.75];
        for factor in [1, 2, 4, 8] {
            let q = scaled_warp(m, factor).map(p.map(|x| x / factor as f64));
            assert!(distance(q.map(|x| x * factor as f64), m.map(p)) < 1e-10);
        }
    }
    #[test]
    fn masked_blur_never_imports_excluded_bright_pixels() {
        let mut a = Image {
            w: 64,
            h: 64,
            v: vec![0.2; 4096],
        };
        let mut mask = vec![true; 4096];
        for y in 27..36 {
            for x in 27..36 {
                mask[y * 64 + x] = false;
                a.v[y * 64 + x] = 1e6;
            }
        }
        for factor in [4, 8] {
            let coarse = downsample_masked(&a, &mask, factor);
            let mut contaminated = coarse.image.clone();
            for (v, &ok) in contaminated.v.iter_mut().zip(&coarse.valid) {
                if !ok {
                    *v = 1e6;
                }
            }
            let extra = gaussian_masked(&contaminated, &coarse.valid, 1, 0.5);
            for blurred in [&coarse, &extra] {
                assert!(blurred.valid.iter().any(|&b| b));
                for (&v, &ok) in blurred.image.v.iter().zip(&blurred.valid) {
                    if ok {
                        assert!((v - 0.2).abs() < 1e-10);
                    }
                }
                assert!(!blurred.valid[(32 / factor) * blurred.image.w + 32 / factor]);
            }
        }
    }
}
