//! Local per-channel differences, evaluated at warped neighbor coordinates.
//! Raw CFA phase is retained; display white balance never enters this score.
use super::geometry::{Affine, Score};
use super::{Image, P};

pub fn raw_rgb(raw: &[u16], w: usize, h: usize, sx: usize, sy: usize) -> [Image; 3] {
    let mut out: [Image; 3] = std::array::from_fn(|_| Image {
        w,
        h,
        v: vec![0.; w * h],
    });
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut count) = ([0.; 3], [0usize; 3]);
            for dy in -1isize..=2 {
                for dx in -1isize..=2 {
                    let a = x.saturating_add_signed(dx).min(w - 1);
                    let b = y.saturating_add_signed(dy).min(h - 1);
                    let c = match (((a + sx) / 2) % 2, ((b + sy) / 2) % 2) {
                        (0, 0) => 0,
                        (1, 1) => 2,
                        _ => 1,
                    };
                    sum[c] += raw[b * w + a] as f64;
                    count[c] += 1;
                }
            }
            for c in 0..3 {
                out[c].v[y * w + x] = sum[c] / (count[c].max(1) as f64 * 1023.);
            }
        }
    }
    out
}
pub struct Frame {
    pub channels: [Image; 3],
    pub valid: Vec<bool>,
}
impl Frame {
    fn sample(&self, p: P) -> Option<[f64; 3]> {
        let im = &self.channels[0];
        if p.iter().any(|x| !x.is_finite())
            || p[0] < 1.
            || p[1] < 1.
            || p[0] >= im.w as f64 - 2.
            || p[1] >= im.h as f64 - 2.
        {
            return None;
        }
        let (x, y) = (p[0] as usize, p[1] as usize);
        if [
            y * im.w + x,
            y * im.w + x + 1,
            (y + 1) * im.w + x,
            (y + 1) * im.w + x + 1,
        ]
        .iter()
        .any(|&k| !self.valid[k])
        {
            return None;
        }
        Some([
            self.channels[0].sample(p)?,
            self.channels[1].sample(p)?,
            self.channels[2].sample(p)?,
        ])
    }
}
struct Sample {
    p: P,
    differences: [Option<[f64; 3]>; 4],
}
pub struct Level {
    pub frames: Vec<Frame>,
    source: Vec<Sample>,
    offset: f64,
    requested: usize,
}
fn offsets(r: f64) -> [P; 4] {
    [[-r, 0.], [r, 0.], [0., -r], [0., r]]
}
impl Level {
    pub fn new(frames: Vec<Frame>, points: &[P], factor: usize) -> Self {
        let offset = (6. / factor as f64).max(1.);
        let source: Vec<_> = points
            .iter()
            .filter_map(|&p| {
                let p = p.map(|x| x / factor as f64);
                let center = frames[0].sample(p)?;
                let differences = offsets(offset).map(|d| {
                    frames[0]
                        .sample([p[0] + d[0], p[1] + d[1]])
                        .map(|v| std::array::from_fn(|c| v[c] - center[c]))
                });
                (differences.iter().flatten().count() >= 2).then_some(Sample { p, differences })
            })
            .collect();
        let requested = source.len();
        Self {
            frames,
            source,
            offset,
            requested,
        }
    }
    pub fn support(&self) -> usize {
        self.source.len()
    }
    pub fn points(&self) -> Vec<P> {
        self.source.iter().map(|s| s.p).collect()
    }
    pub fn score(&self, frame: usize, warp: Affine) -> Score {
        let target = &self.frames[frame];
        let mut loss = 0.;
        let mut used = 0usize;
        for s in &self.source {
            let Some(center) = target.sample(warp.map(s.p)) else {
                continue;
            };
            let mut a = [[0.; 4]; 3];
            let mut b = [[0.; 4]; 3];
            let mut n = 0;
            for (j, d) in offsets(self.offset).into_iter().enumerate() {
                let Some(reference) = s.differences[j] else {
                    continue;
                };
                let Some(v) = target.sample(warp.map([s.p[0] + d[0], s.p[1] + d[1]])) else {
                    continue;
                };
                for c in 0..3 {
                    a[c][n] = reference[c];
                    b[c][n] = v[c] - center[c];
                }
                n += 1;
            }
            if n < 2 {
                continue;
            }
            let mut patch = 0.;
            let mut channels = 0;
            for c in 0..3 {
                let sa = (a[c][..n].iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
                let sb = (b[c][..n].iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
                // A flat target cannot earn a match by producing zero residuals.
                if sa < 1e-5 {
                    continue;
                }
                let denom_a = sa.max(2. / 1023.);
                let denom_b = sb.max(2. / 1023.);
                patch += if sb < 1e-5 {
                    1.
                } else {
                    (0..n)
                        .map(|j| {
                            let e = a[c][j] / denom_a - b[c][j] / denom_b;
                            // Bounded Cauchy residual: a few changing neighbors do
                            // not dominate the shared-motion fit.
                            e * e / (0.25 + e * e)
                        })
                        .sum::<f64>()
                        / n as f64
                };
                channels += 1;
            }
            if channels == 0 {
                continue;
            }
            loss += patch / channels as f64;
            used += 1;
        }
        let coverage = used as f64 / self.requested.max(1) as f64;
        let residual = if used >= 12 { loss / used as f64 } else { 1. };
        Score {
            loss: residual
                + 0.25 * (1. - coverage)
                + if coverage < 0.40 || used < 12 { 3. } else { 0. },
            ncc: 1. - residual,
            coverage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_decode_respects_sensor_phase() {
        for (sx, sy) in [(0, 0), (1, 2), (3, 1)] {
            let raw: Vec<_> = (0..32 * 32)
                .map(
                    |i| match (((i % 32 + sx) / 2) % 2, ((i / 32 + sy) / 2) % 2) {
                        (0, 0) => 100,
                        (1, 1) => 700,
                        _ => 400,
                    },
                )
                .collect();
            let rgb = raw_rgb(&raw, 32, 32, sx, sy);
            for (c, want) in [100., 400., 700.].into_iter().enumerate() {
                assert!((rgb[c].v[16 * 32 + 16] - want / 1023.).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn relative_channels_recover_motion_under_independent_gain_and_offset() {
        let make = |gain: [f64; 3], offset: [f64; 3], shift: P| Frame {
            channels: std::array::from_fn(|c| Image {
                w: 96,
                h: 80,
                v: (0..96 * 80)
                    .map(|i| {
                        let x = i % 96;
                        let y = i / 96;
                        let f = 0.3
                            + 0.06 * ((x as f64 - shift[0]) * (0.13 + c as f64 * 0.07)).sin()
                            + 0.05 * ((y as f64 - shift[1]) * (0.17 + c as f64 * 0.04)).cos();
                        gain[c] * f + offset[c]
                    })
                    .collect(),
            }),
            valid: vec![true; 96 * 80],
        };
        let points: Vec<_> = (20..60)
            .step_by(5)
            .flat_map(|y| (20..76).step_by(5).map(move |x| [x as f64, y as f64]))
            .collect();
        let level = Level::new(
            vec![
                make([1.; 3], [0.; 3], [0.; 2]),
                make([0.6, 1.4, 0.9], [0.11, -0.03, 0.06], [5., -3.]),
                Frame {
                    channels: std::array::from_fn(|_| Image {
                        w: 96,
                        h: 80,
                        v: vec![0.3; 96 * 80],
                    }),
                    valid: vec![true; 96 * 80],
                },
            ],
            &points,
            1,
        );
        let good = level.score(1, Affine::translation([5., -3.]));
        let wrong = level.score(1, Affine::translation([-5., 3.]));
        let blank = level.score(2, Affine::translation([5., -3.]));
        assert!(
            good.loss < 1e-10 && wrong.loss > 0.2 && blank.loss > 0.9,
            "{good:?} {wrong:?} {blank:?}"
        );
    }
}
