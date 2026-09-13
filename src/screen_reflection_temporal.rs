//! Spatially unresolved optical clock: recover a checked temporal color word.
//! No host arrival clock, iris estimates or learned features enter the fit.
use crate::screen_reflection_raw::PackedRaw10;
use std::sync::OnceLock;

pub const SCHEME: &str = "temporal-lfsr10-63-v1";
pub const PERIOD: usize = 1023;
pub const WORD_BITS: usize = 63;
pub const DEFAULT_CODE_HZ: f64 = 5.0;
pub const DEFAULT_AMPLITUDE: f64 = 0.12;
pub const BASE_RGB: [f64; 3] = [0.39, 0.405, 0.42];
pub const OPPONENT_AXIS_RGB: [f64; 3] = [0.55, -0.30, 1.00];

/// Identical opponent-color carrier for standalone and border presentations.
pub fn symbol_rgb(sign: i8, amplitude: f64) -> [f64; 3] {
    std::array::from_fn(|i| BASE_RGB[i] + f64::from(sign) * amplitude * OPPONENT_AXIS_RGB[i])
}

pub fn symbol_pixel(code: u64, session: u8, amplitude: f64) -> u32 {
    let rgb = symbol_rgb(sign(code, session), amplitude);
    rgb.into_iter().fold(0, |pixel, value| {
        (pixel << 8) | (value.clamp(0., 1.) * 255.).round() as u32
    })
}
const MAX_ERRORS: u32 = 6; // Codebook minimum distance is 17, verified exhaustively.

fn codebook() -> &'static ([bool; PERIOD], [u64; PERIOD]) {
    static BOOK: OnceLock<([bool; PERIOD], [u64; PERIOD])> = OnceLock::new();
    BOOK.get_or_init(|| {
        let mut bits = [false; PERIOD];
        let mut state = 1u16;
        for bit in &mut bits {
            *bit = state & 1 != 0;
            state = (state >> 1) | (((state ^ (state >> 3)) & 1) << 9);
        }
        let words = std::array::from_fn(|i| {
            (0..WORD_BITS).fold(0, |w, j| w | ((bits[(i + j) % PERIOD] as u64) << j))
        });
        (bits, words)
    })
}
pub fn sign(code: u64, session: u8) -> i8 {
    if codebook().0[(code as usize + session as usize * 61) % PERIOD] {
        1
    } else {
        -1
    }
}

/// Ratio of native Quad-Bayer plane means; exposure gain approximately cancels.
/// The entire fixed crop contributes, so a blurred/diffuse reflection is usable.
pub fn photometry(raw: PackedRaw10<'_>) -> Option<f64> {
    let mut sums = [0.; 4];
    let mut counts = [0usize; 4];
    // Sample each complete native 4x4 carrier, without a preview or demosaic.
    let sx = (4 - raw.sensor_x as usize % 4) % 4;
    let sy = (4 - raw.sensor_y as usize % 4) % 4;
    for y in (sy..raw.height.saturating_sub(3)).step_by(4) {
        for x in (sx..raw.width.saturating_sub(3)).step_by(4) {
            for (band, (dx, dy)) in [(0, 0), (2, 0), (0, 2), (2, 2)].into_iter().enumerate() {
                let v = (raw.pixel(x + dx, y + dy) as f64
                    + raw.pixel(x + dx + 1, y + dy) as f64
                    + raw.pixel(x + dx, y + dy + 1) as f64
                    + raw.pixel(x + dx + 1, y + dy + 1) as f64)
                    / 4.;
                if (2.0..1000.0).contains(&v) {
                    sums[band] += v;
                    counts[band] += 1;
                }
            }
        }
    }
    if counts.iter().any(|n| *n < 32) {
        return None;
    }
    let mean: [f64; 4] = std::array::from_fn(|b| sums[b] / counts[b] as f64);
    Some((mean[3] + 1.).ln() - 0.5 * ((mean[1] + 1.).ln() + (mean[2] + 1.).ln()))
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub sensor_ns: u64,
    pub value: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub code_mod: u64,
    pub polarity: i8,
    pub bit_errors: u32,
    pub runner_errors: u32,
    pub correlation: f64,
    pub fractional_phase: f64,
    pub phase_width_ticks: f64,
    pub levels_span: f64,
}

pub fn recover(samples: &[Sample], hz: f64, session: u8) -> Option<Fit> {
    if samples.len() < WORD_BITS
        || !hz.is_finite()
        || hz <= 0.
        || samples.iter().any(|s| !s.value.is_finite())
        || samples.windows(2).any(|w| w[1].sensor_ns <= w[0].sensor_ns)
    {
        return None;
    }
    let anchor = samples.first()?.sensor_ns;
    let duration = (samples.last()?.sensor_ns - anchor) as f64 / 1e9;
    if duration * hz < WORD_BITS as f64 + 1. || duration * hz > 256. {
        return None;
    }
    let mut candidates = Vec::new();
    for phase_step in 0..16 {
        let phase = phase_step as f64 / 16.;
        let mut sum = vec![0.; 260];
        let mut count = vec![0usize; 260];
        for s in samples {
            let tick = (s.sensor_ns - anchor) as f64 / 1e9 * hz + phase;
            // Leave transition/exposure mixtures out, without choosing by code.
            if !(0.15..=0.85).contains(&tick.fract()) {
                continue;
            }
            let i = tick.floor() as usize;
            sum[i] += s.value;
            count[i] += 1;
        }
        let last_tick = duration * hz + phase;
        let end = last_tick.floor() as usize;
        if end < WORD_BITS {
            continue;
        }
        let start = end - WORD_BITS;
        if count[start..end].contains(&0) {
            continue;
        }
        let values: Vec<_> = (start..end).map(|i| sum[i] / count[i] as f64).collect();
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let span = sorted[56] - sorted[6];
        if span < 0.002 {
            continue;
        }
        let threshold = (sorted[56] + sorted[6]) * 0.5;
        let word = values
            .iter()
            .enumerate()
            .fold(0u64, |w, (i, v)| w | (((*v > threshold) as u64) << i));
        let mut best = (u32::MAX, 0usize, 1i8);
        let mut runner = u32::MAX;
        for (index, reference) in codebook().1.iter().enumerate() {
            let errors = (word ^ reference).count_ones();
            // Unknown fixed optical/color response polarity is a nuisance
            // parameter. Both branches compete in the SAME nearest-word test.
            for (errors, polarity) in [(errors, 1), (WORD_BITS as u32 - errors, -1)] {
                if errors < best.0 {
                    runner = best.0;
                    best = (errors, index, polarity);
                } else {
                    runner = runner.min(errors);
                }
            }
        }
        if best.0 > MAX_ERRORS || runner < best.0 + 5 {
            continue;
        }
        let signs: Vec<_> = (0..WORD_BITS)
            .map(|i| {
                best.2 as f64
                    * if codebook().0[(best.1 + i) % PERIOD] {
                        1.
                    } else {
                        -1.
                    }
            })
            .collect();
        let mean = values.iter().sum::<f64>() / WORD_BITS as f64;
        let sm = signs.iter().sum::<f64>() / WORD_BITS as f64;
        let dot = values
            .iter()
            .zip(&signs)
            .map(|(v, s)| (v - mean) * (s - sm))
            .sum::<f64>();
        let denom = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            * signs.iter().map(|s| (s - sm).powi(2)).sum::<f64>())
        .sqrt();
        let correlation = dot / denom;
        if !correlation.is_finite() || correlation < 0.70 {
            continue;
        }
        let code = (best.1 + PERIOD - session as usize * 61 % PERIOD + WORD_BITS) % PERIOD;
        // The checked word establishes phase, but must not turn a missing or
        // contradictory current exposure into a fresh optical observation.
        let current = samples.last()?.value;
        if (current - threshold) * (sign(code as u64, session) as f64) * (best.2 as f64)
            < span * 0.15
            || !(0.15..=0.85).contains(&last_tick.fract())
        {
            continue;
        }
        candidates.push(Fit {
            code_mod: code as u64,
            polarity: best.2,
            bit_errors: best.0,
            runner_errors: runner,
            correlation,
            fractional_phase: phase,
            phase_width_ticks: 0.,
            levels_span: span,
        });
    }
    let best = *candidates
        .iter()
        .max_by(|a, b| a.correlation.total_cmp(&b.correlation))?;
    let comparable: Vec<_> = candidates
        .iter()
        .filter(|c| c.bit_errors <= best.bit_errors + 1 && c.correlation >= best.correlation - 0.03)
        .collect();
    // If equally supported phases disagree on this exposure's code, abstain.
    if comparable
        .iter()
        .any(|c| c.code_mod != best.code_mod || c.polarity != best.polarity)
    {
        return None;
    }
    let lo = comparable
        .iter()
        .map(|c| c.fractional_phase)
        .fold(1., f64::min);
    let hi = comparable
        .iter()
        .map(|c| c.fractional_phase)
        .fold(0., f64::max);
    Some(Fit {
        phase_width_ticks: (hi - lo + 1. / 16.).min(1.),
        ..best
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_words_have_checked_distance_seventeen() {
        let book = &codebook().1;
        let mut minimum = 63;
        let mut inverted_minimum = 63;
        for i in 0..PERIOD {
            for j in 0..i {
                let distance = (book[i] ^ book[j]).count_ones();
                minimum = minimum.min(distance);
                inverted_minimum = inverted_minimum.min(63 - distance);
            }
        }
        assert_eq!(minimum, 17);
        assert_eq!(inverted_minimum, 24);
    }
    fn fixture() -> Vec<Sample> {
        (0..400)
            .map(|i| {
                let t = i as f64 * 0.047;
                let code = (t * 5. + 0.37).floor() as u64 + 70;
                Sample {
                    sensor_ns: 10_000_000_000 + (t * 1e9) as u64,
                    value: 0.3 + 0.04 * sign(code, 9) as f64 + 0.0005 * ((i * 17 % 11) as f64 - 5.),
                }
            })
            .collect()
    }
    #[test]
    fn recovers_clock_without_host_time_and_rejects_constant_reversed() {
        let samples = fixture();
        let fit = recover(&samples, 5., 9).unwrap();
        let actual = (399.0_f64 * 0.047 * 5. + 0.37).floor() as u64 + 70;
        assert_eq!(fit.code_mod, actual);
        assert_eq!(fit.bit_errors, 0);
        let inverted: Vec<_> = samples
            .iter()
            .map(|s| Sample {
                value: -s.value,
                ..*s
            })
            .collect();
        let inv = recover(&inverted, 5., 9).unwrap();
        assert_eq!(inv.code_mod, actual);
        assert_eq!(inv.polarity, -1);
        let mut negative = samples.clone();
        for s in &mut negative {
            s.value = 1.;
        }
        assert!(recover(&negative, 5., 9).is_none());
        for i in 0..negative.len() {
            negative[i].value = samples[samples.len() - 1 - i].value;
        }
        assert!(recover(&negative, 5., 9).is_none());
    }
    #[test]
    fn duplicate_and_short_clocks_do_not_lock() {
        let mut s = fixture();
        assert!(recover(&s[..50], 5., 9).is_none());
        s[20].sensor_ns = s[19].sensor_ns;
        assert!(recover(&s, 5., 9).is_none());
    }
}
