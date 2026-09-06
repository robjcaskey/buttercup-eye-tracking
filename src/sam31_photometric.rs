//! Bounded, sensor-referenced photometry for the opt-in live SAM adapter.
//!
//! These are engineering correspondence and adaptation bounds, not calibrated
//! illumination probabilities. Samples at a shared sensor coordinate are not
//! assumed to follow moving anatomy. A coherent, broad set of unclipped samples
//! must agree before an illumination ratio can update the running transform.
//! Repeated or reordered sources never advance adaptation or reference age.

const MAX_GAP_NS: u64 = 900_000_000;
const MAX_SAMPLES: usize = 4096;
const MIN_MATCHED_SAMPLES: usize = 32;
const MAX_LOG_RATE_PER_SECOND: f32 = 3.0;
const MAX_RATIO_MAD: f32 = 0.055;
const MIN_RATIO_AGREEMENT: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Source {
    pub epoch: u64,
    pub prompt_generation: u64,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Parameters {
    pub gains: [f32; 3],
    pub low: [f32; 3],
    pub high: [f32; 3],
}

impl Parameters {
    fn normalized(self, rgb: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|c| {
            ((rgb[c] * self.gains[c] - self.low[c])
                / (self.high[c] - self.low[c]).max(1e-6))
            .clamp(0.0, 1.0)
        })
    }

    fn valid(self) -> bool {
        (0..3).all(|c| {
            self.gains[c].is_finite()
                && (0.25..=4.0).contains(&self.gains[c])
                && self.low[c].is_finite()
                && self.high[c].is_finite()
                && self.high[c] > self.low[c]
        })
    }
}

/// Sorted by (sensor_y, sensor_x); values precede white balance and quantization.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sample {
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub rgb: [f32; 3],
}

impl Sample {
    fn key(self) -> (u32, u32) {
        (self.sensor_y, self.sensor_x)
    }
}

#[derive(Clone, Debug)]
struct Reference {
    source: Source,
    samples: Vec<Sample>,
    parameters: Parameters,
    legacy_parameters: Parameters,
    lighting_source: Source,
    lighting_samples: Vec<Sample>,
    // Effective per-channel RAW bounds. Retaining the target separately avoids
    // losing the unapplied part of a lighting step when adaptation is rate-limited.
    target_raw_low: [f32; 3],
    target_raw_high: [f32; 3],
    target_gains: [f32; 3],
    lighting_reference_sequence: u64,
    lighting_reference_timestamp_ns: u64,
}

#[derive(Default, Debug)]
pub(super) struct State {
    reference: Option<Reference>,
}

#[derive(Clone, Debug)]
pub(super) struct Diagnostics {
    pub reason: &'static str,
    pub source_advanced: bool,
    pub reference_sequence: u64,
    pub reference_timestamp_ns: u64,
    pub source_dt_ns: u64,
    pub crop_overlap_fraction: f32,
    pub common_samples: usize,
    pub usable_samples: [usize; 3],
    pub log_ratio: [Option<f32>; 3],
    pub log_ratio_mad: [Option<f32>; 3],
    pub illumination_supported: bool,
    pub lighting_reference_sequence: u64,
    pub lighting_reference_timestamp_ns: u64,
    pub ratio_reference_sequence: u64,
    pub ratio_reference_timestamp_ns: u64,
    pub lighting_common_samples: usize,
    pub parameters_before: Parameters,
    pub parameters_after: Parameters,
    pub per_crop_candidate: Parameters,
    /// Linear normalized RGB differences on shared sensor samples, not a
    /// motion-compensated or calibrated image-quality error.
    pub common_normalized_mean_absolute_delta: Option<f32>,
    pub candidate_common_normalized_mean_absolute_delta: Option<f32>,
}

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

fn overlap_fraction(a: Source, b: Source) -> f32 {
    let x0 = (a.sensor_x as u64).max(b.sensor_x as u64);
    let y0 = (a.sensor_y as u64).max(b.sensor_y as u64);
    let x1 = (a.sensor_x as u64 + a.width as u64)
        .min(b.sensor_x as u64 + b.width as u64);
    let y1 = (a.sensor_y as u64 + a.height as u64)
        .min(b.sensor_y as u64 + b.height as u64);
    (x1.saturating_sub(x0) * y1.saturating_sub(y0)) as f32
        / ((a.width * a.height).max(b.width * b.height).max(1) as f32)
}

fn common_pairs<'a>(previous: &'a [Sample], current: &'a [Sample]) -> Vec<(&'a Sample, &'a Sample)> {
    let (mut a, mut b) = (0, 0);
    let mut pairs = Vec::with_capacity(previous.len().min(current.len()));
    while a < previous.len() && b < current.len() {
        match previous[a].key().cmp(&current[b].key()) {
            std::cmp::Ordering::Less => a += 1,
            std::cmp::Ordering::Greater => b += 1,
            std::cmp::Ordering::Equal => {
                pairs.push((&previous[a], &current[b]));
                a += 1;
                b += 1;
            }
        }
    }
    pairs
}

fn normalized_delta(
    pairs: &[(&Sample, &Sample)],
    previous: Parameters,
    current: Parameters,
) -> Option<f32> {
    (!pairs.is_empty()).then(|| {
        pairs
            .iter()
            .map(|(a, b)| {
                let a = previous.normalized(a.rgb);
                let b = current.normalized(b.rgb);
                (0..3).map(|c| (a[c] - b[c]).abs()).sum::<f32>()
            })
            .sum::<f32>()
            / (pairs.len() * 3) as f32
    })
}

impl State {
    #[cfg(test)]
    pub fn update(
        &mut self,
        source: Source,
        samples: Vec<Sample>,
        candidate: Parameters,
    ) -> Result<Diagnostics, String> {
        self.update_with_legacy(source, samples, candidate, candidate)
    }

    pub fn update_with_legacy(
        &mut self,
        source: Source,
        samples: Vec<Sample>,
        candidate: Parameters,
        legacy_candidate: Parameters,
    ) -> Result<Diagnostics, String> {
        if source.width == 0
            || source.height == 0
            || !candidate.valid()
            || !legacy_candidate.valid()
            || samples.len() > MAX_SAMPLES
            || samples.windows(2).any(|pair| pair[0].key() >= pair[1].key())
            || samples.iter().any(|sample| {
                !sample.rgb.iter().all(|v| v.is_finite() && *v >= 0.0)
                    || sample.sensor_x < source.sensor_x
                    || sample.sensor_y < source.sensor_y
                    || sample.sensor_x as u64 >= source.sensor_x as u64 + source.width as u64
                    || sample.sensor_y as u64 >= source.sensor_y as u64 + source.height as u64
            })
        {
            return Err("invalid or unbounded sensor photometric observation".to_string());
        }
        let previous = self.reference.as_ref();
        let mut diagnostics = Diagnostics {
            reason: "initialized",
            source_advanced: true,
            reference_sequence: previous.map_or(source.sequence, |r| r.source.sequence),
            reference_timestamp_ns: previous.map_or(source.timestamp_ns, |r| r.source.timestamp_ns),
            source_dt_ns: previous.map_or(0, |r| source.timestamp_ns.saturating_sub(r.source.timestamp_ns)),
            crop_overlap_fraction: previous.map_or(0.0, |r| overlap_fraction(r.source, source)),
            common_samples: 0,
            usable_samples: [0; 3],
            log_ratio: [None; 3],
            log_ratio_mad: [None; 3],
            illumination_supported: false,
            lighting_reference_sequence: previous.map_or(source.sequence, |r| r.lighting_reference_sequence),
            lighting_reference_timestamp_ns: previous.map_or(source.timestamp_ns, |r| r.lighting_reference_timestamp_ns),
            ratio_reference_sequence: previous.map_or(source.sequence, |r| r.lighting_source.sequence),
            ratio_reference_timestamp_ns: previous.map_or(source.timestamp_ns, |r| r.lighting_source.timestamp_ns),
            lighting_common_samples: 0,
            parameters_before: previous.map_or(candidate, |r| r.parameters),
            parameters_after: candidate,
            per_crop_candidate: candidate,
            common_normalized_mean_absolute_delta: None,
            candidate_common_normalized_mean_absolute_delta: None,
        };
        let identity_reset = previous.and_then(|r| {
            if r.source.epoch != source.epoch || r.source.prompt_generation != source.prompt_generation {
                Some("identity-or-prompt-changed")
            } else {
                None
            }
        });
        // Identity changes take precedence, but a repeated/reordered view must
        // neither replace the remembered crop nor refresh its source clock.
        if let Some(previous) = previous.filter(|_| identity_reset.is_none()) {
            if source.timestamp_ns <= previous.source.timestamp_ns
                || source.sequence <= previous.source.sequence
            {
                diagnostics.reason = if source.timestamp_ns == previous.source.timestamp_ns
                    || source.sequence == previous.source.sequence
                {
                    "same-source-no-advance"
                } else {
                    "out-of-order-no-advance"
                };
                diagnostics.source_advanced = false;
                diagnostics.parameters_after = previous.parameters;
                let pairs = common_pairs(&previous.samples, &samples);
                diagnostics.common_samples = pairs.len();
                diagnostics.common_normalized_mean_absolute_delta =
                    normalized_delta(&pairs, previous.parameters, previous.parameters);
                diagnostics.candidate_common_normalized_mean_absolute_delta =
                    normalized_delta(&pairs, previous.legacy_parameters, legacy_candidate);
                return Ok(diagnostics);
            }
        }
        let reset_reason = identity_reset.or_else(|| previous.and_then(|r| {
            if r.source.width != source.width || r.source.height != source.height {
                Some("incompatible-size")
            } else if source.timestamp_ns.saturating_sub(r.source.timestamp_ns) > MAX_GAP_NS {
                Some("source-gap")
            } else if overlap_fraction(r.source, source) == 0.0 {
                Some("no-common-sensor-region")
            } else if source.timestamp_ns.saturating_sub(r.lighting_source.timestamp_ns) > MAX_GAP_NS {
                Some("lighting-reference-expired")
            } else if overlap_fraction(r.lighting_source, source) == 0.0 {
                Some("no-common-lighting-reference-region")
            } else {
                None
            }
        }));
        if previous.is_none() || reset_reason.is_some() {
            diagnostics.reason = reset_reason.unwrap_or("initialized");
            diagnostics.lighting_reference_sequence = source.sequence;
            diagnostics.lighting_reference_timestamp_ns = source.timestamp_ns;
            self.reference = Some(Reference {
                source,
                lighting_source: source,
                lighting_samples: samples.clone(),
                samples,
                parameters: candidate,
                legacy_parameters: legacy_candidate,
                target_raw_low: std::array::from_fn(|c| candidate.low[c] / candidate.gains[c]),
                target_raw_high: std::array::from_fn(|c| candidate.high[c] / candidate.gains[c]),
                target_gains: candidate.gains,
                lighting_reference_sequence: source.sequence,
                lighting_reference_timestamp_ns: source.timestamp_ns,
            });
            return Ok(diagnostics);
        }
        let previous = self.reference.as_ref().unwrap();
        let pairs = common_pairs(&previous.lighting_samples, &samples);
        diagnostics.lighting_common_samples = pairs.len();
        let mut channel_supported = [false; 3];
        for c in 0..3 {
            let mut ratios = Vec::with_capacity(pairs.len());
            let mut quadrants = [false; 4];
            for (a, b) in &pairs {
                // Avoid dark-floor division and clipped highlights. This is
                // RAW10 adapter input, not display RGB or an exposure meter.
                if (8.0..=1000.0).contains(&a.rgb[c]) && (8.0..=1000.0).contains(&b.rgb[c]) {
                    ratios.push((b.rgb[c] / a.rgb[c]).ln());
                    let right = (b.sensor_x as u64 - source.sensor_x as u64) * 2 >= source.width as u64;
                    let bottom = (b.sensor_y as u64 - source.sensor_y as u64) * 2 >= source.height as u64;
                    quadrants[usize::from(right) + usize::from(bottom) * 2] = true;
                }
            }
            diagnostics.usable_samples[c] = ratios.len();
            if let Some(center) = median(&mut ratios) {
                diagnostics.log_ratio[c] = Some(center);
                let mut deviations = ratios.iter().map(|v| (v - center).abs()).collect::<Vec<_>>();
                let mad = median(&mut deviations).unwrap_or(f32::INFINITY);
                diagnostics.log_ratio_mad[c] = Some(mad);
                let agreement = ratios.iter().filter(|&&r| (r - center).abs() <= 0.10).count() as f32
                    / ratios.len() as f32;
                channel_supported[c] = ratios.len() >= MIN_MATCHED_SAMPLES
                    && quadrants.iter().filter(|&&present| present).count() >= 3
                    && mad <= MAX_RATIO_MAD
                    && agreement >= MIN_RATIO_AGREEMENT
                    && center.abs() <= 4.0f32.ln();
            }
        }
        diagnostics.illumination_supported = channel_supported.iter().all(|&supported| supported);
        let mut next = Reference {
            source,
            samples,
            parameters: previous.parameters,
            legacy_parameters: legacy_candidate,
            lighting_source: previous.lighting_source,
            lighting_samples: previous.lighting_samples.clone(),
            target_raw_low: previous.target_raw_low,
            target_raw_high: previous.target_raw_high,
            target_gains: previous.target_gains,
            lighting_reference_sequence: previous.lighting_reference_sequence,
            lighting_reference_timestamp_ns: previous.lighting_reference_timestamp_ns,
        };
        if diagnostics.illumination_supported {
            let ratios = diagnostics.log_ratio.map(|value| value.unwrap().exp());
            for c in 0..3 {
                next.target_raw_low[c] = (next.target_raw_low[c] * ratios[c]).clamp(0.0, 4096.0);
                next.target_raw_high[c] = (next.target_raw_high[c] * ratios[c])
                    .clamp(next.target_raw_low[c] + 1e-3, 4097.0);
                next.target_gains[c] = (next.target_gains[c] * ratios[1] / ratios[c]).clamp(0.25, 4.0);
            }
            diagnostics.reason = "common-sensor-lighting-update";
            next.lighting_reference_sequence = source.sequence;
            next.lighting_reference_timestamp_ns = source.timestamp_ns;
            next.lighting_source = source;
            next.lighting_samples.clone_from(&next.samples);
        } else {
            diagnostics.reason = "insufficient-common-lighting-support";
        }
        // Only source time advances this bound; a queue delay or repeated view
        // of one exposure cannot make a larger change look justified.
        let target_is_recent = source.timestamp_ns.saturating_sub(next.lighting_reference_timestamp_ns)
            <= MAX_GAP_NS;
        let max_log_step = if target_is_recent {
            MAX_LOG_RATE_PER_SECOND * diagnostics.source_dt_ns as f32 * 1e-9
        } else { 0.0 };
        for c in 0..3 {
            let old_gain = previous.parameters.gains[c];
            let gain_step = (next.target_gains[c] / old_gain).ln().clamp(-max_log_step, max_log_step);
            let gain = (old_gain * gain_step.exp()).clamp(0.25, 4.0);
            let old_low = previous.parameters.low[c] / old_gain;
            let old_high = previous.parameters.high[c] / old_gain;
            let range_scale = (next.target_raw_high[c] / old_high.max(1e-3)).ln()
                .clamp(-max_log_step, max_log_step).exp();
            if gain == old_gain && range_scale == 1.0 { continue; }
            // Multiplicative illumination changes the low and high together.
            // Do not independently drag a dark floor toward new crop content.
            next.parameters.gains[c] = gain;
            next.parameters.low[c] = old_low * range_scale * gain;
            next.parameters.high[c] = (old_high * range_scale * gain)
                .max(next.parameters.low[c] + 1e-6);
        }
        diagnostics.parameters_after = next.parameters;
        diagnostics.lighting_reference_sequence = next.lighting_reference_sequence;
        diagnostics.lighting_reference_timestamp_ns = next.lighting_reference_timestamp_ns;
        // Reconstruct pairs after moving samples into the next bounded state.
        let pairs = common_pairs(&previous.samples, &next.samples);
        diagnostics.common_samples = pairs.len();
        diagnostics.common_normalized_mean_absolute_delta =
            normalized_delta(&pairs, previous.parameters, next.parameters);
        diagnostics.candidate_common_normalized_mean_absolute_delta =
            normalized_delta(&pairs, previous.legacy_parameters, legacy_candidate);
        self.reference = Some(next);
        Ok(diagnostics)
    }
}

/// A fixed sensor-origin lattice; choose a power-of-two spacing once per size.
pub(super) fn sampling_stride(width: usize, height: usize) -> usize {
    let mut stride = 8usize;
    while width.div_ceil(stride).saturating_mul(height.div_ceil(stride)) > MAX_SAMPLES {
        stride = stride.saturating_mul(2);
    }
    stride
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(sequence: u64, x: u32) -> Source {
        Source {
            epoch: 1, prompt_generation: 2, sequence,
            timestamp_ns: sequence * 20_000_000,
            sensor_x: x, sensor_y: 0, width: 128, height: 96,
        }
    }

    fn samples(source: Source, light: [f32; 3]) -> Vec<Sample> {
        let mut result = Vec::new();
        for y in (8..source.height - 8).step_by(8) {
            for x in (0..source.width).filter(|x| (x + source.sensor_x as usize) % 8 == 0) {
                let sensor_x = source.sensor_x + x as u32;
                let value = 100.0 + ((sensor_x as usize * 7 + y * 3) % 90) as f32;
                result.push(Sample {
                    sensor_x, sensor_y: y as u32,
                    rgb: std::array::from_fn(|c| value * light[c]),
                });
            }
        }
        result
    }

    fn parameters() -> Parameters {
        Parameters { gains: [1.2, 1.0, 0.8], low: [12.0, 10.0, 8.0], high: [360.0, 300.0, 240.0] }
    }

    #[test]
    fn entering_bright_and_dark_content_does_not_remap_common_pixels() {
        for edge in [0.0, 1023.0] {
            let mut state = State::default();
            let a = source(1, 0);
            state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
            let b = source(2, 16);
            let mut current = samples(b, [1.0; 3]);
            for sample in &mut current {
                if sample.sensor_x >= 128 { sample.rgb = [edge; 3]; }
            }
            let candidate = Parameters { high: [900.0; 3], low: [0.0; 3], ..parameters() };
            let report = state.update(b, current, candidate).unwrap();
            assert_eq!(report.parameters_after, parameters());
            assert_eq!(report.common_normalized_mean_absolute_delta, Some(0.0));
            assert!(report.candidate_common_normalized_mean_absolute_delta.unwrap() > 0.2);
            assert!(report.illumination_supported);
            assert_eq!(report.crop_overlap_fraction, 0.875);
        }
    }

    #[test]
    fn same_source_shift_reuses_parameters_without_refreshing_age_or_crop() {
        let mut state = State::default();
        let a = source(1, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        let b = source(1, 16);
        let report = state.update(b, samples(b, [1.0; 3]), parameters()).unwrap();
        assert_eq!(report.reason, "same-source-no-advance");
        assert!(!report.source_advanced);
        assert_eq!(state.reference.as_ref().unwrap().source, a);
        assert_eq!(report.common_normalized_mean_absolute_delta, Some(0.0));
    }

    #[test]
    fn real_light_step_adapts_with_bounded_source_time_and_preserved_target_debt() {
        let mut state = State::default();
        let a = source(1, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        let b = source(2, 0);
        let report = state.update(b, samples(b, [2.0; 3]), parameters()).unwrap();
        assert!(report.illumination_supported);
        let first_high = report.parameters_after.high[1];
        assert!(first_high > 300.0 && first_high < 320.0);
        for _ in 0..10 {
            let repeated = state.update(b, samples(b, [2.0; 3]), parameters()).unwrap();
            assert_eq!(repeated.parameters_after.high[1], first_high);
        }
        for seq in 3..=16 {
            let frame = source(seq, 0);
            state.update(frame, samples(frame, [2.0; 3]), parameters()).unwrap();
        }
        let final_parameters = state.reference.as_ref().unwrap().parameters;
        assert!((final_parameters.high[1] - 600.0).abs() < 0.001);
        assert!((final_parameters.low[1] - 20.0).abs() < 0.001);
    }

    #[test]
    fn chromatic_light_step_can_adapt_without_crop_statistics() {
        let mut state = State::default();
        let a = source(1, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        let mut b = source(2, 0);
        b.timestamp_ns = a.timestamp_ns + 500_000_000;
        let report = state.update(b, samples(b, [1.4, 1.2, 0.8]), parameters()).unwrap();
        assert!(report.illumination_supported);
        for c in 0..3 {
            let expected = parameters().gains[c] * 1.2 / [1.4, 1.2, 0.8][c];
            assert!((report.parameters_after.gains[c] - expected).abs() < 1e-5);
        }
        assert!(report.common_normalized_mean_absolute_delta.unwrap() < 1e-6);
    }

    #[test]
    fn incoherent_motion_or_clipping_cannot_update_illumination() {
        for clipped in [false, true] {
            let mut state = State::default();
            let a = source(1, 0);
            state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
            let b = source(2, 0);
            let mut current = samples(b, [1.0; 3]);
            for (i, sample) in current.iter_mut().enumerate() {
                sample.rgb = sample.rgb.map(|v| if clipped {1023.0} else {v * (0.5 + (i % 3) as f32 * 0.5)});
            }
            let report = state.update(b, current, parameters()).unwrap();
            assert!(!report.illumination_supported);
            assert_eq!(report.parameters_after, parameters());
        }
    }

    #[test]
    fn real_discontinuities_reset_but_out_of_order_source_does_not_rewind() {
        let mut state = State::default();
        let a = source(5, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        let old = source(4, 16);
        let report = state.update(old, samples(old, [1.0; 3]), parameters()).unwrap();
        assert_eq!(report.reason, "out-of-order-no-advance");
        assert_eq!(state.reference.as_ref().unwrap().source, a);
        for (expected, mut b) in [
            ("identity-or-prompt-changed", source(6, 0)),
            ("source-gap", source(60, 0)),
            ("no-common-sensor-region", source(61, 1000)),
        ] {
            if expected == "identity-or-prompt-changed" { b.prompt_generation += 1; }
            if expected != "identity-or-prompt-changed" { b.prompt_generation = 3; }
            let report = state.update(b, samples(b, [1.0; 3]), parameters()).unwrap();
            assert_eq!(report.reason, expected);
        }
    }

    #[test]
    fn sample_budget_is_bounded_for_native_sensor_sizes() {
        for (width, height) in [(420, 280), (8192, 6144), (16384, 16384)] {
            let stride = sampling_stride(width, height);
            assert!(width.div_ceil(stride) * height.div_ceil(stride) <= MAX_SAMPLES);
            assert!(stride.is_power_of_two());
        }
    }

    #[test]
    fn old_or_duplicate_different_size_cannot_reset_or_refresh_observation() {
        let mut state = State::default();
        let a = source(5, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        for seq in [4, 5] {
            let mut stale = source(seq, 16);
            stale.width = 96;
            let report = state.update(stale, samples(stale, [1.0; 3]), parameters()).unwrap();
            assert!(!report.source_advanced);
            assert_eq!(state.reference.as_ref().unwrap().source, a);
        }
        let mut same_time = source(6, 16);
        same_time.timestamp_ns = a.timestamp_ns;
        let report = state.update(same_time, samples(same_time, [1.0; 3]), parameters()).unwrap();
        assert!(!report.source_advanced);
        assert_eq!(state.reference.as_ref().unwrap().source, a);
    }

    #[test]
    fn unsupported_samples_do_not_refresh_lighting_evidence_clock() {
        let mut state = State::default();
        let a = source(1, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        for seq in 2..=10 {
            let b = source(seq, 0);
            let report = state.update(b, samples(b, [10.0; 3]), parameters()).unwrap();
            assert!(report.source_advanced);
            assert!(!report.illumination_supported);
            assert_eq!(report.lighting_reference_sequence, a.sequence);
            assert_eq!(report.lighting_reference_timestamp_ns, a.timestamp_ns);
        }
    }

    #[test]
    fn rejected_intermediate_light_step_recovers_against_last_accepted_raw() {
        let mut state = State::default();
        let a = source(1, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        let b = source(2, 8);
        let mut rejected = samples(b, [1.4; 3]);
        for (i, sample) in rejected.iter_mut().enumerate() {
            sample.rgb = sample.rgb.map(|v| v * [0.5, 1.0, 1.5][i % 3]);
        }
        let report = state.update(b, rejected, parameters()).unwrap();
        assert!(!report.illumination_supported);
        assert_eq!(state.reference.as_ref().unwrap().source, b);
        assert_eq!(state.reference.as_ref().unwrap().lighting_source, a);
        for stale in [a, b] {
            let held = state.update(stale, samples(stale, [1.4; 3]), parameters()).unwrap();
            assert!(!held.source_advanced);
            assert_eq!(state.reference.as_ref().unwrap().source, b);
            assert_eq!(state.reference.as_ref().unwrap().lighting_source, a);
        }
        let c = source(3, 16);
        let report = state.update(c, samples(c, [1.4; 3]), parameters()).unwrap();
        assert!(report.illumination_supported);
        assert_eq!(report.reference_sequence, b.sequence);
        assert_eq!(report.ratio_reference_sequence, a.sequence);
        assert!((report.parameters_after.high[1] / 300.0).ln() <= 0.060001);
        assert!((report.log_ratio[1].unwrap() - 1.4f32.ln()).abs() < 1e-6);
        assert!((state.reference.as_ref().unwrap().target_raw_high[1] - 420.0).abs() < 1e-3);
        for seq in 4..=12 {
            let frame = source(seq, (seq % 3 * 8) as u32);
            state.update(frame, samples(frame, [1.4; 3]), parameters()).unwrap();
        }
        assert!((state.reference.as_ref().unwrap().parameters.high[1] - 420.0).abs() < 1e-3);
    }

    #[test]
    fn rejected_nudges_expire_reference_on_source_time_and_reinitialize() {
        let mut state = State::default();
        let a = source(1, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        for seq in 2..=46 {
            let frame = source(seq, (seq % 3 * 8) as u32);
            let report = state.update(frame, samples(frame, [10.0; 3]), parameters()).unwrap();
            assert!(!report.illumination_supported);
            assert_eq!(report.lighting_reference_sequence, 1);
        }
        let frame = source(47, 8);
        let candidate = Parameters { high: [600.0; 3], ..parameters() };
        let report = state.update(frame, samples(frame, [1.5; 3]), candidate).unwrap();
        assert_eq!(report.reason, "lighting-reference-expired");
        assert_eq!(report.parameters_after, candidate);
        assert_eq!(report.lighting_reference_sequence, 47);
        let next = source(48, 0);
        assert!(state.update(next, samples(next, [1.6; 3]), candidate).unwrap().illumination_supported);
    }

    #[test]
    fn rejected_nudges_cannot_walk_reference_out_of_view_indefinitely() {
        let mut state = State::default();
        let a = source(1, 0);
        state.update(a, samples(a, [1.0; 3]), parameters()).unwrap();
        for seq in 2..=4 {
            let frame = source(seq, ((seq - 1) * 32) as u32);
            let report = state.update(frame, samples(frame, [10.0; 3]), parameters()).unwrap();
            assert_eq!(report.reason, "insufficient-common-lighting-support");
        }
        let frame = source(5, 128);
        let report = state.update(frame, samples(frame, [1.0; 3]), parameters()).unwrap();
        assert_eq!(report.reason, "no-common-lighting-reference-region");
        assert_eq!(report.lighting_reference_sequence, 5);
    }

    #[test]
    fn matched_per_crop_delta_uses_previous_legacy_not_previous_running() {
        let mut state = State::default();
        let legacy = Parameters { high: [800.0; 3], ..parameters() };
        for seq in 1..=3 {
            let frame = source(seq, 0);
            let report = state.update_with_legacy(frame, samples(frame, [1.0; 3]), parameters(), legacy).unwrap();
            if seq > 1 {
                assert_eq!(report.candidate_common_normalized_mean_absolute_delta, Some(0.0));
            }
        }
    }
}
