//! Evaluation-only target association. Commands never enter native tracking.
use super::{json, BundleSource, Error, Value};
use buttercup_eye_tracking::recorded_bundle::metadata_records;
use std::collections::BTreeMap;

fn ns(v: &Value) -> Result<u64, Error> {
    v.as_u64()
        .or_else(|| v.as_str()?.parse().ok())
        .ok_or_else(|| "missing timestamp".into())
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Visit {
    pub target: usize,
    pub uv: [f64; 2],
    pub start_host_ns: u64,
    pub end_host_ns: u64,
    pub hidden_host_ns: Option<u64>,
    pub viewer_session: String,
}

#[derive(Default)]
pub struct Timeline {
    pub visits: Vec<Visit>,
}

impl Timeline {
    pub fn read(bundle: &BundleSource) -> Result<Self, Error> {
        Self::from_rows(metadata_records(
            bundle.read_entry("metadata.oim1")?.as_slice(),
        ))
    }

    fn from_rows(rows: impl IntoIterator<Item = Result<Value, String>>) -> Result<Self, Error> {
        let mut configurations = BTreeMap::new();
        let mut timeline = Self::default();
        let mut active = None::<usize>;
        let mut last_submit = None;
        for row in rows {
            let r = row?;
            let key = (
                r["viewer_session_id"].to_string(),
                r["configuration_revision"].to_string(),
            );
            match r["event"].as_str() {
                Some("recording_start_snapshot") => {
                    configurations.insert(key, r["configuration"]["calibration"].clone());
                }
                Some("configuration_changed") => {
                    configurations.insert(key, r["data"]["calibration"].clone());
                }
                Some("presentation") => {
                    let at = ns(&r["host_submit_end_unix_ns"])?;
                    if last_submit.is_some_and(|t| at < t) {
                        return Err("presentation clock moved backwards".into());
                    }
                    last_submit = Some(at);
                    let config = configurations
                        .get(&key)
                        .ok_or("missing presentation configuration")?;
                    let visible = r["active_targets"]
                        .as_array()
                        .ok_or("missing target list")?
                        .iter()
                        .filter(|t| t["visible"] == true && t["role"] == "calibration")
                        .collect::<Vec<_>>();
                    if visible.len() > 1 {
                        return Err("multiple calibration targets".into());
                    }
                    let current = if config["phase"] == "collecting" && visible.len() == 1 {
                        let target = config["target_index"]
                            .as_u64()
                            .or_else(|| {
                                visible[0]["id"]
                                    .as_str()?
                                    .strip_prefix("calibration-")?
                                    .parse()
                                    .ok()
                            })
                            .ok_or("missing target index")?
                            as usize;
                        let uv = [
                            visible[0]["normalized"][0]
                                .as_f64()
                                .ok_or("missing target x")?,
                            visible[0]["normalized"][1]
                                .as_f64()
                                .ok_or("missing target y")?,
                        ];
                        if uv
                            .iter()
                            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                        {
                            return Err("invalid target".into());
                        }
                        Some((target, uv, config["thumbnail_opacity"].as_f64()))
                    } else {
                        None
                    };
                    let continuing =
                        active
                            .zip(current)
                            .is_some_and(|(i, (target, uv, opacity))| {
                                let old = &timeline.visits[i];
                                old.target == target
                                    && old.uv == uv
                                    && old.viewer_session == key.0
                                    && !(old.hidden_host_ns.is_some()
                                        && opacity.is_some_and(|v| v > 0.0))
                            });
                    if !continuing {
                        if let Some(i) = active.take() {
                            timeline.visits[i].end_host_ns = at;
                        }
                        if let Some((target, uv, _)) = current {
                            active = Some(timeline.visits.len());
                            timeline.visits.push(Visit {
                                target,
                                uv,
                                start_host_ns: at,
                                end_host_ns: at,
                                hidden_host_ns: None,
                                viewer_session: key.0,
                            });
                        }
                    }
                    if let (Some(i), Some((_, _, opacity))) = (active, current) {
                        timeline.visits[i].end_host_ns = at;
                        if opacity == Some(0.0) {
                            timeline.visits[i].hidden_host_ns.get_or_insert(at);
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(timeline)
    }

    pub fn at_arrival(&self, arrival: u64) -> Value {
        // This is explicitly a sensitivity window, NOT an attested sensor/host
        // clock bound. Arrival includes unknown transport delay. Keep 0..400 ms
        // lag plus 100 ms display allowance and two seconds acclimation visible.
        let early = arrival.saturating_sub(500_000_000);
        let late = arrival;
        let center = arrival.saturating_sub(200_000_000);
        let Some((index, v)) = self
            .visits
            .iter()
            .enumerate()
            .find(|(_, v)| center >= v.start_host_ns && center < v.end_host_ns)
        else {
            return Value::Null;
        };
        let settled_start = v
            .start_host_ns
            .saturating_add(2_100_000_000)
            .max(v.hidden_host_ns.unwrap_or(u64::MAX));
        json!({"visit":index,"target":v.target,"uv":v.uv,"elapsed_ms":(center-v.start_host_ns) as f64/1e6,
            "stable_in_sensitivity_window":early>=settled_start && late.saturating_add(100_000_000)<v.end_host_ns,
            "alignment":"host-arrival proxy; unmeasured transport delay; 0..400ms lag +100ms display sensitivity",
            "fixation":"command only; at least 40% off-target allowance, not a gaze/anatomy label"})
    }

    pub fn report(&self) -> Value {
        json!({"visits":self.visits,"source":"successfully submitted calibration presentations, not recorded predictions",
            "timing":"host-arrival proxy; no optical clock bound; exclude transition/thumbnail/acclimation intervals",
            "anatomy_labels":false,"target_feedback_to_tracker":false})
    }
}

/// Untouched Quad-Bayer channel statistics. No per-frame display white balance,
/// demosaicing, skin/sclera threshold, or tissue label enters these measurements.
pub fn photometry(raw: &[u16], w: usize, h: usize, sx: u32, sy: u32, sensor: [f32; 2]) -> Value {
    let x = sensor[0].round() as i32 - sx as i32;
    let y = sensor[1].round() as i32 - sy as i32;
    if x < 8 || y < 8 || x + 8 >= w as i32 || y + 8 >= h as i32 {
        return Value::Null;
    }
    let mut sum = [0.0; 3];
    let mut sq = [0.0; 3];
    let mut counts = [0usize; 3];
    let mut clipped = 0;
    for yy in y - 8..y + 8 {
        for xx in x - 8..x + 8 {
            let (xe, ye) = (
                (((xx as u32 + sx) / 2) & 1) == 0,
                (((yy as u32 + sy) / 2) & 1) == 0,
            );
            let c = match (xe, ye) {
                (true, true) => 0,
                (false, false) => 2,
                _ => 1,
            };
            let v = f64::from(raw[yy as usize * w + xx as usize]);
            sum[c] += v;
            sq[c] += v * v;
            counts[c] += 1;
            clipped += usize::from(v >= 1018.0);
        }
    }
    let mean = std::array::from_fn::<_, 3, _>(|c| sum[c] / counts[c] as f64);
    let sd = std::array::from_fn::<_, 3, _>(|c| {
        (sq[c] / counts[c] as f64 - mean[c] * mean[c])
            .max(0.0)
            .sqrt()
    });
    json!({"mean_rgb_raw":mean,"sd_rgb_raw":sd,"red_over_green":mean[0]/mean[1].max(1.0),
        "blue_over_green":mean[2]/mean[1].max(1.0),"clipped_fraction":clipped as f64/256.0,
        "black_level":"not subtracted; unknown"})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_target_id_does_not_invent_thumbnail_visibility() {
        let rows = [
            json!({"event":"configuration_changed","viewer_session_id":"s","configuration_revision":1,
            "data":{"calibration":{"phase":"collecting"}}}),
            json!({"event":"presentation","viewer_session_id":"s","configuration_revision":1,
            "host_submit_end_unix_ns":"1000000000","active_targets":[{"id":"calibration-3","role":"calibration","visible":true,"normalized":[0.9,0.1]}]}),
            json!({"event":"presentation","viewer_session_id":"s","configuration_revision":1,
            "host_submit_end_unix_ns":"6000000000","active_targets":[]}),
        ];
        let t = Timeline::from_rows(rows.into_iter().map(Ok)).unwrap();
        assert_eq!(t.visits[0].target, 3);
        assert!(t.visits[0].hidden_host_ns.is_none());
        assert_eq!(
            t.at_arrival(4_000_000_000)["stable_in_sensitivity_window"],
            false
        );
    }

    #[test]
    fn physical_cfa_photometry_survives_odd_crop_origins() {
        for (sx, sy) in [(0, 0), (1, 3), (8, 6)] {
            let mut raw = vec![0; 32 * 32];
            for y in 0..32 {
                for x in 0..32 {
                    raw[y * 32 + x] = match (((x as u32 + sx) / 2) & 1, ((y as u32 + sy) / 2) & 1) {
                        (0, 0) => 300,
                        (1, 1) => 100,
                        _ => 200,
                    };
                }
            }
            let p = photometry(&raw, 32, 32, sx, sy, [sx as f32 + 16., sy as f32 + 16.]);
            assert_eq!(p["mean_rgb_raw"], json!([300., 200., 100.]));
        }
    }
    #[test]
    fn removal_and_thumbnail_visibility_bound_target_association() {
        let config = |opacity| {
            json!({"event":"configuration_changed","viewer_session_id":"s","configuration_revision":opacity,
            "data":{"calibration":{"phase":"collecting","target_index":2,"thumbnail_opacity":opacity}}})
        };
        let submit = |ns: &str, opacity, visible| {
            json!({"event":"presentation","viewer_session_id":"s","configuration_revision":opacity,
            "host_submit_end_unix_ns":ns,"active_targets":if visible {json!([{"role":"calibration","visible":true,"normalized":[0.5,0.5]}])}else{json!([])}})
        };
        let t = Timeline::from_rows(
            [
                config(1),
                submit("1000000000", 1, true),
                config(0),
                submit("1600000000", 0, true),
                submit("5000000000", 0, false),
            ]
            .into_iter()
            .map(Ok),
        )
        .unwrap();
        assert_eq!(t.visits.len(), 1);
        assert_eq!(
            t.at_arrival(3_300_000_000)["stable_in_sensitivity_window"],
            false
        );
        assert_eq!(
            t.at_arrival(4_000_000_000)["stable_in_sensitivity_window"],
            true
        );
        assert!(t.at_arrival(5_600_000_000).is_null());
    }
}
