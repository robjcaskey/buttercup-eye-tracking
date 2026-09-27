//! Candidate-independent outer-band image motion for material diagnostics.
//! No ellipse, target coordinate, or recorded prediction defines its support.
use super::{json, motion, Value};
use std::sync::Arc;

#[derive(Default)]
pub struct Carrier {
    tracker: motion::NativeGlobalSimilarityTracker,
    previous_ns: Option<u64>,
    pub interval: Option<(u64, u64, motion::NativeGlobalSimilarityEvidence)>,
}

impl Carrier {
    pub fn observe(
        &mut self,
        raw: &[u16],
        width: usize,
        height: usize,
        sx: u32,
        sy: u32,
        timestamp_ns: u64,
        reset: bool,
    ) -> Value {
        if reset
            || self
                .previous_ns
                .is_some_and(|t| timestamp_ns <= t || timestamp_ns - t > 250_000_000)
        {
            self.tracker.clear();
            self.previous_ns = None;
        }
        self.tracker.retain_diagnostic_correspondences(true);
        let e = self
            .tracker
            .observe_outer_bands(Arc::new(raw.to_vec()), width, height, sx, sy, 8);
        let m = e.candidate_motion;
        self.interval = self.previous_ns.map(|from| (from, timestamp_ns, e));
        let report = json!({
            "previous_timestamp_ns":self.previous_ns,
            "reliable":e.reliable,"stable_frames":e.stable_frames,
            "candidate_matches":e.candidate_matches,"support":m.support,
            "residual":m.residual,"spatial_span":e.spatial_span,
            "occupied_quadrants":e.occupied_quadrants,
            "center_sensor":e.motion_center_sensor,
            "candidate_tensor":[m.translation[0],m.translation[1],m.diagonal_coefficient_delta,m.rotation_coefficient],
            "points":self.tracker.diagnostic_correspondences().iter().map(|p|json!({
                "previous_sensor":p.previous_sensor_px,"current_sensor":p.current_sensor_px,
                "score":p.photometric_score,"margin":p.distinct_match_margin,
                "inlier":p.global_similarity_inlier,
            })).collect::<Vec<_>>(),
            "scope":"Top/bottom 12.5% of both native ROIs, whole-patch margin 12px, radius 8px. Image motion only; skin, lids, glasses and lighting may confound it. No anatomical ground truth or calibrated 3D head pose. Candidate tensor is diagnostic when reliable=false."
        });
        self.previous_ns = Some(timestamp_ns);
        report
    }
}
