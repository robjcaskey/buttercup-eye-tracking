//! Simple experimental sign latch from source-matched RAW parallax.
//! A circle's two poses imply two eye pivots. Compare their motion with the
//! surrounding RAW image, then hold the winner until sustained contradiction.
//! The pivot ratio and margins are engineering assumptions, not probabilities.
use super::joint::{circle_pose_hypotheses, PinholeCamera};
use crate::{
    geometry::{dot3, Ellipse},
    roi_evidence::NativeGlobalSimilarityEvidence,
};
use std::collections::VecDeque;

#[derive(Default)]
pub(crate) struct Tracker {
    previous: Option<(u64, [[f64; 3]; 2], [[f64; 2]; 2])>,
    selected: Option<usize>,
    velocity: [[f64; 3]; 2],
    pending: Option<usize>,
    votes: usize,
    first_vote: u64,
    samples: VecDeque<(u64, Option<[[f64; 3]; 2]>)>,
    pub(crate) status: &'static str,
    pub(crate) costs: Option<[f64; 2]>,
}
impl Tracker {
    pub(crate) fn last_source(&self) -> Option<u64> {
        self.samples.back().map(|s| s.0)
    }
    pub(crate) fn restart_stream(&mut self) {
        self.previous = None;
        self.samples.clear();
        self.pending = None;
        self.votes = 0;
        // Source clocks cannot bridge a restart. A new pose is required before
        // even a retained latch can publish again.
        self.selected = None;
        self.status = "waiting for parallax";
    }
    pub(crate) fn normals(&self, source: u64) -> Option<[[f64; 3]; 2]> {
        self.samples
            .iter()
            .rev()
            .find(|s| s.0 == source)
            .and_then(|s| s.1)
    }
    pub(crate) fn observe(
        &mut self,
        source: u64,
        ellipse: Ellipse,
        origin: [u32; 2],
        camera: PinholeCamera,
        motion: NativeGlobalSimilarityEvidence,
    ) {
        if self.last_source().is_some_and(|old| source <= old) {
            return;
        }
        let Some(poses) = circle_pose_hypotheses(camera, ellipse, origin) else {
            return;
        };
        let mut normals = poses.map(|p| p.normal);
        // Same 1.83 globe/iris ratio as the existing provisional contact.
        let depth = (1.83_f64.powi(2) - 1.0).sqrt();
        let Some(mut pivots) = poses
            .into_iter()
            .map(|p| {
                camera.project(std::array::from_fn(|i| {
                    p.center_per_radius[i] - depth * p.normal[i]
                }))
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|v| v.try_into().ok())
        else {
            return;
        };
        let pivots_ref: &mut [[f64; 2]; 2] = &mut pivots;
        self.costs = None;
        if let Some((old, previous_normals, previous_pivots)) = self.previous {
            // Follow the evolving ellipse's two solution paths. Near their
            // merger, extrapolate the previous change rather than swapping
            // branches solely because the eigensolver reversed array order.
            let dt = (source - old) as f64 * 1e-9;
            let near = dot3(previous_normals[0], previous_normals[1]) > 20_f64.to_radians().cos();
            let predicted = if near && dt <= 0.75 {
                std::array::from_fn(|i| {
                    crate::geometry::normalized3(std::array::from_fn(|j| {
                        previous_normals[i][j] + self.velocity[i][j] * dt
                    }))
                    .unwrap_or(previous_normals[i])
                })
            } else {
                previous_normals
            };
            // Array order from the conic eigensolve is not branch identity.
            if dot3(predicted[0], normals[1]) + dot3(predicted[1], normals[0])
                > dot3(predicted[0], normals[0]) + dot3(predicted[1], normals[1])
            {
                normals.swap(0, 1);
                pivots_ref.swap(0, 1);
            }
            if dt > 0.001 && dt <= 0.75 && dot3(normals[0], normals[1]) < 3_f64.to_radians().cos() {
                self.velocity = std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        ((normals[i][j] - previous_normals[i][j]) / dt).clamp(-3.0, 3.0)
                    })
                });
            }
            let informative = source - old <= 750_000_000
                && motion.reliable
                && motion.motion.support >= 8
                && motion.motion.residual.is_finite()
                && dot3(normals[0], normals[1]) < 5_f64.to_radians().cos();
            if informative {
                let m = motion.motion;
                let costs = previous_pivots.map(|p| {
                    let x = p[0] - f64::from(motion.motion_center_sensor[0]);
                    let y = p[1] - f64::from(motion.motion_center_sensor[1]);
                    [
                        p[0] + f64::from(m.translation[0])
                            + f64::from(m.diagonal_coefficient_delta) * x
                            - f64::from(m.rotation_coefficient) * y,
                        p[1] + f64::from(m.translation[1])
                            + f64::from(m.rotation_coefficient) * x
                            + f64::from(m.diagonal_coefficient_delta) * y,
                    ]
                });
                let residuals = std::array::from_fn(|i| {
                    (costs[i][0] - pivots[i][0]).hypot(costs[i][1] - pivots[i][1])
                });
                self.costs = Some(residuals);
                let allowance =
                    (2.0 * f64::from(m.residual)).max(0.5) + 0.015 * ellipse.major_radius;
                let winner = usize::from(residuals[1] < residuals[0]);
                let strong = residuals.iter().all(|r| r.is_finite())
                    && residuals[winner] <= 2.0 * allowance
                    && residuals[1 - winner] - residuals[winner] >= allowance;
                if strong {
                    if self.pending != Some(winner) {
                        self.pending = Some(winner);
                        self.votes = 0;
                        self.first_vote = source;
                    }
                    self.votes += 1;
                    let required = if self.selected.is_some() { 6 } else { 4 };
                    if self.votes >= required && source - self.first_vote >= 300_000_000 {
                        if self.selected != Some(winner) {
                            eprintln!("PARALLAX_SIGN source={source} branch={winner} votes={} pivot_residual_px={residuals:?}",self.votes);
                        }
                        self.selected = Some(winner);
                        self.pending = None;
                        self.votes = 0;
                    }
                } else {
                    self.pending = None;
                    self.votes = 0;
                }
            } else {
                self.pending = None;
                self.votes = 0;
            }
        }
        self.previous = Some((source, normals, pivots));
        self.status = if self.selected.is_some() {
            "latched"
        } else {
            "waiting for parallax"
        };
        let chosen = self.selected.map(|i| [normals[i], normals[1 - i]]);
        self.samples.push_back((source, chosen));
        if self.samples.len() > 32 {
            self.samples.pop_front();
        }
    }
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({"status":self.status,"latched":self.selected.is_some(),
            "pending_votes":self.votes,"pivot_residual_px":self.costs,
            "source_ns":self.last_source().map(|s|s.to_string()),"experimental":true})
    }
}
