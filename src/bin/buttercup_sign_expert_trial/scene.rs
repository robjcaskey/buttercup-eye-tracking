//! Exact perspective circle observations and spherical specular reflections.
//! All optical dimensions and known light positions are simulator assumptions.
use super::*;
use conic_solver::joint::{circle_pose_hypotheses, PinholeCamera, ProjectedCircle};
fn norm(v: [f64; 3]) -> [f64; 3] {
    let d = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.map(|x| x / d)
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(a, b).clamp(-1.0, 1.0).acos()
}
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}
fn glint(camera: PinholeCamera, iris: [f64; 3], n: [f64; 3], light: [f64; 3]) -> Option<[f64; 2]> {
    let center = std::array::from_fn::<_, 3, _>(|i| iris[i] - 4.0 * n[i]);
    let mut normal = norm(center.map(|x| -x));
    let mut error = 1.0;
    for _ in 0..64 {
        let point = std::array::from_fn::<_, 3, _>(|i| center[i] + 7.8 * normal[i]);
        let view = norm(point.map(|x| -x));
        let incident = norm(std::array::from_fn(|i| light[i] - point[i]));
        let next = norm(std::array::from_fn(|i| view[i] + incident[i]));
        error = angle(normal, next);
        normal = next;
        if error < 1e-7 {
            break;
        }
    }
    if error > 1e-5 {
        return None;
    }
    camera.project(std::array::from_fn(|i| center[i] + 7.8 * normal[i]))
}
pub(super) fn generate(stats: &[(bool, f64)], first: usize, count: usize) -> Vec<Sample> {
    let camera = PinholeCamera {
        focal_px: [4000.0, 3900.0],
        principal_px: [4000.0, 3000.0],
    };
    let mut samples = Vec::new();
    for session in first..first + count {
        let mut rng = Rng(0x893abc15u64.wrapping_mul(session as u64 + 1));
        let regime = session % 6;
        let phase = rng.uniform() * std::f64::consts::TAU;
        let direction = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        let base = [
            (rng.uniform() - 0.5) * 80.0,
            (rng.uniform() - 0.5) * 80.0,
            -350.0 - rng.uniform() * 200.0,
        ];
        let light = [
            (rng.uniform() - 0.5) * 200.0,
            60.0 + rng.uniform() * 60.0,
            0.0,
        ];
        let mut previous = None::<[f64; 3]>;
        let mut previous_previous = None::<[f64; 3]>;
        let mut last_pivot = None::<([f64; 2], f64)>;
        let mut beam = sign_kinematic_beam::SignKinematicBeam::new(Default::default());
        let mut tracker = perspective_sign::Tracker::default();
        let mut seed = 0;
        for i in 0..80 {
            let t = i as f64 * 0.05;
            let theta = match regime {
                0 => direction * (0.2 - 0.1 * t),
                1 => direction * 0.22 * (t * 2.0 + phase).sin(),
                2 => direction * (0.2 - 0.4 / (1.0 + (-(t - 2.0) * 30.0).exp())),
                _ => direction * 0.2 * (t + phase).sin(),
            };
            let vertical = 0.06 * (t * 0.6 + phase).sin();
            let truth = norm([theta.sin(), vertical.sin(), theta.cos() * vertical.cos()]);
            let pivot = [
                base[0]
                    + if regime == 3 {
                        7.0 * (t * 4.0).sin()
                    } else {
                        0.5 * (t + phase).sin()
                    },
                base[1],
                base[2]
                    + if regime == 3 {
                        10.0 * (t * 2.0).sin()
                    } else {
                        0.0
                    },
            ];
            let center = std::array::from_fn::<_, 3, _>(|j| pivot[j] + 30.0 * truth[j]);
            let mut ellipse = ProjectedCircle::project(camera, center, truth, 6.0, [0, 0])
                .unwrap()
                .ellipse()
                .unwrap();
            let blur = if regime == 2 && (t - 2.0).abs() < 0.15 {
                3.0
            } else {
                1.0
            };
            let sigma = 0.3 * blur;
            ellipse.center.0 += rng.normal() * sigma;
            ellipse.center.1 += rng.normal() * sigma;
            // Perturb symmetric ellipse shape, avoiding a privileged noisy axis.
            let (s, c) = ellipse.angle.sin_cos();
            let a = ellipse.major_radius;
            let b = ellipse.minor_radius;
            let xx = a * c * c + b * s * s + rng.normal() * sigma;
            let yy = a * s * s + b * c * c + rng.normal() * sigma;
            let xy = (a - b) * s * c + rng.normal() * sigma / 2f64.sqrt();
            let delta = ((xx - yy).powi(2) * 0.25 + xy * xy).sqrt();
            ellipse.major_radius = (xx + yy) * 0.5 + delta;
            ellipse.minor_radius = (xx + yy) * 0.5 - delta;
            ellipse.angle = 0.5 * (2.0 * xy).atan2(xx - yy);
            let mut pair = circle_pose_hypotheses(camera, ellipse, [0, 0]).unwrap();
            if pair[0].normal[0] < pair[1].normal[0] {
                pair.swap(0, 1);
            }
            let normals = pair.map(|p| p.normal);
            let centers = pair.map(|p| p.center_per_radius.map(|x| x * 6.0));
            let errors = normals.map(|n| angle(n, truth).to_degrees());
            let target = usize::from(errors[1] < errors[0]);
            if i == 0 {
                seed = target;
            }
            let mut estimate = pivot;
            estimate[0] += rng.normal() * 0.5
                + if regime == 3 {
                    4.0 * (t * 3.0).cos()
                } else {
                    0.0
                };
            estimate[1] += rng.normal() * 0.5;
            estimate[2] += rng.normal() * 2.0;
            let head = camera.project(estimate).unwrap();
            let pivots = std::array::from_fn::<_, 2, _>(|j| {
                camera
                    .project(std::array::from_fn(|k| {
                        centers[j][k] - 30.0 * normals[j][k]
                    }))
                    .unwrap()
            });
            let (missing, clip) = stats[(rng.uniform() * stats.len() as f64) as usize];
            let mut observed = glint(camera, center, truth, light).unwrap();
            observed[0] += rng.normal() * 0.8 * blur;
            observed[1] += rng.normal() * 0.8 * blur;
            if regime == 4 || (regime == 5 && rng.uniform() < 0.35) {
                observed[0] += direction * 15.0;
                observed[1] += 6.0;
            }
            let predicted_glints = std::array::from_fn::<_, 2, _>(|j| {
                glint(camera, centers[j], normals[j], light).unwrap()
            });
            let prior = previous.unwrap_or(normals[seed]);
            let prediction = previous_previous
                .map(|old| norm(std::array::from_fn(|k| 2.0 * prior[k] - old[k])))
                .unwrap_or(prior);
            let costs = [
                normals.map(|n| angle(n, prior).powi(2) / 0.03f64.powi(2)),
                normals.map(|n| angle(n, prediction).powi(2) / 0.03f64.powi(2)),
                pivots.map(|p| dist(p, head) / 25.0),
                predicted_glints.map(|p| dist(p, observed) / 4.0),
            ];
            let source = sign_kinematic_beam::SourceIdentity {
                stream: session as u64 + 1,
                frame: i as u64 + 1,
            };
            let ns = 1_000_000_000 + i as u64 * 50_000_000;
            let transform = last_pivot.map(|(last, z)| {
                let scale = z / estimate[2];
                (
                    scale,
                    [head[0] - scale * last[0], head[1] - scale * last[1]],
                )
            });
            let bo = beam.observe(sign_kinematic_beam::Observation {
                source,
                source_ns: Some(ns),
                fresh: true,
                visible: true,
                hypotheses: std::array::from_fn(|j| sign_kinematic_beam::Hypothesis {
                    normal_camera: normals[j],
                    effective_pivot_sensor_px: pivots[j],
                    pivot_sigma_px: 4.0 * blur,
                }),
                head: transform.map(
                    |(scale, translation_px)| sign_kinematic_beam::HeadTransport {
                        from: sign_kinematic_beam::SourceIdentity {
                            stream: source.stream,
                            frame: source.frame - 1,
                        },
                        to: source,
                        from_source_ns: ns - 50_000_000,
                        to_source_ns: ns,
                        center_sensor_px: [0.0, 0.0],
                        translation_px,
                        scale,
                        angle_rad: 0.0,
                        sigma_px: 7.0,
                        rotation_vector_camera_rad: None,
                    },
                ),
                anchor: if i == 0 {
                    Some(sign_kinematic_beam::IndependentAnchor {
                        source,
                        normal_camera: normals[seed],
                        angular_sigma_rad: 0.02,
                    })
                } else {
                    None
                },
            });
            let ps = perspective_sign::Source {
                stream: source.stream,
                frame: source.frame,
                ns,
            };
            let po = tracker.observe(perspective_sign::Observation {
                source: ps,
                fresh: true,
                seed,
                poses: std::array::from_fn(|j| perspective_sign::Pose {
                    normal: normals[j],
                    pivot_px: pivots[j],
                    pivot_grid_px: std::array::from_fn(|d| {
                        camera
                            .project(std::array::from_fn(|k| {
                                centers[j][k] - (15.0 + 3.75 * d as f64) * normals[j][k]
                            }))
                            .unwrap()
                    }),
                    allowance_px: 4.0 * blur,
                    uncertainty: None,
                }),
                transport: transform.map(|(a, translation)| perspective_sign::Transport {
                    from: perspective_sign::Source {
                        stream: ps.stream,
                        frame: ps.frame - 1,
                        ns: ns - 50_000_000,
                    },
                    to: ps,
                    a,
                    b: 0.0,
                    translation,
                    allowance_px: 7.0,
                }),
                head_rotation: None,
            });
            let reliability = [
                1.0 / blur,
                1.0 / blur,
                if regime == 3 { 0.2 } else { 1.0 },
                if missing {
                    0.0
                } else {
                    (1.0 - clip * 20.0).clamp(0.05, 1.0) * if regime == 4 { 0.15 } else { 1.0 }
                },
                if bo.independent_costs.is_some() {
                    1.0
                } else {
                    0.2
                },
                if po.independent.is_some() { 1.0 } else { 0.2 },
            ];
            let mut x = [0.0; N];
            for j in 0..EXPERTS {
                let cs = if j < 4 {
                    costs[j]
                } else if j == 4 {
                    bo.beam_costs
                } else {
                    po.costs
                };
                let delta = cs[1] - cs[0];
                if delta.is_finite() && reliability[j] > 0.0 {
                    x[j] = (delta / 4.0).tanh();
                    x[j + EXPERTS] = x[j] * reliability[j];
                }
            }
            if i > 0 {
                samples.push(Sample {
                    x,
                    label: if target == 0 { 1.0 } else { 0.0 },
                    regime,
                    session,
                    normal_errors_deg: errors,
                    scene:json!({"ellipse":[ellipse.center.0,ellipse.center.1,ellipse.major_radius,ellipse.minor_radius,ellipse.angle],"glint_available":!missing,"observed_glint":observed,"predicted_glints":predicted_glints,"normal_endpoints":std::array::from_fn::<_,2,_>(|j|camera.project(std::array::from_fn(|k|centers[j][k]+15.0*normals[j][k])).unwrap()),"normals":normals,"true_normal":truth,"center":center,"light":light}),
                    separation_deg: angle(normals[0], normals[1]).to_degrees(),
                });
            }
            previous_previous = previous;
            previous = Some(
                normals[if i == 0 {
                    seed
                } else {
                    usize::from(costs[0][1] < costs[0][0])
                }],
            );
            last_pivot = Some((head, estimate[2]));
        }
    }
    samples
}
