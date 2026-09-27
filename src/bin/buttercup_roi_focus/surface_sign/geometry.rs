//! Continuous sphere-family support. Camera axes: right/down/toward camera.
use buttercup_eye_tracking::focus_region::*;
use serde_json::{json, Value};

pub(super) fn unit(v: V3) -> V3 {
    scale(v, 1. / norm(v))
}
pub(super) fn camera_ray(p: [f64; 2]) -> V3 {
    unit([(p[0] - 4000.) / 4000., (p[1] - 3000.) / 4000., -1.])
}
pub(super) fn project(p: V3) -> [f64; 2] {
    [4000. - 4000. * p[0] / p[2], 3000. - 4000. * p[1] / p[2]]
}
pub(super) fn center(p: GazeRay, r: f64) -> V3 {
    sub(p.origin_iris_radii, scale(p.direction, (r * r - 1.).sqrt()))
}
pub(super) fn hits(p: GazeRay, r: f64, q: V3) -> bool {
    let g = center(p, r);
    let t = dot(q, g);
    t > 0. && dot(g, g) - t * t <= r * r + 1e-10
}

/// Intervals of d=sqrt(R^2-1) for which the forward pixel ray meets the sphere.
/// The discriminant inequality is quadratic in d with coefficient -(q.n)^2.
fn intervals(p: GazeRay, q: V3, lo: f64, hi: f64) -> Vec<[f64; 2]> {
    let c = p.origin_iris_radii;
    let n = p.direction;
    let cq = dot(c, q);
    let nq = dot(n, q);
    let a = -nq * nq;
    let b = -2. * (dot(c, n) - cq * nq);
    let k = dot(c, c) - cq * cq - 1.;
    let mut cuts = vec![lo, hi];
    if a.abs() < 1e-14 {
        if b.abs() > 1e-14 {
            let t = -k / b;
            if t > lo && t < hi {
                cuts.push(t);
            }
        }
    } else {
        let disc = b * b - 4. * a * k;
        if disc >= 0. {
            for t in [(-b - disc.sqrt()) / (2. * a), (-b + disc.sqrt()) / (2. * a)] {
                if t > lo && t < hi {
                    cuts.push(t);
                }
            }
        }
    }
    if nq.abs() > 1e-14 {
        let t = cq / nq;
        if t > lo && t < hi {
            cuts.push(t);
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    cuts.windows(2)
        .filter_map(|w| {
            let d = (w[0] + w[1]) * 0.5;
            ((a * d + b) * d + k <= 1e-10 && cq - d * nq > 0.).then_some([w[0], w[1]])
        })
        .collect()
}

/// Find one shared radius maximizing containment, not one radius per pixel.
pub(super) fn best_coverage(p: GazeRay, pixels: &[[f64; 2]], radii: [f64; 2]) -> Value {
    if pixels.is_empty() {
        return json!({"available":false});
    }
    let lo = (radii[0] * radii[0] - 1.).sqrt();
    let hi = (radii[1] * radii[1] - 1.).sqrt();
    let mut events = vec![(lo, 0i32), (hi, 0)];
    for &xy in pixels {
        for [a, b] in intervals(p, camera_ray(xy), lo, hi) {
            events.push((a, 1));
            events.push((b, -1));
        }
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut i = 0;
    let mut count = 0i32;
    let mut best = 0i32;
    let mut d = (lo + hi) / 2.;
    while i < events.len() {
        let at = events[i].0;
        while i < events.len() && (events[i].0 - at).abs() < 1e-12 {
            count += events[i].1;
            i += 1;
        }
        if i < events.len() && count > best {
            best = count;
            d = (at + events[i].0) * 0.5;
        }
    }
    // Endpoints are included in the continuous family. Verify the sweep result
    // using the original 3D ray-sphere condition, including boundary optima.
    let mut best_r = (1. + d * d).sqrt();
    let mut actual = 0usize;
    for r in [best_r, radii[0], radii[1]] {
        let n = pixels
            .iter()
            .filter(|&&xy| hits(p, r, camera_ray(xy)))
            .count();
        if n > actual {
            actual = n;
            best_r = r;
        }
    }
    assert!(
        actual >= best.max(0) as usize,
        "continuous sphere support sweep failed"
    );
    json!({"available":true,"coverage":actual as f64/pixels.len() as f64,"inside":actual,"samples":pixels.len(),"radius_iris_units":best_r,"center":center(p,best_r),"projected_center":project(center(p,best_r)),"radius_range":radii})
}

fn segment_distance(a: V3, b: V3, c: V3, d: V3) -> f64 {
    let u = sub(b, a);
    let v = sub(d, c);
    let w = sub(a, c);
    let aa = dot(u, u);
    let bb = dot(u, v);
    let cc = dot(v, v);
    let dd = dot(u, w);
    let ee = dot(v, w);
    let mut candidates = vec![];
    for s in [0., 1.] {
        candidates.push((s, ((ee + bb * s) / cc.max(1e-18)).clamp(0., 1.)));
    }
    for t in [0., 1.] {
        candidates.push((((bb * t - dd) / aa.max(1e-18)).clamp(0., 1.), t));
    }
    let det = aa * cc - bb * bb;
    if det > 1e-14 {
        let s = (bb * ee - cc * dd) / det;
        let t = (aa * ee - bb * dd) / det;
        if (0. ..=1.).contains(&s) && (0. ..=1.).contains(&t) {
            candidates.push((s, t));
        }
    }
    candidates
        .into_iter()
        .map(|(s, t)| norm(sub(add(a, scale(u, s)), add(c, scale(v, t)))))
        .fold(f64::INFINITY, f64::min)
}
pub(super) fn center_separation(rays: TheoreticalEllipseExplanations, radii: [f64; 2]) -> Value {
    let ends = rays.rays.map(|p| radii.map(|r| center(p, r)));
    let px = ends.map(|e| {
        e.map(|g| {
            let p = project(g);
            [p[0], p[1], 0.]
        })
    });
    let gap = segment_distance(ends[0][0], ends[0][1], ends[1][0], ends[1][1]);
    let gap_px = segment_distance(px[0][0], px[0][1], px[1][0], px[1][1]);
    json!({"radius_range":radii,"center_segments_iris_units":ends,"projected_center_segments_sensor_px":px,
        "minimum_3d_separation_iris_units":gap,"minimum_projected_separation_px":gap_px,
        "sufficient_independent_center_error_radius_px_strictly_less_than":gap_px*0.5,
        "condition":"Correct candidate belongs to this sphere family and an independently measured projected globe center is within the stated error radius. This is a sufficient separation bound, not a measured center or a calibrated confidence."})
}

pub(super) fn controls() -> Value {
    let p = GazeRay {
        origin_iris_radii: [2., 4., -40.],
        direction: unit([0.1, -0.5, 0.85]),
    };
    let radii = [1.8, 2.4];
    let mut mismatches = 0;
    for y in (2950..3750).step_by(11) {
        for x in (3750..4750).step_by(13) {
            let q = camera_ray([x as f64, y as f64]);
            let list = intervals(
                p,
                q,
                (radii[0] * radii[0] - 1f64).sqrt(),
                (radii[1] * radii[1] - 1f64).sqrt(),
            );
            for i in 0..101 {
                let r = radii[0] + (radii[1] - radii[0]) * i as f64 / 100.;
                let d = (r * r - 1.).sqrt();
                let predicted = list.iter().any(|v| d >= v[0] - 1e-10 && d <= v[1] + 1e-10);
                if predicted != hits(p, r, q) {
                    mismatches += 1;
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "radius interval control disagrees with direct sphere intersections"
    );
    json!({"continuous_radius_membership_mismatches":mismatches,"sphere_pose":p,"radius_range":radii,"checked_radii_per_pixel":101})
}
