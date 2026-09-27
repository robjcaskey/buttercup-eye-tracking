//! Experimental focus volumes from ambiguous circular-iris projections.
//! No monitor pose, target, selected sign or calibration map enters this module.
//! Coordinates are sensor-right/down, +Z toward camera, in iris-radius units.
//! Regions express repeated geometric compatibility, not measured attention.
use crate::geometry::{projected_circle_candidates, Ellipse};
use serde::{Deserialize, Serialize};

pub type V3 = [f64; 3];
pub fn add(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] + b[i])
}
pub fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] - b[i])
}
pub fn scale(a: V3, s: f64) -> V3 {
    a.map(|v| v * s)
}
pub fn dot(a: V3, b: V3) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GazeRay {
    pub origin_iris_radii: V3,
    pub direction: V3,
}
/// Both camera-facing circle interpretations of ONE ellipse. Alternatives
/// are correlated explanations, not two observations or calibrated probabilities.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TheoreticalEllipseExplanations {
    pub rays: [GazeRay; 2],
    pub separation_degrees: f64,
}
impl TheoreticalEllipseExplanations {
    pub fn from_ellipse(
        ellipse_sensor: Ellipse,
        focal: [f64; 2],
        principal: [f64; 2],
    ) -> Option<Self> {
        if ![
            ellipse_sensor.center.0,
            ellipse_sensor.center.1,
            ellipse_sensor.major_radius,
            ellipse_sensor.minor_radius,
            ellipse_sensor.angle,
        ]
        .iter()
        .all(|x| x.is_finite())
            || ellipse_sensor.minor_radius <= 0.
            || ellipse_sensor.major_radius < ellipse_sensor.minor_radius
        {
            return None;
        }
        let poses = projected_circle_candidates(ellipse_sensor, [0, 0], focal, principal)?;
        let rays = poses.map(|(n, c)| GazeRay {
            origin_iris_radii: c,
            direction: n,
        });
        Some(Self {
            separation_degrees: dot(rays[0].direction, rays[1].direction)
                .clamp(-1., 1.)
                .acos()
                .to_degrees(),
            rays,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FocusOptions {
    pub angular_tolerance_degrees: f64,
    pub minimum_crossing_degrees: f64,
    pub maximum_forward_radii: f64,
    pub cluster_radius_radii: f64,
    pub minimum_support: usize,
    pub competing_support_fraction: f64,
}
impl Default for FocusOptions {
    fn default() -> Self {
        Self {
            angular_tolerance_degrees: 2.,
            minimum_crossing_degrees: 3.,
            maximum_forward_radii: 250.,
            cluster_radius_radii: 6.,
            minimum_support: 8,
            competing_support_fraction: 0.35,
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Intersection {
    pub position: V3,
    pub gap_radii: f64,
    pub forward_radii: [f64; 2],
    pub branches: [usize; 2],
    pub quality: f64,
}
pub fn intersect(a: GazeRay, b: GazeRay, options: FocusOptions) -> Option<Intersection> {
    let d = sub(a.origin_iris_radii, b.origin_iris_radii);
    let cosine = dot(a.direction, b.direction);
    let det = 1. - cosine * cosine;
    if det < options.minimum_crossing_degrees.to_radians().sin().powi(2) {
        return None;
    }
    let ad = dot(a.direction, d);
    let bd = dot(b.direction, d);
    let t = (cosine * bd - ad) / det;
    let u = (bd - cosine * ad) / det;
    if ![t, u]
        .iter()
        .all(|x| x.is_finite() && *x > 2. && *x < options.maximum_forward_radii)
    {
        return None;
    }
    let pa = add(a.origin_iris_radii, scale(a.direction, t));
    let pb = add(b.origin_iris_radii, scale(b.direction, u));
    let gap = norm(sub(pa, pb));
    let tolerance = (options.angular_tolerance_degrees.to_radians().tan() * (t + u)).max(0.75);
    if gap > tolerance {
        return None;
    }
    Some(Intersection {
        position: scale(add(pa, pb), 0.5),
        gap_radii: gap,
        forward_radii: [t, u],
        branches: [0, 0],
        quality: (-0.5 * (gap / tolerance).powi(2)).exp(),
    })
}

#[derive(Clone, Debug)]
struct Cluster {
    center: V3,
    samples: Vec<V3>,
    support: usize,
    weight: f64,
    first: f64,
    last: f64,
    occupied_bins: usize,
    last_bin: Option<i64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FocusRegion {
    pub id: usize,
    pub center: V3,
    pub lower: V3,
    pub upper: V3,
    pub support_pairs: usize,
    pub occupied_200ms_bins: usize,
    pub support_weight: f64,
    pub span_seconds: f64,
}
#[derive(Clone, Debug)]
pub struct FocusRegionEstimator {
    pub options: FocusOptions,
    clusters: Vec<Cluster>,
    pub paired_observations: usize,
    pub geometric_pairs: usize,
    pub discarded_cluster_births: usize,
}
impl FocusRegionEstimator {
    pub fn new(options: FocusOptions) -> Self {
        Self {
            options,
            clusters: vec![],
            paired_observations: 0,
            geometric_pairs: 0,
            discarded_cluster_births: 0,
        }
    }
    pub fn observe(
        &mut self,
        seconds: f64,
        a: &TheoreticalEllipseExplanations,
        b: &TheoreticalEllipseExplanations,
    ) -> Vec<Intersection> {
        self.paired_observations += 1;
        let mut points = vec![];
        for i in 0..2 {
            for j in 0..2 {
                if let Some(mut p) = intersect(a.rays[i], b.rays[j], self.options) {
                    p.branches = [i, j];
                    points.push(p);
                }
            }
        }
        if points.is_empty() {
            return points;
        }
        // Candidate numbering must not decide which cluster gets born first.
        points.sort_by(|a, b| {
            a.position[0]
                .total_cmp(&b.position[0])
                .then(a.position[1].total_cmp(&b.position[1]))
                .then(a.position[2].total_cmp(&b.position[2]))
        });
        self.geometric_pairs += 1;
        // Assign all four alternatives against the same PRE-update snapshot.
        // Each exposure pair may update a cluster only once.
        let mut assignments: Vec<(usize, Intersection)> = vec![];
        for p in &points {
            let nearest = self
                .clusters
                .iter()
                .enumerate()
                .map(|(i, c)| (i, norm(sub(c.center, p.position))))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            let id = if let Some((i, d)) =
                nearest.filter(|(_, d)| *d <= self.options.cluster_radius_radii)
            {
                let _ = d;
                i
            } else if self.clusters.len() < 64 {
                self.clusters.push(Cluster {
                    center: p.position,
                    samples: vec![],
                    support: 0,
                    weight: 0.,
                    first: seconds,
                    last: seconds,
                    occupied_bins: 0,
                    last_bin: None,
                });
                self.clusters.len() - 1
            } else {
                self.discarded_cluster_births += 1;
                continue;
            };
            if let Some((_, old)) = assignments.iter_mut().find(|(i, _)| *i == id) {
                if p.quality > old.quality {
                    *old = *p;
                }
            } else {
                assignments.push((id, *p));
            }
        }
        let alternatives = assignments.len().max(1) as f64;
        for (id, p) in assignments {
            let c = &mut self.clusters[id];
            let bin = (seconds / 0.2).floor() as i64;
            let new_bin = c.last_bin != Some(bin);
            if new_bin {
                c.occupied_bins += 1;
                c.last_bin = Some(bin);
            }
            // High-rate repetition cannot dominate a slower fixation interval.
            let w = p.quality / alternatives * if new_bin { 1. } else { 0.05 };
            c.center = scale(
                add(scale(c.center, c.weight), scale(p.position, w)),
                1. / (c.weight + w),
            );
            c.weight += w;
            c.support += 1;
            c.last = seconds;
            if c.samples.len() < 512 {
                c.samples.push(p.position);
            } else if new_bin {
                let slot = (c.support.wrapping_mul(2654435761)) % 512;
                c.samples[slot] = p.position;
            }
        }
        points
    }
    pub fn regions(&self) -> Vec<FocusRegion> {
        let best = self
            .clusters
            .iter()
            .filter(|c| {
                c.support >= self.options.minimum_support
                    && c.occupied_bins >= 3
                    && c.last - c.first >= 0.4
            })
            .map(|c| c.weight)
            .fold(0., f64::max);
        self.clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.support >= self.options.minimum_support
                    && c.occupied_bins >= 3
                    && c.last - c.first >= 0.4
                    && c.weight >= best * self.options.competing_support_fraction
            })
            .map(|(id, c)| {
                let mut lower = [0.; 3];
                let mut upper = [0.; 3];
                for a in 0..3 {
                    let mut xs = c.samples.iter().map(|p| p[a]).collect::<Vec<_>>();
                    xs.sort_by(f64::total_cmp);
                    lower[a] = xs[(xs.len() - 1) * 5 / 100] - 0.5;
                    upper[a] = xs[(xs.len() - 1) * 95 / 100] + 0.5;
                }
                FocusRegion {
                    id,
                    center: c.center,
                    lower,
                    upper,
                    support_pairs: c.support,
                    occupied_200ms_bins: c.occupied_bins,
                    support_weight: c.weight,
                    span_seconds: c.last - c.first,
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RayClassification {
    pub status: &'static str,
    pub region: Option<usize>,
    pub miss_iris_radii: Option<f64>,
    pub miss_degrees: Option<f64>,
    pub forward_iris_radii: Option<f64>,
}
/// Exact minimum distance from a forward half-line to an axis-aligned box.
/// Between box-face crossings the squared distance is a quadratic in t.
pub fn ray_box_distance(ray: GazeRay, region: &FocusRegion) -> (f64, f64) {
    let mut cuts = vec![0.];
    for a in 0..3 {
        if ray.direction[a].abs() > 1e-12 {
            for e in [region.lower[a], region.upper[a]] {
                let t = (e - ray.origin_iris_radii[a]) / ray.direction[a];
                if t > 0. && t.is_finite() {
                    cuts.push(t);
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    cuts.push(f64::INFINITY);
    let distance = |t: f64| {
        let p = add(ray.origin_iris_radii, scale(ray.direction, t));
        (0..3)
            .map(|i| (p[i] - p[i].clamp(region.lower[i], region.upper[i])).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut best = (distance(0.), 0.);
    for w in cuts.windows(2) {
        let m = if w[1].is_finite() {
            (w[0] + w[1]) * 0.5
        } else {
            w[0] + 1.
        };
        let p = add(ray.origin_iris_radii, scale(ray.direction, m));
        let (mut aa, mut bb) = (0., 0.);
        for i in 0..3 {
            let edge = if p[i] < region.lower[i] {
                region.lower[i]
            } else if p[i] > region.upper[i] {
                region.upper[i]
            } else {
                continue;
            };
            aa += ray.direction[i].powi(2);
            bb += ray.direction[i] * (ray.origin_iris_radii[i] - edge);
        }
        let t = if aa > 1e-15 {
            (-bb / aa).clamp(w[0], w[1])
        } else {
            w[0]
        };
        let d = distance(t);
        if d < best.0 {
            best = (d, t);
        }
    }
    best
}
pub fn classify(ray: GazeRay, regions: &[FocusRegion], near_degrees: f64) -> RayClassification {
    let best = regions
        .iter()
        .map(|r| {
            let (d, t) = ray_box_distance(ray, r);
            (r.id, d, t, d.atan2(t.max(1e-9)).to_degrees())
        })
        .min_by(|a, b| a.3.total_cmp(&b.3));
    if let Some((id, d, t, angle)) = best {
        RayClassification {
            status: if d < 1e-8 {
                "inside"
            } else if angle <= near_degrees {
                "nearby"
            } else {
                "outside"
            },
            region: Some(id),
            miss_iris_radii: Some(d),
            miss_degrees: Some(angle),
            forward_iris_radii: Some(t),
        }
    } else {
        RayClassification {
            status: "unresolved_region",
            region: None,
            miss_iris_radii: None,
            miss_degrees: None,
            forward_iris_radii: None,
        }
    }
}
