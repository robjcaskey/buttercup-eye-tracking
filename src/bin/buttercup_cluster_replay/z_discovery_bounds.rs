//! Conditional geometric origin bounds plus bounded photometric subdivision.
//! Image-score pruning is heuristic; analytic pivot polygons are conservative
//! for every motion inside the retained angle/intercept boxes.
use super::{motion_loss, Motion, Picture, P};
use serde::Serialize;
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    pub angle: [f64; 2],
    pub intercept: [[f64; 2]; 2],
}
impl Cell {
    pub fn around(m: Motion) -> Self {
        let b = m.map([0., 0.]);
        Self {
            angle: [
                (m.angle - 3f64.to_radians()).max(-15f64.to_radians()),
                (m.angle + 3f64.to_radians()).min(15f64.to_radians()),
            ],
            intercept: std::array::from_fn(|c| [b[c] - m.crop[c] - 1.5, b[c] - m.crop[c] + 1.5]),
        }
    }
    pub fn center(&self, crop: P) -> Motion {
        Motion {
            pivot: [0., 0.],
            angle: (self.angle[0] + self.angle[1]) * 0.5,
            translation: std::array::from_fn(|c| {
                (self.intercept[c][0] + self.intercept[c][1]) * 0.5
            }),
            crop,
        }
    }
    pub fn scaled(&self, s: f64) -> Self {
        Self {
            angle: self.angle,
            intercept: self.intercept.map(|v| v.map(|x| x * s)),
        }
    }
    fn children(&self) -> Vec<Self> {
        let mut out = Vec::new();
        for bits in 0..8 {
            let mut q = self.clone();
            let mid = (self.angle[0] + self.angle[1]) * 0.5;
            if bits & 1 == 0 {
                q.angle[1] = mid
            } else {
                q.angle[0] = mid
            }
            for c in 0..2 {
                let mid = (self.intercept[c][0] + self.intercept[c][1]) * 0.5;
                if bits & (2 << c) == 0 {
                    q.intercept[c][1] = mid
                } else {
                    q.intercept[c][0] = mid
                }
            }
            out.push(q);
        }
        out
    }
}
#[derive(Clone, Serialize)]
pub struct Bounds {
    pub cells: Vec<Cell>,
    pub origin_polygons: Vec<Vec<P>>,
    pub angle_range: [f64; 2],
    pub intercept_range: [[f64; 2]; 2],
    pub evaluated: usize,
    pub fit_samples: usize,
    pub score_pruned: usize,
    pub budget_pruned: usize,
    pub translation_bound: f64,
    pub heuristic_score_slack: f64,
    pub certified_photometric_bounds: bool,
}
fn clip(poly: Vec<P>, normal: P, limit: f64) -> Vec<P> {
    let mut out = Vec::new();
    if poly.is_empty() {
        return out;
    }
    let dot = |p: P| p[0] * normal[0] + p[1] * normal[1] - limit;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let da = dot(a);
        let db = dot(b);
        if da <= 1e-10 {
            out.push(a);
        }
        if (da < 0.) != (db < 0.) {
            let t = da / (da - db);
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    out
}
// b = (I-R)c + t, |t_i| <= T. Over an angle interval, each entry
// of R changes by <= half interval width (sin/cos are 1-Lipschitz).
// Domain bounds give |(R-Rmid)c| <= halfwidth*(max|cx|+max|cy|).
// Expanding the exact strips by that amount yields a conservative polygon.
pub fn origin_polygon(cell: &Cell, w: f64, h: f64, t: f64) -> Vec<P> {
    let mut poly = vec![
        [-w * 0.5, -h * 0.5],
        [w * 1.5, -h * 0.5],
        [w * 1.5, h * 1.5],
        [-w * 0.5, h * 1.5],
    ];
    let angle = (cell.angle[0] + cell.angle[1]) * 0.5;
    let (s, c) = angle.sin_cos();
    let rows = [[1. - c, s], [-s, 1. - c]];
    let err = (cell.angle[1] - cell.angle[0]) * 0.5 * (w * 1.5 + h * 1.5);
    for axis in 0..2 {
        poly = clip(poly, rows[axis], cell.intercept[axis][1] + t + err);
        poly = clip(
            poly,
            rows[axis].map(|v| -v),
            -cell.intercept[axis][0] + t + err,
        );
    }
    poly
}
pub fn subdivide(
    a: &Picture,
    b: &Picture,
    labels: &[i8],
    id: i8,
    cells: &[Cell],
    crop: P,
    scale: f64,
) -> (Motion, Bounds, Vec<Motion>) {
    let fit_samples = labels
        .iter()
        .enumerate()
        .filter(|(k, l)| **l == id && super::fold(*k % a.w(), *k / a.w()) == 0)
        .count();
    let mut proposals = Vec::new();
    for cell in cells {
        for q in if fit_samples >= 8 {
            cell.children()
        } else {
            vec![cell.clone()]
        } {
            if q.angle[0] > 15f64.to_radians() || q.angle[1] < -15f64.to_radians() {
                continue;
            }
            let m = q.center(crop);
            let loss = motion_loss(a, b, m, labels, id, false);
            proposals.push((loss, q, m));
        }
    }
    proposals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let best = proposals[0].0;
    let best_motion = proposals[0].2;
    let evaluated = proposals.len();
    let score_pruned = proposals.iter().filter(|p| p.0 > best + 0.025).count();
    proposals.retain(|p| p.0 <= best + 0.025);
    let budget_pruned = proposals.len().saturating_sub(16);
    proposals.truncate(16);
    let cells = proposals.iter().map(|p| p.1.clone()).collect::<Vec<_>>();
    let bank = proposals.iter().map(|p| p.2).collect();
    let angle_range = [
        cells
            .iter()
            .map(|c| c.angle[0])
            .fold(f64::INFINITY, f64::min),
        cells
            .iter()
            .map(|c| c.angle[1])
            .fold(f64::NEG_INFINITY, f64::max),
    ];
    let intercept_range = std::array::from_fn(|axis| {
        [
            cells
                .iter()
                .map(|c| c.intercept[axis][0])
                .fold(f64::INFINITY, f64::min),
            cells
                .iter()
                .map(|c| c.intercept[axis][1])
                .fold(f64::NEG_INFINITY, f64::max),
        ]
    });
    let translation_bound = 8. * scale;
    let origin_polygons = cells
        .iter()
        .map(|c| origin_polygon(c, a.w() as f64, a.h() as f64, translation_bound))
        .collect();
    (
        best_motion,
        Bounds {
            cells,
            origin_polygons,
            angle_range,
            intercept_range,
            evaluated,
            fit_samples,
            score_pruned,
            budget_pruned,
            translation_bound,
            heuristic_score_slack: 0.025,
            certified_photometric_bounds: false,
        },
        bank,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    fn inside(poly: &[P], p: P) -> bool {
        !poly.is_empty()
            && (0..poly.len()).all(|i| {
                let a = poly[i];
                let b = poly[(i + 1) % poly.len()];
                (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-8
            })
    }
    #[test]
    fn zero_rotation_does_not_invent_an_origin() {
        let c = Cell {
            angle: [0., 0.],
            intercept: [[2., 2.], [-1., -1.]],
        };
        let p = origin_polygon(&c, 53., 35., 8.);
        assert_eq!(p.len(), 4);
        assert!(inside(&p, [-26., -17.]));
        assert!(inside(&p, [79., 52.]));
    }
    #[test]
    fn exact_rotation_strips_include_all_feasible_origins() {
        for degrees in [-12f64, -3., 4., 11.] {
            let angle = degrees.to_radians();
            for x in [-20., 0., 20., 60.] {
                for y in [-10., 10., 40.] {
                    let m = Motion {
                        pivot: [x, y],
                        angle,
                        translation: [2., -1.],
                        crop: [0., 0.],
                    };
                    let b = m.map([0., 0.]);
                    let c = Cell {
                        angle: [angle, angle],
                        intercept: [[b[0], b[0]], [b[1], b[1]]],
                    };
                    assert!(inside(&origin_polygon(&c, 53., 35., 3.), [x, y]));
                }
            }
        }
    }
    #[test]
    fn interval_origin_enclosure_covers_angle_and_intercept_uncertainty() {
        let c = Cell {
            angle: [0.05, 0.15],
            intercept: [[-3., 4.], [-4., 3.]],
        };
        let poly = origin_polygon(&c, 53., 35., 2.);
        for i in 0..11 {
            let angle = 0.05 + i as f64 * 0.01;
            let (s, co) = angle.sin_cos();
            for x in -26..79 {
                for y in -17..52 {
                    let b = [
                        (1. - co) * x as f64 + s * y as f64,
                        -s * x as f64 + (1. - co) * y as f64,
                    ];
                    if b[0] >= -5. && b[0] <= 6. && b[1] >= -6. && b[1] <= 5. {
                        assert!(inside(&poly, [x as f64, y as f64]));
                    }
                }
            }
        }
    }
    #[test]
    fn impossible_translation_has_empty_origin_set_at_zero_rotation() {
        let c = Cell {
            angle: [0., 0.],
            intercept: [[20., 20.], [0., 0.]],
        };
        assert!(origin_polygon(&c, 53., 35., 2.).is_empty());
    }
}
