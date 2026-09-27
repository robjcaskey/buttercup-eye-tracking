//! Deterministic, bounded classical RAW gradient ellipse proposal.
//! Measured gradient samples stay distinct from the fitted completion.
use buttercup_eye_tracking::geometry::Ellipse;
#[derive(Clone, Copy)]
struct Edge {
    x: f64,
    y: f64,
    gx: f64,
    gy: f64,
}
struct Image {
    p: Vec<f64>,
    w: usize,
    h: usize,
    lo: f64,
    range: f64,
}
impl Image {
    fn sample(&self, x: f64, y: f64) -> Option<f64> {
        if x < 0. || y < 0. || x >= self.w as f64 - 1. || y >= self.h as f64 - 1. {
            return None;
        }
        let (ix, iy) = (x as usize, y as usize);
        let (dx, dy) = (x - ix as f64, y - iy as f64);
        Some(
            (self.p[iy * self.w + ix] * (1. - dx) + self.p[iy * self.w + ix + 1] * dx) * (1. - dy)
                + (self.p[(iy + 1) * self.w + ix] * (1. - dx)
                    + self.p[(iy + 1) * self.w + ix + 1] * dx)
                    * dy,
        )
    }
    fn score(&self, e: Ellipse) -> f64 {
        let (s, c) = e.angle.sin_cos();
        let mut score = 0.;
        let mut counts = [0usize; 3];
        for k in 0..64 {
            let t = k as f64 * std::f64::consts::TAU / 64.;
            let (st, ct) = t.sin_cos();
            let x = e.center.0 + c * e.major_radius * ct - s * e.minor_radius * st;
            let y = e.center.1 + s * e.major_radius * ct + c * e.minor_radius * st;
            let nx = c * ct / e.major_radius - s * st / e.minor_radius;
            let ny = s * ct / e.major_radius + c * st / e.minor_radius;
            let n = nx.hypot(ny);
            let (nx, ny) = (nx / n, ny / n);
            // Upper-lid gradients must not reward a larger iris completion.
            // The upper rim is censored here, not declared absent or negative.
            if ny < -0.25 {
                continue;
            }
            if let (Some(a), Some(b)) = (
                self.sample(x - 0.85 * nx, y - 0.85 * ny),
                self.sample(x + 0.85 * nx, y + 0.85 * ny),
            ) {
                if b > self.lo + 0.35 * self.range && a < self.lo + 0.70 * self.range {
                    let gx = self
                        .sample(x + 0.5, y)
                        .zip(self.sample(x - 0.5, y))
                        .map(|(a, b)| a - b)
                        .unwrap_or(0.);
                    let gy = self
                        .sample(x, y + 0.5)
                        .zip(self.sample(x, y - 0.5))
                        .map(|(a, b)| a - b)
                        .unwrap_or(0.);
                    let alignment = ((gx * nx + gy * ny) / gx.hypot(gy).max(1e-6)).max(0.);
                    let d = ((b - a) / self.range).clamp(0., 0.4) * alignment.powi(4);
                    score += d;
                    if d > 0.05 {
                        if ny.abs() < 0.25 {
                            counts[usize::from(nx > 0.) * 2] += 1;
                        } else if ny > 0.9 {
                            counts[1] += 1;
                        }
                    }
                }
            }
        }
        if counts[0] < 3 || counts[1] < 4 || counts[2] < 3 {
            return -1.;
        }
        score / 64.
    }
}
fn solve(mut a: [[f64; 6]; 5]) -> Option<[f64; 5]> {
    for i in 0..5 {
        let best = (i..5).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        if a[best][i].abs() < 1e-9 {
            return None;
        }
        a.swap(i, best);
        let v = a[i][i];
        for j in i..6 {
            a[i][j] /= v;
        }
        for k in 0..5 {
            if k != i {
                let v = a[k][i];
                for j in i..6 {
                    a[k][j] -= v * a[i][j];
                }
            }
        }
    }
    Some(std::array::from_fn(|i| a[i][5]))
}
fn admissible(e: Ellipse, w: usize, h: usize) -> bool {
    e.center.0 > w as f64 * 0.1
        && e.center.0 < w as f64 * 0.9
        && e.center.1 > h as f64 * 0.1
        && e.center.1 < h as f64 * 0.9
        && e.major_radius >= 6.
        && e.major_radius < h as f64 * 0.49
        && e.minor_radius / e.major_radius > 0.5
        && e.minor_radius <= e.major_radius
}
fn ellipse(edges: &[Edge], w: usize, h: usize) -> Option<Ellipse> {
    let scale = 25.;
    let origin = [w as f64 * 0.5, h as f64 * 0.5];
    let mut a = [[0.; 6]; 5];
    for p in edges {
        let x = (p.x - origin[0]) / scale;
        let y = (p.y - origin[1]) / scale;
        // F(point)=0 and tangent dot grad(F)=0 are both linear in
        // conic coefficients. Tangents constrain the missing upper arc
        // without inventing observed points there.
        for row in [
            [x * x, x * y, y * y, x, y, 1.],
            [
                -p.gy * x,
                0.5 * (p.gx * x - p.gy * y),
                p.gx * y,
                -0.5 * p.gy,
                0.5 * p.gx,
                0.,
            ],
        ] {
            for i in 0..5 {
                for j in 0..6 {
                    a[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let p = solve(a)?;
    let det = p[0] * p[2] - p[1] * p[1] * 0.25;
    if det.abs() < 1e-8 {
        return None;
    }
    let cx = (-0.5 * p[2] * p[3] + 0.25 * p[1] * p[4]) / det;
    let cy = (-0.5 * p[0] * p[4] + 0.25 * p[1] * p[3]) / det;
    let r2 = 1. + p[0] * cx * cx + p[1] * cx * cy + p[2] * cy * cy;
    if r2.abs() < 1e-8 {
        return None;
    }
    let (a, b, c) = (p[0] / r2, p[1] / r2, p[2] / r2);
    let diff = (a - c).hypot(b);
    let small = (a + c - diff) * 0.5;
    let large = (a + c + diff) * 0.5;
    if small <= 0. || large <= 0. {
        return None;
    }
    let e = Ellipse {
        center: (origin[0] + cx * scale, origin[1] + cy * scale),
        major_radius: scale / small.sqrt(),
        minor_radius: scale / large.sqrt(),
        angle: 0.5 * b.atan2(a - c) + std::f64::consts::FRAC_PI_2,
    };
    admissible(e, w, h).then_some(e)
}
pub fn fit(raw: &[u16], w: usize, h: usize) -> Option<(Ellipse, Vec<[f64; 2]>)> {
    let (cw, ch) = (w / 4, h / 4);
    if cw < 32 || ch < 24 || raw.len() != w * h {
        return None;
    }
    let mut p = vec![0.; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            for dy in 0..4 {
                for dx in 0..4 {
                    p[y * cw + x] += raw[(y * 4 + dy) * w + x * 4 + dx] as f64 / 16.;
                }
            }
        }
    }
    let mut sorted = p.clone();
    sorted.sort_by(f64::total_cmp);
    let lo = sorted[sorted.len() / 20];
    let range = sorted[sorted.len() * 19 / 20] - lo;
    if range < 20. {
        return None;
    }
    let original = p.clone();
    for y in 1..ch - 1 {
        for x in 1..cw - 1 {
            p[y * cw + x] = (original[(y - 1) * cw + x]
                + original[(y + 1) * cw + x]
                + original[y * cw + x - 1]
                + original[y * cw + x + 1]
                + 4. * original[y * cw + x])
                / 8.;
        }
    }
    let im = Image {
        p,
        w: cw,
        h: ch,
        lo,
        range,
    };
    let mut edges = Vec::new();
    for y in 2..ch - 2 {
        for x in 2..cw - 2 {
            let gx = (im.p[y * cw + x + 1] - im.p[y * cw + x - 1]) * 0.5;
            let gy = (im.p[(y + 1) * cw + x] - im.p[(y - 1) * cw + x]) * 0.5;
            let n = gx.hypot(gy);
            if n < range * 0.055 {
                continue;
            }
            let (nx, ny) = (gx / n, gy / n);
            let before = im.sample(x as f64 - nx, y as f64 - ny)?;
            let after = im.sample(x as f64 + nx, y as f64 + ny)?;
            if after < lo + range * 0.35 || before > lo + range * 0.70 {
                continue;
            }
            edges.push(Edge {
                x: x as f64,
                y: y as f64,
                gx: nx,
                gy: ny,
            });
        }
    }
    if edges.len() < 30 {
        return None;
    }
    let mut rng = 829416u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut best = None;
    let mut best_score = 0.025;
    let proposals: Vec<_> = edges.iter().copied().filter(|e| e.gy >= -0.25).collect();
    if proposals.len() < 16 {
        return None;
    }
    for _ in 0..4096 {
        let selected: [Edge; 3] =
            std::array::from_fn(|_| proposals[next() as usize % proposals.len()]);
        if let Some(e) = ellipse(&selected, cw, ch) {
            let score = im.score(e);
            if score > best_score {
                best = Some(e);
                best_score = score;
            }
        }
    }
    let mut e = best?;
    for step in [1., 0.5, 0.25] {
        for _ in 0..3 {
            for axis in 0..5 {
                for sign in [-1., 1.] {
                    let mut candidate = e;
                    match axis {
                        0 => candidate.center.0 += step * sign,
                        1 => candidate.center.1 += step * sign,
                        2 => candidate.major_radius += step * sign,
                        3 => candidate.minor_radius += step * sign,
                        _ => candidate.angle += step * sign * 0.06,
                    };
                    if !admissible(candidate, cw, ch) {
                        continue;
                    }
                    let score = im.score(candidate);
                    if score > best_score {
                        e = candidate;
                        best_score = score;
                    }
                }
            }
        }
    }
    let (s, c) = e.angle.sin_cos();
    let mut points = Vec::new();
    for edge in edges {
        let dx = edge.x - e.center.0;
        let dy = edge.y - e.center.1;
        let x = c * dx + s * dy;
        let y = -s * dx + c * dy;
        let rho = (x * x / e.major_radius.powi(2) + y * y / e.minor_radius.powi(2)).sqrt();
        if (rho - 1.).abs() * e.minor_radius > 1. {
            continue;
        }
        let nx = c * x / e.major_radius.powi(2) - s * y / e.minor_radius.powi(2);
        let ny = s * x / e.major_radius.powi(2) + c * y / e.minor_radius.powi(2);
        if (nx * edge.gx + ny * edge.gy) / nx.hypot(ny) > 0.6 {
            points.push([edge.x * 4. + 1.5, edge.y * 4. + 1.5]);
        }
    }
    if points.len() < 16 {
        return None;
    }
    e.center = (e.center.0 * 4. + 1.5, e.center.1 * 4. + 1.5);
    e.major_radius *= 4.;
    e.minor_radius *= 4.;
    Some((e, points))
}
