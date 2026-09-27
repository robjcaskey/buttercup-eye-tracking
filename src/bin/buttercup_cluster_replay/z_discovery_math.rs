//! Whole-image layered motion discovery at coarse resolution. No anatomical ROIs.
use super::{Image, P};
use serde::Serialize;
#[path = "z_discovery_bounds.rs"]
mod bounds;
use bounds::{Bounds, Cell};

#[derive(Clone)]
pub struct Picture {
    pub channels: [Image; 3],
}
impl Picture {
    pub fn w(&self) -> usize {
        self.channels[0].w
    }
    pub fn h(&self) -> usize {
        self.channels[0].h
    }
    fn sample(&self, p: P, c: usize) -> Option<f64> {
        self.channels[c].sample(p)
    }
    fn texture(&self, p: P) -> f64 {
        (0..3)
            .map(|c| {
                let a = self.sample([p[0] - 1., p[1]], c).unwrap_or(0.);
                let b = self.sample([p[0] + 1., p[1]], c).unwrap_or(0.);
                let d = self.sample([p[0], p[1] - 1.], c).unwrap_or(0.);
                let e = self.sample([p[0], p[1] + 1.], c).unwrap_or(0.);
                (a - b).abs() + (d - e).abs()
            })
            .sum::<f64>()
            / 6.
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Motion {
    pub pivot: P,
    pub translation: P,
    pub angle: f64,
    pub crop: P,
}
impl Motion {
    pub fn map(self, p: P) -> P {
        let (s, c) = self.angle.sin_cos();
        let d = [p[0] - self.pivot[0], p[1] - self.pivot[1]];
        [
            self.pivot[0] + c * d[0] - s * d[1] + self.translation[0] + self.crop[0],
            self.pivot[1] + s * d[0] + c * d[1] + self.translation[1] + self.crop[1],
        ]
    }
    pub fn inverse(self, q: P) -> P {
        let (s, c) = self.angle.sin_cos();
        let d = [
            q[0] - self.pivot[0] - self.translation[0] - self.crop[0],
            q[1] - self.pivot[1] - self.translation[1] - self.crop[1],
        ];
        [
            self.pivot[0] + c * d[0] + s * d[1],
            self.pivot[1] - s * d[0] + c * d[1],
        ]
    }
    fn valid(self) -> bool {
        self.angle.abs() <= 15f64.to_radians() && self.translation.iter().all(|t| t.abs() <= 8.)
    }
    pub fn identity(crop: P, w: usize, h: usize) -> Self {
        Self {
            pivot: [w as f64 / 2., h as f64 / 2.],
            translation: [0., 0.],
            angle: 0.,
            crop,
        }
    }
    // Exact reparameterization demonstrates that a 2D pivot is not identified
    // independently of free translation. No anatomical interpretation is made.
    #[cfg(test)]
    fn repivot(self, p: P) -> Self {
        let mut m = Self { pivot: p, ..self };
        let old = self.map([0., 0.]);
        let new = m.map([0., 0.]);
        for c in 0..2 {
            m.translation[c] += old[c] - new[c];
        }
        m
    }
}
// Independent per-channel gain/offset invariance. Neighborhoods are warped,
// including rotation, rather than comparing absolute brightness at one pixel.
pub fn cost(a: &Picture, b: &Picture, p: P, m: Motion) -> Option<f64> {
    let mut result = 0.;
    let mut channels = 0;
    for c in 0..3 {
        let (mut sx, mut sy, mut xx, mut yy, mut xy) = (0., 0., 0., 0., 0.);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let r = [p[0] + dx as f64, p[1] + dy as f64];
                let x = a.sample(r, c)?;
                let y = b.sample(m.map(r), c)?;
                sx += x;
                sy += y;
                xx += x * x;
                yy += y * y;
                xy += x * y;
            }
        }
        let vx = xx - sx * sx / 9.;
        let vy = yy - sy * sy / 9.;
        if vx > 1e-6 && vy > 1e-6 {
            let ncc = ((xy - sx * sy / 9.) / (vx * vy).sqrt()).clamp(-1., 1.);
            result += (1. - ncc).min(1.);
            channels += 1;
        }
    }
    if channels >= 2 {
        Some(result / channels as f64)
    } else {
        None
    }
}
#[derive(Clone, Serialize)]
pub struct Match {
    pub p: P,
    pub q: P,
    pub cost: f64,
    pub fb: f64,
}
fn local_matches(a: &Picture, b: &Picture, crop: P, seeds: Option<&[Motion]>) -> Vec<Match> {
    let mut out = Vec::new();
    let base = Motion::identity(crop, a.w(), a.h());
    let reverse = Motion::identity(crop.map(|v| -v), a.w(), a.h());
    let seeds = seeds.map(|s| s.to_vec()).unwrap_or_default();
    let bank = |reverse_direction: bool| {
        let centers = if seeds.is_empty() {
            vec![if reverse_direction { reverse } else { base }]
        } else if reverse_direction {
            seeds
                .iter()
                .map(|m| Motion {
                    pivot: [0., 0.],
                    translation: m.inverse([0., 0.]),
                    angle: -m.angle,
                    crop: [0., 0.],
                })
                .collect()
        } else {
            seeds.clone()
        };
        let radius = if seeds.is_empty() { 6 } else { 2 };
        let mut out = Vec::new();
        for center in centers {
            for y in -radius..=radius {
                for x in -radius..=radius {
                    out.push(Motion {
                        translation: [
                            center.translation[0] + x as f64,
                            center.translation[1] + y as f64,
                        ],
                        ..center
                    });
                }
            }
        }
        out
    };
    let forward_bank = bank(false);
    let backward_bank = bank(true);
    for y in (3..a.h() - 3).step_by(2) {
        for x in (3..a.w() - 3).step_by(2) {
            let p = [x as f64, y as f64];
            if a.texture(p) < 0.003 {
                continue;
            }
            let mut best = (1., base);
            for &m in &forward_bank {
                let e = cost(a, b, p, m).unwrap_or(1.);
                if e < best.0 {
                    best = (e, m);
                }
            }
            for step in [0.5, 0.25] {
                let start = best.1;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let m = Motion {
                            translation: [
                                start.translation[0] + dx as f64 * step,
                                start.translation[1] + dy as f64 * step,
                            ],
                            ..start
                        };
                        let e = cost(a, b, p, m).unwrap_or(1.);
                        if e < best.0 {
                            best = (e, m);
                        }
                    }
                }
            }
            if best.0 > 0.22 {
                continue;
            }
            let q = best.1.map(p);
            let mut back = (1., [0., 0.]);
            for &m in &backward_bank {
                let e = cost(b, a, q, m).unwrap_or(1.);
                if e < back.0 {
                    back = (e, m.map(q));
                }
            }
            let fb = distance(back.1, p);
            if back.0 < 0.25 && fb <= 1.0 {
                out.push(Match {
                    p,
                    q,
                    cost: best.0,
                    fb,
                });
            }
        }
    }
    out
}
fn distance(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn fit_pair(a: &Match, b: &Match, base: Motion) -> Option<Motion> {
    let p = [b.p[0] - a.p[0], b.p[1] - a.p[1]];
    let q = [b.q[0] - a.q[0], b.q[1] - a.q[1]];
    if p[0].hypot(p[1]) < 6. || q[0].hypot(q[1]) < 6. {
        return None;
    }
    let angle = (p[0] * q[1] - p[1] * q[0]).atan2(p[0] * q[0] + p[1] * q[1]);
    let mut m = Motion { angle, ..base };
    let mapped = m.map(a.p);
    m.translation = [a.q[0] - mapped[0], a.q[1] - mapped[1]];
    m.valid().then_some(m)
}
fn seed_motions(matches: &[Match], base: Motion) -> Vec<Motion> {
    let mut bank = vec![base];
    for s in matches {
        let mut m = base;
        m.translation = [
            s.q[0] - s.p[0] - base.crop[0],
            s.q[1] - s.p[1] - base.crop[1],
        ];
        bank.push(m);
    }
    for i in 0..matches.len() {
        for jump in [7, 19, 43, 71] {
            if !matches.is_empty() {
                if let Some(m) = fit_pair(&matches[i], &matches[(i + jump) % matches.len()], base) {
                    bank.push(m);
                }
            }
        }
    }
    let mut remaining = vec![true; matches.len()];
    let mut chosen = Vec::new();
    for _ in 0..3 {
        let mut best = (0usize, base);
        for &m in &bank {
            let n = matches
                .iter()
                .zip(&remaining)
                .filter(|(s, ok)| **ok && distance(m.map(s.p), s.q) < 0.75)
                .count();
            if n > best.0 {
                best = (n, m);
            }
        }
        if best.0 < 8 {
            break;
        }
        chosen.push(best.1);
        for (s, ok) in matches.iter().zip(&mut remaining) {
            if distance(best.1.map(s.p), s.q) < 1.0 {
                *ok = false;
            }
        }
    }
    if chosen.is_empty() {
        chosen.push(base);
    }
    chosen
}
#[derive(Clone, Serialize)]
pub struct PairOrder {
    pub a: usize,
    pub b: usize,
    pub fit_samples: usize,
    pub held_samples: usize,
    pub fit_margin: f64,
    pub held_margin: f64,
    pub front: Option<usize>,
    pub fit_interval: [f64; 2],
    pub held_interval: [f64; 2],
}
#[derive(Clone, Serialize)]
pub struct Iteration {
    pub motions: Vec<Motion>,
    pub labels: Vec<i8>,
    pub order: Vec<usize>,
    pub allowed_orders: Vec<Vec<usize>>,
    pub ordering: Vec<PairOrder>,
    pub changed: usize,
    pub fit_loss: f64,
    pub held_loss: f64,
    pub coverage: f64,
    pub mask_coverage: f64,
    pub overlapping: usize,
    pub origin_hypotheses: usize,
    pub bounds: Vec<Bounds>,
    pub ordering_conflict: bool,
    pub occluded_fit_samples: usize,
}
#[derive(Serialize)]
pub struct Discovery {
    pub matches: Vec<Match>,
    pub iterations: Vec<Iteration>,
    pub baseline_loss: f64,
    pub baseline_coverage: f64,
    pub translation_loss: f64,
    pub translation_coverage: f64,
}
fn fold(x: usize, y: usize) -> usize {
    (x / 4 + y / 4) % 2
}
fn label_at(labels: &[i8], w: usize, h: usize, p: P) -> i8 {
    let x = p[0].round() as isize;
    let y = p[1].round() as isize;
    if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
        -1
    } else {
        labels[y as usize * w + x as usize]
    }
}
fn assign(
    a: &Picture,
    b: &Picture,
    ms: &[Motion],
    old: &[i8],
    edges: &[(usize, usize)],
) -> Vec<i8> {
    let (w, h) = (a.w(), a.h());
    let mut labels = vec![-1; w * h];
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let p = [x as f64, y as f64];
            if a.texture(p) < 0.002 {
                continue;
            }
            let mut best = (0.36, -1);
            for (id, &m) in ms.iter().enumerate() {
                let mut e = cost(a, b, p, m).unwrap_or(1.);
                // A previous source assignment may persist behind an evidenced
                // foreground layer, at a bounded penalty. Unexplained mismatches
                // remain unknown; no hallucinated newly exposed texture is filled.
                if old[y * w + x] == id as i8 {
                    let q = m.map(p);
                    for &(front, back) in edges {
                        if back != id {
                            continue;
                        }
                        let r = ms[front].inverse(q);
                        if label_at(old, w, h, r) == front as i8
                            && cost(a, b, r, ms[front]).is_some_and(|v| v < 0.15)
                        {
                            e = e.min(0.24);
                            break;
                        }
                    }
                }
                let mut neighbors = 0.;
                let mut disagree = 0.;
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let k = (y as isize + dy) as usize * w + (x as isize + dx) as usize;
                    if old[k] >= 0 {
                        neighbors += 1.;
                        disagree += f64::from(old[k] != id as i8);
                    }
                }
                e += 0.025 * disagree / if neighbors > 0. { neighbors } else { 1. };
                if e < best.0 {
                    best = (e, id as i8);
                }
            }
            labels[y * w + x] = best.1;
        }
    }
    labels
}
fn spatial_labels(matches: &[Match], ms: &[Motion], w: usize, h: usize, scale: f64) -> Vec<i8> {
    let groups = matches
        .iter()
        .map(|s| {
            ms.iter()
                .enumerate()
                .map(|(id, m)| (distance(m.map(s.p), s.q), id))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .filter(|(e, _)| *e < 0.9 * scale)
                .map(|(_, id)| id)
        })
        .collect::<Vec<_>>();
    let mut labels = vec![-1; w * h];
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let p = [x as f64, y as f64];
            let mut votes = vec![0.; ms.len()];
            for (s, group) in matches.iter().zip(&groups) {
                if let Some(id) = group {
                    let d = distance(s.p, p) / scale;
                    if d < 3.5 {
                        votes[*id] += (-d * d / 3.).exp();
                    }
                }
            }
            let winner = votes.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1));
            if let Some((id, v)) = winner {
                if *v > 0.05 && *v > votes.iter().sum::<f64>() * 0.55 {
                    labels[y * w + x] = id as i8;
                }
            }
        }
    }
    labels
}
fn measured_fit_labels(matches: &[Match], labels: &[i8], w: usize, h: usize) -> Vec<i8> {
    let mut fit = vec![-1; w * h];
    for s in matches {
        let x = s.p[0].round() as usize;
        let y = s.p[1].round() as usize;
        if x < w && y < h {
            fit[y * w + x] = labels[y * w + x];
        }
    }
    fit
}
fn motion_loss(a: &Picture, b: &Picture, m: Motion, labels: &[i8], id: i8, held: bool) -> f64 {
    let mut sum = 0.;
    let mut n = 0;
    for y in 2..a.h() - 2 {
        for x in 2..a.w() - 2 {
            if labels[y * a.w() + x] != id || fold(x, y) != usize::from(held) {
                continue;
            }
            sum += cost(a, b, [x as f64, y as f64], m).unwrap_or(0.7).min(0.7);
            n += 1;
        }
    }
    if n < 8 {
        1.
    } else {
        sum / n as f64
    }
}
// Order evidence uses exactly the same overlapping target pixels for both
// hypotheses. Train/held margins are separated, with support and tie rejection.
pub(super) fn order_evidence(
    a: &Picture,
    b: &Picture,
    labels: &[i8],
    ms: &[Motion],
    banks: &[Vec<Motion>],
) -> Vec<PairOrder> {
    let mut out = Vec::new();
    for first in 0..ms.len() {
        for second in first + 1..ms.len() {
            let mut sums = [0., 0.];
            let mut lows = [0., 0.];
            let mut highs = [0., 0.];
            let mut counts = [0usize, 0];
            for y in 2..b.h() - 2 {
                for x in 2..b.w() - 2 {
                    let q = [x as f64, y as f64];
                    let p = ms[first].inverse(q);
                    let r = ms[second].inverse(q);
                    if label_at(labels, a.w(), a.h(), p) != first as i8
                        || label_at(labels, a.w(), a.h(), r) != second as i8
                    {
                        continue;
                    }
                    if let (Some(ca), Some(cb)) =
                        (cost(a, b, p, ms[first]), cost(a, b, r, ms[second]))
                    {
                        let f = fold(x, y);
                        let range = |id: usize| -> Option<(f64, f64)> {
                            let mut low = f64::INFINITY;
                            let mut high = f64::NEG_INFINITY;
                            for &m in &banks[id] {
                                let r = m.inverse(q);
                                if label_at(labels, a.w(), a.h(), r) != id as i8 {
                                    return None;
                                }
                                let v = cost(a, b, r, m)?;
                                low = low.min(v);
                                high = high.max(v);
                            }
                            Some((low, high))
                        };
                        let (Some(ra), Some(rb)) = (range(first), range(second)) else {
                            continue;
                        };
                        sums[f] += cb - ca;
                        lows[f] += rb.0 - ra.1;
                        highs[f] += rb.1 - ra.0;
                        counts[f] += 1;
                    }
                }
            }
            let margins = std::array::from_fn::<_, 2, _>(|i| sums[i] / counts[i].max(1) as f64);
            let intervals = std::array::from_fn::<_, 2, _>(|i| {
                [
                    lows[i] / counts[i].max(1) as f64,
                    highs[i] / counts[i].max(1) as f64,
                ]
            });
            let front = if counts.iter().all(|n| *n >= 6) && intervals.iter().all(|v| v[0] > 0.04) {
                Some(first)
            } else if counts.iter().all(|n| *n >= 6) && intervals.iter().all(|v| v[1] < -0.04) {
                Some(second)
            } else {
                None
            };
            out.push(PairOrder {
                a: first,
                b: second,
                fit_samples: counts[0],
                held_samples: counts[1],
                fit_margin: margins[0],
                held_margin: margins[1],
                front,
                fit_interval: intervals[0],
                held_interval: intervals[1],
            });
        }
    }
    out
}
fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn go(v: Vec<usize>, n: usize, out: &mut Vec<Vec<usize>>) {
        if v.len() == n {
            out.push(v);
            return;
        }
        for k in 0..n {
            if !v.contains(&k) {
                let mut w = v.clone();
                w.push(k);
                go(w, n, out);
            }
        }
    }
    let mut out = Vec::new();
    go(Vec::new(), n, &mut out);
    out
}
fn choose_order(n: usize, evidence: &[PairOrder]) -> Vec<usize> {
    let candidates = permutations(n);
    let mut best = (usize::MAX, Vec::new());
    for p in candidates {
        let violations = evidence
            .iter()
            .filter(|e| {
                e.front.is_some_and(|f| {
                    let back = if f == e.a { e.b } else { e.a };
                    p.iter().position(|&x| x == f) > p.iter().position(|&x| x == back)
                })
            })
            .count();
        if violations < best.0 {
            best = (violations, p);
        }
    }
    best.1
}
fn aggregate(a: &Picture, b: &Picture, labels: &[i8], ms: &[Motion]) -> (f64, f64, f64, f64) {
    let (mut sum, mut count) = ([0., 0.], [0usize, 0]);
    let mut supported = 0;
    let mut fresh = 0;
    let mut total = 0;
    for y in 2..a.h() - 2 {
        for x in 2..a.w() - 2 {
            if a.texture([x as f64, y as f64]) < 0.002 {
                continue;
            }
            total += 1;
            let id = labels[y * a.w() + x];
            let c = if id < 0 {
                0.7
            } else {
                supported += 1;
                cost(a, b, [x as f64, y as f64], ms[id as usize])
                    .unwrap_or(0.7)
                    .min(0.7)
            };
            if id >= 0 && c < 0.36 {
                fresh += 1;
            }
            let f = fold(x, y);
            sum[f] += c;
            count[f] += 1;
        }
    }
    (
        sum[0] / count[0].max(1) as f64,
        sum[1] / count[1].max(1) as f64,
        fresh as f64 / total.max(1) as f64,
        supported as f64 / total.max(1) as f64,
    )
}
pub fn discover(a: &Picture, b: &Picture, crop: P, iterations: usize) -> Discovery {
    discover_impl(a, b, crop, iterations, None, 1.)
}
pub fn refine_level(
    a: &Picture,
    b: &Picture,
    crop: P,
    previous: &Discovery,
    scale: f64,
    iterations: usize,
) -> Discovery {
    let last = previous.iterations.last().unwrap();
    let motions = last
        .motions
        .iter()
        .map(|m| Motion {
            pivot: m.pivot.map(|v| v * scale),
            translation: m.translation.map(|v| v * scale),
            crop,
            angle: m.angle,
        })
        .collect();
    let oldw = (a.w() + 1) / 2;
    let oldh = (a.h() + 1) / 2;
    let labels = (0..a.w() * a.h())
        .map(|k| {
            label_at(
                &last.labels,
                oldw,
                oldh,
                [(k % a.w()) as f64 / scale, (k / a.w()) as f64 / scale],
            )
        })
        .collect();
    let cells = last
        .bounds
        .iter()
        .map(|b| b.cells.iter().map(|c| c.scaled(scale)).collect())
        .collect();
    discover_impl(
        a,
        b,
        crop,
        iterations,
        Some((
            motions,
            labels,
            cells,
            last.ordering
                .iter()
                .filter_map(|e| e.front.map(|f| (f, if f == e.a { e.b } else { e.a })))
                .collect(),
        )),
        scale,
    )
}
fn discover_impl(
    a: &Picture,
    b: &Picture,
    crop: P,
    iterations: usize,
    initial: Option<(Vec<Motion>, Vec<i8>, Vec<Vec<Cell>>, Vec<(usize, usize)>)>,
    scale: f64,
) -> Discovery {
    let base = Motion::identity(crop, a.w(), a.h());
    let parent_seeds = initial.as_ref().map(|v| {
        v.2.iter()
            .flat_map(|cells| cells.iter().map(|c| c.center(crop)))
            .collect::<Vec<_>>()
    });
    let matches = local_matches(a, b, crop, parent_seeds.as_deref());
    let mut ms = seed_motions(&matches, base);
    let unknown = vec![-1; a.w() * a.h()];
    let bl = assign(a, b, &[base], &unknown, &[]);
    let (_, baseline_loss, baseline_coverage, _) = aggregate(a, b, &bl, &[base]);
    let translations = matches
        .iter()
        .map(|s| [s.q[0] - s.p[0] - crop[0], s.q[1] - s.p[1] - crop[1]])
        .collect::<Vec<_>>();
    let median = |c: usize| {
        let mut v = translations.iter().map(|p| p[c]).collect::<Vec<_>>();
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            0.
        } else {
            v[v.len() / 2]
        }
    };
    let tm = Motion {
        translation: [median(0), median(1)],
        ..base
    };
    let tl = assign(a, b, &[tm], &unknown, &[]);
    let (_, translation_loss, translation_coverage, _) = aggregate(a, b, &tl, &[tm]);
    let (mut labels, mut cells, mut edges) = if let Some((seed, labels, cells, edges)) = initial {
        ms = seed;
        (labels, cells, edges)
    } else {
        let labels = spatial_labels(&matches, &ms, a.w(), a.h(), scale);
        (
            labels,
            ms.iter().map(|&m| vec![Cell::around(m)]).collect(),
            Vec::new(),
        )
    };
    let mut history = Vec::new();
    for _ in 0..iterations {
        let mut fit_labels = measured_fit_labels(&matches, &labels, a.w(), a.h());
        let mut occluded_fit_samples = 0;
        for k in 0..fit_labels.len() {
            if fit_labels[k] < 0 {
                continue;
            }
            let id = fit_labels[k] as usize;
            let p = [(k % a.w()) as f64, (k / a.w()) as f64];
            let q = ms[id].map(p);
            if edges.iter().any(|&(front, back)| {
                back == id && label_at(&labels, a.w(), a.h(), ms[front].inverse(q)) == front as i8
            }) {
                fit_labels[k] = -1;
                occluded_fit_samples += 1;
            }
        }

        let mut hypotheses = 0;
        let mut sets = Vec::new();
        let mut banks = Vec::new();
        for id in 0..ms.len() {
            let (m, set, bank) =
                bounds::subdivide(a, b, &fit_labels, id as i8, &cells[id], crop, scale);
            hypotheses += set.evaluated;
            ms[id] = m;
            cells[id] = set.cells.clone();
            sets.push(set);
            banks.push(bank);
        }
        let mut ordering = order_evidence(a, b, &labels, &ms, &banks);
        for e in &mut ordering {
            if sets[e.a].fit_samples < 8 || sets[e.b].fit_samples < 8 {
                e.front = None;
            }
        }
        let order = choose_order(ms.len(), &ordering);
        let ordering_conflict = ordering.iter().any(|e| {
            e.front.is_some_and(|f| {
                let back = if f == e.a { e.b } else { e.a };
                order.iter().position(|&x| x == f) > order.iter().position(|&x| x == back)
            })
        });
        if ordering_conflict {
            for e in &mut ordering {
                e.front = None;
            }
        }
        let allowed_orders = permutations(ms.len())
            .into_iter()
            .filter(|p| {
                ordering.iter().all(|e| {
                    e.front.is_none_or(|f| {
                        let back = if f == e.a { e.b } else { e.a };
                        p.iter().position(|&x| x == f) < p.iter().position(|&x| x == back)
                    })
                })
            })
            .collect();
        edges = ordering
            .iter()
            .filter_map(|e| e.front.map(|f| (f, if f == e.a { e.b } else { e.a })))
            .collect();
        let mut next = spatial_labels(&matches, &ms, a.w(), a.h(), scale);
        // Retain a source-mask hypothesis behind an evidenced occluder;
        // do not count it as a fresh patch correspondence.
        let occlusion = assign(a, b, &ms, &labels, &edges);
        for k in 0..next.len() {
            if next[k] < 0 && labels[k] >= 0 && occlusion[k] == labels[k] {
                let id = labels[k] as usize;
                let p = [(k % a.w()) as f64, (k / a.w()) as f64];
                let q = ms[id].map(p);
                if edges.iter().any(|&(front, back)| {
                    back == id
                        && label_at(&labels, a.w(), a.h(), ms[front].inverse(q)) == front as i8
                }) {
                    next[k] = labels[k];
                }
            }
        }
        let changed = next.iter().zip(&labels).filter(|(a, b)| a != b).count();
        let (fit_loss, held_loss, coverage, mask_coverage) = aggregate(a, b, &labels, &ms);
        let overlapping = ordering
            .iter()
            .map(|e| e.fit_samples + e.held_samples)
            .sum();
        history.push(Iteration {
            motions: ms.clone(),
            labels: labels.clone(),
            order,
            allowed_orders,
            ordering,
            changed,
            fit_loss,
            held_loss,
            coverage,
            mask_coverage,
            overlapping,
            origin_hypotheses: hypotheses,
            bounds: sets,
            ordering_conflict,
            occluded_fit_samples,
        });
        labels = next;
    }
    Discovery {
        matches,
        iterations: history,
        baseline_loss,
        baseline_coverage,
        translation_loss,
        translation_coverage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pivot_gauge_and_inverse_do_not_change_projected_motion() {
        let m = Motion {
            pivot: [17., -8.],
            translation: [2., -3.],
            angle: 0.13,
            crop: [4., 2.],
        };
        let n = m.repivot([-20., 50.]);
        for p in [[0., 0.], [30., 20.], [52., 34.]] {
            assert!(distance(m.map(p), n.map(p)) < 1e-10);
            assert!(distance(m.inverse(m.map(p)), p) < 1e-10);
        }
    }
    #[test]
    fn relative_rgb_patch_cost_ignores_independent_channel_gain_and_offset() {
        let make = |shift: P, gains: [f64; 3], offset: [f64; 3]| Picture {
            channels: std::array::from_fn(|c| Image {
                w: 53,
                h: 35,
                v: (0..53 * 35)
                    .map(|k| {
                        let x = (k % 53) as f64 - shift[0];
                        let y = (k / 53) as f64 - shift[1];
                        (0.4 + 0.1 * (x * 0.31 + y * 0.17 + c as f64).sin()
                            + 0.05 * (y * 0.53 - x * 0.07).cos())
                            * gains[c]
                            + offset[c]
                    })
                    .collect(),
            }),
        };
        let a = make([0., 0.], [1.; 3], [0.; 3]);
        let b = make([2., -1.], [0.7, 1.4, 1.1], [0.1, -0.03, 0.02]);
        let m = Motion {
            translation: [2., -1.],
            ..Motion::identity([0., 0.], 53, 35)
        };
        assert!(cost(&a, &b, [20., 16.], m).unwrap() < 1e-9);
        assert!(cost(&a, &b, [20., 16.], Motion::identity([0., 0.], 53, 35)).unwrap() > 0.01);
    }
    #[test]
    fn inferred_region_is_not_counted_as_fresh_image_support() {
        let a = Picture {
            channels: std::array::from_fn(|_| Image {
                w: 53,
                h: 35,
                v: (0..53 * 35)
                    .map(|k| 0.5 + 0.2 * ((k % 53) as f64 * 0.5 + (k / 53) as f64 * 0.3).sin())
                    .collect(),
            }),
        };
        let b = Picture {
            channels: a.channels.clone().map(|mut c| {
                for v in &mut c.v {
                    *v = 1. - *v;
                }
                c
            }),
        };
        let (_, _, fresh, mask) = aggregate(
            &a,
            &b,
            &vec![0; 53 * 35],
            &[Motion::identity([0., 0.], 53, 35)],
        );
        assert_eq!(fresh, 0.);
        assert_eq!(mask, 1.);
    }
    #[test]
    fn blank_scene_does_not_invent_groups_or_depth() {
        let a = Picture {
            channels: std::array::from_fn(|_| Image {
                w: 53,
                h: 35,
                v: vec![0.4; 53 * 35],
            }),
        };
        let d = discover(&a, &a, [0., 0.], 2);
        let r = d.iterations.last().unwrap();
        assert!(d.matches.is_empty());
        assert_eq!(r.bounds[0].fit_samples, 0);
        assert!(
            (r.bounds[0].angle_range[1] - r.bounds[0].angle_range[0] - 6f64.to_radians()).abs()
                < 1e-12
        );
        assert_eq!(r.coverage, 0.);
        assert!(r.ordering.is_empty());
        assert!(r.labels.iter().all(|x| *x < 0));
    }
}
