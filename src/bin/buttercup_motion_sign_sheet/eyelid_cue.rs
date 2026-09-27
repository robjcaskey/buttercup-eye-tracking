//! Offline cue audit, not an anatomical eyelid detector or live sign authority.
//! Compare the observed semantic boundary with the unchanged fitted limbus.
use super::Ellipse;
use serde_json::{json, Value};
type P = [f64; 2];
fn median(mut x: Vec<f64>) -> Option<f64> {
    if x.is_empty() {
        return None;
    }
    x.sort_by(f64::total_cmp);
    Some(x[x.len() / 2])
}
pub fn vertical_edges(e: Ellipse, x: f64) -> Option<P> {
    let (s, c) = e.angle.sin_cos();
    let a = c * c / e.major_radius.powi(2) + s * s / e.minor_radius.powi(2);
    let b = c * s * (1. / e.major_radius.powi(2) - 1. / e.minor_radius.powi(2));
    let d = s * s / e.major_radius.powi(2) + c * c / e.minor_radius.powi(2);
    let x = x - e.center.0;
    let disc = (b * x).powi(2) - d * (a * x * x - 1.);
    (disc >= 0.).then(|| {
        [
            e.center.1 + (-b * x - disc.sqrt()) / d,
            e.center.1 + (-b * x + disc.sqrt()) / d,
        ]
    })
}
fn rgb(v: u32) -> [f64; 3] {
    [
        (v >> 16 & 255) as f64,
        (v >> 8 & 255) as f64,
        (v & 255) as f64,
    ]
}
fn luma(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}
fn chroma(c: [f64; 3]) -> [f64; 3] {
    let n = c.iter().sum::<f64>().max(1.);
    c.map(|x| x / n)
}
fn color_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}
/// Image-y sign: +1 means down in the image, not world-down or screen gaze.
pub fn measure(e: Ellipse, boundary: &[P], pixels: &[u32], w: usize, h: usize, origin: P) -> Value {
    let half = e.major_radius * 0.45;
    let mut upper = vec![];
    let mut lower = vec![];
    let mut traces = vec![];
    for i in 0..13 {
        let x = e.center.0 - half + 2. * half * i as f64 / 12.;
        let Some(edge) = vertical_edges(e, x) else {
            continue;
        };
        let ys = boundary
            .iter()
            .filter(|p| (p[0] - x).abs() < 3.5)
            .map(|p| p[1])
            .collect::<Vec<_>>();
        let Some(lo) = ys.iter().copied().reduce(f64::min) else {
            continue;
        };
        let hi = ys.iter().copied().reduce(f64::max).unwrap();
        // Both sides must be sampled; a missing arc cannot masquerade as a lid.
        if lo > e.center.1 || hi < e.center.1 || ys.len() < 2 {
            continue;
        }
        upper.push(lo - edge[0]);
        lower.push(edge[1] - hi);
        traces.push(json!({"x":x,"observed":[lo,hi],"fitted":edge}));
    }
    let u = median(upper.clone());
    let l = median(lower.clone());
    let mut status = "insufficient current mask-boundary support";
    let mut direction = None;
    if let (Some(u), Some(l)) = (u, l) {
        if traces.len() >= 7 {
            status = "overlap is balanced or too small";
            if u < -4. || l < -4. {
                status = "boundary and fitted ellipse disagree";
            } else if u.max(0.) + l.max(0.) > e.minor_radius * 0.55 {
                status = "both sides restricted; possible blink or bad mask";
            } else if (l - u).abs() > 4. && l.max(u) > 5. {
                direction = Some(if l > u { 1 } else { -1 });
                status = if l > u {
                    "lower mask overlap suggests image-down"
                } else {
                    "upper mask overlap suggests image-up"
                };
            }
        }
    }
    // Current-image color references: lateral sclera-like brightness vs nearby
    // skin. This is deliberately a displayed heuristic, not semantic sclera truth.
    let sample = |x: f64, y: f64| -> Option<[f64; 3]> {
        let x = (x - origin[0]).round() as i32;
        let y = (y - origin[1]).round() as i32;
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| rgb(pixels[y as usize * w + x as usize]))
    };
    let mut lateral = vec![];
    let mut skin = vec![];
    for side in [-1., 1.] {
        for i in 0..12 {
            for j in -5..=5 {
                if let Some(c) = sample(
                    e.center.0 + side * e.major_radius * (1.05 + 0.4 * i as f64 / 11.),
                    e.center.1 + e.minor_radius * 0.05 * j as f64,
                ) {
                    lateral.push(c);
                }
                if let Some(c) = sample(
                    e.center.0 + e.major_radius * 0.06 * j as f64,
                    e.center.1 + side * e.minor_radius * (1.55 + 0.3 * i as f64 / 11.),
                ) {
                    skin.push(c);
                }
            }
        }
    }
    lateral.sort_by(|a, b| luma(*b).total_cmp(&luma(*a)));
    let reference = |v: &[[f64; 3]]| -> Option<[f64; 3]> {
        if v.len() < 24 {
            return None;
        }
        Some(std::array::from_fn(|i| {
            median(v.iter().map(|c| c[i]).collect()).unwrap()
        }))
    };
    let bright = reference(&lateral[..(lateral.len() / 3).max(1).min(lateral.len())]);
    let skin = reference(&skin);
    let mut white =
        json!({"status":"color references unavailable or inseparable","direction":null});
    if let (Some(sc), Some(sk)) = (bright, skin) {
        let separation = color_distance(chroma(sc), chroma(sk));
        if separation > 0.045 && luma(sc) > luma(sk) * 1.05 {
            let mut gaps = [vec![], vec![]];
            let mut white_points = vec![];
            for i in 0..11 {
                let x = e.center.0 - e.major_radius * 0.4 + e.major_radius * 0.8 * i as f64 / 10.;
                let Some(edges) = vertical_edges(e, x) else {
                    continue;
                };
                for k in 0..2 {
                    let sign = if k == 0 { -1. } else { 1. };
                    let mut length = 0.;
                    let mut misses = 0;
                    for step in 2..=(e.minor_radius * 0.6).min(60.) as usize {
                        let y = edges[k] + sign * step as f64;
                        let Some(c) = sample(x, y) else { break };
                        let a = color_distance(chroma(c), chroma(sc));
                        let b = color_distance(chroma(c), chroma(sk));
                        let matches = luma(c) > 0.6 * luma(sc) && a + 0.015 < b && a < 0.14;
                        if matches {
                            length = step as f64;
                            misses = 0;
                            white_points.push([x, y]);
                        } else {
                            misses += 1;
                        }
                        if misses >= 2 {
                            break;
                        }
                    }
                    gaps[k].push(length);
                }
            }
            let top = median(gaps[0].clone());
            let bottom = median(gaps[1].clone());
            let d = top.zip(bottom).and_then(|(a, b)| {
                ((a - b).abs() > 4. && a.max(b) > 5.).then_some(if a > b { 1 } else { -1 })
            });
            white = json!({"status":"current-image color-gap proxy","upper_px":top,"lower_px":bottom,"direction":d,"points":white_points,"reference_separation":separation});
            if direction.is_none()
                && traces.len() >= 7
                && u.is_some_and(|x| (-4. ..4.).contains(&x))
                && l.is_some_and(|x| (-4. ..4.).contains(&x))
                && d.is_some()
            {
                direction = d;
                status = "sclera-color gap tiebreaker; overlap small";
            }
        }
    }
    json!({"status":status,"image_y_direction":direction,"upper_overlap_px":u,"lower_overlap_px":l,"supported_columns":traces.len(),"traces":traces,"white_gap":white,"contract":"Semantic-boundary overlap and current RAW color heuristic; neither proves an anatomical eyelid nor absolute sign. No held observation or candidate radius normalization."})
}
