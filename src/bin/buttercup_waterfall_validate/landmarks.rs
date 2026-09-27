//! Two-phase dot extraction and a deterministic third-phase Bezier illustration.
use super::*;
fn chroma(p: [f64; 3], blue: bool) -> f64 {
    if blue {
        (p[2] - p[0]).min(p[2] - p[1])
    } else {
        (p[0] - p[1]).min(p[2] - p[1])
    }
}
fn blobs(weights: &[f64], w: usize, h: usize) -> Vec<Value> {
    let mut seen = vec![false; w * h];
    let mut result = Vec::new();
    for seed in 0..weights.len() {
        if seen[seed] || weights[seed] <= 0. {
            continue;
        }
        let mut queue = vec![seed];
        seen[seed] = true;
        let (mut head, mut xmin, mut xmax, mut ymin, mut ymax) = (0, w, 0, h, 0);
        let (mut sum, mut sx, mut sy) = (0., 0., 0.);
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            let (x, y) = (i % w, i / w);
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
            sum += weights[i];
            sx += weights[i] * x as f64;
            sy += weights[i] * y as f64;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = (x as isize + dx, y as isize + dy);
                    if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                        let j = yy as usize * w + xx as usize;
                        if !seen[j] && weights[j] > 0. {
                            seen[j] = true;
                            queue.push(j);
                        }
                    }
                }
            }
        }
        let (bw, bh) = (xmax - xmin + 1, ymax - ymin + 1);
        if queue.len() >= 5
            && queue.len() <= 500
            && bw.max(bh) <= 22
            && bw.min(bh) >= 2
            && bw.max(bh) as f64 / bw.min(bh) as f64 <= 2.5
            && queue.len() as f64 / (bw * bh) as f64 >= 0.3
        {
            result.push(
                json!({"xy":[sx/sum,sy/sum],"pixels":queue.len(),"bounds":[xmin,ymin,bw,bh]}),
            );
        }
    }
    result
}
fn xy(v: &Value) -> [f64; 2] {
    [v["xy"][0].as_f64().unwrap(), v["xy"][1].as_f64().unwrap()]
}
fn dot(
    rgb: &mut [u8],
    mask: &mut [u8],
    sw: usize,
    rect: [usize; 4],
    p: [f64; 2],
    color: [u8; 3],
    radius: f64,
) {
    let [x, y, w, h] = rect;
    for yy in 0..h {
        for xx in 0..w {
            let alpha = (radius + 0.5 - (xx as f64 - p[0]).hypot(yy as f64 - p[1])).clamp(0., 1.);
            if alpha == 0. {
                continue;
            }
            let i = ((y + yy) * sw + x + xx) * 3;
            for c in 0..3 {
                rgb[i + c] =
                    ((1. - alpha) * rgb[i + c] as f64 + alpha * color[c] as f64).round() as u8;
                mask[i + c] = 255;
            }
        }
    }
}
pub(crate) fn run(args: &[String]) -> Res<bool> {
    if args.len() != 6 && args.len() != 7 {
        return Err("usage: buttercup-waterfall-validate --landmarks ORIGINAL.png GENERATED.png PREPARATION.json NEW_OUT [PHASE1_POINTS.json]".into());
    }
    let out = Path::new(&args[5]);
    if out.exists() {
        return Err("output must be new".into());
    }
    if !fs::canonicalize(out.parent().ok_or("missing output parent")?)?
        .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("output must be beneath outputs".into());
    }
    let receipt: Value = serde_json::from_slice(&fs::read(&args[4])?)?;
    let op = Path::new(&args[2]);
    if receipt["sheet"]["sha256"].as_str() != Some(&hash(&fs::read(op)?)) {
        return Err("original hash mismatch".into());
    }
    let (w, h, a) = read(op)?;
    let (ew, eh, b) = read(Path::new(&args[3]))?;
    let prior: Option<Value> = args
        .get(6)
        .map(|p| {
            fs::read(p).and_then(|b| serde_json::from_slice(&b).map_err(std::io::Error::other))
        })
        .transpose()?;
    if let Some(p) = &prior {
        if p["original_sha256"].as_str() != Some(&hash(&fs::read(op)?)) {
            return Err("phase1 original mismatch".into());
        }
    }
    let mut points_image = a.clone();
    let mut point_mask = vec![0; w * h * 3];
    let mut pink_curve = a.clone();
    let mut black_curve = a.clone();
    let mut curve_mask = point_mask.clone();
    let mut records = Vec::new();
    for row in receipt["frames"].as_array().ok_or("missing frames")? {
        let r = row["tile_xywh"].as_array().ok_or("missing rectangle")?;
        if r.len() != 4 {
            return Err("invalid rectangle".into());
        }
        let mut rect = [0; 4];
        for (i, v) in r.iter().enumerate() {
            rect[i] = v.as_u64().ok_or("invalid rectangle number")? as usize;
        }
        let [x, y, tw, th] = rect;
        if tw < 24
            || th < 24
            || x.checked_add(tw).is_none_or(|v| v > w)
            || y.checked_add(th).is_none_or(|v| v > h)
        {
            return Err("tile out of bounds".into());
        }
        let reg = Registration {
            a: &a,
            b: &b,
            w,
            h,
            ew,
            eh,
            x,
            y,
            tw,
            th,
        };
        let (t, baseline, score) = reg.align();
        let heldout = reg.score(t, 1);
        let aligned = score >= 0.8
            && heldout >= 0.8
            && t.dx.abs() < 8.99
            && t.dy.abs() < 8.99
            && (t.scale - 1.).abs() < 0.0499;
        let mut pink_weights = vec![0.; tw * th];
        let mut blue_weights = pink_weights.clone();
        if aligned {
            for yy in 0..th {
                for xx in 0..tw {
                    let (gx, gy) = reg.mapped(xx as f64, yy as f64, t);
                    let left = x as f64 * ew as f64 / w as f64;
                    let right = (x + tw) as f64 * ew as f64 / w as f64;
                    let top = y as f64 * eh as f64 / h as f64;
                    let bottom = (y + th) as f64 * eh as f64 / h as f64;
                    if gx < left || gx > right - 1. || gy < top || gy > bottom - 1. {
                        continue;
                    }
                    if let Some(bb) = sample(&b, ew, eh, gx, gy) {
                        let i = ((y + yy) * w + x + xx) * 3;
                        let aa = [a[i] as f64, a[i + 1] as f64, a[i + 2] as f64];
                        for blue in [false, true] {
                            let strength = chroma(bb, blue);
                            let gain = strength - chroma(aa, blue);
                            let bright = if blue {
                                bb[2] > 170. && bb[0] < 120.
                            } else {
                                bb[0] > 175. && bb[2] > 150.
                            };
                            if strength > 70. && gain > 45. && bright {
                                if blue {
                                    blue_weights[yy * tw + xx] = gain;
                                } else {
                                    pink_weights[yy * tw + xx] = gain;
                                }
                            }
                        }
                    }
                }
            }
        }
        let detected_pink = blobs(&pink_weights, tw, th);
        let detected_blue = blobs(&blue_weights, tw, th);
        let mut corners = Vec::new();
        let mut diagnostics = Vec::new();
        if let Some(p) = &prior {
            if let Some(prev) = p["frames"]
                .as_array()
                .and_then(|rs| rs.iter().find(|v| v["tile"] == row["tile"]))
            {
                corners = prev["corners"].as_array().cloned().unwrap_or_default();
            }
        }
        let max_corners = if prior.is_some() { 2 } else { 1 };
        for p in &detected_pink {
            let c = xy(p);
            if corners
                .iter()
                .any(|q| (xy(q)[0] - c[0]).hypot(xy(q)[1] - c[1]) < 8.)
            {
                continue;
            }
            corners.push(p.clone());
        }
        if corners.len() > max_corners {
            diagnostics.push("too_many_pink_candidates");
            corners.clear();
        }
        let apex = if prior.is_some() && detected_blue.len() == 1 {
            Some(xy(&detected_blue[0]))
        } else {
            None
        };
        if prior.is_some() && apex.is_none() {
            diagnostics.push("missing_or_ambiguous_blue_apex");
        }
        if !aligned {
            diagnostics.push("alignment_rejected");
            corners.clear();
        }
        for p in &corners {
            dot(
                &mut points_image,
                &mut point_mask,
                w,
                rect,
                xy(p),
                [255, 0, 255],
                3.,
            );
        }
        if let Some(p) = apex {
            dot(
                &mut points_image,
                &mut point_mask,
                w,
                rect,
                p,
                [0, 90, 255],
                3.,
            );
        }
        let mut curve = Value::Null;
        let mut status = if prior.is_none() {
            "phase1_marks_only"
        } else {
            "insufficient_landmarks"
        };
        if aligned && corners.len() == 2 {
            if let Some(apex) = apex {
                let mut p0 = xy(&corners[0]);
                let mut p2 = xy(&corners[1]);
                if p0[0] > p2[0] {
                    std::mem::swap(&mut p0, &mut p2);
                }
                let chord = [p2[0] - p0[0], p2[1] - p0[1]];
                let length = chord[0].hypot(chord[1]);
                let along =
                    ((apex[0] - p0[0]) * chord[0] + (apex[1] - p0[1]) * chord[1]) / length.powi(2);
                let sag =
                    ((apex[0] - p0[0]) * chord[1] - (apex[1] - p0[1]) * chord[0]).abs() / length;
                if length > 20. && (0.05..0.95).contains(&along) && sag < length * 0.5 {
                    let control = [
                        2. * apex[0] - 0.5 * (p0[0] + p2[0]),
                        2. * apex[1] - 0.5 * (p0[1] + p2[1]),
                    ];
                    let mut last = p0;
                    for n in 1..=400 {
                        let t = n as f64 / 400.;
                        let p = std::array::from_fn::<_, 2, _>(|c| {
                            (1. - t).powi(2) * p0[c]
                                + 2. * (1. - t) * t * control[c]
                                + t * t * p2[c]
                        });
                        let dist = (p[0] - last[0]).hypot(p[1] - last[1]);
                        let steps = (dist * 2.).ceil().max(1.) as usize;
                        for k in 0..=steps {
                            let q = std::array::from_fn::<_, 2, _>(|c| {
                                last[c] + (p[c] - last[c]) * k as f64 / steps as f64
                            });
                            draw_point(
                                &mut pink_curve,
                                &mut black_curve,
                                &mut curve_mask,
                                w,
                                rect,
                                q,
                            );
                        }
                        last = p;
                    }
                    curve = json!({"type":"quadratic_bezier","start":p0,"control":control,"end":p2,"apex_at_t_half":apex,"note":"Interpolates both marked corners and apex exactly; predicted anatomy remains unverified."});
                    status = "curve_fitted";
                } else {
                    diagnostics.push("inconsistent_corner_apex_geometry");
                }
            }
        }
        records.push(json!({"tile":row["tile"],"source":row["source"],"tile_xywh":rect,"corners":corners,"apex_xy":apex,"detected_pink_candidates":detected_pink,"detected_blue_candidates":detected_blue,"curve":curve,"status":status,"diagnostics":diagnostics,"alignment":{"scale":t.scale,"dx":t.dx,"dy":t.dy,"baseline":baseline,"fit":score,"heldout":heldout},"coordinates":"original ROI pixels; float dot centroid is not anatomical subpixel accuracy"}));
    }
    fs::create_dir(out)?;
    png::save(&out.join("raw-with-landmarks.png"), w, h, &points_image);
    png::save(&out.join("landmark-mask.png"), w, h, &point_mask);
    png::save(&out.join("curve-pink.png"), w, h, &pink_curve);
    png::save(&out.join("curve-black.png"), w, h, &black_curve);
    png::save(&out.join("curve-mask.png"), w, h, &curve_mask);
    fs::copy(op, out.join("original.png"))?;
    if let Some(previous) = args.get(6) {
        fs::copy(
            Path::new(previous)
                .parent()
                .ok_or("missing prior directory")?
                .join("raw-with-landmarks.png"),
            out.join("phase-1.png"),
        )?;
    } else {
        fs::copy(out.join("raw-with-landmarks.png"), out.join("phase-1.png"))?;
    }
    for i in 0..w * h {
        if point_mask[i * 3] == 0 && points_image[i * 3..i * 3 + 3] != a[i * 3..i * 3 + 3] {
            return Err("landmark preservation failed".into());
        }
        if curve_mask[i * 3] == 0
            && (pink_curve[i * 3..i * 3 + 3] != a[i * 3..i * 3 + 3]
                || black_curve[i * 3..i * 3 + 3] != a[i * 3..i * 3 + 3])
        {
            return Err("curve preservation failed".into());
        }
    }
    let report = json!({"schema":"buttercup-three-phase-landmarks-v1","original_sha256":hash(&fs::read(op)?),"generated_sha256":hash(&fs::read(&args[3])?),"generated_dimensions":[ew,eh],"dimensions":[w,h],"phase":if prior.is_some(){2}else{1},"outside_masks_changed_pixels":0,"frames":records});
    fs::write(
        out.join("landmarks.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let mut html = String::from(
        r#"<!doctype html><html lang="en"><meta charset="utf-8"><title>Three-phase eye meridian</title>
<style>body{margin:0;background:#181818;color:#eee;font:16px system-ui}header{position:sticky;top:0;padding:12px;background:#222;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}button,select{font:inherit;padding:6px}#label{min-width:13em}#view{display:block;margin:auto;width:min(100%,1536px)}#view.native{width:auto;max-width:none}table{border-collapse:collapse;margin:16px}td,th{text-align:left;padding:6px 12px;border-bottom:1px solid #444}</style>
<header><select id="phase"><option value="3">3 · Fitted curve</option><option value="2">2 · Corners + predicted apex</option><option value="1">1 · First corner</option><option value="0">Original</option></select><button id="pause">Pause</button><button id="size">Native size</button><span id="label">Loading…</span></header>
<img id="view" alt="Three-phase meridian comparison">
<p style="margin:16px">Pink dots: model-proposed corners. Blue dot: predicted apex. Curves require two extracted corner dots and a consistent apex; missing inputs are listed below. These are illustrations, not validated anatomy.</p>
<table><thead><tr><th>Tile</th><th>Corners</th><th>Apex</th><th>Curve</th></tr></thead><tbody>"#,
    );
    for row in &records {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            row["tile"],
            row["corners"].as_array().unwrap().len(),
            if row["apex_xy"].is_null() {
                "Missing"
            } else {
                "Predicted"
            },
            row["status"].as_str().unwrap()
        ));
    }
    html.push_str(r#"</tbody></table><script>
const files=['original.png','phase-1.png','raw-with-landmarks.png','curve-pink.png','curve-black.png'],images=files.map(src=>{const i=new Image();i.src=src;return i;});const view=document.getElementById('view'),phase=document.getElementById('phase'),pause=document.getElementById('pause'),label=document.getElementById('label');let black=false,playing=true,ready=false,timer;
function render(){const p=Number(phase.value),index=p===3?(black?4:3):p;view.src=files[index];label.textContent=p===3?(black?'Curve · black':'Curve · pink'):['Original','First corner','Corners + apex'][p];}
function schedule(){clearInterval(timer);if(ready&&playing&&phase.value==='3'&&!document.hidden)timer=setInterval(()=>{black=!black;render();},1000);}
phase.onchange=()=>{render();schedule();};pause.onclick=()=>{playing=!playing;pause.textContent=playing?'Pause':'Resume';schedule();};document.getElementById('size').onclick=e=>{const n=view.classList.toggle('native');e.target.textContent=n?'Fit to window':'Native size';};document.addEventListener('visibilitychange',schedule);
Promise.all(images.map(i=>i.decode())).then(()=>{ready=true;render();schedule();}).catch(e=>label.textContent=e.message);
</script></html>"#);
    fs::write(out.join("three-phase.html"), html)?;
    println!(
        "{} tiles: {} with corners, {} apex marks, {} fitted curves",
        records.len(),
        records
            .iter()
            .filter(|r| !r["corners"].as_array().unwrap().is_empty())
            .count(),
        records.iter().filter(|r| !r["apex_xy"].is_null()).count(),
        records
            .iter()
            .filter(|r| r["status"] == "curve_fitted")
            .count()
    );
    Ok(true)
}
fn draw_point(
    pink: &mut [u8],
    black: &mut [u8],
    mask: &mut [u8],
    sw: usize,
    rect: [usize; 4],
    p: [f64; 2],
) {
    let [x, y, w, h] = rect;
    let (cx, cy) = (p[0].round() as isize, p[1].round() as isize);
    for dy in -2..=2 {
        for dx in -2..=2 {
            let (xx, yy) = (cx + dx, cy + dy);
            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                continue;
            }
            let a = (1.5 - (xx as f64 - p[0]).hypot(yy as f64 - p[1])).clamp(0., 1.);
            if a == 0. {
                continue;
            }
            let i = ((y + yy as usize) * sw + x + xx as usize) * 3;
            let col = [255., 90., 210.];
            for c in 0..3 {
                pink[i + c] = ((1. - a) * pink[i + c] as f64 + a * col[c]).round() as u8;
                black[i + c] = ((1. - a) * black[i + c] as f64).round() as u8;
                mask[i + c] = 255;
            }
        }
    }
}
