//! Extract illustrative pink strokes; move only the strokes, never the RAW preview.
use super::*;
#[path = "landmarks.rs"]
pub(super) mod landmarks;
#[derive(Clone, Copy)]
struct Transform {
    scale: f64,
    dx: f64,
    dy: f64,
}
struct Registration<'a> {
    a: &'a [u8],
    b: &'a [u8],
    w: usize,
    h: usize,
    ew: usize,
    eh: usize,
    x: usize,
    y: usize,
    tw: usize,
    th: usize,
}
fn sample(rgb: &[u8], w: usize, h: usize, x: f64, y: f64) -> Option<[f64; 3]> {
    if x < 0. || y < 0. || x > (w - 1) as f64 || y > (h - 1) as f64 {
        return None;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    Some(std::array::from_fn(|c| {
        let at = |xx, yy| rgb[(yy * w + xx) * 3 + c] as f64;
        (1. - fy) * ((1. - fx) * at(x0, y0) + fx * at(x1, y0))
            + fy * ((1. - fx) * at(x0, y1) + fx * at(x1, y1))
    }))
}
fn excess(p: [f64; 3]) -> f64 {
    (p[0] - p[1]).min(p[2] - p[1])
}
fn luma(p: [f64; 3]) -> f64 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}
impl Registration<'_> {
    fn mapped(&self, x: f64, y: f64, t: Transform) -> (f64, f64) {
        let (cx, cy) = ((self.tw - 1) as f64 / 2., (self.th - 1) as f64 / 2.);
        let xx = self.x as f64 + cx + (x - cx) * t.scale + t.dx;
        let yy = self.y as f64 + cy + (y - cy) * t.scale + t.dy;
        (
            (xx + 0.5) * self.ew as f64 / self.w as f64 - 0.5,
            (yy + 0.5) * self.eh as f64 / self.h as f64 - 0.5,
        )
    }
    fn pairs(&self, t: Transform, step: usize, parity: usize) -> Vec<(f64, f64)> {
        let mut pairs = Vec::new();
        for y in (8..self.th - 8).step_by(step) {
            for x in (8..self.tw - 8).step_by(step) {
                if (x / step + y / step) % 2 != parity {
                    continue;
                }
                let i = ((self.y + y) * self.w + self.x + x) * 3;
                let aa = [self.a[i] as f64, self.a[i + 1] as f64, self.a[i + 2] as f64];
                // Outside central well: identity evidence is neighboring texture, not the painted shaft.
                let nx = (x as f64 - self.tw as f64 / 2.) / (self.tw as f64 / 2.);
                let ny = (y as f64 - self.th as f64 / 2.) / (self.th as f64 / 2.);
                if nx * nx + ny * ny < 0.16 || excess(aa) > 20. {
                    continue;
                }
                let (gx, gy) = self.mapped(x as f64, y as f64, t);
                if let Some(bb) = sample(self.b, self.ew, self.eh, gx, gy) {
                    if excess(bb) > 20. {
                        continue;
                    }
                    pairs.push((luma(aa), luma(bb)));
                }
            }
        }
        pairs
    }
    fn score(&self, t: Transform, parity: usize) -> f64 {
        let pairs = self.pairs(t, 6, parity);
        if pairs.len() < 200 {
            return -1.;
        }
        let n = pairs.len() as f64;
        let (ma, mb) = (
            pairs.iter().map(|p| p.0).sum::<f64>() / n,
            pairs.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let (mut aa, mut bb, mut ab) = (0., 0., 0.);
        for (a, b) in pairs {
            aa += (a - ma).powi(2);
            bb += (b - mb).powi(2);
            ab += (a - ma) * (b - mb);
        }
        ab / (aa * bb).sqrt().max(1e-12)
    }
    fn align(&self) -> (Transform, f64, f64) {
        let mut best = Transform {
            scale: 1.,
            dx: 0.,
            dy: 0.,
        };
        let baseline = self.score(best, 0);
        let mut score = baseline;
        for s in -2..=2 {
            for dy in -4..=4 {
                for dx in -4..=4 {
                    let t = Transform {
                        scale: 1. + s as f64 * 0.02,
                        dx: dx as f64 * 2.,
                        dy: dy as f64 * 2.,
                    };
                    let q = self.score(t, 0);
                    if q > score {
                        score = q;
                        best = t;
                    }
                }
            }
        }
        let coarse = best;
        for s in -2..=2 {
            for dy in -4..=4 {
                for dx in -4..=4 {
                    let t = Transform {
                        scale: coarse.scale + s as f64 * 0.005,
                        dx: coarse.dx + dx as f64 * 0.25,
                        dy: coarse.dy + dy as f64 * 0.25,
                    };
                    let q = self.score(t, 0);
                    if q > score {
                        score = q;
                        best = t;
                    }
                }
            }
        }
        (best, baseline, score)
    }
}
fn retain_strokes(alpha: &mut [f64], w: usize, h: usize) -> usize {
    let mut seen = vec![false; alpha.len()];
    let mut removed = 0;
    for seed in 0..alpha.len() {
        if seen[seed] || alpha[seed] == 0. {
            continue;
        }
        let mut queue = vec![seed];
        seen[seed] = true;
        let (mut minx, mut maxx, mut miny, mut maxy) = (w, 0, h, 0);
        let mut head = 0;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            let (x, y) = (i % w, i / w);
            minx = minx.min(x);
            maxx = maxx.max(x);
            miny = miny.min(y);
            maxy = maxy.max(y);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = (x as isize + dx, y as isize + dy);
                    if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                        let j = yy as usize * w + xx as usize;
                        if !seen[j] && alpha[j] > 0. {
                            seen[j] = true;
                            queue.push(j);
                        }
                    }
                }
            }
        }
        // Reject isolated color flecks; retain connected line fragments, not broad pink glare.
        let long = (maxx - minx + 1).max(maxy - miny + 1);
        let area = (maxx - minx + 1) * (maxy - miny + 1);
        if queue.len() < 6
            || long < 6
            || queue.len() as f64 / area as f64 > 0.75 && queue.len() > 40
        {
            for i in queue {
                alpha[i] = 0.;
                removed += 1;
            }
        }
    }
    removed
}
pub(super) fn run(args: &[String]) -> Res<bool> {
    if args.len() != 6 {
        return Err("usage: buttercup-waterfall-validate --composite ORIGINAL.png GENERATED.png PREPARATION_PROVENANCE.json NEW_OUTPUT_DIR".into());
    }
    let out = Path::new(&args[5]);
    if out.exists() {
        return Err("output must be new".into());
    }
    if !fs::canonicalize(out.parent().ok_or("missing parent")?)?
        .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("output must be beneath checked outputs".into());
    }
    let original = Path::new(&args[2]);
    let generated = Path::new(&args[3]);
    let receipt: Value = serde_json::from_slice(&fs::read(&args[4])?)?;
    if receipt["sheet"]["sha256"].as_str() != Some(&hash(&fs::read(original)?)) {
        return Err("original PNG hash does not match preparation receipt".into());
    }
    let (w, h, a) = read(original)?;
    let (ew, eh, b) = read(generated)?;
    let mut blended = a.clone();
    let mut mask = vec![0u8; w * h * 3];
    let mut alpha_image = mask.clone();
    let mut aligned_preview = a.clone();
    let mut occupied = vec![false; w * h];
    let mut records = Vec::new();
    let frames = receipt["frames"].as_array().ok_or("missing frames")?;
    for row in frames {
        let rect = row["tile_xywh"]
            .as_array()
            .ok_or("missing tile rectangle")?;
        if rect.len() != 4 {
            return Err("invalid tile rectangle".into());
        }
        let mut r = [0usize; 4];
        for (i, v) in rect.iter().enumerate() {
            r[i] = v.as_u64().ok_or("invalid rectangle number")? as usize;
        }
        let [x, y, tw, th] = r;
        if tw < 24
            || th < 24
            || x.checked_add(tw).is_none_or(|v| v > w)
            || y.checked_add(th).is_none_or(|v| v > h)
        {
            return Err("tile out of bounds".into());
        }
        for yy in y..y + th {
            for xx in x..x + tw {
                let i = yy * w + xx;
                if occupied[i] {
                    return Err("overlapping tiles".into());
                }
                occupied[i] = true;
            }
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
        let baseheld = reg.score(
            Transform {
                scale: 1.,
                dx: 0.,
                dy: 0.,
            },
            1,
        );
        let boundary = t.dx.abs() >= 8.99 || t.dy.abs() >= 8.99 || (t.scale - 1.).abs() >= 0.0499;
        let accepted = score >= 0.8 && heldout >= 0.8 && heldout + 0.005 >= baseheld && !boundary;
        let mut alpha = vec![0.; tw * th];
        let mut mapped = vec![[0.; 3]; tw * th];
        if accepted {
            for yy in 0..th {
                for xx in 0..tw {
                    let (gx, gy) = reg.mapped(xx as f64, yy as f64, t);
                    // Never read neighboring tiles to invent missing edge evidence.
                    let left = x as f64 * ew as f64 / w as f64;
                    let right = (x + tw) as f64 * ew as f64 / w as f64;
                    let top = y as f64 * eh as f64 / h as f64;
                    let bottom = (y + th) as f64 * eh as f64 / h as f64;
                    if gx < left || gy < top || gx > right - 1. || gy > bottom - 1. {
                        continue;
                    }
                    if let Some(bb) = sample(&b, ew, eh, gx, gy) {
                        let i = ((y + yy) * w + x + xx) * 3;
                        let aa = [a[i] as f64, a[i + 1] as f64, a[i + 2] as f64];
                        mapped[yy * tw + xx] = bb;
                        let gain = excess(bb) - excess(aa);
                        if excess(bb) > 22. && gain > 15. && bb[0] > 110. {
                            alpha[yy * tw + xx] =
                                ((gain - 15.) / (110. - excess(aa)).max(30.)).clamp(0., 1.);
                        }
                    }
                }
            }
        }
        let removed = retain_strokes(&mut alpha, tw, th);
        let masked = alpha.iter().filter(|&&v| v > 0.).count();
        let accepted = accepted && masked > 0 && masked * 4 < tw * th;
        for yy in 0..th {
            for xx in 0..tw {
                let j = yy * tw + xx;
                let i = ((y + yy) * w + x + xx) * 3;
                if accepted {
                    if mapped[j] != [0.; 3] {
                        for c in 0..3 {
                            aligned_preview[i + c] = mapped[j][c].round().clamp(0., 255.) as u8;
                        }
                    }
                    let opacity = alpha[j];
                    if opacity > 0. {
                        let color = [255., 100., 210.];
                        for c in 0..3 {
                            blended[i + c] = ((1. - opacity) * a[i + c] as f64 + opacity * color[c])
                                .round() as u8;
                            mask[i + c] = 255;
                            alpha_image[i + c] = (opacity * 255.).round() as u8;
                        }
                    }
                }
            }
        }
        records.push(json!({"tile":row["tile"],"tile_xywh":r,"source":row["source"],"status":if accepted{"composited"}else{"rejected_alignment_or_mask"},"transform_original_to_generated":{"scale_about_tile_center":t.scale,"dx_native_px":t.dx,"dy_native_px":t.dy,"global_resize_xy":[ew as f64/w as f64,eh as f64/h as f64]},"baseline_correlation":baseline,"fit_correlation":score,"heldout_correlation":heldout,"heldout_baseline":baseheld,"search_boundary":boundary,"mask_pixels":if accepted{masked}else{0},"removed_color_flecks":removed}));
    }
    let mut outside = 0;
    let mut inside = 0;
    for i in 0..w * h {
        if a[i * 3..i * 3 + 3] != blended[i * 3..i * 3 + 3] {
            if mask[i * 3] == 0 {
                outside += 1;
            } else {
                inside += 1;
            }
        }
    }
    if outside != 0 {
        return Err("compositor invariant violated: original changed outside mask".into());
    }
    fs::create_dir(out)?;
    png::save(&out.join("raw-with-pink-strokes.png"), w, h, &blended);
    png::save(&out.join("stroke-mask.png"), w, h, &mask);
    png::save(&out.join("stroke-alpha.png"), w, h, &alpha_image);
    png::save(
        &out.join("aligned-generated-diagnostic.png"),
        w,
        h,
        &aligned_preview,
    );
    fs::copy(original, out.join("original.png"))?;
    fs::write(
        out.join("blink-comparison.html"),
        r#"<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Pink / black extracted strokes</title>
<style>body{margin:0;background:#181818;color:#eee;font:16px system-ui}header{padding:12px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}#label{width:18em}button{padding:7px;font:inherit}.stack{position:relative;isolation:isolate;margin:auto;width:min(100vw,1536px)}img{display:block;width:100%;height:auto}.layer{position:absolute;inset:0}#black{filter:invert(1);mix-blend-mode:multiply;visibility:hidden}.native{width:max-content!important}.native img{width:auto}</style>
<header><span id="label">Loading…</span><button id="pause" disabled>Pause</button><button id="size">Native size</button><span>Pink / black every second · Same original background</span></header>
<div class="stack" id="stack"><img id="base" src="original.png" alt="Unchanged original"><img id="pink" class="layer" src="raw-with-pink-strokes.png" alt="Extracted pink strokes on original"><img id="black" class="layer" src="stroke-alpha.png" alt="Same extracted strokes in black"></div>
<script>
const base=document.getElementById('base'),pink=document.getElementById('pink'),black=document.getElementById('black'),label=document.getElementById('label'),pause=document.getElementById('pause');let shownBlack=false,timer=null,playing=true,ready=false;
function render(){pink.style.visibility=shownBlack?'hidden':'visible';black.style.visibility=shownBlack?'visible':'hidden';label.textContent=shownBlack?'Original + black strokes':'Original + pink strokes';}
function schedule(){clearInterval(timer);if(ready&&playing&&!document.hidden)timer=setInterval(()=>{shownBlack=!shownBlack;render();},1000);}
pause.onclick=()=>{playing=!playing;pause.textContent=playing?'Pause':'Resume';schedule();};
document.getElementById('size').onclick=e=>{const native=document.getElementById('stack').classList.toggle('native');e.target.textContent=native?'Fit to window':'Native size';};
document.addEventListener('visibilitychange',schedule);
Promise.all([base.decode(),pink.decode(),black.decode()]).then(()=>{for(const image of [pink,black])if(base.naturalWidth!==image.naturalWidth||base.naturalHeight!==image.naturalHeight)throw Error('Dimensions differ');ready=true;pause.disabled=false;render();schedule();}).catch(e=>label.textContent=e.message);
</script></html>"#,
    )?;
    fs::write(
        out.join("composite-report.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"buttercup-waterfall-composite-v1","original_sha256":hash(&fs::read(original)?),"generated_sha256":hash(&fs::read(generated)?),"original_dimensions":[w,h],"generated_dimensions":[ew,eh],"output_dimensions":[w,h],"outside_mask_changed_pixels":outside,"inside_mask_changed_pixels":inside,"method":"Per-tile bounded translation and isotropic scale in native coordinates after global size mapping; ZNCC of nonpink surrounding texture, alternating samples held out. Extract new pink excess, remove flecks, composite constant pink only over unchanged original RGB.","limits":"Extracted mask is heuristic, not generator alpha. No anatomical accuracy claim. Scale/shift cannot repair nonlinear edits or wrong tile content. Generated coordinates in low-resolution sources have no recovered subpixel detail. Rejected tiles remain untouched.","frames":records}),
        )?,
    )?;
    let composed = records
        .iter()
        .filter(|r| r["status"] == "composited")
        .count();
    println!(
        "Composited {composed}/{} tiles; {outside} changes outside extracted mask",
        records.len()
    );
    Ok(composed == records.len())
}
