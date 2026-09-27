//! Native RAW -> existing SAM RGB adapter -> regenerated SAM -> shared contour
//! fit. The teacher sees no recorded predictions or calibration targets.
use super::{data, native, sam_export, train::Dataset, Result};
#[path = "canvas.rs"]
mod canvas;
use buttercup_eye_tracking::{recorded_bundle::BundleSource, sam31_outer as sam};
use canvas::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    fs,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};
use tch::{CModule, Device, IValue, Kind, Tensor};

pub const TW: usize = 420;
pub const TH: usize = 280;
pub struct Input {
    pub raw: Vec<u16>,
    pub frame: Value,
    pub hash: String,
}

/// One RAW exposure through the established color adapter, in planar RGB.
/// Repeated history inputs only satisfy the adapter's shape; no extra exposure
/// or temporal evidence is introduced.
pub fn input_rgb(input: &Input) -> Result<Vec<u8>> {
    let f = &input.frame;
    let n = |key: &str| data::num(&f[key]).ok_or_else(|| format!("RAW {key}"));
    let (w, h) = (n("width")? as usize, n("height")? as usize);
    if w * 2 != h * 3 || input.raw.len() != w * h {
        return Err("SAM adapter requires matching native 3:2 ROI".into());
    }
    let raw = Arc::new(sam::RawFrame {
        eye_index: n("eye_id")?.saturating_sub(1) as usize,
        sequence: n("sequence")?,
        timestamp_ns: n("timestamp_ns")?,
        sensor_x: n("sensor_x")? as u32,
        sensor_y: n("sensor_y")? as u32,
        width: w,
        height: h,
        registration_anchor: None,
        pupil_component_seed: None,
        pixels: Arc::new(input.raw.clone()),
    });
    let strip = sam::diagnostic_quantized_adapters(&vec![raw; sam::HISTORY_FRAMES])?
        .into_iter()
        .find(|(name, _)| *name == "quad_rgb")
        .ok_or("quad RGB adapter")?
        .1;
    let (sw, sh) = (sam::FRAME_WIDTH, sam::FRAME_HEIGHT);
    let fw = sw * sam::HISTORY_FRAMES;
    let mut rgb = vec![0; 3 * TW * TH];
    for c in 0..3 {
        for y in 0..TH {
            for x in 0..TW {
                rgb[c * TW * TH + y * TW + x] =
                    strip[c * fw * sh + (y * sh / TH) * fw + x * sw / TW];
            }
        }
    }
    Ok(rgb)
}

/// Cold teacher generation. Only original bytes and acquisition metadata are
/// reopened; no archived predictions, masks, affine or personal model is read.
pub fn prepare(d: &mut Dataset, out: &Path, teacher: &Teacher) -> Result<Value> {
    let start = Instant::now();
    let mut archives = BTreeMap::<String, Vec<usize>>::new();
    for (i, f) in d.frames.iter().enumerate() {
        archives
            .entry(f.source["archive"].as_str().ok_or("RAW archive")?.into())
            .or_default()
            .push(i);
    }
    let mut log = BufWriter::new(fs::File::create(out.join("sam-teacher.jsonl"))?);
    let mut totals = [0usize; 3];
    let mut audits = Vec::new();
    for (ai, (archive, ids)) in archives.iter().enumerate() {
        let bundle = BundleSource::open(Path::new(archive))?;
        let mut selected = Vec::new();
        for base in [0, ids.len() / 2, ids.len().saturating_sub(3)] {
            for i in base..(base + 3).min(ids.len()) {
                if !selected.contains(&i) {
                    selected.push(i);
                }
            }
        }
        let mut page = Canvas::new(1800, 1680)?;
        page.clear();
        page.text(
            20.,
            30.,
            23.,
            WHITE,
            "Cold SAM teacher | neighboring RAW, predicted contours and both conic hypotheses",
        );
        page.text(20., 58., 16., MUTED, "Cyan: fitted curve. Pink: SAM contour samples. Orange: classical control. Arrows: nominal camera-normal XY, not verified gaze.");
        let before = totals;
        for (j, &index) in ids.iter().enumerate() {
            let frame = &mut d.frames[index];
            let f = &frame.source["frame"];
            let get = |key: &str| data::num(&f[key]).ok_or_else(|| format!("RAW {key}"));
            let bytes = bundle.read_range(
                f["stream"].as_str().ok_or("RAW stream")?,
                get("offset")?,
                get("length")? as usize,
            )?;
            if data::digest(&bytes) != frame.hash {
                return Err("RAW changed between cold image and SAM preparation".into());
            }
            let input = Input {
                raw: native::unpack(&bytes, f)?,
                frame: f.clone(),
                hash: frame.hash.clone(),
            };
            let seg = teacher
                .segment(std::slice::from_ref(&input), None)?
                .pop()
                .ok_or("missing SAM result")?;
            let admitted = seg.fit.as_ref().is_some_and(|f| f.admissible());
            totals[0] += 1;
            totals[1] += usize::from(seg.fit.is_some());
            totals[2] += usize::from(admitted);
            serde_json::to_writer(
                &mut log,
                &json!({"source":frame.source,"sam_score":seg.score,"mask_sha256":data::digest(&seg.mask),"fit":seg.fit,"admissible":admitted}),
            )?;
            log.write_all(b"\n")?;
            if let Some(k) = selected.iter().position(|&v| v == j) {
                teacher_panel(&mut page, k, &input, &seg)?;
                page.png(&out.join(format!("sam-teacher-{ai:02}.png")))?;
            }
            frame.conic = seg.fit;
            if totals[0] % 100 == 0 {
                log.flush()?;
                let progress = json!({"frames_fits_admitted":totals,"frames_total":d.frames.len(),"completed_archives":ai,"archives":archives.len(),"seconds":start.elapsed().as_secs_f64()});
                data::write(out.join("sam-teacher-progress.json"), &progress)?;
                eprintln!(
                    "SAM cold teacher: {}/{} frames, {} fits, {} rim-admitted, {:.1}s",
                    totals[0],
                    d.frames.len(),
                    totals[1],
                    totals[2],
                    start.elapsed().as_secs_f64()
                );
            }
        }
        audits.push(json!({"archive":archive,"frames_fits_admitted":std::array::from_fn::<_,3,_>(|i|totals[i]-before[i]),"contact_sheet":format!("sam-teacher-{ai:02}.png")}));
    }
    log.flush()?;
    let result = json!({"provider":"sam31-single","export_receipt":teacher.receipt,"frames_fits_admitted":totals,"archives":audits,"teacher_log_sha256":sam_export::hash(&out.join("sam-teacher.jsonl"))?,"seconds":start.elapsed().as_secs_f64(),"scope":"all unique native RAW contributing eligible target pairs; other movie exposures remain visible without a SAM conic","limits":"SAM masks and nominal conics are conditional predictions, not human localization, independent scale or measured 3D signs."});
    data::write(out.join("sam-teacher.json"), &result)?;
    Ok(result)
}

fn teacher_panel(page: &mut Canvas, k: usize, input: &Input, seg: &Segment) -> Result<()> {
    let x = 20. + (k % 3) as f64 * 600.;
    let y = 100. + (k / 3) as f64 * 520.;
    let w = data::num(&input.frame["width"]).ok_or("width")? as usize;
    let h = data::num(&input.frame["height"]).ok_or("height")? as usize;
    let scale = 560. / w as f64;
    let hh = h as f64 * scale;
    page.text(
        x,
        y - 12.,
        16.,
        WHITE,
        &format!(
            "eye {} seq {} | SAM {:.3}",
            input.frame["eye_id"], input.frame["sequence"], seg.score
        ),
    );
    page.image(
        &native::preview(&input.raw, w, h),
        w / 4,
        h / 4,
        x,
        y,
        560.,
        hh,
    );
    let classical = native::for_supervision(&input.raw, &input.frame);
    for (fit, color) in [(classical.as_ref(), ORANGE), (seg.fit.as_ref(), CYAN)] {
        if let Some(fit) = fit {
            let curve = fit
                .ellipse()
                .dense_points(129)
                .into_iter()
                .map(|(a, b)| [x + a * scale, y + b * scale])
                .collect::<Vec<_>>();
            page.clipped(x, y, 560., hh, |c| c.path(&curve, 2., color));
        }
    }
    if let Some(fit) = &seg.fit {
        page.clipped(x, y, 560., hh, |c| {
            for p in &fit.points {
                c.dot(x + p[0] * scale, y + p[1] * scale, 1.4, PINK, true);
            }
        });
        page.text(
            x,
            y + hh + 22.,
            15.,
            if fit.admissible() { GREEN } else { ORANGE },
            &format!(
                "RAW rim L/B/R {:.2?} | {}",
                fit.outward_support_left_bottom_right,
                if fit.admissible() {
                    "admitted"
                } else {
                    "excluded"
                }
            ),
        );
        for (i, n) in fit.normals.iter().enumerate() {
            let center = [x + 25. + i as f64 * 285., y + hh + 63.];
            let color = if i == 0 { CYAN } else { PINK };
            page.cross(center[0], center[1], 4., MUTED);
            page.arrow(
                center,
                [center[0] + n[0] * 65., center[1] + n[1] * 65.],
                color,
            );
            page.text(
                center[0] + 40.,
                center[1] + 4.,
                12.,
                color,
                &format!("N{} {:.2?}", i + 1, n),
            );
        }
    } else {
        page.text(
            x,
            y + hh + 25.,
            16.,
            ORANGE,
            "No admitted SAM contour / native conic",
        );
    }
    Ok(())
}
pub struct Segment {
    pub mask: Vec<u8>,
    pub score: f32,
    pub fit: Option<native::NativeFit>,
}
pub struct Teacher {
    model: CModule,
    pub receipt: Value,
}
impl Teacher {
    pub fn open(export: &Path) -> Result<Self> {
        let receipt: Value = serde_json::from_slice(&fs::read(export.join("export.json"))?)?;
        if receipt["schema"] != "buttercup-sam31-cold-export-v1"
            || receipt["checkpoint_sha256"] != sam_export::CHECKPOINT_SHA
            || receipt["upstream_revision"] != sam_export::REVISION
            || receipt["prompts"] != json!(sam_export::PROMPTS)
            || receipt["adapter_sha256"] != sam_export::adapter_hash()
            || receipt["model_sha256"] != sam_export::hash(&export.join("detector.pt"))?
        {
            return Err(
                "SAM export identity/recipe mismatch; regenerate with current sam-export".into(),
            );
        }
        // LibTorch's CUDA dispatch library can be dropped by --as-needed.
        // Retain it for the duration of this offline teacher process.
        let library = CString::new("libtorch_cuda.so")?;
        unsafe {
            if libc::dlopen(library.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL).is_null() {
                let err = libc::dlerror();
                return Err(if err.is_null() {
                    "load libtorch CUDA dispatch".into()
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned().into()
                });
            }
        }
        tch::set_num_threads(4);
        tch::jit::set_tensor_expr_fuser_enabled(false);
        tch::jit::set_graph_executor_optimize(false);
        let mut model = CModule::load_on_device(export.join("detector.pt"), Device::Cuda(0))?;
        model.set_eval();
        Ok(Self { model, receipt })
    }
    pub fn segment(&self, inputs: &[Input], diagnostic: Option<&Path>) -> Result<Vec<Segment>> {
        if inputs.is_empty() || inputs.len() > 12 {
            return Err("SAM mosaic needs 1..12 RAW frames".into());
        }
        let columns = if inputs.len() == 1 { 1 } else { 3 };
        let rows = inputs.len().div_ceil(columns);
        let mosaic_width = columns * TW;
        let mosaic_height = rows * TH;
        let mut mosaic = vec![0u8; 3 * mosaic_width * mosaic_height];
        for (tile, input) in inputs.iter().enumerate() {
            let rgb = input_rgb(input)?;
            for y in 0..TH {
                for x in 0..TW {
                    for c in 0..3 {
                        let src = c * TW * TH + y * TW + x;
                        let dst = c * mosaic_width * mosaic_height
                            + (tile / columns * TH + y) * mosaic_width
                            + tile % columns * TW
                            + x;
                        mosaic[dst] = rgb[src];
                    }
                }
            }
        }
        if let Some(p) = diagnostic {
            let bgra = (0..mosaic_width * mosaic_height)
                .flat_map(|i| {
                    [
                        mosaic[2 * mosaic_width * mosaic_height + i],
                        mosaic[mosaic_width * mosaic_height + i],
                        mosaic[i],
                        255,
                    ]
                })
                .collect::<Vec<u8>>();
            let mut page = Canvas::new(mosaic_width, mosaic_height)?;
            page.image(
                &bgra,
                mosaic_width,
                mosaic_height,
                0.,
                0.,
                mosaic_width as f64,
                mosaic_height as f64,
            );
            page.png(&p.with_extension("input.png"))?;
        }
        let _guard = tch::no_grad_guard();
        let tensor = Tensor::from_slice(&mosaic)
            .view([1, 3, mosaic_height as i64, mosaic_width as i64])
            .to_device(Device::Cuda(0))
            .to_kind(Kind::Float)
            / 255.;
        let tensor = tensor.internal_upsample_bilinear2d_aa([1008, 1008], false, None, None);
        let tensor = ((tensor - 0.5) / 0.5).to_kind(Kind::BFloat16);
        let output = self.model.forward_is(&[IValue::Tensor(tensor)])?;
        let IValue::Tuple(mut output) = output else {
            return Err("SAM output tuple".into());
        };
        if output.len() != 2 {
            return Err("SAM output count".into());
        }
        let IValue::Tensor(masks) = output.pop().unwrap() else {
            return Err("SAM masks".into());
        };
        let IValue::Tensor(scores) = output.pop().unwrap() else {
            return Err("SAM scores".into());
        };
        let size = masks.size();
        if size.len() != 4 || size[0] != 2 || scores.size() != size[..2] {
            return Err("SAM output shape".into());
        }
        let (queries, mh, mw) = (size[1] as usize, size[2] as usize, size[3] as usize);
        let masks = masks
            .get(0)
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .contiguous();
        let scores =
            Vec::<f32>::try_from(scores.get(0).to_kind(Kind::Float).to_device(Device::Cpu))?;
        let mut logits = vec![0f32; queries * mw * mh];
        masks.copy_data(&mut logits, queries * mw * mh);
        if let Some(p) = diagnostic {
            let mut ranked = (0..queries).collect::<Vec<_>>();
            ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let mut query_rows = Vec::new();
            for &q in ranked.iter().take(20) {
                let l = &logits[q * mw * mh..(q + 1) * mw * mh];
                let mut areas = vec![0usize; columns * rows];
                for y in 0..mh {
                    for x in 0..mw {
                        if l[y * mw + x] > 0. {
                            areas[(y * rows / mh) * columns + x * columns / mw] += 1;
                        }
                    }
                }
                query_rows.push(
                    json!({"query":q,"score":scores[q],"positive_mask_pixels_by_tile":areas}),
                );
            }
            data::write(p.with_extension("queries.json"), &query_rows)?;
            let q = ranked[0];
            let bgra = logits[q * mw * mh..(q + 1) * mw * mh]
                .iter()
                .flat_map(|v| {
                    if *v > 0. {
                        [110, 220, 150, 255]
                    } else {
                        [18, 20, 25, 255]
                    }
                })
                .collect::<Vec<u8>>();
            let mut page = Canvas::new(mosaic_width, mosaic_height)?;
            page.image(
                &bgra,
                mw,
                mh,
                0.,
                0.,
                mosaic_width as f64,
                mosaic_height as f64,
            );
            page.png(&p.with_extension("top-mask.png"))?;
        }
        let mut result = (0..inputs.len())
            .map(|_| Segment {
                mask: vec![0; TW * TH],
                score: 0.,
                fit: None,
            })
            .collect::<Vec<_>>();
        for q in 0..queries {
            if !scores[q].is_finite() || scores[q] < 0.15 {
                continue;
            }
            let l = &logits[q * mw * mh..(q + 1) * mw * mh];
            let mut areas = vec![0usize; columns * rows];
            for y in 0..mh {
                for x in 0..mw {
                    if l[y * mw + x] > 0. {
                        areas[(y * rows / mh) * columns + x * columns / mw] += 1;
                    }
                }
            }
            let tile = (0..areas.len()).max_by_key(|&i| areas[i]).unwrap();
            let area = areas.iter().sum::<usize>();
            if tile >= inputs.len() || area < 64 || areas[tile] * 5 < area * 4 {
                continue;
            }
            let mut mask = vec![0u8; TW * TH];
            for y in 0..TH {
                for x in 0..TW {
                    let xx = ((tile % columns * TW + x) as f64 + 0.5) * mw as f64
                        / mosaic_width as f64
                        - 0.5;
                    let yy = ((tile / columns * TH + y) as f64 + 0.5) * mh as f64
                        / mosaic_height as f64
                        - 0.5;
                    let (x0, y0) = (xx.floor().max(0.) as usize, yy.floor().max(0.) as usize);
                    let (x1, y1) = ((x0 + 1).min(mw - 1), (y0 + 1).min(mh - 1));
                    let (a, b) = (
                        (xx - x0 as f64).clamp(0., 1.) as f32,
                        (yy - y0 as f64).clamp(0., 1.) as f32,
                    );
                    let v = (1. - b) * ((1. - a) * l[y0 * mw + x0] + a * l[y0 * mw + x1])
                        + b * ((1. - a) * l[y1 * mw + x0] + a * l[y1 * mw + x1]);
                    mask[y * TW + x] = u8::from(v > 0.);
                }
            }
            let input = &inputs[tile];
            let fit = sam::diagnostic_fit_single_frame_mask(&mask, TW, TH).and_then(|review| {
                let scale = data::num(&input.frame["width"])? as f64 / sam::FRAME_WIDTH as f64;
                let mut e = review.ellipse;
                e.center = (
                    (e.center.0 + 0.5) * scale - 0.5,
                    (e.center.1 + 0.5) * scale - 0.5,
                );
                e.major_radius *= scale;
                e.minor_radius *= scale;
                let points = review
                    .retained_points
                    .iter()
                    .map(|&(x, y)| [(x + 0.5) * scale - 0.5, (y + 0.5) * scale - 0.5])
                    .collect();
                native::from_segmentation(&input.raw, &input.frame, e, points).map(|mut fit| {
                    fit.provider = if inputs.len() == 1 {
                        "sam31-single"
                    } else {
                        "sam31-mosaic"
                    }
                    .into();
                    fit.sam_score = Some(scores[q]);
                    fit
                })
            });
            let admitted = fit.as_ref().is_some_and(|f| f.admissible());
            let old_admitted = result[tile].fit.as_ref().is_some_and(|f| f.admissible());
            if (admitted && !old_admitted)
                || (admitted == old_admitted && scores[q] > result[tile].score)
            {
                result[tile] = Segment {
                    mask,
                    score: scores[q],
                    fit,
                };
            }
        }
        Ok(result)
    }
}
pub fn run(args: &[String]) -> Result<()> {
    if !(4..=5).contains(&args.len()) {
        return Err("sam-native CORPUS EXPORT NEW_OUT [ARCHIVE]".into());
    }
    let out = data::output(&args[3])?;
    let start = Instant::now();
    let teacher = Teacher::open(Path::new(&args[2]))?;
    let (sources, inventory) = data::scan(&args[1])?;
    data::write(out.join("inventory.json"), &inventory)?;
    let mut log = BufWriter::new(fs::File::create(out.join("sam-native.jsonl"))?);
    let mut totals = [0usize; 3];
    for (si, s) in sources
        .iter()
        .filter(|s| !s.eligible.is_empty() && args.get(4).is_none_or(|p| s.archive.contains(p)))
        .enumerate()
    {
        let bundle = BundleSource::open(Path::new(&s.archive))?;
        let mut chosen = Vec::new();
        for t in 0..s.spans.len() {
            for eye in 1..=2 {
                let ids = (0..s.eligible.len())
                    .filter(|&j| {
                        s.targets[j] == t
                            && data::num(&s.frames[s.eligible[j][1]]["eye_id"]) == Some(eye)
                    })
                    .collect::<Vec<_>>();
                if !ids.is_empty() {
                    chosen.push(ids[ids.len() / 2]);
                }
            }
        }
        let mut inputs = Vec::new();
        for &j in chosen.iter().take(12) {
            let f = &s.frames[s.eligible[j][1]];
            let raw = bundle.read_range(
                f["stream"].as_str().ok_or("stream")?,
                data::num(&f["offset"]).ok_or("offset")?,
                data::num(&f["length"]).ok_or("length")? as usize,
            )?;
            inputs.push(Input {
                raw: native::unpack(&raw, f)?,
                frame: f.clone(),
                hash: data::digest(&raw),
            });
        }
        let mut segments = Vec::new();
        let batch = if args[0] == "sam-native-single" {
            1
        } else {
            12
        };
        for (chunk, frames) in inputs.chunks(batch).enumerate() {
            segments.extend(
                teacher.segment(frames, Some(&out.join(format!("sam-{si:02}-{chunk:02}"))))?,
            );
        }
        let mut page = Canvas::new(1800, 2040)?;
        page.clear();
        page.text(
            20.,
            30.,
            23.,
            WHITE,
            "Regenerated SAM3.1 on native RAW | mask contours and shared ellipse fit",
        );
        page.text(20.,58.,16.,MUTED,"Cyan curve: SAM-derived fit. Pink dots: mask contour samples. Orange: classical RAW proposal. These are not human labels.");
        for (k, (input, seg)) in inputs.iter().zip(&segments).enumerate() {
            let x = 20. + (k % 3) as f64 * 600.;
            let y = 100. + (k / 3) as f64 * 480.;
            let (w, h) = (
                data::num(&input.frame["width"]).unwrap() as usize,
                data::num(&input.frame["height"]).unwrap() as usize,
            );
            let scale = 560. / w as f64;
            let hh = h as f64 * scale;
            page.image(
                &native::preview(&input.raw, w, h),
                w / 4,
                h / 4,
                x,
                y,
                560.,
                hh,
            );
            let baseline = native::for_supervision(&input.raw, &input.frame);
            for (f, color) in [(baseline.as_ref(), ORANGE), (seg.fit.as_ref(), CYAN)] {
                if let Some(f) = f {
                    let curve = f
                        .ellipse()
                        .dense_points(129)
                        .into_iter()
                        .map(|(a, b)| [x + a * scale, y + b * scale])
                        .collect::<Vec<_>>();
                    page.clipped(x, y, 560., hh, |c| c.path(&curve, 2., color));
                }
            }
            if let Some(f) = &seg.fit {
                page.clipped(x, y, 560., hh, |c| {
                    for p in &f.points {
                        c.dot(x + p[0] * scale, y + p[1] * scale, 1.4, PINK, true);
                    }
                });
            }
            let mask = seg
                .mask
                .iter()
                .flat_map(|v| {
                    if *v != 0 {
                        [110, 220, 150, 255]
                    } else {
                        [18, 20, 25, 255]
                    }
                })
                .collect::<Vec<u8>>();
            page.image(&mask, TW, TH, x, y + hh + 5., 105., 70.);
            page.text(
                x,
                y - 12.,
                16.,
                WHITE,
                &format!(
                    "eye {} seq {} target {:?}",
                    input.frame["eye_id"],
                    input.frame["sequence"],
                    s.spans[s.targets[chosen[k]]].uv
                ),
            );
            let admitted = seg.fit.as_ref().is_some_and(|f| f.admissible());
            page.text(
                x + 120.,
                y + hh + 25.,
                16.,
                if admitted { GREEN } else { ORANGE },
                &format!(
                    "SAM {:.3} | {}",
                    seg.score,
                    if admitted {
                        "RAW rim audit passes"
                    } else {
                        "not admitted"
                    }
                ),
            );
            if let Some(f) = &seg.fit {
                page.text(
                    x + 120.,
                    y + hh + 47.,
                    14.,
                    MUTED,
                    &format!(
                        "rim support L/B/R: {:.2?}",
                        f.outward_support_left_bottom_right
                    ),
                );
            }
            totals[0] += 1;
            totals[1] += usize::from(seg.fit.is_some());
            totals[2] += usize::from(admitted);
            serde_json::to_writer(
                &mut log,
                &json!({"archive":s.archive,"frame":input.frame,"raw_sha256":input.hash,"target":s.spans[s.targets[chosen[k]]],"sam_score":seg.score,"sam_mask_sha256":data::digest(&seg.mask),"fit":seg.fit,"admissible":admitted,"classical":baseline}),
            )?;
            log.write_all(b"\n")?;
        }
        page.png(&out.join(format!("sam-native-{si:02}.png")))?;
        eprintln!("SAM RAW probe {si}: frames/fits/admitted {totals:?}");
    }
    log.flush()?;
    data::write(
        out.join("result.json"),
        &json!({"frames_fits_admitted":totals,"seconds":start.elapsed().as_secs_f64(),"teacher":teacher.receipt,"limits":"RAW localization inspection only; mask contours are SAM predictions, nominal conics are conditional, no independent sign/scale/human localization truth."}),
    )
}
