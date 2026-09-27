//! Shared affine shape/motion reconstruction with missing observations.
//! Rank-2 planar baseline, rank-3 shape candidate, held-out measurements and
//! explicit metric-upgrade checks. Completed entries are never observations.
use opencv::{
    calib3d,
    core::{self, Mat},
    prelude::*,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fmt::Write as _,
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
type E = Box<dyn std::error::Error>;
type P = [f64; 2];
struct Data {
    frames: Vec<Value>,
    ids: Vec<u64>,
    source: usize,
    observed: Vec<Vec<Option<P>>>,
    initial: Vec<P>,
}
struct Model {
    shape: Vec<Vec<f64>>,
    cameras: Vec<Vec<f64>>,
    rank: usize,
    iterations: usize,
}
fn solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, E> {
    if a.is_empty() || a.len() < a[0].len() {
        return Err("underdetermined least squares".into());
    }
    let rhs = b.iter().map(|v| [*v]).collect::<Vec<_>>();
    let mut result = Mat::default();
    if !core::solve(
        &Mat::from_slice_2d(a)?,
        &Mat::from_slice_2d(&rhs)?,
        &mut result,
        core::DECOMP_SVD,
    )? {
        return Err("SVD solve failed".into());
    }
    let v = (0..a[0].len())
        .map(|j| Ok(*result.at_2d::<f64>(j as i32, 0)?))
        .collect::<Result<Vec<_>, opencv::Error>>()?;
    if v.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite factorization".into());
    }
    Ok(v)
}
fn predict(m: &Model, f: usize, p: usize) -> P {
    std::array::from_fn(|axis| {
        m.cameras[2 * f + axis][m.rank]
            + (0..m.rank)
                .map(|k| m.cameras[2 * f + axis][k] * m.shape[p][k])
                .sum::<f64>()
    })
}
fn held(d: &Data, f: usize, p: usize, fold: Option<usize>) -> bool {
    fold.is_some_and(|fold| {
        d.frames[f]["sequence"].as_u64().unwrap() > 1036
            && f % 2 == 1
            && d.ids[p] as usize % 3 == fold
    })
}
fn fit(d: &Data, rank: usize, fold: Option<usize>) -> Result<Model, E> {
    let mut m = Model {
        shape: d
            .initial
            .iter()
            .enumerate()
            .map(|(p, v)| {
                (0..rank)
                    .map(|k| {
                        if k < 2 {
                            v[k]
                        } else {
                            0.1 * (p as f64 * 2.39996).sin()
                        }
                    })
                    .collect()
            })
            .collect(),
        cameras: vec![vec![0.; rank + 1]; 2 * d.frames.len()],
        rank,
        iterations: 0,
    };
    let mut previous = f64::INFINITY;
    for iteration in 0..100 {
        for f in 0..d.frames.len() {
            for axis in 0..2 {
                let mut a = vec![];
                let mut b = vec![];
                for p in 0..d.ids.len() {
                    if held(d, f, p, fold) {
                        continue;
                    }
                    if let Some(v) = d.observed[f][p] {
                        let mut row = m.shape[p].clone();
                        row.push(1.);
                        a.push(row);
                        b.push(v[axis]);
                    }
                }
                m.cameras[2 * f + axis] = solve(&a, &b)?;
            }
        }
        for p in 0..d.ids.len() {
            let mut a = vec![];
            let mut b = vec![];
            for f in 0..d.frames.len() {
                if held(d, f, p, fold) {
                    continue;
                }
                if let Some(v) = d.observed[f][p] {
                    for axis in 0..2 {
                        a.push(m.cameras[2 * f + axis][..rank].to_vec());
                        b.push(v[axis] - m.cameras[2 * f + axis][rank]);
                    }
                }
            }
            m.shape[p] = solve(&a, &b)?;
        }
        // Fix translation/scale gauges without changing any image prediction.
        for k in 0..rank {
            let mean = m.shape.iter().map(|s| s[k]).sum::<f64>() / m.shape.len() as f64;
            let scale = (m.shape.iter().map(|s| (s[k] - mean).powi(2)).sum::<f64>()
                / m.shape.len() as f64)
                .sqrt()
                .max(1e-8);
            for c in &mut m.cameras {
                c[rank] += c[k] * mean;
                c[k] *= scale;
            }
            for s in &mut m.shape {
                s[k] = (s[k] - mean) / scale;
            }
        }
        let mut cost = 0.;
        for f in 0..d.frames.len() {
            for p in 0..d.ids.len() {
                if held(d, f, p, fold) {
                    continue;
                }
                if let Some(v) = d.observed[f][p] {
                    let q = predict(&m, f, p);
                    cost += (q[0] - v[0]).powi(2) + (q[1] - v[1]).powi(2);
                }
            }
        }
        m.iterations = iteration + 1;
        if iteration > 10 && (previous - cost).abs() < 1e-10 * (1. + previous) {
            break;
        }
        previous = cost;
    }
    Ok(m)
}
fn errors(d: &Data, m: &Model, fold: Option<usize>) -> Vec<f64> {
    let mut result = vec![];
    for f in 0..d.frames.len() {
        for p in 0..d.ids.len() {
            if fold.is_some() && !held(d, f, p, fold) {
                continue;
            }
            if let Some(v) = d.observed[f][p] {
                let q = predict(m, f, p);
                result.push(100. * (q[0] - v[0]).hypot(q[1] - v[1]));
            }
        }
    }
    result
}
fn stats(mut e: Vec<f64>) -> Value {
    e.sort_by(f64::total_cmp);
    if e.is_empty() {
        return json!({"count":0});
    }
    json!({"count":e.len(),"median_px":e[e.len()/2],"rms_px":(e.iter().map(|x|x*x).sum::<f64>()/e.len() as f64).sqrt(),"p90_px":e[(e.len()-1)*9/10]})
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn product(a: [f64; 3], b: [f64; 3]) -> [f64; 6] {
    [
        a[0] * b[0],
        a[1] * b[1],
        a[2] * b[2],
        a[0] * b[1] + a[1] * b[0],
        a[0] * b[2] + a[2] * b[0],
        a[1] * b[2] + a[2] * b[1],
    ]
}
fn metric(d: &Data, m: &Model, scaled: bool) -> Result<Value, E> {
    let mut a = vec![];
    let mut b = vec![];
    for f in 0..d.frames.len() {
        let x = std::array::from_fn(|k| m.cameras[2 * f][k]);
        let y = std::array::from_fn(|k| m.cameras[2 * f + 1][k]);
        let xx = product(x, x);
        let yy = product(y, y);
        let xy = product(x, y);
        if scaled {
            // Homogeneous equal-row-length/orthogonality constraints; trace(L)=3
            // fixes scale exactly, without forcing an indefinite metric positive.
            for c in [std::array::from_fn::<_, 6, _>(|k| xx[k] - yy[k]), xy] {
                a.push(vec![c[0] - c[2], c[1] - c[2], c[3], c[4], c[5]]);
                b.push(-3. * c[2]);
            }
        } else {
            for (c, v) in [(xx, 1.), (yy, 1.), (xy, 0.)] {
                a.push(c.to_vec());
                b.push(v);
            }
        }
    }
    let v = solve(&a, &b)?;
    let l = if scaled {
        [
            [v[0], v[2], v[3]],
            [v[2], v[1], v[4]],
            [v[3], v[4], 3. - v[0] - v[1]],
        ]
    } else {
        [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]]
    };
    let (mut values, mut vectors) = (Mat::default(), Mat::default());
    core::eigen(&Mat::from_slice_2d(&l)?, &mut values, &mut vectors)?;
    let eig = (0..3)
        .map(|i| Ok(*values.at_2d::<f64>(i, 0)?))
        .collect::<Result<Vec<_>, opencv::Error>>()?;
    let mut report = json!({"metric_matrix":l,"metric_eigenvalues":eig,"positive_definite":eig.iter().all(|v|*v>1e-8)});
    if eig.iter().any(|v| *v <= 1e-8) {
        report["status"]=json!("No positive-definite metric upgrade; affine fit does not establish rigid 3D motion under this projection model.");
        return Ok(report);
    }
    let q: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| *vectors.at_2d::<f64>(i as i32, j as i32).unwrap())
    });
    let root: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| q[k][i] * eig[k].sqrt() * q[k][j]).sum())
    });
    let inv: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| q[k][i] / eig[k].sqrt() * q[k][j]).sum())
    });
    let shapes = m
        .shape
        .iter()
        .map(|s| inv.map(|r| (0..3).map(|k| r[k] * s[k]).sum::<f64>()))
        .collect::<Vec<_>>();
    let mut rotations = vec![];
    let mut scales = vec![];
    let mut residuals = vec![];
    for f in 0..d.frames.len() {
        let x: [f64; 3] =
            std::array::from_fn(|k| (0..3).map(|j| m.cameras[2 * f][j] * root[j][k]).sum());
        let y: [f64; 3] =
            std::array::from_fn(|k| (0..3).map(|j| m.cameras[2 * f + 1][j] * root[j][k]).sum());
        let scale = if scaled { (norm(x) + norm(y)) / 2. } else { 1. };
        let x = x.map(|v| v / norm(x));
        let y0 = std::array::from_fn(|k| y[k] - dot(x, y) * x[k]);
        let y = y0.map(|v| v / norm(y0));
        let z = [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ];
        rotations.push([x, y, z]);
        scales.push(scale);
        for p in 0..d.ids.len() {
            if let Some(v) = d.observed[f][p] {
                let pred = [
                    scale * dot(x, shapes[p]) + m.cameras[2 * f][3],
                    scale * dot(y, shapes[p]) + m.cameras[2 * f + 1][3],
                ];
                residuals.push(100. * (pred[0] - v[0]).hypot(pred[1] - v[1]));
            }
        }
    }
    let reference = rotations[d.source];
    let mut poses = vec![];
    for (f, r) in rotations.iter().enumerate() {
        let relative: [[f64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| dot(r[i], reference[j])));
        let mut rv = Mat::default();
        calib3d::rodrigues_def(&Mat::from_slice_2d(&relative)?, &mut rv)?;
        let vector = (0..3)
            .map(|i| rv.at_2d::<f64>(i, 0).unwrap().to_degrees())
            .collect::<Vec<_>>();
        poses.push(json!({"sequence":d.frames[f]["sequence"],"rotation_vector_degrees":vector,"relative_image_scale":scales[f]/scales[d.source]}));
    }
    report["metric_reprojection"] = stats(residuals);
    report["poses"] = json!(poses);
    report["status"]=json!("Conditional metric upgrade; reflection ambiguity and optical-axis translation remain unobserved. No metric distance or anatomical claim.");
    Ok(report)
}
pub fn run(input: &str, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("output must be new".into());
    }
    if !out
        .parent()
        .ok_or("missing output parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    core::set_num_threads(1)?;
    let root: Value = serde_json::from_slice(&fs::read(input)?)?;
    let s = root["series"]
        .as_array()
        .ok_or("missing series")?
        .iter()
        .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .ok_or("missing series")?;
    let all_frames = s["frames"].as_array().unwrap();
    let source = all_frames
        .iter()
        .position(|f| f["sequence"] == s["source_sequence"])
        .unwrap();
    let origin = [
        all_frames[source]["input"]["frame"]["sensor_x"]
            .as_f64()
            .unwrap()
            + 210.,
        all_frames[source]["input"]["frame"]["sensor_y"]
            .as_f64()
            .unwrap()
            + 140.,
    ];
    let mut sets = vec![];
    for field in ["outer_tracks", "tracks"] {
        let candidates = s[field]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| {
                t["frames"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|v| v["accepted"] == true)
                    .count()
                    >= 6
            })
            .collect::<Vec<_>>();
        let fi = (0..all_frames.len())
            .filter(|&f| {
                candidates
                    .iter()
                    .filter(|t| t["frames"][f]["accepted"] == true)
                    .count()
                    >= 8
            })
            .collect::<Vec<_>>();
        let d = Data {
            frames: fi.iter().map(|&f| all_frames[f].clone()).collect(),
            ids: candidates
                .iter()
                .map(|t| t["id"].as_u64().unwrap())
                .collect(),
            source: fi
                .iter()
                .position(|&f| f == source)
                .ok_or("missing source support")?,
            observed: fi
                .iter()
                .map(|&f| {
                    candidates
                        .iter()
                        .map(|t| {
                            if t["frames"][f]["accepted"] == true {
                                Some(std::array::from_fn(|a| {
                                    (t["frames"][f]["sensor"][a].as_f64().unwrap() - origin[a])
                                        / 100.
                                }))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect(),
            initial: candidates
                .iter()
                .map(|t| {
                    std::array::from_fn(|a| {
                        (t["source_sensor"][a].as_f64().unwrap() - origin[a]) / 100.
                    })
                })
                .collect(),
        };
        let mut models = vec![];
        for rank in [2, 3] {
            let model = fit(&d, rank, None)?;
            let mut cv = vec![];
            let mut folds = vec![];
            let mut held_predictions = vec![];
            for fold in 0..3 {
                let held_model = fit(&d, rank, Some(fold))?;
                let e = errors(&d, &held_model, Some(fold));
                folds.push(stats(e.clone()));
                cv.extend(e);
                for f in 0..d.frames.len() {
                    for p in 0..d.ids.len() {
                        if held(&d, f, p, Some(fold)) {
                            if let Some(v) = d.observed[f][p] {
                                let q = predict(&held_model, f, p);
                                held_predictions.push(json!({"fold":fold,"sequence":d.frames[f]["sequence"],"raw_sha256":d.frames[f]["raw_sha256"],"id":d.ids[p],"observed_sensor":[100.*v[0]+origin[0],100.*v[1]+origin[1]],"predicted_sensor":[100.*q[0]+origin[0],100.*q[1]+origin[1]]}));
                            }
                        }
                    }
                }
            }
            let reprojections=d.frames.iter().enumerate().map(|(f,frame)|json!({"sequence":frame["sequence"],"raw_sha256":frame["raw_sha256"],"points":d.ids.iter().enumerate().map(|(p,id)|{let pred=predict(&model,f,p);json!({"id":id,"observed_sensor":d.observed[f][p].map(|v|[100.*v[0]+origin[0],100.*v[1]+origin[1]]),"predicted_sensor":[100.*pred[0]+origin[0],100.*pred[1]+origin[1]]})}).collect::<Vec<_>>()})).collect::<Vec<_>>();
            let mut result = json!({"rank":rank,"iterations":model.iterations,"training":stats(errors(&d,&model,None)),"heldout":stats(cv),"folds":folds,"reprojections":reprojections,"heldout_predictions":held_predictions});
            if rank == 3 {
                result["orthographic_metric"] = metric(&d, &model, false)?;
                result["scaled_orthographic_metric"] = metric(&d, &model, true)?;
            }
            println!(
                "{field} rank {rank}: train {} heldout {}",
                result["training"], result["heldout"]
            );
            if rank == 3 {
                println!(
                    "metrics: orthographic {}, scaled {}",
                    result["orthographic_metric"]["metric_eigenvalues"],
                    result["scaled_orthographic_metric"]["metric_eigenvalues"]
                );
            }
            models.push(result);
        }
        sets.push(json!({"group":field,"frame_count":d.frames.len(),"sequences":d.frames.iter().map(|f|f["sequence"].clone()).collect::<Vec<_>>(),"point_count":d.ids.len(),"ids":d.ids,"observed_count":d.observed.iter().flatten().filter(|v|v.is_some()).count(),"models":models}));
    }
    fs::create_dir(out)?;
    render_validation(&sets, all_frames, out)?;
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"input":input,"method":"Shared rank-2/rank-3 affine factorization via alternating SVD least squares over all observed views, plus orthographic/scaled-orthographic metric-upgrade attempts. Missing entries are never counted as evidence.","validation":"Three deterministic held-out point folds in alternating frames after source 1036; source observations and original cohort-selection interval are never held out. Camera rows and shared shape fit without held coordinates. Conditional within-recording validation; overlapping patches and no human 3D truth.","units":"native pixel reprojection errors; fixed 100px numeric scaling only. No estimated iris radius used for normalization.","reference":"Tomasi and Kanade: Shape and Motion from Image Streams (factorization); this missing-data ALS diagnostic is not a complete calibrated perspective bundle adjustment.","sets":sets}),
        )?,
    )?;
    Ok(())
}

/// Both predictions for each dot were made without that dot's coordinates in
/// this exposure. The RAW and measured coordinates remain separate from fits.
fn render_validation(sets: &[Value], frames: &[Value], out: &Path) -> Result<(), E> {
    let mut svg = "<svg xmlns='http://www.w3.org/2000/svg' width='1860' height='1190'><rect width='100%' height='100%' fill='#171b24'/><g fill='white' font-family='sans-serif'><text x='25' y='36' font-size='25'>Shared reconstruction: predict measurements deliberately withheld from the fit</text><text x='25' y='68' font-size='18'>White = measured · teal ring = shared planar fit · orange cross = shared rank-3 fit · actual pixel displacement</text><text x='25' y='98' font-size='17'>Same point IDs, RAW exposures and held-out coordinates for both models. Three folds pooled; no completed track is an observation.</text>".to_string();
    for (col, seq) in [1050, 1060, 1072].iter().enumerate() {
        let frame = frames
            .iter()
            .find(|f| f["sequence"] == *seq)
            .ok_or("missing validation exposure")?;
        let input = &frame["input"];
        let meta = &input["frame"];
        let n = |k: &str| meta[k].as_u64().unwrap() as usize;
        let (w, h) = (n("width"), n("height"));
        let origin = [n("sensor_x") as f64, n("sensor_y") as f64];
        let mut file = fs::File::open(input["raw_file"].as_str().unwrap())?;
        file.seek(SeekFrom::Start(input["raw_offset"].as_u64().unwrap()))?;
        let mut bytes = vec![0; input["raw_length"].as_u64().unwrap() as usize];
        file.read_exact(&mut bytes)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            frame["raw_sha256"].as_str().unwrap()
        );
        let pixels = crate::raw10::try_unpack_raw10(&bytes, w, h, n("stride"))?;
        let rgb = crate::raw_preview::color_preview(
            &pixels,
            w,
            h,
            origin[0] as u32,
            origin[1] as u32,
            100,
            None,
        );
        write!(svg, "<defs><g id='raw-{seq}' shape-rendering='crispEdges'>")?;
        for y in 0..h {
            for x in 0..w {
                write!(
                    svg,
                    "<rect x='{x}' y='{y}' width='1' height='1' fill='#{:06x}'/>",
                    rgb[y * w + x]
                )?;
            }
        }
        svg.push_str("</g></defs>");
        for (row, set) in sets.iter().enumerate() {
            let (x, y) = (25. + 615. * col as f64, 165. + row as f64 * 495.);
            let scale = 580. / w as f64;
            let height = h as f64 * scale;
            let group = if row == 0 {
                "Outer top / bottom bands"
            } else {
                "Iris region / reflection"
            };
            write!(svg,"<text x='{x}' y='{}' font-size='20'>{group} · source {seq}</text><clipPath id='clip-{row}-{col}'><rect x='{x}' y='{y}' width='580' height='{height}'/></clipPath><g clip-path='url(#clip-{row}-{col})'><use href='#raw-{seq}' transform='translate({x},{y}) scale({scale})'/>",y-15.)?;
            let mut errors = vec![];
            let mut counts = vec![];
            for (mi, model) in set["models"].as_array().unwrap().iter().enumerate() {
                let points = model["heldout_predictions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|p| p["sequence"] == *seq)
                    .collect::<Vec<_>>();
                counts.push(points.len());
                let mut es = vec![];
                for point in points {
                    assert_eq!(point["raw_sha256"], frame["raw_sha256"]);
                    let observed: P =
                        std::array::from_fn(|a| point["observed_sensor"][a].as_f64().unwrap());
                    let predicted: P =
                        std::array::from_fn(|a| point["predicted_sensor"][a].as_f64().unwrap());
                    let a = [
                        x + (observed[0] - origin[0]) * scale,
                        y + (observed[1] - origin[1]) * scale,
                    ];
                    let b = [
                        x + (predicted[0] - origin[0]) * scale,
                        y + (predicted[1] - origin[1]) * scale,
                    ];
                    es.push((observed[0] - predicted[0]).hypot(observed[1] - predicted[1]));
                    let color = if mi == 0 { "#62f4d3" } else { "#ffa74f" };
                    write!(svg,"<path d='M{},{} L{},{}' stroke='{color}' stroke-width='1.2' opacity='.9' fill='none'/>",a[0],a[1],b[0],b[1])?;
                    if mi == 0 {
                        write!(svg,"<circle cx='{}' cy='{}' r='4' fill='none' stroke='{color}' stroke-width='1.5'/><circle cx='{}' cy='{}' r='1.8' fill='white' stroke='#111' stroke-width='.6'/>",b[0],b[1],a[0],a[1])?;
                    } else {
                        write!(svg,"<path d='M{},{} l6,6 m-6,0 l6,-6' fill='none' stroke='{color}' stroke-width='1.6'/>",b[0]-3.,b[1]-3.)?;
                    }
                }
                errors.push(stats(es));
            }
            svg.push_str("</g>");
            for (i, e) in errors.iter().enumerate() {
                let color = if i == 0 { "#62f4d3" } else { "#ffa74f" };
                write!(svg,"<text x='{x}' y='{}' font-size='17' fill='{color}'>{}: {} withheld points · RMS {:.2}px · median {:.2}px</text>",y+height+24.+i as f64*24.,if i==0{"Planar"}else{"Rank 3"},counts[i],e["rms_px"].as_f64().unwrap_or(0.),e["median_px"].as_f64().unwrap_or(0.))?;
            }
        }
    }
    svg.push_str("<text x='25' y='1172' font-size='17'>Rank 3 permits affine depth variation; it does not establish anatomical depth or a unique metric rotation. Predictions outside the image are clipped.</text></g></svg>");
    fs::write(out.join("heldout-raw-overlays.svg"), svg)?;
    Ok(())
}
