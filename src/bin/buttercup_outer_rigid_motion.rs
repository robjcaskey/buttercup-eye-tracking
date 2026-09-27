//! Offline 3D two-view motion diagnostic from measured outer-region tracks.
//! Translation has unit-baseline scale, not millimetres. No live solver changes.
#[cfg(feature = "checkerboard")]
#[path = "buttercup_outer_rigid_motion/factorization.rs"]
mod factorization;
#[path = "../raw10.rs"]
mod raw10;
#[path = "../raw_preview.rs"]
mod raw_preview;
#[cfg(feature = "checkerboard")]
#[path = "buttercup_outer_rigid_motion/stabilized.rs"]
mod stabilized;
#[cfg(not(feature = "checkerboard"))]
fn main() {
    eprintln!("Build with the checkerboard feature (OpenCV calib3d).");
    std::process::exit(2);
}
#[cfg(feature = "checkerboard")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::run()
}
#[cfg(feature = "checkerboard")]
mod app {
    use opencv::{
        calib3d,
        core::{self, Mat, Point2d, Vector},
        prelude::*,
    };
    use serde_json::{json, Value};
    use std::{fmt::Write as _, fs, path::Path};
    type Err = Box<dyn std::error::Error>;
    #[derive(Clone)]
    struct Track {
        id: u64,
        a: [f64; 2],
        b: [f64; 2],
    }
    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a.into_iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn norm(a: [f64; 3]) -> f64 {
        dot(a, a).sqrt()
    }
    fn median(mut a: Vec<f64>) -> Option<f64> {
        a.retain(|v| v.is_finite());
        a.sort_by(f64::total_cmp);
        (!a.is_empty()).then(|| a[a.len() / 2])
    }
    fn quantile(mut a: Vec<f64>, q: f64) -> Option<f64> {
        a.retain(|v| v.is_finite());
        a.sort_by(f64::total_cmp);
        (!a.is_empty()).then(|| a[((a.len() - 1) as f64 * q).round() as usize])
    }
    fn matrix(m: &Mat) -> opencv::Result<[[f64; 3]; 3]> {
        let mut a = [[0.; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = *m.at_2d::<f64>(i as i32, j as i32)?
            }
        }
        Ok(a)
    }
    fn matvec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        m.map(|r| dot(r, v))
    }
    fn transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
    }
    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn ray(p: [f64; 2], f: f64) -> [f64; 3] {
        [(p[0] - 4000.) / f, (p[1] - 3000.) / f, 1.]
    }
    fn sampson(e: [[f64; 3]; 3], a: [f64; 2], b: [f64; 2], f: f64) -> f64 {
        let x = ray(a, f);
        let y = ray(b, f);
        let ex = matvec(e, x);
        let ey = matvec(transpose(e), y);
        dot(y, ex).abs()
            / (ex[0] * ex[0] + ex[1] * ex[1] + ey[0] * ey[0] + ey[1] * ey[1])
                .sqrt()
                .max(1e-14)
            * f
    }
    struct Fit {
        r: [[f64; 3]; 3],
        t: [f64; 3],
        rotation_vector_deg: [f64; 3],
        e: [[f64; 3]; 3],
        inliers: usize,
        positive: usize,
        errors: Vec<f64>,
        depths: Vec<f64>,
    }
    fn fit(ts: &[Track], f: f64, threshold: f64) -> Result<Fit, Err> {
        if ts.len() < 8 {
            return Err("fewer than 8 measured points".into());
        }
        let a: Vector<Point2d> = ts.iter().map(|p| Point2d::new(p.a[0], p.a[1])).collect();
        let b: Vector<Point2d> = ts.iter().map(|p| Point2d::new(p.b[0], p.b[1])).collect();
        let k = Mat::from_slice_2d(&[[f, 0., 4000.], [0., f, 3000.], [0., 0., 1.]])?;
        core::set_rng_seed(19)?;
        let mut mask = Mat::default();
        let e = calib3d::find_essential_mat(
            &a,
            &b,
            &k,
            calib3d::RANSAC,
            0.999,
            threshold,
            4000,
            &mut mask,
        )?;
        if e.rows() != 3 || e.cols() != 3 {
            return Err("essential matrix was missing or multiple".into());
        }
        let inliers = core::count_non_zero(&mask)? as usize;
        let (mut r, mut t, mut triangulated) = (Mat::default(), Mat::default(), Mat::default());
        // Default 50-baseline distance gate would discard nearly all tiny-motion
        // eye points. Retain distant points for diagnostics; do NOT call them stable.
        let positive = calib3d::recover_pose_triangulated(
            &e,
            &a,
            &b,
            &k,
            &mut r,
            &mut t,
            1e6,
            &mut mask,
            &mut triangulated,
        )? as usize;
        let mut rv = Mat::default();
        calib3d::rodrigues_def(&r, &mut rv)?;
        let t = [
            *t.at_2d::<f64>(0, 0)?,
            *t.at_2d::<f64>(1, 0)?,
            *t.at_2d::<f64>(2, 0)?,
        ];
        let r = matrix(&r)?;
        let e = matrix(&e)?;
        let rotation_vector_deg = [
            *rv.at_2d::<f64>(0, 0)?,
            *rv.at_2d::<f64>(1, 0)?,
            *rv.at_2d::<f64>(2, 0)?,
        ]
        .map(f64::to_degrees);
        let errors = ts.iter().map(|p| sampson(e, p.a, p.b, f)).collect();
        let mut depths = vec![];
        if triangulated.rows() == 4 {
            for j in 0..triangulated.cols() {
                if *mask.at_2d::<u8>(j, 0)? == 0 {
                    continue;
                }
                let w = *triangulated.at_2d::<f64>(3, j)?;
                if w.abs() > 1e-12 {
                    depths.push(*triangulated.at_2d::<f64>(2, j)? / w);
                }
            }
        }
        Ok(Fit {
            r,
            t,
            rotation_vector_deg,
            e,
            inliers,
            positive,
            errors,
            depths,
        })
    }
    fn fit_json(f: &Fit) -> Value {
        json!({"rotation_matrix":f.r,"rotation_vector_degrees":f.rotation_vector_deg,"rotation_angle_degrees":norm(f.rotation_vector_deg),"translation_unit_direction":f.t,"ransac_inliers":f.inliers,"positive_depth_inliers":f.positive,"median_sampson_px":median(f.errors.clone()),"p90_sampson_px":quantile(f.errors.clone(),0.9),"per_point_sampson_px":f.errors,"median_positive_depth_in_baseline_units":median(f.depths.clone())})
    }
    fn rotation_difference(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
        let trace: f64 = a
            .into_iter()
            .flatten()
            .zip(b.into_iter().flatten())
            .map(|(x, y)| x * y)
            .sum();
        ((trace - 1.) * 0.5).clamp(-1., 1.).acos().to_degrees()
    }
    fn direction_difference(a: [f64; 3], b: [f64; 3]) -> f64 {
        (dot(a, b) / (norm(a) * norm(b)))
            .clamp(-1., 1.)
            .acos()
            .to_degrees()
    }
    fn rigid2d(train: &[Track], test: &[Track]) -> Value {
        let n = train.len() as f64;
        let ca = [
            train.iter().map(|p| p.a[0]).sum::<f64>() / n,
            train.iter().map(|p| p.a[1]).sum::<f64>() / n,
        ];
        let cb = [
            train.iter().map(|p| p.b[0]).sum::<f64>() / n,
            train.iter().map(|p| p.b[1]).sum::<f64>() / n,
        ];
        let (mut aa, mut bb) = (0., 0.);
        for p in train {
            let x = [p.a[0] - ca[0], p.a[1] - ca[1]];
            let y = [p.b[0] - cb[0], p.b[1] - cb[1]];
            aa += x[0] * y[0] + x[1] * y[1];
            bb += x[0] * y[1] - x[1] * y[0];
        }
        let angle = bb.atan2(aa);
        let (s, c) = angle.sin_cos();
        let errors = test
            .iter()
            .map(|p| {
                let x = [p.a[0] - ca[0], p.a[1] - ca[1]];
                (cb[0] + c * x[0] - s * x[1] - p.b[0]).hypot(cb[1] + s * x[0] + c * x[1] - p.b[1])
            })
            .collect::<Vec<_>>();
        json!({"rotation_degrees":angle.to_degrees(),"source_centroid":ca,"current_centroid":cb,"translation_at_track_centroid_px":[cb[0]-ca[0],cb[1]-ca[1]],"evaluation_median_euclidean_px":median(errors.clone()),"evaluation_errors_px":errors})
    }
    fn rng(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / (1u64 << 53) as f64
    }
    fn radius(p: [f64; 2], e: &Value) -> f64 {
        let x = p[0] - e["center_sensor_px"][0].as_f64().unwrap();
        let y = p[1] - e["center_sensor_px"][1].as_f64().unwrap();
        let (s, c) = e["angle"].as_f64().unwrap().sin_cos();
        ((c * x + s * y) / e["a"].as_f64().unwrap())
            .hypot((-s * x + c * y) / e["b"].as_f64().unwrap())
    }
    fn analyze(ts: &[Track]) -> Result<Value, Err> {
        analyze_with_jitter(ts, 0.25)
    }
    fn analyze_with_jitter(ts: &[Track], jitter: f64) -> Result<Value, Err> {
        let base = fit(ts, 4000., 1.)?;
        let mut variants = vec![];
        for focal in [3500., 4000., 4500.] {
            for threshold in [0.5, 1., 2.] {
                let result = fit(ts, focal, threshold);
                variants.push(match result{Ok(x)=>json!({"kind":"intrinsics_and_ransac_threshold","focal_px":focal,"threshold_px":threshold,"fit":fit_json(&x),"rotation_difference_degrees":rotation_difference(base.r,x.r),"translation_direction_difference_degrees":direction_difference(base.t,x.t)}),Err(e)=>json!({"error":e.to_string()})});
            }
        }
        // Bounded perturbation sensitivity, not a confidence interval: coordinate
        // uncertainty is unknown. Uniform +/-0.25 native px in each exposure.
        for seed in 0..20 {
            let mut state = seed + 100;
            let mut perturbed = ts.to_vec();
            for p in &mut perturbed {
                for v in p.a.iter_mut().chain(p.b.iter_mut()) {
                    *v += (rng(&mut state) - 0.5) * 2. * jitter;
                }
            }
            variants.push(match fit(&perturbed,4000.,1.){Ok(x)=>json!({"kind":"coordinate_perturbation","seed":seed,"jitter_bound_px":jitter,"fit":fit_json(&x),"rotation_difference_degrees":rotation_difference(base.r,x.r),"translation_direction_difference_degrees":direction_difference(base.t,x.t)}),Err(e)=>json!({"kind":"coordinate_perturbation","error":e.to_string()})});
        }
        let mut leave_one_out = vec![];
        for i in 0..ts.len() {
            let train: Vec<_> = ts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| p.clone())
                .collect();
            let held = &ts[i];
            let baseline = rigid2d(&train, std::slice::from_ref(held));
            let fitted = fit(&train, 4000., 1.);
            leave_one_out.push(match fitted{Ok(x)=>json!({"heldout_id":held.id,"heldout_sampson_px":sampson(x.e,held.a,held.b,4000.),"fit":fit_json(&x),"rigid_2d_baseline":baseline,"rotation_difference_degrees":rotation_difference(base.r,x.r),"translation_direction_difference_degrees":direction_difference(base.t,x.t)}),Err(e)=>json!({"heldout_id":held.id,"error":e.to_string(),"rigid_2d_baseline":baseline})});
        }
        let displacement = ts
            .iter()
            .map(|p| (p.b[0] - p.a[0]).hypot(p.b[1] - p.a[1]))
            .collect::<Vec<_>>();
        Ok(
            json!({"point_count":ts.len(),"baseline":fit_json(&base),"rigid_2d":rigid2d(ts,ts),"median_observed_displacement_px":median(displacement),"sensitivity":variants,"leave_one_out":leave_one_out,"tracks":ts.iter().map(|p|json!({"id":p.id,"previous":p.a,"current":p.b})).collect::<Vec<_>>() }),
        )
    }
    pub fn run() -> Result<(), Err> {
        let args: Vec<_> = std::env::args().collect();
        if args.len() == 4 && args[1] == "--multiframe" {
            return crate::factorization::run(&args[2], Path::new(&args[3]));
        }
        if args.len() == 5 && args[1] == "--motion-series" {
            return motion_series(&args[2], &args[3], Path::new(&args[4]));
        }
        if (args.len() == 4 || args.len() == 5) && args[1] == "--iris-features" {
            return iris_features(
                &args[2],
                Path::new(&args[3]),
                args.get(4).map(String::as_str),
            );
        }
        if args.len() != 4 {
            return Err(
                "usage: buttercup_outer_rigid_motion MOTION_JSON POSES_JSON NEW_OUTPUT".into(),
            );
        }
        core::set_num_threads(1)?;
        let out = Path::new(&args[3]);
        if out.exists() {
            return Err("output must be new".into());
        }
        if !out
            .parent()
            .ok_or("missing parent")?
            .canonicalize()?
            .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
        {
            return Err("use checked bulk outputs".into());
        }
        fs::create_dir(out)?;
        let motion: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
        let poses: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
        let pair = motion["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| {
                p["capture"] == "later-recording" && p["eye"] == 2 && p["previous_sequence"] == 1034
            })
            .ok_or("missing selected pair")?;
        let find = |seq: u64| {
            poses["frames"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| {
                    p["selection"]["capture"] == "later-recording"
                        && p["eye"] == 2
                        && p["sequence"] == seq
                })
                .unwrap()
        };
        let previous = find(1034);
        let current = find(1035);
        let mut sets = vec![];
        for name in ["general_layer", "geometrically_outer"] {
            let ts: Vec<_> = pair["tracks"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|t| {
                    let a = std::array::from_fn(|i| t["previous_sensor"][i].as_f64().unwrap());
                    let b = std::array::from_fn(|i| t["current_sensor"][i].as_f64().unwrap());
                    let accept = if name == "general_layer" {
                        t["layer"] == 0
                    } else {
                        t["layer"] != 2
                            && radius(a, &previous["ellipse"]) > 1.15
                            && radius(b, &current["ellipse"]) > 1.15
                    };
                    accept.then(|| Track {
                        id: t["id"].as_u64().unwrap(),
                        a,
                        b,
                    })
                })
                .collect();
            let mut result = analyze(&ts)
                .unwrap_or_else(|e| json!({"error":e.to_string(),"point_count":ts.len()}));
            result["selection"] = json!(name);
            sets.push(result);
        }
        let report = json!({"source":"later-recording eye2 1034->1035","dt_ms":pair["dt_ms"],"raw_sha256":pair["raw_sha256"],"scope":"calibrated essential-matrix rigid-motion fit; X2=R*X1+t; camera axes x right y down z away; t is unit-baseline direction, NOT metric displacement","assumptions":"same rigid material, fx=fy4000,cx4000,cy3000, no lens distortion; neither rigidity nor depths nor intrinsics validated","sets":sets});
        fs::write(
            out.join("results.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        let mut svg=String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1300' height='860'><rect width='1300' height='860' fill='#161922'/><g fill='white' font-family='sans-serif'><text x='25' y='35' font-size='25'>Outer-track 3D rigid motion — source 1034 → 1035, eye 2</text>");
        for (i, set) in report["sets"].as_array().unwrap().iter().enumerate() {
            let y = 85 + i * 350;
            let base = &set["baseline"];
            write!(
                svg,
                "<text x='25' y='{y}' font-size='20'>{}: {} tracks</text>",
                set["selection"].as_str().unwrap(),
                set["point_count"]
            )?;
            let lines = [
                format!(
                    "Rotation vector [degrees]: {}",
                    base["rotation_vector_degrees"]
                ),
                format!(
                    "Translation direction [unit length]: {}",
                    base["translation_unit_direction"]
                ),
                format!(
                    "Inliers: {} · positive depth: {} · median epipolar error: {} px",
                    base["ransac_inliers"],
                    base["positive_depth_inliers"],
                    base["median_sampson_px"]
                ),
                format!(
                    "2D rigid motion: {} degrees; centroid shift {} px",
                    set["rigid_2d"]["rotation_degrees"],
                    set["rigid_2d"]["translation_at_track_centroid_px"]
                ),
                format!(
                    "Rotation change under ±0.25px perturbations: up to {:.2} degrees",
                    set["sensitivity"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|x| x["kind"] == "coordinate_perturbation")
                        .filter_map(|x| x["rotation_difference_degrees"].as_f64())
                        .fold(0., f64::max)
                ),
                format!(
                    "Translation-direction change under perturbations: up to {:.2} degrees",
                    set["sensitivity"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|x| x["kind"] == "coordinate_perturbation")
                        .filter_map(|x| x["translation_direction_difference_degrees"].as_f64())
                        .fold(0., f64::max)
                ),
            ];
            for (j, line) in lines.iter().enumerate() {
                write!(
                    svg,
                    "<text x='25' y='{}' font-size='16'>{line}</text>",
                    y + 35 + j * 30
                )?;
            }
        }
        svg.push_str("<text x='25' y='795' font-size='17'>A numerical fit is not a unique or reliable 3D motion measurement. No millimetre translation without depth.</text><text x='25' y='825' font-size='17'>Perturbations and leave-one-out checks measure sensitivity, not calibrated confidence. See results.json.</text></g></svg>");
        fs::write(out.join("fit-summary.svg"), svg)?;
        crate::stabilized::render(previous, current, &report["sets"][1]["tracks"], out)?;
        println!("{}",serde_json::to_string_pretty(&report["sets"].as_array().unwrap().iter().map(|s|json!({"selection":s["selection"],"point_count":s["point_count"],"baseline":s["baseline"],"rigid_2d":s["rigid_2d"]})).collect::<Vec<_>>())?);
        Ok(())
    }

    fn motion_series(input: &str, cohorts: &str, out: &Path) -> Result<(), Err> {
        if out.exists() {
            return Err("output must be new".into());
        }
        if !out
            .parent()
            .ok_or("output parent missing")?
            .canonicalize()?
            .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
        {
            return Err("use checked bulk outputs".into());
        }
        core::set_num_threads(1)?;
        let root: Value = serde_json::from_slice(&fs::read(input)?)?;
        let groups: Value = serde_json::from_slice(&fs::read(cohorts)?)?;
        let s = root["series"]
            .as_array()
            .ok_or("missing series")?
            .iter()
            .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
            .ok_or("missing selected series")?;
        let frames = s["frames"].as_array().unwrap();
        let source = frames
            .iter()
            .position(|f| f["sequence"] == s["source_sequence"])
            .ok_or("missing seed")?;
        let ids = groups["groups"][0]["ids"]
            .as_array()
            .ok_or("missing iris cohort")?;
        let reference = [
            frames[source]["input"]["frame"]["sensor_x"]
                .as_f64()
                .unwrap()
                + frames[source]["input"]["frame"]["width"].as_f64().unwrap() / 2.,
            frames[source]["input"]["frame"]["sensor_y"]
                .as_f64()
                .unwrap()
                + frames[source]["input"]["frame"]["height"].as_f64().unwrap() / 2.,
        ];
        let mut rows = vec![];
        for (j, f) in frames.iter().enumerate() {
            assert_eq!(f["raw_sha256"], groups["frames"][j]["raw_sha256"]);
            let mut row = json!({"sequence":f["sequence"],"raw_sha256":f["raw_sha256"],"source_sequence":frames[source]["sequence"],"reference":j==source});
            for (name, field) in [("outer", "outer_tracks"), ("iris", "tracks")] {
                let tracks = s[field]
                    .as_array()
                    .ok_or("missing point tracks")?
                    .iter()
                    .filter(|t| name == "outer" || ids.contains(&t["id"]))
                    .filter(|t| {
                        t["frames"][source]["accepted"] == true
                            && t["frames"][j]["accepted"] == true
                    })
                    .map(|t| Track {
                        id: t["id"].as_u64().unwrap(),
                        a: std::array::from_fn(|a| {
                            t["frames"][source]["sensor"][a].as_f64().unwrap()
                        }),
                        b: std::array::from_fn(|a| t["frames"][j]["sensor"][a].as_f64().unwrap()),
                    })
                    .collect::<Vec<_>>();
                let mut value = if j == source {
                    json!({"reference":true})
                } else {
                    analyze_with_jitter(&tracks, if name == "outer" { 1. } else { 0.25 })
                        .unwrap_or_else(|e| json!({"error":e.to_string()}))
                };
                value["point_count"] = json!(tracks.len());
                value["ids"] = json!(tracks.iter().map(|t| t.id).collect::<Vec<_>>());
                if tracks.len() >= 3 {
                    let planar = rigid2d(&tracks, &tracks);
                    let theta = planar["rotation_degrees"].as_f64().unwrap().to_radians();
                    let ca: [f64; 2] =
                        std::array::from_fn(|a| planar["source_centroid"][a].as_f64().unwrap());
                    let cb: [f64; 2] =
                        std::array::from_fn(|a| planar["current_centroid"][a].as_f64().unwrap());
                    let d = [reference[0] - ca[0], reference[1] - ca[1]];
                    value["motion_at_fixed_roi_center_px"] = json!([
                        cb[0] + theta.cos() * d[0] - theta.sin() * d[1] - reference[0],
                        cb[1] + theta.sin() * d[0] + theta.cos() * d[1] - reference[1]
                    ]);
                    value["rigid_2d"] = planar;
                }
                if name == "outer" {
                    let top = tracks.iter().filter(|t| t.a[1] < reference[1]).count();
                    value["top_count"] = json!(top);
                    value["bottom_count"] = json!(tracks.len() - top);
                }
                row[name] = value;
            }
            println!(
                "source {}: outer {} points, r={} t={}; iris r={} t={}",
                f["sequence"],
                row["outer"]["point_count"],
                row["outer"]["baseline"]["rotation_vector_degrees"],
                row["outer"]["baseline"]["translation_unit_direction"],
                row["iris"]["baseline"]["rotation_vector_degrees"],
                row["iris"]["baseline"]["translation_unit_direction"]
            );
            rows.push(row);
        }
        let report = json!({"input":input,"cohorts":cohorts,"source_index":source,"frames":rows,"fixed_roi_center_sensor":reference,"outer_band_tracking":s["outer_band_tracking"],"scope":"Separate source-to-current essential-matrix candidate solves, in original sensor coordinates; Xcurrent=R Xsource+t. t is unit direction with unknown metric length, not a trajectory in cm. No global affine applied to either input. No 3D subtraction between independently scaled groups.","assumptions":"pinhole fx=fy4000,cx4000,cy3000,no distortion; conditional rigid correspondence hypothesis. No measured material identities, depths or metric scale. Sensitivity probes are not probabilities."});
        fs::create_dir(out)?;
        fs::write(
            out.join("results.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        Ok(())
    }

    fn homography(ts: &[Track]) -> Result<Value, Err> {
        if ts.len() < 8 {
            return Err("fewer than 8 points for a checked planar fit".into());
        }
        let a: Vector<Point2d> = ts.iter().map(|p| Point2d::new(p.a[0], p.a[1])).collect();
        let b: Vector<Point2d> = ts.iter().map(|p| Point2d::new(p.b[0], p.b[1])).collect();
        let mut mask = Mat::default();
        core::set_rng_seed(19)?;
        let h = calib3d::find_homography_ext(&a, &b, calib3d::RANSAC, 1., &mut mask, 4000, 0.999)?;
        if h.rows() != 3 {
            return Err("homography missing".into());
        }
        let hm = matrix(&h)?;
        let transfer = |m: [[f64; 3]; 3], p: &Track| {
            let q = matvec(m, [p.a[0], p.a[1], 1.]);
            (q[0] / q[2] - p.b[0]).hypot(q[1] / q[2] - p.b[1])
        };
        let errors = ts.iter().map(|p| transfer(hm, p)).collect::<Vec<_>>();
        let k = Mat::from_slice_2d(&[[4000., 0., 4000.], [0., 4000., 3000.], [0., 0., 1.]])?;
        let (mut rs, mut trans, mut normals) = (
            Vector::<Mat>::new(),
            Vector::<Mat>::new(),
            Vector::<Mat>::new(),
        );
        calib3d::decompose_homography_mat(&h, &k, &mut rs, &mut trans, &mut normals)?;
        let mut candidates = vec![];
        for i in 0..rs.len() {
            let r = rs.get(i)?;
            let t = trans.get(i)?;
            let nn = normals.get(i)?;
            let tv = [
                *t.at_2d::<f64>(0, 0)?,
                *t.at_2d::<f64>(1, 0)?,
                *t.at_2d::<f64>(2, 0)?,
            ];
            let nv = [
                *nn.at_2d::<f64>(0, 0)?,
                *nn.at_2d::<f64>(1, 0)?,
                *nn.at_2d::<f64>(2, 0)?,
            ];
            let rm = matrix(&r)?;
            let mut rv = Mat::default();
            calib3d::rodrigues_def(&r, &mut rv)?;
            let rot = [
                *rv.at_2d::<f64>(0, 0)?,
                *rv.at_2d::<f64>(1, 0)?,
                *rv.at_2d::<f64>(2, 0)?,
            ]
            .map(f64::to_degrees);
            let mut positive = 0;
            let mut depths = vec![];
            for p in ts {
                let ray = ray(p.a, 4000.);
                let den = dot(nv, ray);
                if den.abs() < 1e-12 {
                    continue;
                }
                let point = ray.map(|v| v / den);
                let q = matvec(rm, point);
                if point[2] > 0. && q[2] + tv[2] > 0. {
                    positive += 1;
                    depths.push(point[2]);
                }
            }
            candidates.push(json!({"index":i,"rotation_vector_deg":rot,"rotation_angle_deg":norm(rot),"translation_over_plane_distance":tv,"normal_first_camera":nv,"positive_depth_points":positive,"median_positive_depth_over_plane_distance":median(depths),"note":"plane n dot X = d; translation is t/d, no metric scale"}));
        }
        let mut heldout = vec![];
        for i in 0..ts.len() {
            let aa: Vector<Point2d> = ts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| Point2d::new(p.a[0], p.a[1]))
                .collect();
            let bb: Vector<Point2d> = ts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| Point2d::new(p.b[0], p.b[1]))
                .collect();
            let mut m = Mat::default();
            core::set_rng_seed(19)?;
            match calib3d::find_homography_ext(&aa, &bb, calib3d::RANSAC, 1., &mut m, 4000, 0.999) {
                Ok(h) if h.rows() == 3 => heldout
                    .push(json!({"id":ts[i].id,"transfer_error_px":transfer(matrix(&h)?,&ts[i])})),
                _ => heldout.push(json!({"id":ts[i].id,"error":"fit failed"})),
            }
        }
        Ok(
            json!({"ransac_inliers":core::count_non_zero(&mask)?,"matrix_first_to_second":hm,"median_transfer_error_px":median(errors),"decompositions":candidates,"leave_one_out":heldout,"median_heldout_transfer_px":median(heldout.iter().filter_map(|v|v["transfer_error_px"].as_f64()).collect())}),
        )
    }

    fn iris_features(input: &str, out: &Path, cohorts: Option<&str>) -> Result<(), Err> {
        core::set_num_threads(1)?;
        if out.exists() {
            return Err("output must be new".into());
        }
        if !out
            .parent()
            .ok_or("output parent missing")?
            .canonicalize()?
            .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
        {
            return Err("use checked bulk outputs".into());
        }
        fs::create_dir(out)?;
        let root: Value = serde_json::from_slice(&fs::read(input)?)?;
        let series = root["series"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["capture"] == "later-recording" && v["eye"] == 2)
            .ok_or("selected series missing")?;
        let frame_index = |seq: u64| {
            series["frames"]
                .as_array()
                .unwrap()
                .iter()
                .position(|f| f["sequence"] == seq)
                .unwrap()
        };
        let source = frame_index(1034);
        let next = frame_index(1035);
        let rows = series["tracks"].as_array().unwrap();
        let coords = |v: &Value| [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()];
        let mut sets = vec![];
        for selection in [
            "all_38",
            "bright_support",
            "nonbright_support",
            "survive_all_five",
            "reverse_error_at_most_half_pixel",
        ] {
            let ts = rows
                .iter()
                .filter(|t| t["frames"][next]["accepted"] == true)
                .filter(|t| match selection {
                    "bright_support" => t["kind"] == "bright/reflection",
                    "nonbright_support" => t["kind"] != "bright/reflection",
                    "survive_all_five" => t["frames"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|v| v["accepted"] == true),
                    "reverse_error_at_most_half_pixel" => {
                        t["frames"][next]["forward_back_px"].as_f64().unwrap() <= 0.5
                    }
                    _ => true,
                })
                .map(|t| Track {
                    id: t["id"].as_u64().unwrap(),
                    a: coords(&t["frames"][source]["sensor"]),
                    b: coords(&t["frames"][next]["sensor"]),
                })
                .collect::<Vec<_>>();
            let mut result = analyze(&ts)
                .unwrap_or_else(|e| json!({"error":e.to_string(),"point_count":ts.len()}));
            result["selection"] = json!(selection);
            result["planar_fit"] =
                homography(&ts).unwrap_or_else(|e| json!({"error":e.to_string()}));
            println!(
                "{selection}: {} points, rotation {}, translation {}",
                ts.len(),
                result["baseline"]["rotation_vector_degrees"],
                result["baseline"]["translation_unit_direction"]
            );
            sets.push(result);
        }
        if let Some(path) = cohorts {
            let groups: Value = serde_json::from_slice(&fs::read(path)?)?;
            assert_eq!(
                groups["frames"][source]["raw_sha256"],
                series["frames"][source]["raw_sha256"]
            );
            assert_eq!(
                groups["frames"][next]["raw_sha256"],
                series["frames"][next]["raw_sha256"]
            );
            for kind in ["all_four_increment_groups", "first_two_increment_groups"] {
                for (i, g) in groups["variants"][0][kind]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    let ids = g["ids"].as_array().unwrap();
                    let ts = rows
                        .iter()
                        .filter(|t| ids.contains(&t["id"]) && t["frames"][next]["accepted"] == true)
                        .map(|t| Track {
                            id: t["id"].as_u64().unwrap(),
                            a: coords(&t["frames"][source]["sensor"]),
                            b: coords(&t["frames"][next]["sensor"]),
                        })
                        .collect::<Vec<_>>();
                    let mut result = analyze(&ts)
                        .unwrap_or_else(|e| json!({"error":e.to_string(),"point_count":ts.len()}));
                    result["selection"] = json!(format!("motion_group_{kind}_{}", i + 1));
                    result["planar_fit"] =
                        homography(&ts).unwrap_or_else(|e| json!({"error":e.to_string()}));
                    println!(
                        "{}: {} points, rotation {}, translation {}",
                        result["selection"],
                        ts.len(),
                        result["baseline"]["rotation_vector_degrees"],
                        result["baseline"]["translation_unit_direction"]
                    );
                    sets.push(result);
                }
            }
        }
        let mut temporal = vec![];
        for (a, b) in [
            (1032, 1033),
            (1033, 1034),
            (1034, 1035),
            (1035, 1036),
            (1032, 1036),
        ] {
            let ai = frame_index(a);
            let bi = frame_index(b);
            // Same persistent identities in every temporal fit.
            let ts = rows
                .iter()
                .filter(|t| {
                    t["frames"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|v| v["accepted"] == true)
                })
                .map(|t| Track {
                    id: t["id"].as_u64().unwrap(),
                    a: coords(&t["frames"][ai]["sensor"]),
                    b: coords(&t["frames"][bi]["sensor"]),
                })
                .collect::<Vec<_>>();
            let result = fit(&ts, 4000., 1.)
                .map(|v| fit_json(&v))
                .unwrap_or_else(|e| json!({"error":e.to_string()}));
            temporal.push(json!({"sequences":[a,b],"points":ts.len(),"fit":result,"rigid2d":rigid2d(&ts,&ts),"planar_fit":homography(&ts).unwrap_or_else(|e|json!({"error":e.to_string()}))}));
        }
        let report = json!({"input":input,"source":"later-recording eye2 1034->1035; source-selected image features","raw_sha256":[series["frames"][source]["raw_sha256"],series["frames"][next]["raw_sha256"]],"scope":"X2=R X1+t, x right y down z away; essential translation has unit length and unknown metric scale; 2D image matches, not premeasured 3D points","assumptions":"rigid correspondences, pinhole K fx=fy4000,cx4000,cy3000, no distortion; bright/context patches may violate material rigidity; no anatomical prior or metric scale used","sets":sets,"persistent_temporal":temporal});
        fs::write(
            out.join("results.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        let all = &report["sets"][0];
        let base = &all["baseline"];
        let jit = all["sensitivity"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v["kind"] == "coordinate_perturbation")
            .collect::<Vec<_>>();
        let vecstr = |v: &Value| {
            v.as_array()
                .map(|v| {
                    format!(
                        "[{}]",
                        v.iter()
                            .map(|x| format!("{:+.3}", x.as_f64().unwrap()))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .unwrap_or_else(|| "unavailable".into())
        };
        let mut svg="<svg xmlns='http://www.w3.org/2000/svg' width='1300' height='1200'><rect width='100%' height='100%' fill='#171b24'/><g fill='white' font-family='sans-serif'><text x='25' y='36' font-size='25'>Can 38 image matches identify a 3D rigid motion?</text>".to_string();
        let lines=[format!("Baseline rotation vector (degrees): {}",vecstr(&base["rotation_vector_degrees"])),format!("Baseline translation direction (unit length, NOT millimeters): {}",vecstr(&base["translation_unit_direction"])),format!("RANSAC inliers: {} / 38; positive-depth support: {} / 38",base["ransac_inliers"],base["positive_depth_inliers"]),format!("Median epipolar error: {:.3}px; 2D rigid prediction error: {:.3}px",base["median_sampson_px"].as_f64().unwrap(),all["rigid_2d"]["evaluation_median_euclidean_px"].as_f64().unwrap()),format!("Under ±0.25px coordinate perturbations: rotation changes up to {:.2}°, direction up to {:.2}°",jit.iter().filter_map(|v|v["rotation_difference_degrees"].as_f64()).fold(0.,f64::max),jit.iter().filter_map(|v|v["translation_direction_difference_degrees"].as_f64()).fold(0.,f64::max))];
        for (i, line) in lines.iter().enumerate() {
            write!(
                svg,
                "<text x='25' y='{}' font-size='17'>{line}</text>",
                80 + i * 32
            )?;
        }
        svg.push_str("<text x='25' y='280' font-size='21'>Planar homography decompositions (same 2D warp)</text>");
        for (i, p) in all["planar_fit"]["decompositions"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            write!(svg,"<text x='25' y='{}' font-size='16'>#{}: rotation {:.3}°; t/d {}; positive depth {} / 38</text>",315+i*35,i,p["rotation_angle_deg"].as_f64().unwrap(),vecstr(&p["translation_over_plane_distance"]),p["positive_depth_points"])?;
        }
        svg.push_str("<text x='25' y='490' font-size='21'>Same 28 persistent identities across neighboring exposures</text>");
        for (i, p) in report["persistent_temporal"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            write!(
                svg,
                "<text x='25' y='{}' font-size='16'>{}: rotation {}; t-direction {}</text>",
                530 + i * 38,
                p["sequences"],
                vecstr(&p["fit"]["rotation_vector_degrees"]),
                vecstr(&p["fit"]["translation_unit_direction"])
            )?;
        }
        svg.push_str(
            "<text x='25' y='760' font-size='21'>After selecting coherent motion groups</text>",
        );
        for (i, g) in report["sets"]
            .as_array()
            .unwrap()
            .iter()
            .skip(5)
            .enumerate()
        {
            let jitter = g["sensitivity"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|v| v["kind"] == "coordinate_perturbation");
            let max_angle = jitter
                .clone()
                .filter_map(|v| v["rotation_difference_degrees"].as_f64())
                .fold(0., f64::max);
            let max_t = jitter
                .filter_map(|v| v["translation_direction_difference_degrees"].as_f64())
                .fold(0., f64::max);
            write!(svg,"<text x='25' y='{}' font-size='17'>{} points: rotation {}; jitter changes R up to {:.2}°, t direction {:.2}°</text>",805+i*38,g["point_count"],vecstr(&g["baseline"]["rotation_vector_degrees"]),max_angle,max_t)?;
        }
        svg.push_str("<text x='25' y='950' font-size='18'>Small image residuals do not establish a stable or unique 3D motion. Patches overlap and include bright context.</text><text x='25' y='992' font-size='18'>Epipolar error is a one-dimensional constraint; it is not directly comparable to a two-dimensional prediction error.</text><text x='25' y='1034' font-size='18'>Assumed intrinsics; no metric scale. Perturbation ranges are sensitivity checks, not calibrated confidence intervals.</text><text x='25' y='1090' font-size='18'>RAW overlays and patch correspondences: iris-feature-motion-20260916-ready. Full candidate details: results.json.</text></g></svg>");
        fs::write(out.join("fit-summary.svg"), svg)?;
        Ok(())
    }
}
