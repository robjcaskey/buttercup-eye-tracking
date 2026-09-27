//! Source-matched vessel-front-end assay; landmarks are proposals, not tissue
//! labels or validated physical identities. No learned assets are consumed.
use super::{
    json, motion, point, ppm, quantiles, raw_preview, read_rows, BundleSource, Error, Value,
};
use buttercup_eye_tracking::{
    raw10, raw_sclera_red_canny::ScleraRedCannyTracker,
    raw_sclera_vein_graph::ScleraVeinGraphTracker,
};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};

fn seed(v: &Value) -> Option<motion::IrisEllipseSeed> {
    Some(motion::IrisEllipseSeed {
        center: (v["center"][0].as_f64()?, v["center"][1].as_f64()?),
        major_radius: v["major"].as_f64()?,
        minor_radius: v["minor"].as_f64()?,
        angle: v["angle"].as_f64()?,
    })
}

pub fn run(args: &[String]) -> Result<(), Error> {
    if args.len() != 5 {
        return Err("usage: --vessel-assay BUNDLE MATCHED_REPLAY NEW_OUTPUT".into());
    }
    let bundle = BundleSource::open(Path::new(&args[2]))?;
    let rows = read_rows(&Path::new(&args[3]).join("frames.jsonl"))?;
    let out = Path::new(&args[4]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output needs parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("use new checked runtime output".into());
    }
    fs::create_dir(out)?;
    let mut trackers: [[ScleraVeinGraphTracker; 2]; 2] = Default::default();
    let mut red_trackers: [ScleraRedCannyTracker; 2] = Default::default();
    let mut counts = [0usize; 2];
    let mut elapsed = Vec::new();
    let mut ridge_elapsed = Vec::new();
    let mut proposal_counts: [Vec<f64>; 5] = Default::default();
    let mut near_generic_counts: [usize; 2] = [0; 2];
    let mut log = fs::File::create(out.join("frames.jsonl"))?;
    for r in rows {
        let s = &r["source"];
        let u = |k: &str| s[k].as_u64().ok_or("missing source field");
        let eye = match u("eye_id")? {
            1 => 0,
            2 => 1,
            _ => return Err("unsupported eye id".into()),
        };
        let (w, h, sx, sy, t) = (
            u("width")? as usize,
            u("height")? as usize,
            u("sensor_x")? as u32,
            u("sensor_y")? as u32,
            u("timestamp_ns")?,
        );
        let packed = bundle.read_range(
            s["stream"].as_str().ok_or("missing stream")?,
            u("offset")?,
            u("length")? as usize,
        )?;
        if format!("{:x}", Sha256::digest(&packed))
            != r["raw_sha256"].as_str().ok_or("missing source digest")?
        {
            return Err("RAW identity mismatch".into());
        }
        let raw = raw10::try_unpack_raw10(&packed, w, h, u("stride")? as usize)?;
        if r["reset"] == true {
            red_trackers[eye].clear();
        }
        let red = red_trackers[eye].observe(&raw, w, h, sx, sy, t);
        let ridge_started = std::time::Instant::now();
        let ridges = super::vessel_ridges::detect(&raw, w, h, sx, sy);
        let ridge_ms = ridge_started.elapsed().as_secs_f64() * 1000.;
        ridge_elapsed.push(ridge_ms);
        proposal_counts[2].push(red.anchors.len() as f64);
        proposal_counts[3].push(ridges.len() as f64);
        proposal_counts[4].push(
            r["tensor_points"]
                .as_array()
                .ok_or("missing generic points")?
                .len() as f64,
        );
        let mut variants = Vec::new();
        for (variant, iris) in [seed(&r["seed"]), None].into_iter().enumerate() {
            if r["reset"] == true {
                trackers[eye][variant].clear();
            }
            let v = trackers[eye][variant].observe(&raw, w, h, sx, sy, t, iris);
            elapsed.push(v.elapsed_us as f64 / 1000.);
            proposal_counts[variant].push(v.landmarks.len() as f64);
            let points=v.landmarks.iter().map(|p| {
                let sensor=[p.point[0]+sx as f32,p.point[1]+sy as f32];
                let nearest_generic=r["tensor_points"].as_array().unwrap().iter().filter_map(|m| {
                    Some((m["current_sensor"][0].as_f64()?-sensor[0] as f64).hypot(m["current_sensor"][1].as_f64()?-sensor[1] as f64))
                }).min_by(f64::total_cmp);
                json!({"id":p.id,"sensor":sensor,"degree":p.degree,"branch_distance":p.distance_to_branch_px,
                    "strength":p.strength,"matched":p.matched,"motion":p.motion_px,"nearest_generic_px":nearest_generic})
            }).collect::<Vec<_>>();
            near_generic_counts[variant] += points
                .iter()
                .filter(|p| p["nearest_generic_px"].as_f64().is_some_and(|d| d <= 4.))
                .count();
            variants.push(json!({"iris_exclusion":variant==0,"ridge_cells":v.ridge_cells,"branch_points":v.branch_points,
                "matched":v.matched_landmarks,"stable":v.stable_landmarks,"ms":v.elapsed_us as f64/1000.,"points":points}));
            if counts[eye] % 8 == 0 {
                let mut image = raw_preview::color_preview(&raw, w, h, sx, sy, 100, None);
                for segment in &v.segments {
                    for k in 0..=8 {
                        let a = k as f32 / 8.;
                        point(
                            &mut image,
                            w,
                            h,
                            (
                                (segment.start[0] * (1. - a) + segment.end[0] * a) as f64,
                                (segment.start[1] * (1. - a) + segment.end[1] * a) as f64,
                            ),
                            0x3f8da0,
                            0,
                        );
                    }
                }
                for p in &v.landmarks {
                    point(
                        &mut image,
                        w,
                        h,
                        (p.point[0] as f64, p.point[1] as f64),
                        if p.matched { 0xffcc60 } else { 0xff6090 },
                        1,
                    );
                }
                ppm(
                    &out.join(format!(
                        "eye-{}-{:04}-vessels-{variant}.ppm",
                        eye + 1,
                        counts[eye]
                    )),
                    &image,
                    w,
                    h,
                )?;
            }
        }
        if counts[eye] % 8 == 0 {
            let mut image = raw_preview::color_preview(&raw, w, h, sx, sy, 100, None);
            for p in &red.anchors {
                point(
                    &mut image,
                    w,
                    h,
                    (p.point[0] as f64, p.point[1] as f64),
                    0xff6090,
                    1,
                );
            }
            ppm(
                &out.join(format!("eye-{}-{:04}-red.ppm", eye + 1, counts[eye])),
                &image,
                w,
                h,
            )?;
            let mut image = raw_preview::color_preview(&raw, w, h, sx, sy, 100, None);
            for p in &ridges {
                point(
                    &mut image,
                    w,
                    h,
                    (p.point[0] as f64, p.point[1] as f64),
                    0x40ffc0,
                    1,
                );
            }
            ppm(
                &out.join(format!("eye-{}-{:04}-ridges.ppm", eye + 1, counts[eye])),
                &image,
                w,
                h,
            )?;
        }
        serde_json::to_writer(
            &mut log,
            &json!({"source":s,"raw_sha256":r["raw_sha256"],"target":r["target"],"variants":variants,
                "red":{"edges":red.accepted_edge_cells,"ms":red.elapsed_us as f64/1000.,"points":red.anchors.iter().map(|p|json!({"id":p.id,"sensor":[p.point[0]+sx as f32,p.point[1]+sy as f32],"stable":p.stable,"persistent":p.persistent})).collect::<Vec<_>>()},
                "chromatic_ridges":{"ms":ridge_ms,"points":ridges.iter().map(|p|json!({"sensor":[p.point[0]+sx as f32,p.point[1]+sy as f32],"tangent":p.tangent,"contrast":p.contrast})).collect::<Vec<_>>()}}),
        )?;
        writeln!(log)?;
        counts[eye] += 1;
    }
    let names = [
        "blue_with_iris_exclusion",
        "blue_without_iris_exclusion",
        "red_canny",
        "chromatic_ridges",
        "generic_matched_points",
    ];
    let proposal_counts = names
        .into_iter()
        .zip(proposal_counts)
        .map(|(name, counts)| {
            (
                name.to_owned(),
                json!({"total":counts.iter().sum::<f64>(),"per_frame":quantiles(counts)}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let summary = json!({"bundle":args[2],"reference":args[3],"counts":counts,"blue_detector_ms":quantiles(elapsed),"chromatic_ridge_ms":quantiles(ridge_elapsed),"proposal_counts":proposal_counts,"blue_landmarks_within_4px_of_generic_point":near_generic_counts,
        "scope":"Existing physical-blue graph with/without native iris exclusion, existing red Canny, and experimental phase-aligned chromatic ridges on identical native RAW. Counts are proposals, not vessel recall or localization accuracy. Graph associations are not verified physical identities. Targets are context only and never detector input. Experimental ridges are not connected to live tracking."});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
