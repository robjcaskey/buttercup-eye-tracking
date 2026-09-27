//! Compare native projections on identical RAW supports and independent RAW
//! motion. No candidate radius defines its own image-scale normalization.
use super::*;
use super::playback::{ns,pair};
use crate::geometry::Ellipse;

const KINDS: [&str;3]=["OuterLimbus","InnerLimbus","PupillaryBoundary"];

/// A precision-only comparison must leave every source's MAP and native
/// projected evidence unchanged. More admitted sources are not accuracy truth.
pub(crate) fn compare_precision<I:Iterator<Item=String>>(mut args:I)->Result<(),String> {
    let baseline=PathBuf::from(args.next().ok_or("expected BASE_PLAYBACK CANDIDATE_PLAYBACK NEW_REPORT")?);
    let candidate=PathBuf::from(args.next().ok_or("missing candidate playback")?);
    let output=PathBuf::from(args.next().ok_or("missing new precision report")?);
    if args.next().is_some() {return Err("unexpected precision comparison argument".into());}
    let allowed=fs::canonicalize("outputs").map_err(|e|e.to_string())?;
    if !fs::canonicalize(output.parent().ok_or("report needs a parent")?).map_err(|e|e.to_string())?.starts_with(allowed) {
        return Err("precision report must be beneath outputs".into());
    }
    let reports=[read_json(&baseline)?,read_json(&candidate)?];
    for key in ["schema","session","cache","selected_eye","native_frames","camera_mount_assumption","pupil_evidence_ablation"] {
        if reports[0][key]!=reports[1][key] {return Err(format!("precision arms differ in {key}"));}
    }
    let rows=reports.each_ref().map(|r|read_rows(Path::new(r["bridge"].as_str().ok_or("missing bridge")?)))
        .into_iter().collect::<Result<Vec<_>,String>>()?;
    if rows[0].len()!=rows[1].len() || rows[0].is_empty() {return Err("precision arms have different source coverage".into());}
    let mut ready:[std::collections::BTreeSet<u64>;2]=std::array::from_fn(|_|Default::default());
    let mut samples:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
    let mut fresh=std::collections::BTreeSet::new();
    for (a,b) in rows[0].iter().zip(&rows[1]) {
        for key in ["input_index","source_ns","camera_mount_assumption","intrinsics_diagnostic","radial_camera_diagnostic","integration_recipe","timing_validation"] {
            if a[key]!=b[key] {return Err(format!("precision arms differ in source/configuration {key}"));}
        }
        for key in ["sources","target","eye_centers","surface_normals","gaze_directions","source_projected_conics","arcs","cost"] {
            if a["joint"][key]!=b["joint"][key] {return Err(format!("precision trial changed native geometry {key}"));}
        }
        for (arm,row) in [a,b].into_iter().enumerate() {
            if let Some(source)=super::playback::current_qualified_source(row)? {ready[arm].insert(source);}
        }
        let time=ns(&a["source_ns"])?;
        if a["joint"]["calibration_source_group_complete"]==true
            && a["joint"]["sources"].as_array().is_some_and(|s|s.len()==2 && s.iter().all(|s|ns(&s["sensor_timestamp_ns"]).ok()==Some(time)))
            && fresh.insert(time) {
            for (arm,row) in [a,b].into_iter().enumerate() {
                if let Some(n)=row["joint"]["posterior"]["samples"].as_f64() {samples[arm].push(n);}
            }
        }
    }
    let result=json!({"schema":"buttercup-calibration-precision-comparison-v1","baseline":baseline,"candidate":candidate,
        "native_source_rows":rows[0].len(),"all_map_geometry_and_raw_supports_identical":true,
        "fresh_ready_sources":ready.each_ref().map(|r|r.len()),
        "newly_ready_source_ns":ready[1].difference(&ready[0]).map(u64::to_string).collect::<Vec<_>>(),
        "lost_ready_source_ns":ready[0].difference(&ready[1]).map(u64::to_string).collect::<Vec<_>>(),
        "actual_integration_draws_per_current_pair":samples.map(stats),
        "budget_overrides":rows.iter().map(|r|r[0]["integration_budget_override"].clone()).collect::<Vec<_>>(),
        "native_fits":reports.each_ref().map(|r|r["fit"].clone()),
        "held_target_fits":reports.each_ref().map(|r|r["leave_one_target_out"].clone()),
        "limits":["budgets are maximums; unchanged early stopping does not create an independent fixed-budget reference",
            "conditional numerical support is not measured gaze accuracy; admitted sample changes can alter calibration clusters",
            "fixed recorded stimulus; no new target visits, no end-to-end live timing claim",
            "the enclosing acquisition test uses an unpresented target and does not simulate recorded target progression"]});
    let mut writer=fs::OpenOptions::new().create_new(true).write(true).open(output).map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut writer,&result).map_err(|e|e.to_string())?;
    Ok(())
}

struct SourceShape {
    sequence:u64,
    ellipses:[Option<Ellipse>;3],
    points:[Vec<(f64,f64)>;3],
}

fn shapes(path:&Path, raw:&BTreeMap<(u64,u64),&Value>) -> Result<BTreeMap<(u64,u64),SourceShape>,String> {
    let mut result=BTreeMap::new();
    for row in read_rows(path)? {
        let time=ns(&row["source_ns"])?;let joint=&row["joint"];
        if joint["calibration_source_group_complete"]!=true {continue;}
        let Some(sources)=joint["sources"].as_array() else {continue;};
        if sources.len()!=2 || !sources.iter().all(|s|ns(&s["sensor_timestamp_ns"]).ok()==Some(time)) {continue;}
        for eye in 0..2 {
            let roi=eye as u64+1;let key=(roi,time);
            let input=*raw.get(&key).ok_or("projected source absent from RAW motion ledger")?;
            let frame=&input["frame"];
            let projection=&joint["source_projected_conics"]["eyes"][eye];
            let source=&projection["source"];
            let clock=joint_gaze_live::clock(input["clock_lineage"].as_str().ok_or("RAW clock lineage missing")?);
            if ns(&source["roi_id"])?!=roi || ns(&source["sequence"])?!=ns(&frame["sequence"])?
                || ns(&source["sensor_timestamp_ns"])?!=time || ns(&source["clock_domain"])?!=clock.domain
                || ns(&source["clock_epoch"])?!=clock.epoch
                || pair(&projection["sensor_origin_px"])!=Some((ns(&frame["sensor_x"])? as f64,ns(&frame["sensor_y"])? as f64))
                || pair(&projection["dimensions_px"])!=Some((ns(&frame["width"])? as f64,ns(&frame["height"])? as f64)) {
                return Err("projection does not match its exact native RAW source/ROI/clock".into());
            }
            let ellipses=std::array::from_fn(|i| {
                let v=&projection["ellipses"][i];
                let e=Ellipse {center:pair(&v["center"])?,major_radius:v["major_radius"].as_f64()?,
                    minor_radius:v["minor_radius"].as_f64()?,angle:v["angle_rad"].as_f64()?};
                (e.major_radius.is_finite() && e.minor_radius>0.0 && e.major_radius>=e.minor_radius && e.angle.is_finite()).then_some(e)
            });
            let mut points:[Vec<(f64,f64)>;3]=std::array::from_fn(|_|Vec::new());
            for arc in joint["arcs"].as_array().ok_or("projected native geometry lacks arc ledger")? {
                if arc["used"]!=true || ns(&arc["roi_id"])?!=roi {continue;}
                let index=KINDS.iter().position(|kind|arc["kind"].as_str()==Some(kind)).ok_or("unknown boundary kind")?;
                for point in arc["points_roi_px"].as_array().ok_or("missing RAW support points")? {
                    points[index].push(pair(point).ok_or("invalid RAW support point")?);
                }
            }
            result.entry(key).or_insert(SourceShape {sequence:ns(&frame["sequence"])?,ellipses,points});
        }
    }
    Ok(result)
}

pub(crate) fn compare<I:Iterator<Item=String>>(mut args:I)->Result<(),String> {
    let baseline=PathBuf::from(args.next().ok_or("expected BASE_BRIDGE CANDIDATE_BRIDGE RAW_MOTION NEW_REPORT")?);
    let candidate=PathBuf::from(args.next().ok_or("missing candidate bridge")?);
    let motion_path=PathBuf::from(args.next().ok_or("missing independent RAW motion")?);
    let output=PathBuf::from(args.next().ok_or("missing new geometry report")?);
    if args.next().is_some() {return Err("unexpected geometry comparison argument".into());}
    let allowed=fs::canonicalize("outputs").map_err(|e|e.to_string())?;
    if !fs::canonicalize(output.parent().ok_or("report needs a parent")?).map_err(|e|e.to_string())?.starts_with(allowed) {
        return Err("geometry report must be beneath outputs".into());
    }
    let motion=read_rows(&motion_path)?;
    let mut raw=BTreeMap::new();
    for row in &motion {
        let frame=&row["input"]["frame"];let key=(ns(&frame["eye_id"])?,ns(&frame["timestamp_ns"])?);
        if raw.insert(key,&row["input"]).is_some() {return Err("duplicate native motion source".into());}
    }
    let arms=[shapes(&baseline,&raw)?,shapes(&candidate,&raw)?];
    let mut eyes=Vec::new();
    for roi in 1..=2 {
        let mut residuals:[[Vec<f64>;3];2]=std::array::from_fn(|_|std::array::from_fn(|_|Vec::new()));
        let mut area_steps:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
        let mut regressions=Vec::new();let mut last:Option<(u64,[Option<f64>;2])>=None;
        let mut present=[0;2];let mut count=0;
        for row in motion.iter().filter(|r|r["input"]["frame"]["eye_id"]==roi) {
            let time=ns(&row["to_source_ns"])?;let key=(roi,time);count+=1;
            let source=arms.each_ref().map(|a|a.get(&key));
            let radius=source.map(|s|s.and_then(|s|s.ellipses[0]).map(|e|e.major_radius));
            for arm in 0..2 {if radius[arm].is_some() {present[arm]+=1;}}
            if let [Some(base),Some(cand)]=source {
                for kind in 0..3 {
                    if base.points[kind].is_empty() {continue;}
                    let [Some(a),Some(b)]=[base.ellipses[kind],cand.ellipses[kind]] else {continue;};
                    let errors=[a,b].map(|e|base.points[kind].iter().map(|&p|crate::conic_solver::ellipse_residual(p,e))
                        .sum::<f64>()/base.points[kind].len() as f64);
                    for arm in 0..2 {residuals[arm][kind].push(errors[arm]);}
                    if kind==0 {regressions.push((errors[1]-errors[0],time,base.sequence,errors));}
                }
            }
            if let Some((before,previous))=last {
                if time<=before {return Err("motion source clock is not increasing per ROI".into());}
                if row["reliable"]==true && ns(&row["from_source_ns"]).ok()==Some(before) && time-before<=500_000_000 {
                    let m=&row["motion"];
                    let scale=(1.0+m["diagonal_coefficient_delta"].as_f64().ok_or("missing RAW scale")?)
                        .hypot(m["rotation_coefficient"].as_f64().ok_or("missing RAW rotation")?);
                    if (0.8..=1.25).contains(&scale) {
                        if let ([Some(a),Some(b)],[Some(c),Some(d)])=(previous,radius) {
                            area_steps[0].push((2.0*((c/a).ln()-scale.ln())).abs());
                            area_steps[1].push((2.0*((d/b).ln()-scale.ln())).abs());
                        }
                    }
                }
            }
            last=Some((time,radius));
        }
        regressions.sort_by(|a,b|b.0.total_cmp(&a.0));
        eyes.push(json!({"roi_id":roi,"native_raw_sources":count,"outer_projection_sources":present,
            "baseline_point_residual_px":residuals.map(|r|r.map(stats)),
            "matched_sn_feida_abs_log_steps":area_steps.map(stats),
            "largest_outer_residual_regressions":regressions.iter().take(8).map(|(delta,time,sequence,error)|
                json!({"source_ns":time.to_string(),"sequence":sequence,"delta_px":delta,"residual_px":error})).collect::<Vec<_>>()}));
    }
    let result=json!({"schema":"buttercup-native-calibration-geometry-comparison-v1","baseline":baseline,"candidate":candidate,
        "raw_motion":motion_path,"arms":["baseline","candidate"],"boundary_order":KINDS,"eyes":eyes,
        "human_contour_labels":0,
        "contracts":["exact native ROI/source/clock binding; held publications never refresh shapes",
            "localization diagnostic is shared radial pixel residual to baseline RAW supports, not human localization truth",
            "SN-FEIDA step uses fitted native outer major radius and the same independent RAW similarity determinant in both arms",
            "adjacent matched sources only; unreliable scale, gaps over 500ms and missing projections do not bridge dropouts",
            "pupil aperture changes are not outer iris area changes; conditional fitted curves are not new observations"]});
    let mut writer=fs::OpenOptions::new().create_new(true).write(true).open(output).map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut writer,&result).map_err(|e|e.to_string())?;
    Ok(())
}
