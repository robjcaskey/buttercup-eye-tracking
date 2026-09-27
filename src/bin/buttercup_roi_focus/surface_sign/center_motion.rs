//! Conditional temporal globe-center assay; no physical sign labels or live changes.
use super::*;
use buttercup_eye_tracking::raw_motion_octrees::{
    NativeGlobalSimilarityEvidence, NativeGlobalSimilarityTracker, NativePatchCorrespondence,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, serde::Serialize)]
struct Transport {
    a: f64,
    b: f64,
    x: f64,
    y: f64,
}
impl Transport {
    const ID: Self = Self {
        a: 1.,
        b: 0.,
        x: 0.,
        y: 0.,
    };
    fn from_evidence(e: NativeGlobalSimilarityEvidence) -> Self {
        assert!(e.reliable);
        let m = e.motion;
        let (a, b) = (
            1. + f64::from(m.diagonal_coefficient_delta),
            f64::from(m.rotation_coefficient),
        );
        let [x, y] = e.motion_center_sensor.map(f64::from);
        Self {
            a,
            b,
            x: f64::from(m.translation[0]) + (1. - a) * x + b * y,
            y: f64::from(m.translation[1]) - b * x + (1. - a) * y,
        }
    }
    fn apply(self, p: [f64; 2]) -> [f64; 2] {
        [
            self.a * p[0] - self.b * p[1] + self.x,
            self.b * p[0] + self.a * p[1] + self.y,
        ]
    }
    fn compose(self, prior: Self) -> Self {
        let [x, y] = self.apply([prior.x, prior.y]);
        Self {
            a: self.a * prior.a - self.b * prior.b,
            b: self.b * prior.a + self.a * prior.b,
            x,
            y,
        }
    }
}

struct Node {
    row: Value,
    class: String,
    raw: Arc<Vec<u16>>,
    rays: TheoreticalEllipseExplanations,
    ns: u64,
    chain: usize,
    transport: Transport,
    bands: NativeGlobalSimilarityEvidence,
    whole: NativeGlobalSimilarityEvidence,
    matches: Vec<NativePatchCorrespondence>,
    reason: &'static str,
}
fn evidence(e: NativeGlobalSimilarityEvidence) -> Value {
    let m = e.candidate_motion;
    json!({"reliable":e.reliable,"matches":e.candidate_matches,"inliers":m.support,
        "residual_px":(m.support>0).then_some(m.residual),"translation_px":m.translation,
        "scale":(1.+f64::from(m.diagonal_coefficient_delta)).hypot(f64::from(m.rotation_coefficient)),
        "rotation_degrees":f64::from(m.rotation_coefficient).atan2(1.+f64::from(m.diagonal_coefficient_delta)).to_degrees(),
        "span":e.spatial_span,"quadrants":e.occupied_quadrants,"center_sensor":e.motion_center_sensor})
}
fn band_contains(row: &Value, p: [f32; 2]) -> bool {
    let f = &row["frame"];
    let y = p[1] - n(&f["sensor_y"]) as f32;
    let h = n(&f["height"]) as f32;
    y <= h * 0.125 - 12. + 0.001 || y >= h * 0.875 + 12. - 0.001
}
fn image(c: &mut Canvas, node: &Node, x: f64, y: f64) {
    let f = &node.row["frame"];
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let color = preview::color_preview(
        &node.raw,
        w,
        h,
        n(&f["sensor_x"]) as u32,
        n(&f["sensor_y"]) as u32,
        100,
        None,
    );
    let bgra = color
        .into_iter()
        .flat_map(|v| {
            [
                (v & 255) as u8,
                ((v >> 8) & 255) as u8,
                ((v >> 16) & 255) as u8,
                255,
            ]
        })
        .collect::<Vec<_>>();
    c.image(&bgra, w, h, x, y, w as f64 * 2., h as f64 * 2.);
    c.shade(x, y + h as f64 * 0.25, w as f64 * 2., h as f64 * 1.5, 0.45);
    let e = shape(&node.row["fit"]["ellipse"]).unwrap();
    let pts = (0..=180)
        .map(|j| {
            let t = j as f64 * std::f64::consts::TAU / 180.;
            let (s, a) = e.angle.sin_cos();
            [
                x + 2. * (e.center.0 + a * e.major_radius * t.cos() - s * e.minor_radius * t.sin()),
                y + 2. * (e.center.1 + s * e.major_radius * t.cos() + a * e.minor_radius * t.sin()),
            ]
        })
        .collect::<Vec<_>>();
    c.path(&pts, 1.2, WHITE);
    for p in node.row["fit"]["points"].as_array().unwrap() {
        c.dot(
            x + 2. * p[0].as_f64().unwrap(),
            y + 2. * p[1].as_f64().unwrap(),
            1.7,
            ORANGE,
            true,
        );
    }
}
fn panel_point(node: &Node, p: [f64; 2], x: f64, y: f64) -> [f64; 2] {
    [
        x + 2. * (p[0] - n(&node.row["frame"]["sensor_x"]) as f64),
        y + 2. * (p[1] - n(&node.row["frame"]["sensor_y"]) as f64),
    ]
}
fn render(previous: &Node, current: &Node, out: &Path) -> Result<()> {
    let mut c = Canvas::new(1800, 940)?;
    c.clear();
    c.text(
        25.,
        32.,
        24.,
        WHITE,
        "Measured outer-band motion and the two hypothesized globe centers",
    );
    c.text(
        25.,
        64.,
        18.,
        MUTED,
        &format!(
            "{} / eye {} / records {} -> {} / {:.1} ms / {}",
            current.row["provider"].as_str().unwrap(),
            current.row["eye"],
            previous.row["record"],
            current.row["record"],
            (current.ns - previous.ns) as f64 / 1e6,
            current.reason
        ),
    );
    image(&mut c, previous, 25., 120.);
    image(&mut c, current, 930., 120.);
    c.text(25., 103., 18., WHITE, "Previous admitted RAW exposure");
    c.text(930., 103., 18., WHITE, "Current admitted RAW exposure");
    for m in &current.matches {
        let color = if m.global_similarity_inlier {
            GREEN
        } else {
            RED
        };
        let a = panel_point(previous, m.previous_sensor_px.map(f64::from), 25., 120.);
        let b = panel_point(current, m.current_sensor_px.map(f64::from), 930., 120.);
        c.dot(a[0], a[1], 4., color, false);
        c.dot(b[0], b[1], 4., color, false);
        let start = panel_point(current, m.previous_sensor_px.map(f64::from), 930., 120.);
        c.arrow(start, b, color);
    }
    for (j, col) in [CYAN, PINK].into_iter().enumerate() {
        for (node, x) in [(previous, 25.), (current, 930.)] {
            let g = project(center(node.rays.rays[j], 2.15));
            let p = panel_point(node, g, x, 120.);
            c.cross(p[0], p[1], 10., col);
            c.text(
                p[0] + 12.,
                p[1] - 4.,
                18.,
                col,
                if j == 0 { "A" } else { "B" },
            );
        }
        if current.bands.reliable {
            let old = Transport::from_evidence(current.bands)
                .apply(project(center(previous.rays.rays[j], 2.15)));
            let p = panel_point(current, old, 930., 120.);
            c.dot(p[0], p[1], 10., col, false);
        }
    }
    c.text(
        25.,
        721.,
        19.,
        GREEN,
        &format!(
            "Outer bands: {} / {} inliers, residual {} px, accepted={}",
            current.bands.candidate_motion.support,
            current.bands.candidate_matches,
            if current.bands.candidate_motion.support > 0 {
                format!("{:.2}", current.bands.candidate_motion.residual)
            } else {
                "unavailable".to_string()
            },
            current.bands.reliable
        ),
    );
    c.text(
        25.,
        754.,
        18.,
        MUTED,
        &format!(
            "Whole-ROI comparison: {} / {} inliers, accepted={} (never used as fallback)",
            current.whole.candidate_motion.support,
            current.whole.candidate_matches,
            current.whole.reliable
        ),
    );
    c.text(25.,791.,18.,WHITE,"Orange dots: measured segmentation boundary. White curve: fitted ellipse. Crosses: 3D sphere centers projected into RAW.");
    c.text(25.,824.,18.,MUTED,"A/B may swap order between exposures. Globe/iris radius 2.15; nominal intrinsics. Rings, if present: transported prior centers.");
    c.text(25.,857.,18.,MUTED,"Preview is demosaiced for display. Matching uses native-coordinate CFA-neutralized patches; no downsampled pyramid.");
    c.text(25.,890.,18.,MUTED,"Skin, eyelid and glasses motion are not guaranteed rigid head motion. Neither candidate label is physical sign truth.");
    c.png(out)
}

fn motion_controls() -> Result<Value> {
    // Declared fixture geometry, not learned anatomy or fabricated RAW evidence.
    let texture = |x: i32, y: i32| {
        let mut v = (x as u32).wrapping_mul(0x9e3779b9) ^ (y as u32).wrapping_mul(0x85ebca6b);
        v ^= v >> 16;
        v = v.wrapping_mul(0xc2b2ae35);
        v ^= v >> 13;
        150 + (v % 700) as u16
    };
    let make = |sx: i32, sy: i32, dx: i32, dy: i32| {
        Arc::new(
            (0..420 * 280)
                .map(|i| texture(sx + i % 420 - dx, sy + i / 420 - dy))
                .collect::<Vec<_>>(),
        )
    };
    let mut results = vec![];
    for radius in [4, 8] {
        for displacement in [[0, 0], [4, -4]] {
            let mut tracker = NativeGlobalSimilarityTracker::default();
            let first =
                tracker.observe_outer_bands(make(3948, 3294, 0, 0), 420, 280, 3948, 3294, radius);
            assert!(!first.reliable, "one exposure cannot measure motion");
            let e = tracker.observe_outer_bands(
                make(3960, 3298, displacement[0], displacement[1]),
                420,
                280,
                3960,
                3298,
                radius,
            );
            let m = e.motion;
            if !e.reliable
                || (f64::from(m.translation[0]) - f64::from(displacement[0])).abs() > 0.15
                || (f64::from(m.translation[1]) - f64::from(displacement[1])).abs() > 0.15
                || m.rotation_coefficient.abs() > 0.002
                || m.diagonal_coefficient_delta.abs() > 0.002
            {
                return Err(format!("outer-band known-motion control failed: radius {radius}, displacement {displacement:?}, {e:?}").into());
            }
            results.push(json!({"patch_radius":radius,"displacement":displacement,"reframe_sensor_px":[12,4],"measured":evidence(e)}));
        }
    }
    Ok(
        json!({"fixtures":results,"translation_tolerance_px":0.15,"coefficient_tolerance":0.002,"not_real_sign_accuracy":true}),
    )
}

pub(crate) fn run(area_dir: &str, fresh_dir: &str, output: &str, patch_radius: i32) -> Result<()> {
    if patch_radius != 4 && patch_radius != 8 {
        return Err("PATCH_RADIUS must be 4 or 8".into());
    }
    let area = Path::new(area_dir);
    let out = Path::new(output);
    if out.exists() {
        return Err("output exists".into());
    }
    let summary = load(&area.join("summary.json"))?;
    if summary["complete"] != true || summary["schema"] != "buttercup-area-first-focus-v1" {
        return Err("completed area-first input required".into());
    }
    let fresh = fs::read(Path::new(fresh_dir).join("frames.jsonl"))?;
    if archive::digest(&fresh) != summary["fresh_frames_sha256"] {
        return Err("fresh input hash changed".into());
    }
    let classes = rows(&area.join("classifications.jsonl"))?;
    let sources = classes
        .iter()
        .filter(|v| v["class"] == "multiple")
        .map(|v| n(&v["source"]))
        .collect::<BTreeSet<_>>();
    let classes = classes
        .into_iter()
        .map(|v| {
            (
                (n(&v["record"]), v["provider"].as_str().unwrap().to_owned()),
                v,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<_, Vec<Value>>::new();
    for v in rows(&area.join("retained-inputs.jsonl"))?
        .into_iter()
        .filter(|v| sources.contains(&n(&v["source"])))
    {
        if v["area_admission"]["accepted"] != true {
            return Err("unadmitted observation".into());
        }
        groups
            .entry((
                v["provider"].as_str().unwrap().to_owned(),
                n(&v["source"]),
                n(&v["epoch"]),
                n(&v["eye"]),
            ))
            .or_default()
            .push(v);
    }
    fs::create_dir(out)?;
    let controls = motion_controls()?;
    fs::write(
        out.join("controls.json"),
        serde_json::to_vec_pretty(&controls)?,
    )?;
    let mut bundles = BTreeMap::new();
    let mut cache = BTreeMap::<String, Arc<Vec<u16>>>::new();
    let mut writer = BufWriter::new(fs::File::create(out.join("motion.jsonl"))?);
    let mut review = vec![];
    let mut counts = BTreeMap::new();
    let mut total = 0;
    for (group, mut rows) in groups {
        rows.sort_by_key(|r| r["source_ns"].as_str().unwrap().parse::<u64>().unwrap());
        let mut bands = NativeGlobalSimilarityTracker::default();
        bands.retain_diagnostic_correspondences(true);
        let mut whole = NativeGlobalSimilarityTracker::default();
        let mut nodes = Vec::<Node>::new();
        let mut shown = BTreeSet::new();
        for row in rows {
            let f = &row["frame"];
            let raw_hash = row["raw_sha256"].as_str().unwrap();
            if !cache.contains_key(raw_hash) {
                let source = row["raw_source"].as_str().unwrap();
                if !bundles.contains_key(source) {
                    bundles.insert(source.to_owned(), BundleSource::open(Path::new(source))?);
                }
                let bytes = bundles[source].read_range(
                    row["stream_entry"].as_str().unwrap(),
                    n(&f["offset"]),
                    n(&f["length"]) as usize,
                )?;
                if archive::digest(&bytes) != raw_hash {
                    return Err("RAW hash mismatch".into());
                }
                cache.insert(
                    raw_hash.to_owned(),
                    Arc::new(raw10::try_unpack_raw10(
                        &bytes,
                        n(&f["width"]) as usize,
                        n(&f["height"]) as usize,
                        n(&f["stride"]) as usize,
                    )?),
                );
            }
            let raw = cache[raw_hash].clone();
            let ns = row["source_ns"].as_str().unwrap().parse::<u64>()?;
            let gap = nodes.last().map(|p| {
                ns.checked_sub(p.ns)
                    .filter(|v| *v > 0)
                    .expect("strict source ordering")
            });
            if gap.is_some_and(|g| g > 300_000_000) {
                bands.clear();
                whole.clear();
            }
            let b = bands.observe_outer_bands(
                raw.clone(),
                n(&f["width"]) as usize,
                n(&f["height"]) as usize,
                n(&f["sensor_x"]) as u32,
                n(&f["sensor_y"]) as u32,
                patch_radius,
            );
            let w = whole.observe(
                raw.clone(),
                n(&f["width"]) as usize,
                n(&f["height"]) as usize,
                n(&f["sensor_x"]) as u32,
                n(&f["sensor_y"]) as u32,
            );
            let matches = bands.diagnostic_correspondences().to_vec();
            if let Some(prior) = nodes.last() {
                for m in &matches {
                    assert!(
                        band_contains(&prior.row, m.previous_sensor_px)
                            && band_contains(&row, m.current_sensor_px)
                    );
                }
            }
            let (chain, transport, reason) = match nodes.last() {
                None => (0, Transport::ID, "first_observation"),
                Some(p) if gap.unwrap() > 300_000_000 => (p.chain + 1, Transport::ID, "source_gap"),
                Some(p) if !b.reliable => (p.chain + 1, Transport::ID, "unreliable_outer_motion"),
                Some(p) => (
                    p.chain,
                    Transport::from_evidence(b).compose(p.transport),
                    "measured_outer_motion",
                ),
            };
            let class = &classes[&(n(&row["record"]), group.0.clone())];
            if class["raw_sha256"] != row["raw_sha256"] {
                return Err("class source mismatch".into());
            }
            let mut e = shape(&row["fit"]["ellipse"])?;
            e.center.0 += n(&f["sensor_x"]) as f64;
            e.center.1 += n(&f["sensor_y"]) as f64;
            let rays = TheoreticalEllipseExplanations::from_ellipse(e, [4000.; 2], [4000., 3000.])
                .ok_or("circle solve failed")?;
            let node = Node {
                row,
                class: class["class"].as_str().unwrap().to_owned(),
                raw,
                rays,
                ns,
                chain,
                transport,
                bands: b,
                whole: w,
                matches,
                reason,
            };
            count(&mut counts, format!("{}/{}", group.0, reason));
            let points=node.matches.iter().map(|m|json!({"previous":m.previous_sensor_px,"current":m.current_sensor_px,"inlier":m.global_similarity_inlier,"photometric_score":m.photometric_score,"distinct_margin":m.distinct_match_margin})).collect::<Vec<_>>();
            let result = json!({"record":node.row["record"],"provider":group.0,"source":group.1,"epoch":group.2,"eye":group.3,"source_ns":node.row["source_ns"],"previous_record":nodes.last().map(|p|&p.row["record"]),"raw_sha256":node.row["raw_sha256"],"area_admission":node.row["area_admission"],"class":node.class,"gap_ms":gap.map(|n|n as f64/1e6),"reason":reason,"chain":chain,"chain_transport":transport,"bands":evidence(b),"whole_roi_comparison":evidence(w),"matches":points,"physical_sign_truth":null});
            writeln!(writer, "{}", serde_json::to_string(&result)?)?;
            if let Some(p) = nodes.last() {
                if shown.insert(reason) || nodes.len().is_multiple_of(32) {
                    let name = format!("motion-{}-{}.png", node.row["record"], group.0);
                    render(p, &node, &out.join(&name))?;
                    review.push(json!({"image":name,"reason":reason,"record":node.row["record"],"provider":group.0,"eye":group.3}));
                }
            }
            nodes.push(node);
            total += 1;
        }
        eprintln!(
            "center motion {:?}: {} area-admitted rows",
            group,
            nodes.len()
        );
    }
    writer.flush()?;
    fs::write(out.join("review.json"), serde_json::to_vec_pretty(&review)?)?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"schema":"buttercup-surface-center-motion-v1","counts":counts,"rows":total,"unique_raw_exposures":cache.len(),"patch_radius":patch_radius,"feature_columns":if patch_radius==8 {20} else {10},"controls":controls,"source_subset":sources,"maximum_pair_gap_ms":300,"support":"top/bottom 12.5 percent, 12px full-patch safety margin in both frames","area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"retained_inputs_sha256":archive::digest(&fs::read(area.join("retained-inputs.jsonl"))?),"executable_sha256":archive::digest(&fs::read(std::env::current_exe()?)?),"physical_sign_truth":null,"transport_diagnostic_only":true,"sign_estimation_implemented":false}),
        )?,
    )?;
    Ok(())
}
