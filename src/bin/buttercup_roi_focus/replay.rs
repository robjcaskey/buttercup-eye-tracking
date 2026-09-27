use super::archive::{self, Frame, Manifest};
use crate::Result;
use buttercup_eye_tracking::{
    focus_region::*, geometry::Ellipse, raw10, recorded_bundle::BundleSource,
};
use serde_json::json;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{BufWriter, Write},
    path::Path,
};
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
use canvas::*;

pub(super) fn explanations(f: &Frame, focal: f64) -> Option<TheoreticalEllipseExplanations> {
    TheoreticalEllipseExplanations::from_ellipse(f.shape()?, [focal; 2], [4000., 3000.])
}
fn usable(f: &Frame) -> bool {
    f.flags & 5 == 5 && (f.provider != 1 || f.flags & 2 != 0)
}
pub(super) fn verdict(
    f: &Frame,
    rays: &Option<TheoreticalEllipseExplanations>,
    regions: &[FocusRegion],
) -> (&'static str, Vec<RayClassification>) {
    let Some(rays) = rays else {
        return (
            if f.shape().is_some() {
                "invalid_projection"
            } else {
                "missing_ellipse"
            },
            vec![],
        );
    };
    let classes = rays.rays.map(|r| classify(r, regions, 2.)).to_vec();
    let n = classes
        .iter()
        .filter(|c| c.status == "inside" || c.status == "nearby")
        .count();
    (
        if !usable(f) {
            "unusable_evidence"
        } else if regions.is_empty() {
            "unresolved_region"
        } else {
            ["zero", "one", "multiple"][n]
        },
        classes,
    )
}
fn counts_add(counts: &mut BTreeMap<String, usize>, k: &str) {
    *counts.entry(k.into()).or_default() += 1;
}

pub fn run(input: &str, output: &str) -> Result<()> {
    let out = Path::new(output);
    if out.exists() {
        return Err("output exists".into());
    }
    fs::create_dir_all(out)?;
    let (manifest, frames) = archive::read(Path::new(input))?;
    let mut groups: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    for (i, f) in frames.iter().enumerate() {
        groups.entry((f.source, f.epoch)).or_default().push(i);
    }
    let rays = frames
        .iter()
        .map(|f| explanations(f, 4000.))
        .collect::<Vec<_>>();
    let mut classifications = BufWriter::new(fs::File::create(out.join("classifications.jsonl"))?);
    let mut evolution = BufWriter::new(fs::File::create(out.join("region-evolution.jsonl"))?);
    let mut intersections = BufWriter::new(fs::File::create(out.join("intersections.jsonl"))?);
    let mut all_counts = BTreeMap::new();
    let mut prefix_counts = BTreeMap::new();
    let mut reports = vec![];
    let mut thumbnails = vec![];
    let mut snapshots = BTreeMap::new();
    let options = FocusOptions::default();
    for ((source, epoch), ids) in groups {
        let first_ns = ids.iter().map(|&i| frames[i].ns).min().unwrap_or(0);
        let mut pairs = BTreeMap::<u64, [Option<usize>; 2]>::new();
        let mut seen = HashSet::new();
        let mut duplicates = 0;
        for &i in &ids {
            let f = &frames[i];
            if !seen.insert(f.key()) {
                duplicates += 1;
                continue;
            }
            if usable(f) && rays[i].is_some() && (1..=2).contains(&f.eye) {
                pairs.entry(f.ns).or_insert([None; 2])[f.eye as usize - 1] = Some(i);
            }
        }
        let pairs = pairs
            .into_iter()
            .filter_map(|(ns, v)| Some((ns, [v[0]?, v[1]?])))
            .collect::<Vec<_>>();
        let mut estimator = FocusRegionEstimator::new(options);
        let mut prefix_regions = vec![];
        let cutoff = pairs.len() * 7 / 10;
        let cutoff_ns = pairs.get(cutoff).map(|p| p.0).unwrap_or(u64::MAX);
        for (pi, &(ns, pair)) in pairs.iter().enumerate() {
            if pi == cutoff {
                prefix_regions = estimator.regions();
            }
            let seconds = ns.saturating_sub(first_ns) as f64 * 1e-9;
            let points = estimator.observe(
                seconds,
                rays[pair[0]].as_ref().unwrap(),
                rays[pair[1]].as_ref().unwrap(),
            );
            serde_json::to_writer(
                &mut intersections,
                &json!({"source":source,"epoch":epoch,"source_ns":ns.to_string(),"frame_records":pair,"possibilities":points}),
            )?;
            writeln!(intersections)?;
            if pi % 25 == 0 || pi + 1 == pairs.len() {
                serde_json::to_writer(
                    &mut evolution,
                    &json!({"source":source,"epoch":epoch,"paired_observations":pi+1,"source_ns":ns.to_string(),"regions":estimator.regions()}),
                )?;
                writeln!(evolution)?;
            }
        }
        let regions = estimator.regions();
        let mut counts = BTreeMap::new();
        let mut heldout = BTreeMap::new();
        let mut by_provider = BTreeMap::<u8, BTreeMap<String, usize>>::new();
        let mut example_classes = HashSet::new();
        for &i in &ids {
            let f = &frames[i];
            let (class, candidates) = verdict(f, &rays[i], &regions);
            counts_add(&mut counts, class);
            counts_add(&mut all_counts, class);
            counts_add(by_provider.entry(f.provider).or_default(), class);
            if f.ns >= cutoff_ns {
                let (prefix, _) = verdict(f, &rays[i], &prefix_regions);
                counts_add(&mut heldout, prefix);
                counts_add(&mut prefix_counts, prefix);
            }
            let record = json!({"record":i,"source":source,"epoch":epoch,"eye":f.eye,"sequence":f.sequence,"source_ns":f.ns.to_string(),"provider":f.provider,"ellipse_sensor_px":f.shape().map(|_|f.ellipse),"disk_area":f.area.json(),"explanations":rays[i],"classification":class,"valid_interpretations":if ["zero","one","multiple"].contains(&class){Some(candidates.iter().filter(|c|c.status=="inside"||c.status=="nearby").count())}else{None},"interpretations":candidates,"distinct_directions":rays[i].map(|r|r.separation_degrees>1.),"retrospective_region_fit":true});
            serde_json::to_writer(&mut classifications, &record)?;
            writeln!(classifications)?;
            if f.provider == 1
                && usable(f)
                && !regions.is_empty()
                && example_classes.insert(class)
                && thumbnails.len() < 36
            {
                thumbnails.push((i, class.to_string(), regions.clone()));
            }
        }
        if !regions.is_empty() {
            snapshots.insert((source, epoch), regions.clone());
        }
        reports.push(json!({"source":source,"epoch":epoch,"path":manifest.sources[source as usize].path,"frames":ids.len(),"paired_fresh_usable_exposures":pairs.len(),"duplicate_source_keys":duplicates,"geometric_pairs":estimator.geometric_pairs,"discarded_cluster_births":estimator.discarded_cluster_births,"regions":regions,"classes":counts,"by_provider":by_provider,"prefix70_regions":prefix_regions,"last30_frame_classes_from_prefix70":heldout}));
    }
    classifications.flush()?;
    evolution.flush()?;
    intersections.flush()?;
    let summary = json!({"schema":"buttercup-focus-volume-experiment-v1","binary":input,"binary_sha256":archive::digest(&fs::read(input)?),"options":options,"sources":manifest.sources.len(),"frames":frames.len(),"groups":reports.len(),"groups_with_regions":snapshots.len(),"classes":all_counts,"prefix70_last30_classes":prefix_counts,"per_group":reports,
        "units":"iris radii in nominal camera coordinates; miss_degrees=atan2(shortest forward-ray-to-region distance, forward distance). Inside/nearby means inferred focus volume, NOT sensor image crop or known physical screen.",
        "limits":["Historical diagnostic geometry, no newly certified model ancestry; no training or promotion.","Recorded contact reconstructions are counterfactual weak-perspective ellipses and reported separately from SAM3 fits.","Only exact same source timestamp/epoch eye1-eye2 pairs initialize 3D regions; monocular or unsynchronized data remains unresolved.","The 2-degree nearby tolerance, 6-radius clustering and 35-percent competing-support threshold are declared engineering assumptions, not calibrated uncertainty.","A coherent wrong branch can produce a second focus region, including a camera-adjacent ghost. Multiple regions and multiple interpretations remain valid rather than claiming the sign is known.","No measured monitor plane, metric iris radius, kappa, distortion or independent sign/attention truth. Nominal focal and shared inter-eye radius can bias depth.","Full second-pass classifications are retrospective/self-influenced. Prefix70/last30 reports use only earlier pairs for region fitting but are compatibility, not gaze accuracy.","Ellipses are unchanged; this method cannot improve localization or SN-FEIDA. Independent scale and human localization/sign labels are missing in this replay."]});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    fs::write(
        out.join("archive-manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    render(&manifest, &frames, &rays, &thumbnails, &snapshots, out)?;
    eprintln!(
        "REPLAY DONE {} frames, {} groups with regions; {:?}",
        frames.len(),
        snapshots.len(),
        all_counts
    );
    Ok(())
}

fn preview(bytes: &[u8], f: &Frame) -> Result<(Vec<u8>, usize, usize)> {
    let raw = raw10::try_unpack_raw10(
        bytes,
        f.width as usize,
        f.height as usize,
        f.stride as usize,
    )?;
    let w = f.width as usize / 4;
    let h = f.height as usize / 4;
    let mut values = raw.iter().step_by(8).copied().collect::<Vec<_>>();
    values.sort_unstable();
    let lo = values[values.len() / 50] as f64;
    let hi = values[values.len() * 49 / 50] as f64;
    let mut bgra = vec![];
    for y in 0..h {
        for x in 0..w {
            let mut v = 0.;
            for dy in 0..4 {
                for dx in 0..4 {
                    v += raw[(y * 4 + dy) * f.width as usize + x * 4 + dx] as f64 / 16.;
                }
            }
            let q = (255. * ((v - lo) / (hi - lo).max(1.)).clamp(0., 1.)) as u8;
            bgra.extend_from_slice(&[q, q, q, 255]);
        }
    }
    Ok((bgra, w, h))
}
fn render(
    manifest: &Manifest,
    frames: &[Frame],
    rays: &[Option<TheoreticalEllipseExplanations>],
    examples: &[(usize, String, Vec<FocusRegion>)],
    snapshots: &BTreeMap<(u32, u32), Vec<FocusRegion>>,
    out: &Path,
) -> Result<()> {
    let native_bytes = fs::read(&manifest.native_conics_path)?;
    if archive::digest(&native_bytes) != manifest.native_conics_sha256 {
        return Err("source conic export changed before RAW review".into());
    }
    let mut hashes = BTreeMap::new();
    for line in native_bytes
        .split(|x| *x == b'\n')
        .filter(|l| !l.is_empty())
    {
        let r: serde_json::Value = serde_json::from_slice(line)?;
        let f = &r["source"]["frame"];
        let key = &f["source_clock"]["source_key"];
        let Some(epoch) = key["stream_epoch"].as_str() else {
            continue;
        };
        let n = super::pack::number;
        let id = (
            epoch.to_owned(),
            n(&f["eye_id"]).unwrap_or(0) as u16,
            n(&key["sensor_timestamp_ns"])
                .or_else(|| n(&f["timestamp_ns"]))
                .unwrap_or(0),
            n(&key["sequence"])
                .or_else(|| n(&f["sequence"]))
                .unwrap_or(0),
            [
                n(&f["sensor_x"]).unwrap_or(0) as u32,
                n(&f["sensor_y"]).unwrap_or(0) as u32,
            ],
            n(&f["width"]).unwrap_or(0) as u32,
            n(&f["height"]).unwrap_or(0) as u32,
        );
        if let Some(hash) = r["source"]["raw_sha256"].as_str() {
            hashes.insert(id, hash.to_owned());
        }
    }
    let mut verified = vec![];
    for (page, chunk) in examples.chunks(6).enumerate() {
        let mut c = Canvas::new(1800, 1160)?;
        c.clear();
        c.text(
            20.,
            34.,
            25.,
            WHITE,
            "Source-matched RAW and ambiguous focus-volume intersections",
        );
        c.text(20.,63.,17.,MUTED,"White ellipse = historical SAM fit. Cyan/pink = its two theoretical rays. Volumes are inferred, not a measured screen.");
        for (k, (i, class, regions)) in chunk.iter().enumerate() {
            let f = &frames[*i];
            let source = &manifest.sources[f.source as usize];
            let bundle = BundleSource::open(Path::new(&source.path))?;
            let raw = bundle.read_range(
                &format!("{}{}", source.prefix, source.streams[f.stream as usize]),
                f.offset,
                f.length as usize,
            )?;
            let expected = hashes
                .get(&(
                    manifest.epochs[f.epoch as usize].clone(),
                    f.eye,
                    f.ns,
                    f.sequence,
                    f.origin,
                    f.width,
                    f.height,
                ))
                .ok_or("RAW review missing original conic identity")?;
            let actual = archive::digest(&raw);
            if &actual != expected {
                return Err("RAW preview differs from original conic exposure".into());
            }
            verified.push(json!({"record":i,"source":f.source,"eye":f.eye,"sequence":f.sequence,"raw_sha256":actual,"matches_original_sam_conic_raw":true,"class":class,"area":f.area.json()}));
            let (p, w, h) = preview(&raw, f)?;
            let x = 20. + (k % 3) as f64 * 600.;
            let y = 100. + (k / 3) as f64 * 510.;
            let sh = 310.;
            let sw = f.width as f64 / f.height as f64 * sh;
            c.image(&p, w, h, x, y, sw, sh);
            let e = f.shape().unwrap();
            let pixel_scale = sh / f.height as f64;
            let line = e
                .dense_points(160)
                .iter()
                .map(|(xx, yy)| {
                    [
                        x + (*xx - f.origin[0] as f64) * pixel_scale,
                        y + (*yy - f.origin[1] as f64) * pixel_scale,
                    ]
                })
                .collect::<Vec<_>>();
            c.clipped(x, y, sw, sh, |c| c.path(&line, 2., WHITE));
            c.text(
                x,
                y + 335.,
                18.,
                WHITE,
                &format!(
                    "source {} eye {} seq {}: {}",
                    f.source, f.eye, f.sequence, class
                ),
            );
            for b in 0..2 {
                let ray = rays[*i].unwrap().rays[b];
                let cl = classify(ray, regions, 2.);
                c.text(
                    x,
                    y + 361. + b as f64 * 25.,
                    16.,
                    [CYAN, PINK][b],
                    &format!(
                        "{}: {} | miss {:.2} r / {:.2} deg",
                        b,
                        cl.status,
                        cl.miss_iris_radii.unwrap_or(0.),
                        cl.miss_degrees.unwrap_or(0.)
                    ),
                );
            }
            let boxrect = [x, y + 414., 550., 62.];
            c.rect(
                boxrect[0],
                boxrect[1],
                boxrect[2],
                boxrect[3],
                [0.09, 0.12, 0.16],
            );
            let all = regions
                .iter()
                .flat_map(|r| [r.lower, r.upper])
                .chain(rays[*i].unwrap().rays.map(|r| r.origin_iris_radii))
                .collect::<Vec<_>>();
            let minx = all.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) - 2.;
            let maxx = all.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max) + 2.;
            let minz = all.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min) - 2.;
            let maxz = all.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max) + 2.;
            let plot_scale = (530. / (maxx - minx)).min(50. / (maxz - minz));
            let pp = |p: V3| {
                [
                    x + 275. + (p[0] - (minx + maxx) / 2.) * plot_scale,
                    y + 445. + (p[2] - (minz + maxz) / 2.) * plot_scale,
                ]
            };
            for r in regions {
                let a = pp(r.lower);
                let b = pp(r.upper);
                c.rect(
                    a[0],
                    a[1],
                    (b[0] - a[0]).max(2.),
                    (b[1] - a[1]).max(2.),
                    [0.3, 0.4, 0.33],
                );
            }
            for b in 0..2 {
                let r = rays[*i].unwrap().rays[b];
                let cl = classify(r, regions, 2.);
                c.arrow(
                    pp(r.origin_iris_radii),
                    pp(add(
                        r.origin_iris_radii,
                        scale(r.direction, cl.forward_iris_radii.unwrap_or(50.)),
                    )),
                    [CYAN, PINK][b],
                );
            }
        }
        c.text(20.,1140.,17.,MUTED,"Bottom strips: camera X/Z projection in iris-radius units. No frame is counted as a fresh exposure twice.");
        c.png(&out.join(format!("raw-review-{:02}.png", page + 1)))?;
    }
    let mut c = Canvas::new(1800, 1100)?;
    c.clear();
    c.text(
        20.,
        35.,
        26.,
        WHITE,
        "Inferred competing 3D focus regions — camera X/Z",
    );
    c.text(20.,65.,17.,MUTED,"Each panel is one recording/source epoch. Retained regions are green. Origin (camera) is white; units are iris radii.");
    for (k, ((source, _), regions)) in snapshots
        .iter()
        .filter(|(_, r)| !r.is_empty())
        .take(12)
        .enumerate()
    {
        let x = 20. + (k % 4) as f64 * 450.;
        let y = 105. + (k / 4) as f64 * 325.;
        let minx = regions.iter().map(|r| r.lower[0]).fold(-15., f64::min);
        let maxx = regions.iter().map(|r| r.upper[0]).fold(15., f64::max);
        let minz = regions.iter().map(|r| r.lower[2]).fold(-20., f64::min);
        let maxz = regions.iter().map(|r| r.upper[2]).fold(20., f64::max);
        let plot_scale = (390. / (maxx - minx)).min(215. / (maxz - minz));
        let project = |p: V3| {
            [
                x + 215. + (p[0] - (minx + maxx) / 2.) * plot_scale,
                y + 145. + (p[2] - (minz + maxz) / 2.) * plot_scale,
            ]
        };
        c.text(
            x,
            y,
            18.,
            WHITE,
            &format!("Recording {}: {} regions", source, regions.len()),
        );
        c.rect(x, y + 15., 430., 265., [0.09, 0.12, 0.16]);
        let o = project([0.; 3]);
        c.cross(o[0], o[1], 7., WHITE);
        for r in regions {
            let a = project(r.lower);
            let b = project(r.upper);
            c.rect(
                a[0],
                a[1],
                (b[0] - a[0]).max(2.),
                (b[1] - a[1]).max(2.),
                [0.23, 0.45, 0.32],
            );
            let p = project(r.center);
            c.text(p[0], p[1], 13., GREEN, &format!("{}", r.id));
        }
        c.text(
            x,
            y + 300.,
            14.,
            MUTED,
            &format!("X [{minx:.0},{maxx:.0}]   Z [{minz:.0},{maxz:.0}]"),
        );
    }
    c.png(&out.join("focus-regions.png"))?;
    fs::write(
        out.join("visual-review.json"),
        serde_json::to_vec_pretty(&verified)?,
    )?;
    Ok(())
}

fn synthetic_ellipse(center: V3, normal: V3) -> Ellipse {
    // Cone of a unit circle: (n.c)^2 I - (n.c)(c n' + n c') + (c.c-1)n n'.
    let nc = dot(normal, center);
    let cc = dot(center, center) - 1.;
    let q: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            nc * nc * if i == j { 1. } else { 0. }
                - nc * (center[i] * normal[j] + normal[i] * center[j])
                + cc * normal[i] * normal[j]
        })
    });
    let (a, b, d) = (q[0][0], q[0][1], q[1][1]);
    let (x, y) = (-q[0][2], -q[1][2]);
    let det = a * d - b * b;
    let cx = (b * y - d * x) / det;
    let cy = (b * x - a * y) / det;
    let constant = q[2][2] + x * cx + y * cy;
    let angle = 0.5 * (2. * b).atan2(a - d);
    let root = ((a - d).powi(2) + 4. * b * b).sqrt();
    let l1 = (a + d + root) * 0.5;
    let l2 = (a + d - root) * 0.5;
    let r1 = (-constant / l1).sqrt() * 4000.;
    let r2 = (-constant / l2).sqrt() * 4000.;
    Ellipse {
        center: (4000. + 4000. * cx, 3000. + 4000. * cy),
        major_radius: r1.max(r2),
        minor_radius: r1.min(r2),
        angle: if r1 >= r2 {
            angle
        } else {
            angle + std::f64::consts::FRAC_PI_2
        },
    }
}
pub fn controls(output: &str) -> Result<()> {
    let mut estimator = FocusRegionEstimator::new(FocusOptions::default());
    let mut cases = vec![];
    let mut max_error = 0f64;
    for i in 0..540 {
        let target = [
            ((i / 20) % 3) as f64 * 10. - 10.,
            ((i / 60) % 3) as f64 * 8. - 16.,
            0.,
        ];
        let centers = [[-6., -4., -80.], [6., -4., -80.]];
        let pair = centers.map(|c| {
            let d = sub(target, c);
            let n = scale(d, 1. / norm(d));
            let e = synthetic_ellipse(c, n);
            let rays = TheoreticalEllipseExplanations::from_ellipse(e, [4000.; 2], [4000., 3000.])
                .unwrap();
            let (branch, error) = (0..2)
                .map(|b| {
                    (
                        b,
                        norm(sub(n, rays.rays[b].direction))
                            + norm(sub(c, rays.rays[b].origin_iris_radii)),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            max_error = max_error.max(error);
            (rays, branch)
        });
        estimator.observe(i as f64 * 0.05, &pair[0].0, &pair[1].0);
        cases.push(pair);
    }
    if max_error > 1e-6 {
        return Err(format!("synthetic projection/unprojection disagrees: {max_error}").into());
    }
    let regions = estimator.regions();
    let mut valid_true = 0;
    let mut valid_wrong = 0;
    for pair in &cases {
        for &(r, b) in pair {
            valid_true += usize::from(classify(r.rays[b], &regions, 2.).status != "outside");
            valid_wrong += usize::from(classify(r.rays[1 - b], &regions, 2.).status != "outside");
        }
    }
    let r = FocusRegion {
        id: 0,
        center: [0., 0., 10.],
        lower: [-1., -1., 9.],
        upper: [1., 1., 11.],
        support_pairs: 10,
        occupied_200ms_bins: 10,
        support_weight: 10.,
        span_seconds: 2.,
    };
    let straight = GazeRay {
        origin_iris_radii: [0.; 3],
        direction: [0., 0., 1.],
    };
    let outside = GazeRay {
        origin_iris_radii: [3., 0., 0.],
        direction: [0., 0., 1.],
    };
    let backward = GazeRay {
        origin_iris_radii: [0.; 3],
        direction: [0., 0., -1.],
    };
    let d = [
        ray_box_distance(straight, &r),
        ray_box_distance(outside, &r),
        ray_box_distance(backward, &r),
    ];
    if d[0].0 > 1e-10 || (d[1].0 - 2.).abs() > 1e-10 || (d[2].0 - 9.).abs() > 1e-10 {
        return Err("half-ray distance control failed".into());
    }
    let result = json!({"exact_circle_unprojection_max_error":max_error,"ray_box_distances":d,"synthetic_pairs":cases.len(),"true_rays_compatible":valid_true,"wrong_rays_compatible":valid_wrong,"rays_per_branch":cases.len()*2,"regions":regions,"interpretation":"Known planar targets, exact conics and nominal optics; both true and ghost regions may be geometrically coherent. This control checks math and exposes non-identifiability; not real-corpus gaze accuracy."});
    fs::write(output, serde_json::to_vec_pretty(&result)?)?;
    eprintln!("synthetic control: true {valid_true}, alternative {valid_wrong}, unprojection error {max_error}");
    Ok(())
}

/// Matched diagnostic arms score exactly the same saved SAM source exposures.
/// No target coordinates, learned sign scores or previous branch choices enter.
pub fn compare(input: &str, output: &str) -> Result<()> {
    let (manifest, frames) = archive::read(Path::new(input))?;
    let mut groups = BTreeMap::<(u32, u32), Vec<usize>>::new();
    let mut area_present = 0;
    let mut area_missing = 0;
    let mut area_scaled = 0;
    for (i, f) in frames.iter().enumerate() {
        groups.entry((f.source, f.epoch)).or_default().push(i);
        if let Some(e) = f.shape() {
            let expected = archive::DiskAreaRecord::from_shape(Some(e));
            if (f.area.frontal_equivalent_disk_px2 - expected.frontal_equivalent_disk_px2).abs()
                > 1e-9
                || !f.area.frontal_equivalent_disk_px2.is_finite()
                || (f.area.projected_disk_px2 - expected.projected_disk_px2).abs() > 1e-9
            {
                return Err("stored disk area differs from original ellipse".into());
            }
            area_present += 1;
        } else {
            if f.area.flags != 0 || f.area.frontal_equivalent_disk_px2.is_finite() {
                return Err("missing ellipse has fabricated area".into());
            }
            area_missing += 1;
        }
        if f.area.flags & 2 != 0 {
            if f.area.scale_reference as usize >= manifest.scale_references.len() {
                return Err("normalized area has missing scale reference".into());
            }
            area_scaled += 1;
        }
    }
    let names = [
        "nominal4000_all",
        "focal3600_all",
        "focal4400_all",
        "nominal4000_sam_only",
        "nominal4000_reversed_alternatives",
    ];
    let focals = [4000., 3600., 4400., 4000., 4000.];
    let mut reports = vec![];
    let mut totals: Vec<BTreeMap<String, usize>> = vec![BTreeMap::new(); 5];
    let mut flips = [0usize; 5];
    let mut comparable = [0usize; 5];
    let mut disagreements = [0usize; 5];
    let mut fresh_ids = HashSet::new();
    for ((source, epoch), ids) in groups {
        if !ids
            .iter()
            .any(|&i| frames[i].provider == 1 && usable(&frames[i]))
        {
            continue;
        }
        let first = ids.iter().map(|&i| frames[i].ns).min().unwrap();
        let mut base_pairs = BTreeMap::<u64, [Option<usize>; 2]>::new();
        for &i in &ids {
            let f = &frames[i];
            if usable(f) && (1..=2).contains(&f.eye) {
                base_pairs.entry(f.ns).or_insert([None; 2])[f.eye as usize - 1] = Some(i);
            }
        }
        let pairs = base_pairs
            .into_iter()
            .filter_map(|(ns, p)| Some((ns, [p[0]?, p[1]?])))
            .collect::<Vec<_>>();
        let mut arm_results = vec![];
        for arm in 0..5 {
            let local = ids
                .iter()
                .filter_map(|&i| Some((i, explanations(&frames[i], focals[arm])?)))
                .collect::<BTreeMap<_, _>>();
            let mut estimator = FocusRegionEstimator::new(FocusOptions::default());
            for (ns, pair) in &pairs {
                if arm == 3 && pair.iter().any(|&i| frames[i].provider != 1) {
                    continue;
                }
                if let (Some(a), Some(b)) = (local.get(&pair[0]), local.get(&pair[1])) {
                    let (mut a, mut b) = (*a, *b);
                    if arm == 4 {
                        a.rays.swap(0, 1);
                        b.rays.swap(0, 1);
                    }
                    estimator.observe(ns.saturating_sub(first) as f64 * 1e-9, &a, &b);
                }
            }
            let regions = estimator.regions();
            let mut masks = BTreeMap::new();
            let mut classes = BTreeMap::new();
            let mut unique = HashSet::new();
            for &i in &ids {
                let f = &frames[i];
                if f.provider != 1 || !usable(f) || !unique.insert(f.key()) {
                    continue;
                }
                let rays = local.get(&i).copied();
                let (status, candidates) = verdict(f, &rays, &regions);
                counts_add(&mut classes, status);
                counts_add(&mut totals[arm], status);
                if let Some(r) = rays.filter(|_| !regions.is_empty()) {
                    let mask = candidates.iter().enumerate().fold(0u8, |m, (b, c)| {
                        m | if c.status == "inside" || c.status == "nearby" {
                            1 << b
                        } else {
                            0
                        }
                    });
                    masks.insert(i, (mask, r));
                }
                if arm == 0 {
                    fresh_ids.insert((source, f.key()));
                }
            }
            arm_results.push((masks,json!({"arm":names[arm],"regions":regions.len(),"pairs":estimator.paired_observations,"classes":classes,"discarded_cluster_births":estimator.discarded_cluster_births})));
        }
        for arm in 1..5 {
            for (i, (base, b)) in &arm_results[0].0 {
                if let Some((mask, a)) = arm_results[arm].0.get(i) {
                    let same = dot(b.rays[0].direction, a.rays[0].direction)
                        + dot(b.rays[1].direction, a.rays[1].direction);
                    let swapped = dot(b.rays[0].direction, a.rays[1].direction)
                        + dot(b.rays[1].direction, a.rays[0].direction);
                    let mask = if swapped > same {
                        ((mask & 1) << 1) | ((mask & 2) >> 1)
                    } else {
                        *mask
                    };
                    comparable[arm] += 1;
                    disagreements[arm] += usize::from(mask != *base);
                    flips[arm] += usize::from(
                        base.count_ones() == 1 && mask.count_ones() == 1 && mask != *base,
                    );
                }
            }
        }
        reports.push(json!({"source":source,"epoch":epoch,"arms":arm_results.into_iter().map(|x|x.1).collect::<Vec<_>>()}));
    }
    if disagreements[4] != 0 {
        return Err("branch numbering changes classifications".into());
    }
    let result = json!({"schema":"buttercup-focus-matched-sensitivity-v1","binary_sha256":archive::digest(&fs::read(input)?),"scored_source_matched_sam_frames":fresh_ids.len(),"area_validation":{"present":area_present,"missing":area_missing,"independently_scaled":area_scaled,"all_stored_values_match_original_ellipses":true},"arms":(0..5).map(|i|json!({"name":names[i],"classes":totals[i],"both_arms_have_regions":comparable[i],"changed_valid_candidate_set":disagreements[i],"opposite_single_winner":flips[i]})).collect::<Vec<_>>(),"groups":reports,"limits":"Focal perturbation and historical-provider ablations measure sensitivity, not gaze accuracy. Only saved admissible SAM exposure IDs are scored; metadata-identical aliases were removed. Recorded reconstructions may support the all-provider region estimator. No measured signs or scale."});
    fs::write(output, serde_json::to_vec_pretty(&result)?)?;
    eprintln!(
        "COMPARE {} SAM frames; changed sets {:?}; single-winner flips {:?}; {} areas",
        fresh_ids.len(),
        disagreements,
        flips,
        area_present
    );
    Ok(())
}
