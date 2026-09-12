//! Experimental F view. No candidate from this module mutates tracking, conic
//! authority, laser/mouse output or calibration. All pixels use the SAM source.
use super::*;
use limbus_refiner::{Context, Field, Model, Support};
use std::cell::RefCell;

struct Cache {
    model: Result<Model, String>,
    eyes: [Option<(Arc<sam31_outer::ProposalMasks>, Arc<Field>)>; 2],
}
thread_local! {static CACHE: RefCell<Option<Cache>> = const {RefCell::new(None)};}

fn field_for(frame: &EyeFrame) -> Result<Arc<Field>, String> {
    let proposal = frame
        .sam31_proposal_masks
        .as_ref()
        .ok_or("WAITING FOR SAM SOURCE")?;
    let review = proposal
        .outer_fit
        .as_ref()
        .ok_or("NO SAM LIMBUS TO REFINE")?;
    let eye = proposal.eye_index.min(1);
    CACHE.with(|cache| {
        let mut state = cache.borrow_mut();
        let cache = state.get_or_insert_with(|| Cache {
            model: Model::load(&limbus_refiner::default_model_path()),
            eyes: [None, None],
        });
        if let Some((source, field)) = &cache.eyes[eye] {
            if Arc::ptr_eq(source, proposal) {
                return Ok(Arc::clone(field));
            }
        }
        let model = cache.model.as_ref().map_err(|e| e.clone())?;
        let context = Context::from_coarse_scale(
            frame.centimeter_scale.map(|s| Support {
                estimate: s.estimate_px / 10.0,
                half_width: (s.maximum_px - s.minimum_px) / 20.0,
            }),
            4000.0,
        );
        // No calibrated camera-distance or source-aligned VCM observation
        // exists in ProposalMasks. Never borrow a newer camera setting.
        let field = Arc::new(limbus_refiner::refine(
            model,
            &proposal.source_raw,
            proposal.source_width,
            proposal.source_height,
            review.ellipse,
            &review.retained_points,
            context,
        ));
        cache.eyes[eye] = Some((Arc::clone(proposal), Arc::clone(&field)));
        Ok(field)
    })
}

fn preview_surface(
    frame: &EyeFrame,
    proposal: &sam31_outer::ProposalMasks,
    candidate: geometry::Ellipse,
) -> Option<SurfaceGazeSample> {
    let mut surface = frame.virtual_contact_surface_gaze?;
    if surface.source_timestamp_ns != Some(proposal.source_timestamp_ns)
        || !surface.sign_resolved
        || !surface.relative_gaze.is_camera_facing()
    {
        return None;
    }
    let reference = surface.relative_gaze;
    let tilt = (1.0 - (candidate.minor_radius / candidate.major_radius).powi(2))
        .max(0.0)
        .sqrt();
    let (s, c) = candidate.angle.sin_cos();
    let mut normal = (-s * tilt, c * tilt);
    // Preserve the established same-exposure sign branch; this patch model
    // has no signed gaze supervision and may not manufacture a sign decision.
    if normal.0 * reference.right + normal.1 * reference.down < 0.0 {
        normal = (-normal.0, -normal.1);
    }
    surface.relative_gaze =
        eye_scene_model::RelativeGazeVector::from_projected(normal.0, normal.1)?;
    surface.frontal_equivalent_disk_area_px2 =
        std::f64::consts::PI * candidate.major_radius.powi(2);
    surface.quantized_frontal_disk_radius_px = candidate.major_radius;
    Some(surface)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn draw(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    scale: usize,
    frame: &EyeFrame,
) -> usize {
    let count = draw_sam31_outer_iris_fit(
        pixels,
        width,
        height,
        x,
        y,
        scale,
        frame.sequence,
        frame.sam31_proposal_masks.as_deref(),
    );
    let field = match field_for(frame) {
        Ok(field) => field,
        Err(error) => {
            draw_text(
                pixels,
                width,
                height,
                x + 4,
                y + 8,
                "TWEAKED CONTACT / UNAVAILABLE",
                0x00ff_c857,
            );
            draw_text(
                pixels,
                width,
                height,
                x + 4,
                y + 24,
                if error == "NO SAM LIMBUS TO REFINE" || error == "WAITING FOR SAM SOURCE" {
                    &error
                } else {
                    "MODEL MISSING OR INCOMPATIBLE"
                },
                0x00ff_c857,
            );
            return count;
        }
    };
    let project = |p: (f64, f64)| {
        (
            x + (p.0 * scale as f64).round() as i32,
            y + (p.1 * scale as f64).round() as i32,
        )
    };
    let mut contact_drawn = false;
    {
        // An abstention retains the exact source's original contact. It is not
        // a newly refined observation and is explicitly labeled below.
        let candidate = field.candidate.unwrap_or(field.baseline);
        let boundary = candidate.dense_points(240);
        if let Some(pose) = frame
            .sam31_proposal_masks
            .as_deref()
            .and_then(|p| preview_surface(frame, p, candidate))
            .and_then(|s| provisional_surface_pose(Some(s), &boundary))
        {
            let proposal = frame
                .sam31_proposal_masks
                .as_ref()
                .expect("field requires a source");
            draw_rotation_meridians(
                pixels,
                width,
                height,
                x,
                y,
                scale,
                Some(pose.rotation_center),
                None,
                Some(pose.relative_gaze),
                Some(pose.sphere_radius),
                Some((proposal.source_sequence % 180) as f64 * std::f64::consts::PI / 180.0),
                true,
                None,
                &[],
                &boundary,
                proposal.source_width,
                proposal.source_height,
            );
            contact_drawn = true;
        }
        for k in 0..boundary.len() {
            let a = project(boundary[k]);
            let b = project(boundary[(k + 1) % boundary.len()]);
            draw_line_clipped(
                pixels,
                width,
                height,
                a.0,
                a.1,
                b.0,
                b.1,
                if field.candidate.is_some() {
                    0x00ff_e080
                } else {
                    0x0000_ffff
                },
            );
        }
    }
    // A sparse optical heightmap: inward/outward displacement vectors plus
    // onset (blue), surface apex (gold), submerged limit (purple). The latter
    // is a diagnostic landmark, never a sample supplied to the conic refit.
    for sample in &field.samples {
        if let Some(shift) = sample.correction_px {
            let a = project(sample.origin);
            let b = project((
                sample.origin.0 + sample.normal.0 * shift,
                sample.origin.1 + sample.normal.1 * shift,
            ));
            draw_line_clipped(
                pixels,
                width,
                height,
                a.0,
                a.1,
                b.0,
                b.1,
                if shift < 0.0 {
                    0x0000_bfff
                } else {
                    0x00ff_a050
                },
            );
            fill_rect(pixels, width, height, b.0 - 1, b.1 - 1, 3, 3, 0x00ff_e080);
        }
        for (role, color) in [
            (1, 0x0000_d8c0),
            (2, 0x00ff_9850),
            (3, 0x0070_9fff),
            (4, 0x00ff_e080),
            (5, 0x00bb_70ff),
        ] {
            let landmark = &sample.landmarks[role];
            if landmark.supported() {
                let p = project((
                    sample.origin.0 + sample.normal.0 * landmark.offset_px,
                    sample.origin.1 + sample.normal.1 * landmark.offset_px,
                ));
                fill_rect(pixels, width, height, p.0 - 1, p.1 - 1, 2, 2, color);
            }
        }
    }
    draw_text(
        pixels,
        width,
        height,
        x + 4,
        y + 8,
        if field.candidate.is_some() {
            "TWEAKED CONTACT / PREVIEW"
        } else {
            "ABSTAINED / ORIGINAL CONTACT"
        },
        0x00ff_e080,
    );
    draw_text(
        pixels,
        width,
        height,
        x + 4,
        y + 24,
        &format!(
            "LOCAL {}/{}  {:.1}MS  MAX 4PX",
            field
                .samples
                .iter()
                .filter(|p| p.correction_px.is_some())
                .count(),
            field.samples.len(),
            field.elapsed_ms
        ),
        0x00ff_ffff,
    );
    draw_text(
        pixels,
        width,
        height,
        x + 4,
        y + 40,
        "TEAL/ORANGE INNER/OUTER BAND",
        0x00ff_ffff,
    );
    draw_text(
        pixels,
        width,
        height,
        x + 4,
        y + 56,
        if field.candidate.is_some() && !contact_drawn {
            "RIM ONLY / WAIT SOURCE SIGN"
        } else {
            "GOLD RIM / PURPLE SUBMERGED"
        },
        0x00ff_ffff,
    );
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_corpus_preview_is_cached_and_never_borrows_newer_crop_pixels() {
        // Opt-in integration fixture, not a repository dependency or a new
        // annotation UI. No label file is read by this renderer test.
        let Some(root) = std::env::var_os("BUTTERCUP_LIMBUS_UI_CORPUS_DIR") else {
            return;
        };
        let root = PathBuf::from(root);
        let reference = std::fs::read_to_string(root.join("sam-baseline.jsonl")).unwrap();
        let row = reference
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .find(|r| r["input"]["frame"]["sequence"] == 195)
            .unwrap();
        let src = &row["input"];
        assert_eq!(src["raw_offset"], 0);
        let f = &src["frame"];
        let w = f["width"].as_u64().unwrap() as usize;
        let h = f["height"].as_u64().unwrap() as usize;
        let packed = std::fs::read(src["raw_file"].as_str().unwrap()).unwrap();
        assert_eq!(packed.len(), src["raw_length"].as_u64().unwrap() as usize);
        let raw =
            raw10::try_unpack_raw10(&packed, w, h, f["stride"].as_u64().unwrap() as usize).unwrap();
        let point = |v: &serde_json::Value| (v[0].as_f64().unwrap(), v[1].as_f64().unwrap());
        let c = &row["candidates"][0];
        let e = &c["baseline_ellipse"];
        let ellipse = geometry::Ellipse {
            center: point(&e["center"]),
            major_radius: e["major_radius"].as_f64().unwrap(),
            minor_radius: e["minor_radius"].as_f64().unwrap(),
            angle: e["angle"].as_f64().unwrap(),
        };
        let retained = c["baseline_retained"]
            .as_array()
            .unwrap()
            .iter()
            .map(point)
            .collect();
        let proposal = Arc::new(sam31_outer::ProposalMasks {
            source_timestamp_ns: f["timestamp_ns"].as_u64().unwrap(),
            source_sequence: 195,
            source_width: w,
            source_height: h,
            source_raw: Arc::new(raw),
            eye_index: 1,
            source_sensor_origin: (
                f["sensor_x"].as_u64().unwrap() as u32,
                f["sensor_y"].as_u64().unwrap() as u32,
            ),
            outer_fit: Some(sam31_outer::OuterMaskFitReview {
                ellipse,
                retained_points: Arc::new(retained),
                flat_tire_points: Arc::new(vec![]),
                conic_segments: Arc::new(vec![]),
                upper_flat_tire: false,
                lower_flat_tire: false,
                source_component_area_px: std::f64::consts::PI
                    * ellipse.major_radius
                    * ellipse.minor_radius,
            }),
            ..Default::default()
        });
        let mut frame = crate::tests::control_eye_frame(195);
        frame.width = w;
        frame.height = h;
        frame.sam31_proposal_masks = Some(Arc::clone(&proposal));
        // The sign is a synthetic rendering fixture, not a gaze truth label.
        // Shape/pixels/corrections below are the actual native corpus result.
        frame.virtual_contact_surface_gaze = Some(SurfaceGazeSample {
            source_timestamp_ns: Some(proposal.source_timestamp_ns),
            frontal_equivalent_disk_area_px2: std::f64::consts::PI * ellipse.major_radius.powi(2),
            area_bucket: 0,
            quantized_frontal_disk_radius_px: ellipse.major_radius,
            near_surface_point_sensor_px: ellipse.center,
            relative_gaze: RelativeGazeVector::from_projected(0.0, -0.6).unwrap(),
            sign_resolved: true,
            sign_epoch: 7,
            kinematic_sign_correction: [false; 2],
            sign_diagnostics: None,
        });
        CACHE.with(|cache| *cache.borrow_mut() = None);
        let field = field_for(&frame).unwrap();
        assert!(
            field.candidate.is_some(),
            "corpus preview must exercise the trained model, not only fallback"
        );
        assert!(
            Arc::ptr_eq(&field, &field_for(&frame).unwrap()),
            "same source is inferred only once"
        );
        let (width, height) = (w * 2, h * 2);
        let render = |frame: &EyeFrame| {
            let mut p = vec![0; width * height];
            draw(&mut p, width, height, 0, 0, 2, frame);
            draw_text(
                &mut p,
                width,
                height,
                4,
                72,
                "RENDER TEST / SYNTHETIC SIGN",
                0x00ff_c857,
            );
            p
        };
        let pixels = render(&frame);
        assert!(
            pixels.iter().filter(|p| **p == 0x00ff_00ff).count() > 30,
            "convex meridians must be drawn"
        );
        assert!(
            pixels.iter().filter(|p| **p == 0x00ff_e080).count() > 30,
            "refined rim must be drawn"
        );
        frame.width += 40;
        frame.height += 24;
        frame.sensor_x += 100;
        frame.sensor_y += 200;
        frame.quad_color = Arc::new(vec![0; frame.width * frame.height]);
        assert_eq!(
            render(&frame),
            pixels,
            "review pixels, geometry and clipping all use the SAM exposure, not newer ROI"
        );
        export_eye_ppm(
            &root.join("refined-corpus-preview.ppm"),
            &pixels,
            width,
            height,
        )
        .unwrap();
        let mut baseline = vec![0; width * height];
        draw_sam31_outer_iris_fit(&mut baseline, width, height, 0, 0, 2, 195, Some(&proposal));
        export_eye_ppm(
            &root.join("baseline-corpus-preview.ppm"),
            &baseline,
            width,
            height,
        )
        .unwrap();
    }

    #[test]
    fn changed_ellipse_cannot_borrow_a_newer_surface_sign() {
        let mut frame = crate::tests::control_eye_frame(120);
        let proposal = sam31_outer::ProposalMasks {
            source_timestamp_ns: 10,
            ..Default::default()
        };
        let ellipse = geometry::Ellipse {
            center: (100.0, 100.0),
            major_radius: 80.0,
            minor_radius: 60.0,
            angle: 0.0,
        };
        assert!(preview_surface(&frame, &proposal, ellipse).is_none());
        let surface = SurfaceGazeSample {
            source_timestamp_ns: Some(11),
            frontal_equivalent_disk_area_px2: std::f64::consts::PI * 80.0 * 80.0,
            area_bucket: 0,
            quantized_frontal_disk_radius_px: 80.0,
            near_surface_point_sensor_px: (100.0, 100.0),
            relative_gaze: RelativeGazeVector::from_projected(0.0, -0.6).unwrap(),
            sign_resolved: true,
            sign_epoch: 7,
            kinematic_sign_correction: [false; 2],
            sign_diagnostics: None,
        };
        frame.virtual_contact_surface_gaze = Some(surface);
        assert!(
            preview_surface(&frame, &proposal, ellipse).is_none(),
            "newer sign cannot be borrowed"
        );
        frame.virtual_contact_surface_gaze = Some(SurfaceGazeSample {
            source_timestamp_ns: Some(10),
            ..surface
        });
        let refined = preview_surface(&frame, &proposal, ellipse).unwrap();
        assert!(refined.relative_gaze.is_camera_facing());
        assert!(
            refined.relative_gaze.down < 0.0,
            "same-exposure sign branch must survive"
        );
        assert_eq!(refined.sign_epoch, surface.sign_epoch);
        frame.virtual_contact_surface_gaze = Some(SurfaceGazeSample {
            sign_resolved: false,
            ..refined
        });
        assert!(
            preview_surface(&frame, &proposal, ellipse).is_none(),
            "unsigned is not resolved"
        );
        frame.virtual_contact_surface_gaze = Some(SurfaceGazeSample {
            relative_gaze: RelativeGazeVector {
                toward_camera: -refined.relative_gaze.toward_camera,
                ..refined.relative_gaze
            },
            ..refined
        });
        assert!(
            preview_surface(&frame, &proposal, ellipse).is_none(),
            "concave source must never draw"
        );
    }
}
