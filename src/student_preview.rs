//! Isolated, source-registered Eye Student inspection layers. F only chooses
//! pixels to draw: this module cannot publish observations or modify gaze,
//! calibration, the model, or its conic/temporal state.
use super::*;
use std::cell::RefCell;

const LIMBUS: u32 = 0x0000_ffff;
const PUPIL: u32 = 0x00ff_c857;
const KEEP: u32 = 0x005f_ff69;
const REJECT: u32 = 0x00ff_3ca6;
const REFINEMENT_BEFORE: u32 = 0x0040_dfff;
const REFINEMENT_AFTER: u32 = 0x00ff_60b5;
const OTHER_MASKS: [u32; 4] = [0x00ff_a020, 0x00dd_60ff, 0x00ff_4050, 0x0060_a0ff];

struct CachedSource {
    proposal: Arc<sam31_outer::ProposalMasks>,
    mode: ViewMode,
    edge: EdgeMapVariant,
    pixels: Arc<Vec<u32>>,
}

// At most one color conversion per eye/source/view, not one per redraw. This
// cache contains presentation pixels only; it has no motion or model memory.
thread_local! {
    static SOURCES: RefCell<[Option<CachedSource>; 2]> = const { RefCell::new([None, None]) };
}

pub(super) fn source(frame: &EyeFrame) -> Option<&Arc<sam31_outer::ProposalMasks>> {
    let p = frame.sam31_proposal_masks.as_ref()?;
    (p.eye_index < 2
        && frame.eye_id == p.eye_index as u32 + 1
        && p.source_width > 0
        && p.source_height > 0
        && p.source_width.checked_mul(p.source_height) == Some(p.source_raw.len())
        && frame
            .gaze_authority_sam_prompt_generation
            .is_none_or(|g| g == p.prompt_generation))
    .then_some(p)
}

pub(super) fn source_dimensions(frame: &EyeFrame) -> Option<(usize, usize)> {
    source(frame).map(|p| (p.source_width, p.source_height))
}

pub(super) fn source_pixels(frame: &EyeFrame, mode: ViewMode) -> Option<Arc<Vec<u32>>> {
    let p = source(frame)?;
    Some(SOURCES.with(|sources| {
        let mut sources = sources.borrow_mut();
        if let Some(cached) = &sources[p.eye_index] {
            if Arc::ptr_eq(&cached.proposal, p)
                && cached.mode == mode
                && cached.edge == frame.edge_map_variant
            {
                return Arc::clone(&cached.pixels);
            }
        }
        let color = || {
            color_preview(
                &p.source_raw,
                p.source_width,
                p.source_height,
                p.source_sensor_origin.0,
                p.source_sensor_origin.1,
                100,
                None,
            )
        };
        let mut pixels = match mode {
            ViewMode::QuadColor
            | ViewMode::BlueFilter
            | ViewMode::RedFilter
            | ViewMode::GreenFilter => color(),
            ViewMode::RawColor => raw10_color_preview(
                &p.source_raw,
                p.source_width,
                p.source_sensor_origin.0,
                p.source_sensor_origin.1,
                100,
            ),
            ViewMode::RawLuma => raw10_luma_preview(&p.source_raw, 100),
            ViewMode::QuadLuma => {
                quad_luma_preview(&p.source_raw, p.source_width, p.source_height, 100)
            }
            ViewMode::Canny => {
                let luma = quad_luma_preview(&p.source_raw, p.source_width, p.source_height, 100);
                edge_map_preview(
                    &luma,
                    p.source_width,
                    p.source_height,
                    frame.edge_map_variant,
                )
            }
            ViewMode::SpecularMap
            | ViewMode::CrossPolarized
            | ViewMode::Diffuse
            | ViewMode::IlluminationMap
            | ViewMode::AlbedoMap => {
                // No current-frame motion or illumination buffer is borrowed.
                // The diffuse fallback is spatial, not temporally compensated.
                let views = specular_map::SpecularMapTracker::default().observe(
                    &color(),
                    p.source_width,
                    p.source_height,
                    p.source_sensor_origin.0,
                    p.source_sensor_origin.1,
                    None,
                );
                match mode {
                    ViewMode::SpecularMap => views.specular_map,
                    ViewMode::CrossPolarized => views.cross_polarized,
                    ViewMode::Diffuse => views.diffuse,
                    ViewMode::IlluminationMap => views.illumination,
                    ViewMode::AlbedoMap => views.albedo,
                    _ => unreachable!(),
                }
            }
        };
        for pixel in &mut pixels {
            *pixel = mode.filter_pixel(*pixel);
        }
        let pixels = Arc::new(pixels);
        sources[p.eye_index] = Some(CachedSource {
            proposal: Arc::clone(p),
            mode,
            edge: frame.edge_map_variant,
            pixels: Arc::clone(&pixels),
        });
        pixels
    }))
}

struct Layer<'a> {
    pixels: &'a mut [u32],
    width: usize,
    height: usize,
}

impl Layer<'_> {
    fn line(&mut self, a: (f64, f64), b: (f64, f64), color: u32) {
        if [a.0, a.1, b.0, b.1].into_iter().all(f64::is_finite) {
            draw_line_clipped(
                self.pixels,
                self.width,
                self.height,
                a.0.round() as i32,
                a.1.round() as i32,
                b.0.round() as i32,
                b.1.round() as i32,
                color,
            );
        }
    }

    fn ellipse(&mut self, ellipse: geometry::Ellipse, color: u32) {
        if ![
            ellipse.center.0,
            ellipse.center.1,
            ellipse.major_radius,
            ellipse.minor_radius,
            ellipse.angle,
        ]
        .into_iter()
        .all(f64::is_finite)
            || ellipse.major_radius <= 0.0
            || ellipse.minor_radius <= 0.0
        {
            return;
        }
        let points = ellipse.dense_points(240);
        for index in 0..points.len() {
            self.line(points[index], points[(index + 1) % points.len()], color);
        }
    }

    fn points(&mut self, points: &[(f64, f64)], color: u32) {
        for &(x, y) in points {
            if x.is_finite() && y.is_finite() {
                fill_rect(
                    self.pixels,
                    self.width,
                    self.height,
                    x.round() as i32 - 1,
                    y.round() as i32 - 1,
                    2,
                    2,
                    color,
                );
            }
        }
    }

    fn status(&mut self, message: &str) {
        // Missing evidence is explicit. Successful lean layers have no status
        // text covering the anatomy; the surrounding UI supplies view/clock.
        let rows = message.lines().map(str::to_owned).collect::<Vec<_>>();
        for (row, line) in viewer_ui::wrapped_lines(&rows, self.width.saturating_sub(6) / 6)
            .iter()
            .enumerate()
            .take(self.height.saturating_sub(6) / 9)
        {
            draw_text_scaled(
                self.pixels,
                self.width,
                self.height,
                3,
                3 + row as i32 * 9,
                line,
                PUPIL,
                1,
            );
        }
    }

    fn masks(
        &mut self,
        p: &sam31_outer::ProposalMasks,
        selected_only: bool,
        outline: bool,
    ) -> usize {
        let Some(answer) = p.semantic.as_ref() else {
            self.status("NO STUDENT MASK");
            return 0;
        };
        if answer.width == 0 || answer.height == 0 {
            return 0;
        }
        let mut count = 0;
        for mask in &answer.masks {
            let selected = answer.selected_query == Some(mask.query);
            if selected_only && !selected {
                continue;
            }
            if answer.width.checked_mul(answer.height) != Some(mask.pixels.len()) {
                continue;
            }
            let color = if selected {
                LIMBUS
            } else {
                OTHER_MASKS[mask.query % OTHER_MASKS.len()]
            };
            count += 1;
            if outline {
                // Boundary indices have no contour ordering. Trace exposed
                // mask-cell edges instead of joining arbitrary neighboring
                // indices or painting enlarged white/colored point markers.
                for &index in mask.boundary_pixels.iter() {
                    let i = index as usize;
                    if i >= mask.pixels.len() || mask.pixels[i] == 0 {
                        continue;
                    }
                    let (x, y) = (i % answer.width, i / answer.width);
                    let (x0, x1) = (
                        x * self.width / answer.width,
                        (x + 1) * self.width / answer.width,
                    );
                    let (y0, y1) = (
                        y * self.height / answer.height,
                        (y + 1) * self.height / answer.height,
                    );
                    let a = (x0 as f64, y0 as f64);
                    let b = (x1.saturating_sub(1) as f64, y0 as f64);
                    let c = (x1.saturating_sub(1) as f64, y1.saturating_sub(1) as f64);
                    let d = (x0 as f64, y1.saturating_sub(1) as f64);
                    if y == 0 || mask.pixels[i - answer.width] == 0 {
                        self.line(a, b, color);
                    }
                    if y + 1 == answer.height || mask.pixels[i + answer.width] == 0 {
                        self.line(d, c, color);
                    }
                    if x == 0 || mask.pixels[i - 1] == 0 {
                        self.line(a, d, color);
                    }
                    if x + 1 == answer.width || mask.pixels[i + 1] == 0 {
                        self.line(b, c, color);
                    }
                }
            } else {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let i = (y * answer.height / self.height) * answer.width
                            + x * answer.width / self.width;
                        if mask.pixels[i] == 0 {
                            continue;
                        }
                        let pixel = &mut self.pixels[y * self.width + x];
                        let alpha: u32 = if selected { 64 } else { 36 };
                        let channel = |shift: u32| {
                            (((*pixel >> shift) & 255u32) * (255u32 - alpha)
                                + ((color >> shift) & 255u32) * alpha
                                + 127u32)
                                / 255u32
                        };
                        *pixel = channel(16) << 16 | channel(8) << 8 | channel(0);
                    }
                }
            }
        }
        if selected_only && count == 0 {
            self.status("NO SELECTED STUDENT MASK");
        }
        count
    }

    fn contact(&mut self, frame: &EyeFrame, p: &sam31_outer::ProposalMasks) -> usize {
        let Some(review) = p.outer_fit.as_ref() else {
            self.status("NO LIMBUS FIT");
            return 0;
        };
        let joint_ellipse = joint_gaze_live::source_ellipse(frame);
        let ellipse = joint_ellipse.unwrap_or(review.ellipse);
        self.ellipse(ellipse, LIMBUS);
        let surface = match contact_surface(frame, p, joint_ellipse.is_some()) {
            Ok(surface) => surface,
            Err(reason) => {
                self.status(reason);
                return 0;
            }
        };
        let boundary = ellipse.dense_points(240);
        let Some(pose) = provisional_surface_pose(Some(surface), &boundary) else {
            self.status("SOURCE CONTACT GEOMETRY UNAVAILABLE");
            return 0;
        };
        // Four thin globe meridians only: no retained/rejected points, pupil
        // ring, center circles, moving sweep/trail, or implicit laser layer.
        draw_rotation_meridians(
            self.pixels,
            self.width,
            self.height,
            0,
            0,
            1,
            Some(pose.rotation_center),
            None,
            Some(pose.relative_gaze),
            Some(pose.sphere_radius),
            None,
            true,
            None,
            &[],
            &boundary,
            self.width,
            self.height,
        );
        if !surface.sign_resolved {
            // The stereo scene also displays the conditional candidate. Keep
            // it inspectable here, but never promote it to accepted gaze.
            self.status(
                if surface
                    .sign_diagnostics
                    .is_some_and(|d| d.orientation_reference_ready == Some(false))
                {
                    "UNCONFIRMED DIRECTION\nSHIFT+M TO REORIENT"
                } else {
                    "UNCONFIRMED DIRECTION"
                },
            );
        }
        1
    }
}

fn contact_surface(
    frame: &EyeFrame,
    p: &sam31_outer::ProposalMasks,
    matched_joint: bool,
) -> Result<SurfaceGazeSample, &'static str> {
    if frame.gaze_policy_error.is_some() {
        return Err("WAITING FOR CURRENT GAZE SETTINGS");
    }
    if frame.joint_gaze_active && !matched_joint {
        return Err("WAITING FOR MATCHING STEREO SOURCE");
    }
    let surface = frame
        .virtual_contact_surface_gaze
        .filter(|s| s.source_timestamp_ns == Some(p.source_timestamp_ns))
        .ok_or("WAITING FOR MATCHING SURFACE SOURCE")?;
    if !surface.relative_gaze.is_camera_facing() {
        return Err("NO CAMERA-FACING SOURCE SURFACE");
    }
    Ok(surface)
}

// Read only the worker's immutable decision. A rejected proposal is not an
// "after" edge, and selecting this view must never load/run the refiner.
fn refinement_edges(
    p: &sam31_outer::ProposalMasks,
) -> Option<(
    &sam31_outer::OuterMaskFitReview,
    Option<&sam31_outer::OuterMaskFitReview>,
    String,
)> {
    let current = p.outer_fit.as_ref()?;
    match p.limbus_refinement.as_ref() {
        Some(attempt) if attempt.applied => {
            let before = &attempt.baseline;
            if before.retained_points.len() != current.retained_points.len() {
                return Some((
                    before,
                    None,
                    "SOURCE EDGE CORRESPONDENCE UNAVAILABLE".into(),
                ));
            }
            let maximum = before
                .retained_points
                .iter()
                .zip(current.retained_points.iter())
                .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                .fold(0_f64, f64::max);
            Some((
                before,
                Some(current),
                format!("REFINED / MAX SHIFT {maximum:.2} PX"),
            ))
        }
        Some(attempt) => Some((
            &attempt.baseline,
            None,
            format!("UNCHANGED / {}", attempt.status),
        )),
        None => Some((
            current,
            None,
            "BASELINE ONLY / SHIFT+F TO TOGGLE REFINEMENT".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_refinement_edges(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    scale: usize,
    p: &sam31_outer::ProposalMasks,
) -> usize {
    let comparison = refinement_edges(p);
    let mut count = 0;
    if let Some((before, after, _)) = &comparison {
        // Draw at presentation resolution so a subpixel native correction
        // remains visible when the source ROI is magnified. Trace only the
        // retained samples and their original runs; never close missing arcs
        // with a fitted ellipse, an eyelid chord, or a rejected model field.
        let project = |q: (f64, f64)| {
            (q.0.is_finite()
                && q.1.is_finite()
                && q.0 >= 0.
                && q.1 >= 0.
                && q.0 < p.source_width as f64
                && q.1 < p.source_height as f64)
                .then_some((x as f64 + q.0 * scale as f64, y as f64 + q.1 * scale as f64))
        };
        let mut paint = |q: (f64, f64), color: u32| {
            let (px, py) = (q.0.round() as i32, q.1.round() as i32);
            if px >= x.max(0)
                && py >= y.max(0)
                && px < (x + (p.source_width * scale) as i32).min(width as i32)
                && py < (y + (p.source_height * scale) as i32).min(height as i32)
            {
                pixels[py as usize * width + px as usize] = color;
            }
        };
        for (review, color, dashed) in [
            Some((*before, REFINEMENT_BEFORE, false)),
            after.map(|a| (a, REFINEMENT_AFTER, true)),
        ]
        .into_iter()
        .flatten()
        {
            for run in review.conic_segments.iter() {
                let mut travelled = 0.;
                for pair in run.windows(2) {
                    let Some((a, b)) = review
                        .retained_points
                        .get(pair[0])
                        .zip(review.retained_points.get(pair[1]))
                    else {
                        continue;
                    };
                    let Some((a, b)) = project(*a).zip(project(*b)) else {
                        continue;
                    };
                    let length = (a.0 - b.0).hypot(a.1 - b.1);
                    let steps = length.ceil().clamp(1., 8192.) as usize;
                    for k in 0..=steps {
                        let t = k as f64 / steps as f64;
                        if !dashed || ((travelled + t * length) as usize / 4) % 2 == 0 {
                            paint((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)), color);
                        }
                    }
                    travelled += length;
                }
            }
            // Isolated retained observations remain visible even without a run.
            for q in review.retained_points.iter().filter_map(|q| project(*q)) {
                paint(q, color);
            }
        }
        count = before.retained_points.len();
    }
    let font = scale.clamp(1, 2);
    let row = 9 * font as i32;
    let columns = ((p.source_width * scale).saturating_sub(8) / (6 * font)).max(1);
    let label = comparison
        .as_ref()
        .map_or("NO SOURCE LIMBUS EDGES".to_owned(), |(_, _, status)| {
            status.clone()
        });
    let status = format!("SOURCE {} / {label}", p.source_sequence);
    let status_lines = viewer_ui::wrapped_lines(&[status], columns);
    fill_rect(
        pixels,
        width,
        height,
        x,
        y,
        (p.source_width * scale) as i32,
        (status_lines.len().min(2) as i32 * row + 6).min((p.source_height * scale) as i32),
        0x0010_1010,
    );
    for (i, line) in status_lines.iter().take(2).enumerate() {
        draw_text_scaled(
            pixels,
            width,
            height,
            x + 3,
            y + 3 + i as i32 * row,
            line,
            0x00ff_ffff,
            font as i32,
        );
    }
    let legend = viewer_ui::wrapped_lines(&["CYAN BEFORE / PINK DASHES AFTER".into()], columns);
    let legend_rows = legend.len().min(2);
    let legend_height = (legend_rows as i32 * row + 3).min((p.source_height * scale) as i32);
    fill_rect(
        pixels,
        width,
        height,
        x,
        y + (p.source_height * scale) as i32 - legend_height,
        (p.source_width * scale) as i32,
        legend_height,
        0x0010_1010,
    );
    for (i, line) in legend.iter().take(legend_rows).enumerate() {
        draw_text_scaled(
            pixels,
            width,
            height,
            x + 3,
            y + (p.source_height * scale) as i32 - (legend_rows - i) as i32 * row,
            line,
            REFINEMENT_BEFORE,
            font as i32,
        );
    }
    count
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
    mode: ViewMode,
    overlay: RoiOverlayMode,
) -> usize {
    let (Some(p), Some(base)) = (source(frame), source_pixels(frame, mode)) else {
        draw_text_scaled(
            pixels,
            width,
            height,
            x + 3,
            y + 3,
            "WAITING FOR DETECTOR SOURCE",
            PUPIL,
            1,
        );
        return 0;
    };
    let mut image = base.as_ref().clone();
    let mut layer = Layer {
        pixels: &mut image,
        width: p.source_width,
        height: p.source_height,
    };
    let count = match overlay {
        RoiOverlayMode::LimbusRefinementEdges => 0, // drawn after magnifying the source pixels below
        RoiOverlayMode::StereoContributions => {
            viewer_ui::draw_stereo_segments(layer.pixels, layer.width, layer.height, frame, p)
        }
        RoiOverlayMode::SamOuterIrisMasks => layer.masks(p, true, false),
        RoiOverlayMode::SamSegmentationOnly => layer.masks(p, false, false),
        RoiOverlayMode::StudentMaskOutline => layer.masks(p, false, true),
        RoiOverlayMode::StudentPupilOnly => {
            if let Some(pupil) = p.inner_pupil_fit {
                layer.ellipse(pupil.ellipse, PUPIL);
                1
            } else {
                layer.status("NO SOURCE PUPIL VOID FIT");
                0
            }
        }
        RoiOverlayMode::SamDeflattenedVirtualContact => layer.contact(frame, p),
        RoiOverlayMode::SamTweakedContactGeometry => {
            limbus_refiner_view::draw(layer.pixels, layer.width, layer.height, 0, 0, 1, frame)
        }
        RoiOverlayMode::StudentSourceRaw => 0,
        RoiOverlayMode::SamOuterIrisFit
        | RoiOverlayMode::SamConicSegments
        | RoiOverlayMode::StudentEllipseOnly => {
            if let Some(review) = &p.outer_fit {
                match overlay {
                    RoiOverlayMode::StudentEllipseOnly => {
                        layer.ellipse(review.ellipse, LIMBUS);
                    }
                    RoiOverlayMode::SamOuterIrisFit => {
                        layer.ellipse(review.ellipse, LIMBUS);
                        layer.points(&review.retained_points, KEEP);
                        layer.points(&review.flat_tire_points, REJECT);
                    }
                    RoiOverlayMode::SamConicSegments => {
                        for (index, segment) in review.conic_segments.iter().enumerate() {
                            for pair in segment.windows(2) {
                                if let (Some(a), Some(b)) = (
                                    review.retained_points.get(pair[0]),
                                    review.retained_points.get(pair[1]),
                                ) {
                                    layer.line(
                                        *a,
                                        *b,
                                        conic_segment_color(index, review.conic_segments.len()),
                                    );
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                review.retained_points.len()
            } else {
                layer.status("NO SOURCE LIMBUS FIT");
                0
            }
        }
        _ => unreachable!("not a Student source inspection layer"),
    };
    let scale = scale.max(1);
    for row in 0..p.source_height {
        for column in 0..p.source_width {
            fill_rect(
                pixels,
                width,
                height,
                x + (column * scale) as i32,
                y + (row * scale) as i32,
                scale as i32,
                scale as i32,
                image[row * p.source_width + column],
            );
        }
    }
    if overlay == RoiOverlayMode::LimbusRefinementEdges {
        return draw_refinement_edges(pixels, width, height, x, y, scale, p);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> EyeFrame {
        let (width, height) = (160, 120);
        let ellipse = geometry::Ellipse {
            center: (80.0, 64.0),
            major_radius: 45.0,
            minor_radius: 28.0,
            angle: 0.18,
        };
        let pupil = geometry::Ellipse {
            center: (80.0, 66.0),
            major_radius: 14.0,
            minor_radius: 9.0,
            angle: 0.18,
        };
        let dense = ellipse.dense_points(128);
        let retained = dense
            .iter()
            .copied()
            .filter(|p| p.1 > 42.0)
            .collect::<Vec<_>>();
        let excluded = (55..=104)
            .step_by(2)
            .map(|x| (x as f64, 42.0))
            .collect::<Vec<_>>();
        let raw = (0..width * height)
            .map(|i| {
                let point = ((i % width) as f64, (i / width) as f64);
                if geometry::ellipse_coordinate(point, pupil) <= 1.0 {
                    120
                } else if geometry::ellipse_coordinate(point, ellipse) <= 1.0 && point.1 >= 42.0 {
                    240 + (((i % width) / 4 + (i / width) / 4) % 5) as u16 * 15
                } else {
                    500
                }
            })
            .collect::<Vec<_>>();
        let mask = (0..80 * 60)
            .map(|i| {
                let p = ((i % 80) as f64 * 2.0 + 1.0, (i / 80) as f64 * 2.0 + 1.0);
                (geometry::ellipse_coordinate(p, ellipse) <= 1.0 && p.1 >= 42.0) as u8
            })
            .collect::<Vec<_>>();
        let boundary = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                (v != 0
                    && (i < 80
                        || i + 80 >= mask.len()
                        || i % 80 == 0
                        || i % 80 == 79
                        || mask[i - 1] == 0
                        || mask[i + 1] == 0
                        || mask[i - 80] == 0
                        || mask[i + 80] == 0))
                    .then_some(i as u32)
            })
            .collect();
        let proposal = Arc::new(sam31_outer::ProposalMasks {
            prompt_generation: 2,
            eye_index: 0,
            source_sequence: 195,
            source_timestamp_ns: 1_000_000_000,
            source_sensor_origin: (3000, 2400),
            source_width: width,
            source_height: height,
            source_raw: Arc::new(raw),
            semantic: Some(sam31_outer::SemanticProposalMasks {
                prompt_index: sam31_outer::OUTER_IRIS_PROMPT,
                width: 80,
                height: 60,
                selected_query: Some(0),
                tile_slice: None,
                masks: vec![sam31_outer::ProposalMask {
                    query: 0,
                    score: 0.9,
                    pixels: Arc::new(mask),
                    boundary_pixels: Arc::new(boundary),
                }],
            }),
            outer_fit: Some(sam31_outer::OuterMaskFitReview {
                ellipse,
                source_component_area_px: 3500.0,
                conic_segments: Arc::new(vec![
                    (0..retained.len() / 2).collect(),
                    (retained.len() / 2..retained.len()).collect(),
                ]),
                retained_points: Arc::new(retained),
                flat_tire_points: Arc::new(excluded),
                upper_flat_tire: true,
                lower_flat_tire: false,
            }),
            inner_pupil_fit: Some(sam31_outer::PupilVoidFitReview {
                ellipse: pupil,
                raw_support: sam31_outer::RawRingSupport {
                    score: 0.8,
                    points: 32,
                    positive_fraction: 0.9,
                    strong_sectors: 12,
                },
            }),
            ..Default::default()
        });
        let mut frame = crate::tests::control_eye_frame(200);
        frame.segmentation_mode = SegmentationMode::EyeStudent;
        frame.eye_identity_present = true;
        frame.width = width + 32;
        frame.height = height + 24;
        frame.quad_color = Arc::new(vec![0x0012_3456; frame.width * frame.height]);
        frame.sensor_x = 3036;
        frame.sensor_y = 2420;
        frame.timestamp_ns = proposal.source_timestamp_ns + 100_000_000;
        frame.gaze_authority_sam_prompt_generation = Some(proposal.prompt_generation);
        frame.virtual_contact_surface_gaze = Some(SurfaceGazeSample {
            source_timestamp_ns: Some(proposal.source_timestamp_ns),
            frontal_equivalent_disk_area_px2: std::f64::consts::PI * ellipse.major_radius.powi(2),
            area_bucket: 1,
            quantized_frontal_disk_radius_px: ellipse.major_radius,
            near_surface_point_sensor_px: (3080.0, 2464.0),
            relative_gaze: RelativeGazeVector::from_projected(-0.1, 0.5).unwrap(),
            sign_resolved: true,
            sign_epoch: 7,
            kinematic_sign_correction: [false; 2],
            sign_diagnostics: None,
        });
        frame.sam31_proposal_masks = Some(proposal);
        frame.eye_laser_enabled = true;
        frame.centimeter_scale = Some(CentimeterScaleEstimate {
            estimate_px: 72.0,
            minimum_px: 50.0,
            maximum_px: 96.0,
            movement_fraction: 0.2,
            reacquisition_count: 2,
            semantic_center_sensor: [3080.0, 2464.0],
        });
        // These deliberately incompatible Native predictions must not leak
        // into a Student source-specific inspection layer.
        frame.outer_iris_points = Arc::new(
            geometry::Ellipse {
                center: (33.0, 24.0),
                major_radius: 24.0,
                minor_radius: 20.0,
                angle: 1.2,
            }
            .dense_points(64),
        );
        frame
    }

    fn render(frame: &EyeFrame, overlay: RoiOverlayMode) -> Vec<u32> {
        let (w, h) = if overlay.uses_student_source() {
            source_dimensions(frame).unwrap()
        } else {
            (frame.width, frame.height)
        };
        let (canvas_w, canvas_h) = (w + 16, h + 40);
        let mut pixels = vec![0; canvas_w * canvas_h];
        draw_eye_with_spatial_debug(
            &mut pixels,
            canvas_w,
            canvas_h,
            frame,
            8,
            28,
            ViewMode::QuadColor,
            "",
            false,
            true,
            1,
            overlay,
            None,
        );
        (0..h)
            .flat_map(|y| {
                pixels[(y + 28) * canvas_w + 8..(y + 28) * canvas_w + 8 + w]
                    .iter()
                    .copied()
            })
            .collect()
    }

    #[test]
    fn student_cycle_adds_sparse_layers_without_changing_sam() {
        let method = SegmentationMode::EyeStudent;
        let available = RoiOverlayMode::available(method);
        assert_eq!(available.len(), 15);
        assert!(available.contains(&RoiOverlayMode::StereoContributions));
        assert!(available.contains(&RoiOverlayMode::PinkWellSightline));
        assert_eq!(available.last(), Some(&RoiOverlayMode::FullDiagnostics));
        let mut mode = available[0];
        for (i, expected) in available.iter().enumerate() {
            assert_eq!(mode, *expected);
            assert_eq!(mode.position_for(method), (i + 1, 15));
            assert!(!mode.label_for(method).contains("ENTER EDIT"));
            mode = mode.cycled_for(method);
        }
        assert_eq!(mode, available[0]);
        assert_eq!(RoiOverlayMode::available(SegmentationMode::Sam31).len(), 11);
        for mode in [
            RoiOverlayMode::StudentMaskOutline,
            RoiOverlayMode::StudentEllipseOnly,
            RoiOverlayMode::StudentPupilOnly,
            RoiOverlayMode::StudentSourceRaw,
        ] {
            assert!(!RoiOverlayMode::available(SegmentationMode::Sam31).contains(&mode));
            assert_eq!(
                mode.normalized_for(SegmentationMode::Sam31),
                RoiOverlayMode::FullDiagnostics
            );
        }
    }

    #[test]
    fn student_sparse_views_draw_only_the_selected_evidence_layer() {
        let frame = fixture();
        let source = render(&frame, RoiOverlayMode::StudentSourceRaw);
        assert_eq!(source, *source_pixels(&frame, ViewMode::QuadColor).unwrap());
        let geom_colors = [LIMBUS, PUPIL, KEEP, REJECT, 0x00ff_00ff];
        let mask = render(&frame, RoiOverlayMode::SamOuterIrisMasks);
        assert_ne!(mask, source);
        for color in geom_colors {
            assert!(
                !mask.contains(&color),
                "default mask has unexpected {color:x} overlay"
            );
        }
        let outline = render(&frame, RoiOverlayMode::StudentMaskOutline);
        let ellipse = render(&frame, RoiOverlayMode::StudentEllipseOnly);
        for lean in [&outline, &ellipse] {
            assert!(lean.contains(&LIMBUS));
            for color in [PUPIL, KEEP, REJECT, 0x00ff_00ff] {
                assert!(!lean.contains(&color));
            }
        }
        assert_ne!(
            outline, ellipse,
            "measured flat-tired outline is not the inferred complete ellipse"
        );
        let fit = render(&frame, RoiOverlayMode::SamOuterIrisFit);
        for color in [LIMBUS, KEEP, REJECT] {
            assert!(fit.contains(&color));
        }
        let changed = |image: &[u32]| image.iter().zip(&source).filter(|(a, b)| a != b).count();
        assert!(
            changed(&ellipse) < changed(&fit),
            "ellipse-only should actually remove point clutter"
        );
        let arcs = render(&frame, RoiOverlayMode::SamConicSegments);
        for color in [LIMBUS, KEEP, REJECT, 0x00ff_00ff] {
            assert!(!arcs.contains(&color));
        }
        assert!(changed(&arcs) < changed(&fit));
        let pupil = render(&frame, RoiOverlayMode::StudentPupilOnly);
        assert!(pupil.contains(&PUPIL));
        for color in [LIMBUS, KEEP, REJECT, 0x00ff_00ff] {
            assert!(!pupil.contains(&color));
        }
        let contact = render(&frame, RoiOverlayMode::SamDeflattenedVirtualContact);
        assert!(contact.contains(&0x00ff_00ff));
        for color in [KEEP, REJECT] {
            assert!(!contact.contains(&color));
        }
        let mut without_laser = frame.clone();
        without_laser.eye_laser_enabled = false;
        assert_eq!(
            render(&without_laser, RoiOverlayMode::SamDeflattenedVirtualContact),
            contact
        );
        assert_eq!(
            render(&frame, RoiOverlayMode::Clean),
            *frame.quad_color,
            "clean live ROI must not contain scale/reticles either"
        );
        if let Some(path) = std::env::var_os("BUTTERCUP_STUDENT_VIEW_TEST_EXPORT") {
            let path = PathBuf::from(path);
            std::fs::create_dir_all(&path).unwrap();
            for (name, pixels) in [
                ("source", source),
                ("mask", mask),
                ("outline", outline),
                ("ellipse", ellipse),
                ("fit-review", fit),
                ("arcs", arcs),
                ("pupil", pupil),
                ("contact", contact),
            ] {
                export_eye_ppm(&path.join(format!("student-{name}.ppm")), &pixels, 160, 120)
                    .unwrap();
            }
        }
    }

    #[test]
    fn student_source_views_survive_reframing_without_mutating_gaze_or_clocks() {
        let frame = fixture();
        let original_gaze = mouse_gaze_surface(&frame);
        let original_proposal = Arc::clone(frame.sam31_proposal_masks.as_ref().unwrap());
        let mut reframed = frame.clone();
        reframed.width += 64;
        reframed.height += 32;
        reframed.quad_color = Arc::new(vec![0x0098_7654; reframed.width * reframed.height]);
        reframed.sensor_x += 200;
        reframed.sensor_y += 100;
        reframed.timestamp_ns += 30_000_000;
        reframed.sequence += 1;
        for &mode in RoiOverlayMode::available(SegmentationMode::EyeStudent) {
            if !mode.uses_student_source() {
                continue;
            }
            assert_eq!(
                render(&frame, mode),
                render(&reframed, mode),
                "{mode:?} borrowed a newer crop/clock"
            );
        }
        assert_eq!(mouse_gaze_surface(&frame), original_gaze);
        assert_eq!(mouse_gaze_surface(&reframed), original_gaze);
        assert!(Arc::ptr_eq(
            frame.sam31_proposal_masks.as_ref().unwrap(),
            &original_proposal
        ));
        assert_eq!(original_proposal.source_sequence, 195);
        assert_eq!(original_proposal.source_timestamp_ns, 1_000_000_000);
        assert_eq!(
            frame.gaze_authority_generation,
            reframed.gaze_authority_generation
        );
        assert_eq!(source_dimensions(&reframed), Some((160, 120)));
        assert!(Arc::ptr_eq(
            &source_pixels(&frame, ViewMode::QuadColor).unwrap(),
            &source_pixels(&frame, ViewMode::QuadColor).unwrap()
        ));
    }

    #[test]
    fn refinement_edges_use_only_the_accepted_source_decision() {
        let mut frame = fixture();
        let p = Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap());
        let before = p.outer_fit.as_ref().unwrap().clone();
        let mut after = before.clone();
        after.retained_points = Arc::new(
            before
                .retained_points
                .iter()
                .map(|q| (q.0 + 0.35, q.1 + 0.25))
                .collect(),
        );
        after.ellipse.center.0 += 0.35;
        after.ellipse.center.1 += 0.25;
        // A deliberately unrelated field must not replace the worker's final
        // accepted edge product or make a rejected candidate look accepted.
        p.limbus_refinement = Some(sam31_outer::limbus_refinement::Attempt {
            baseline: before.clone(),
            applied: true,
            status: "TEST ACCEPTED".into(),
            field: Some(Arc::new(limbus_refiner::Field {
                baseline: before.ellipse,
                unmodified_refit: None,
                candidate: Some(before.ellipse),
                samples: vec![],
                corrected_points: vec![(2., 2.)],
                elapsed_ms: 0.,
                status: "UNRELATED FIELD",
            })),
        });
        p.outer_fit = Some(after.clone());
        let (a, b, status) = refinement_edges(p).unwrap();
        assert_eq!(a.retained_points, before.retained_points);
        assert_eq!(b.unwrap().retained_points, after.retained_points);
        assert!(status.contains("0.43 PX"));
        let expected = render(&frame, RoiOverlayMode::LimbusRefinementEdges);
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let mut moved = frame.clone();
            moved.segmentation_mode = method;
            moved.width += 100;
            moved.height += 80;
            moved.sensor_x += 400;
            moved.sensor_y += 200;
            moved.quad_color = Arc::new(vec![0; moved.width * moved.height]);
            moved.timestamp_ns += 9_000_000;
            let authority = mouse_gaze_surface(&moved);
            assert_eq!(
                render(&moved, RoiOverlayMode::LimbusRefinementEdges),
                expected
            );
            assert_eq!(mouse_gaze_surface(&moved), authority);
        }
        let p = Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap());
        p.limbus_refinement.as_mut().unwrap().applied = false;
        p.limbus_refinement.as_mut().unwrap().status =
            "REFINEMENT REJECTED: SOURCE RAW SUPPORT".into();
        p.outer_fit = Some(before.clone());
        let (a, b, status) = refinement_edges(p).unwrap();
        assert_eq!(a.retained_points, before.retained_points);
        assert!(b.is_none());
        assert!(status.starts_with("UNCHANGED"));
        assert!(!render(&frame, RoiOverlayMode::LimbusRefinementEdges).contains(&REFINEMENT_AFTER));
        let p = Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap());
        p.limbus_refinement = None;
        assert!(refinement_edges(p).unwrap().1.is_none());
        p.outer_fit = None;
        assert!(refinement_edges(p).is_none());
    }

    #[test]
    fn refinement_edges_preserve_gaps_and_reveal_subpixel_changes_when_magnified() {
        let mut frame = fixture();
        let p = Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap());
        let mut before = p.outer_fit.as_ref().unwrap().clone();
        before.retained_points = Arc::new(vec![(25., 55.), (45., 55.), (95., 55.), (115., 55.)]);
        before.conic_segments = Arc::new(vec![vec![0, 1], vec![2, 3]]);
        let mut after = before.clone();
        after.retained_points = Arc::new(
            before
                .retained_points
                .iter()
                .map(|q| (q.0, q.1 + 0.35))
                .collect(),
        );
        p.outer_fit = Some(after);
        p.limbus_refinement = Some(sam31_outer::limbus_refinement::Attempt {
            baseline: before,
            field: None,
            applied: true,
            status: "TEST SUBPIXEL SHIFT".into(),
        });
        let p = frame.sam31_proposal_masks.as_ref().unwrap();
        let (w, h, scale) = (640, 480, 4);
        let mut image = vec![0; w * h];
        draw_refinement_edges(&mut image, w, h, 0, 0, scale, p);
        assert_eq!(image[220 * w + 140], REFINEMENT_BEFORE);
        assert_eq!(image[221 * w + 100], REFINEMENT_AFTER);
        for row in 218..224 {
            assert!(
                image[row * w + 200..row * w + 370].iter().all(|p| *p == 0),
                "excluded gap must not get an ellipse/chord/pupil overlay"
            );
        }
        if let Some(root) = std::env::var_os("BUTTERCUP_STUDENT_VIEW_TEST_EXPORT") {
            let root = PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            let mut image = vec![0; w * h];
            draw(
                &mut image,
                w,
                h,
                0,
                0,
                scale,
                &frame,
                ViewMode::QuadColor,
                RoiOverlayMode::LimbusRefinementEdges,
            );
            export_eye_ppm(&root.join("refinement-subpixel-gaps.ppm"), &image, w, h).unwrap();
        }
    }

    #[test]
    fn pink_well_uses_the_same_registered_source_for_both_detectors() {
        let frame = fixture();
        let expected = render(&frame, RoiOverlayMode::PinkWellSightline);
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let mut moved = frame.clone();
            moved.segmentation_mode = method;
            moved.width += 64;
            moved.height += 32;
            moved.sensor_x += 200;
            moved.sensor_y += 100;
            moved.sequence += 20;
            moved.quad_color = Arc::new(vec![0x00987654; moved.width * moved.height]);
            let authority = mouse_gaze_surface(&moved);
            assert_eq!(render(&moved, RoiOverlayMode::PinkWellSightline), expected);
            assert_eq!(mouse_gaze_surface(&moved), authority);
            assert_eq!(
                moved.sam31_proposal_masks.as_ref().unwrap().source_sequence,
                195
            );
        }
    }

    #[test]
    fn contact_previews_the_conditional_candidate_without_authorizing_gaze() {
        let mut frame = fixture();
        let proposal = Arc::clone(frame.sam31_proposal_masks.as_ref().unwrap());
        assert!(contact_surface(&frame, &proposal, false).is_ok());
        let surface = frame.virtual_contact_surface_gaze.as_mut().unwrap();
        surface.sign_resolved = false;
        assert!(
            !contact_surface(&frame, &proposal, false)
                .unwrap()
                .sign_resolved
        );
        assert!(render(&frame, RoiOverlayMode::SamDeflattenedVirtualContact).contains(&0x00ff_00ff));
        assert!(
            gaze_output_direction(&frame).is_none(),
            "preview must not authorize a mouse direction"
        );
        frame
            .virtual_contact_surface_gaze
            .as_mut()
            .unwrap()
            .sign_diagnostics = Some(crate::eye_scene_model::SurfaceSignDiagnostics {
            orientation_reference_ready: Some(false),
            continuous_gaze: None,
            evidence: crate::eye_scene_model::SurfaceSignEvidence::Unresolved,
            selected_branch: 0,
            branch_residual_ema_px: [f64::NAN; 2],
            source_motion_residual_px: None,
            temporal_margin_px: None,
            pending_anchor_votes: 0,
            near_frontal_continuation: false,
        });
        assert!(
            !contact_surface(&frame, &proposal, false)
                .unwrap()
                .sign_resolved
        );
        frame.joint_gaze_active = true;
        assert_eq!(
            contact_surface(&frame, &proposal, false).unwrap_err(),
            "WAITING FOR MATCHING STEREO SOURCE"
        );
        assert!(
            !contact_surface(&frame, &proposal, true)
                .unwrap()
                .sign_resolved
        );
        frame
            .virtual_contact_surface_gaze
            .as_mut()
            .unwrap()
            .source_timestamp_ns = Some(1);
        assert_eq!(
            contact_surface(&frame, &proposal, true).unwrap_err(),
            "WAITING FOR MATCHING SURFACE SOURCE"
        );
    }

    #[test]
    fn student_contact_never_borrows_other_exposure_or_concave_surface() {
        let mut frame = fixture();
        let good = frame.virtual_contact_surface_gaze.unwrap();
        for surface in [
            SurfaceGazeSample {
                source_timestamp_ns: Some(1_000_000_001),
                ..good
            },
            SurfaceGazeSample {
                source_timestamp_ns: None,
                ..good
            },
            SurfaceGazeSample {
                relative_gaze: RelativeGazeVector {
                    toward_camera: -good.relative_gaze.toward_camera,
                    ..good.relative_gaze
                },
                ..good
            },
        ] {
            frame.virtual_contact_surface_gaze = Some(surface);
            assert!(
                !render(&frame, RoiOverlayMode::SamDeflattenedVirtualContact)
                    .contains(&0x00ff_00ff)
            );
        }
        frame.eye_id = 2;
        assert!(
            source_dimensions(&frame).is_none(),
            "wrong-eye proposal cannot be displayed"
        );
        frame.eye_id = 1;
        frame.gaze_authority_sam_prompt_generation = Some(3);
        assert!(
            source_dimensions(&frame).is_none(),
            "wrong prompt generation cannot be displayed"
        );
        frame.gaze_authority_sam_prompt_generation = Some(2);
        Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).source_raw = Arc::new(vec![]);
        assert!(
            source_dimensions(&frame).is_none(),
            "invalid source dimensions cannot be displayed"
        );
    }
}
