//! Source-matched iris/pupil circle projection review. No sign authority or feedback.
use crate::conic_solver::joint::{circle_pose_hypotheses, PinholeCamera};
use crate::geometry::{add3, dot3, norm3, normalized3, scale3, sub3, Ellipse};
use crate::sam31_outer::ProposalMasks;

pub(crate) fn startup_requested() -> bool {
    std::env::var("BUTTERCUP_VOID_SIGHTLINE").as_deref() == Ok("1") || nested_enabled()
}
fn nested_enabled() -> bool {
    std::env::var("BUTTERCUP_NESTED_PUPIL_COMPARE").as_deref() == Ok("1")
}

#[derive(Clone, Copy, Debug)]
struct Projection {
    center: [f64; 2],
    tip: [f64; 2],
    normal: [f64; 3],
    center_per_radius: [f64; 3],
}

/// Centers are the projected centers of the two reconstructed iris/pupil circles,
/// not the 2D conic centroid and not a hidden retinal or globe-rotation center.
/// Unit circle radius fixes a gauge only; no metric depth is measured here.
fn projections(
    camera: PinholeCamera,
    ellipse: Ellipse,
    origin: [u32; 2],
) -> Option<[Projection; 2]> {
    let poses = circle_pose_hypotheses(camera, ellipse, origin)?;
    let project = |p| {
        camera
            .project(p)
            .map(|q| [q[0] - origin[0] as f64, q[1] - origin[1] as f64])
    };
    let make = |i: usize| {
        let p = poses[i];
        Some(Projection {
            center: project(p.center_per_radius)?,
            // An outward axis of two circle radii: not visual-axis calibration.
            tip: project(std::array::from_fn(|j| {
                p.center_per_radius[j] + 2.0 * p.normal[j]
            }))?,
            normal: p.normal,
            center_per_radius: p.center_per_radius,
        })
    };
    Some([make(0)?, make(1)?])
}

/// Eigenvector signs (and therefore solver branch indices) are local to one
/// conic. They do not identify the same direction in the other eye or after
/// a re-fit. Give display colours a shared camera-coordinate convention:
/// pink is the more upward normal, with leftward breaking a horizontal tie.
/// This only permutes the two exact poses; it is not correspondence, sign
/// evidence, temporal tracking, or a constraint on the actual stereo solve.
fn display_indices(candidates: [Projection; 2]) -> [usize; 2] {
    let dy = candidates[0].normal[1] - candidates[1].normal[1];
    let reverse = if dy.abs() > 1e-6 {
        dy > 0.
    } else {
        candidates[0].normal[0] > candidates[1].normal[0]
    };
    if reverse {
        [1, 0]
    } else {
        [0, 1]
    }
}

fn display_order(candidates: [Projection; 2]) -> [Projection; 2] {
    display_indices(candidates).map(|i| candidates[i])
}

#[derive(Clone, Copy, Debug)]
struct WellMeridian {
    ledge: [f64; 3],
    rim: [f64; 3],
    bottom: [f64; 3],
    sphere_center: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
struct WellPoint {
    position: [f64; 3],
    opacity: f64,
    shaft: bool,
}

/// The surface stroke is a great-circle arc on the illustrative globe, with
/// both ends registered to the fitted image boundaries. At the aperture it
/// makes a sharp 90-degree turn along the inward local surface normal. There
/// is no rounded lip or depth fade; only antialiasing/occlusion affect opacity.
fn well_profile(m: WellMeridian) -> Vec<WellPoint> {
    let radius = norm3(sub3(m.rim, m.sphere_center));
    let Some(start) = normalized3(sub3(m.ledge, m.sphere_center)) else {
        return vec![];
    };
    let Some(end) = normalized3(sub3(m.rim, m.sphere_center)) else {
        return vec![];
    };
    let cosine = dot3(start, end).clamp(-1., 1.);
    let angle = cosine.acos();
    let Some(tangent) = normalized3(sub3(end, scale3(start, cosine))) else {
        return vec![];
    };
    let mut points = Vec::with_capacity(97);
    for k in 0..=48 {
        let theta = angle * k as f64 / 48.;
        points.push(WellPoint {
            position: if k == 0 {
                m.ledge
            } else if k == 48 {
                m.rim
            } else {
                add3(
                    m.sphere_center,
                    scale3(
                        add3(scale3(start, theta.cos()), scale3(tangent, theta.sin())),
                        radius,
                    ),
                )
            },
            opacity: 0.9,
            shaft: k == 48,
        });
    }
    for k in 1..=48 {
        let t = k as f64 / 48.;
        points.push(WellPoint {
            position: add3(m.rim, scale3(sub3(m.bottom, m.rim), t)),
            opacity: 0.9,
            shaft: true,
        });
    }
    points
}

fn ellipse_radius_squared(e: Ellipse, q: [f64; 2]) -> f64 {
    let (s, c) = e.angle.sin_cos();
    let (x, y) = (q[0] - e.center.0, q[1] - e.center.1);
    ((c * x + s * y) / e.major_radius).powi(2) + ((-s * x + c * y) / e.minor_radius).powi(2)
}

/// Lift the flat circle solve onto one spherical surface through the pupil
/// rim. Curvature is an illustration, not another anatomical measurement.
/// Ray/sphere intersections preserve the observed outer endpoints too. Each
/// meridian stays in a fixed plane through the sphere center (no spiral).
fn well_meridians(
    camera: PinholeCamera,
    origin: [u32; 2],
    outer: Ellipse,
    pupil: Ellipse,
    pose: Projection,
) -> Vec<WellMeridian> {
    if ![outer.major_radius, outer.minor_radius]
        .iter()
        .all(|r| r.is_finite() && *r > 0.)
        || !ellipse_radius_squared(outer, pose.center).is_finite()
        || ellipse_radius_squared(outer, pose.center) >= 1.
    {
        return vec![];
    }
    let plane = |q: [f64; 2]| {
        let ray = camera.unproject([q[0] + origin[0] as f64, q[1] + origin[1] as f64], 1.);
        let t = dot3(pose.normal, pose.center_per_radius) / dot3(pose.normal, ray);
        (t.is_finite() && t > 0.).then(|| scale3(ray, t))
    };
    let (s, c) = outer.angle.sin_cos();
    let normalized = |q: [f64; 2]| {
        [
            (c * q[0] + s * q[1]) / outer.major_radius,
            (-s * q[0] + c * q[1]) / outer.minor_radius,
        ]
    };
    let o = normalized([
        pose.center[0] - outer.center.0,
        pose.center[1] - outer.center.1,
    ]);
    let flat: Vec<_> = pupil
        .dense_points(12)
        .into_iter()
        .filter_map(|q| {
            let q = [q.0, q.1];
            let d = [q[0] - pose.center[0], q[1] - pose.center[1]];
            let v = normalized(d);
            let a = v[0] * v[0] + v[1] * v[1];
            let b = o[0] * v[0] + o[1] * v[1];
            let discriminant = b * b - a * (o[0] * o[0] + o[1] * o[1] - 1.);
            let t = (-b + discriminant.sqrt()) / a;
            if !t.is_finite() || t <= 1. {
                return None;
            }
            let rim = plane(q)?;
            Some((
                plane([pose.center[0] + t * d[0], pose.center[1] + t * d[1]])?,
                rim,
            ))
        })
        .collect();
    // A common radius for all meridians, in pupil-radius units. The 1.5
    // factor is a display choice. Increase it only when an outer ray misses
    // the cap or a pupil rim lies on its back; never move a fitted endpoint.
    let mut radius = 1.5
        * flat
            .iter()
            .map(|(ledge, _)| norm3(sub3(*ledge, pose.center_per_radius)))
            .fold(1., f64::max);
    for _ in 0..8 {
        let sphere_center = sub3(
            pose.center_per_radius,
            scale3(pose.normal, (radius * radius - 1.).sqrt()),
        );
        let lifted: Option<Vec<_>> = flat
            .iter()
            .map(|&(ledge, rim)| {
                let local_normal = normalized3(sub3(rim, sphere_center))?;
                if dot3(local_normal, scale3(rim, -1.)) <= 0. {
                    return None;
                }
                let ray = normalized3(ledge)?;
                let along = dot3(ray, sphere_center);
                let offset = sub3(sphere_center, scale3(ray, along));
                let discriminant = radius * radius - dot3(offset, offset);
                let depth = along - discriminant.sqrt();
                if !depth.is_finite() || depth <= 0. {
                    return None;
                }
                Some(WellMeridian {
                    ledge: scale3(ray, depth),
                    rim,
                    bottom: sub3(rim, scale3(local_normal, 1.8)),
                    sphere_center,
                })
            })
            .collect();
        if let Some(meridians) = lifted {
            return meridians;
        }
        radius *= 1.25;
    }
    vec![]
}

fn draw_well(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    p: &ProposalMasks,
    camera: PinholeCamera,
    candidates: [Projection; 2],
) {
    let Some((outer, pupil)) = p.outer_fit.as_ref().zip(p.inner_pupil_fit) else {
        return;
    };
    let origin = [p.source_sensor_origin.0, p.source_sensor_origin.1];
    let mask = p.semantic.as_ref().and_then(|answer| {
        answer
            .masks
            .iter()
            .find(|m| Some(m.query) == answer.selected_query)
            .filter(|m| {
                answer.width > 0
                    && answer.height > 0
                    && answer.width.checked_mul(answer.height) == Some(m.pixels.len())
            })
            .map(|m| (answer, m))
    });
    let visible = |x: usize, y: usize| {
        mask.is_none_or(|(answer, m)| {
            m.pixels[(y * answer.height / height) * answer.width + x * answer.width / width] != 0
        })
    };
    // A single coherent illustrative well using the shared pink direction. Drawing
    // both opposite shafts in one aperture destroys the depth cue. Both
    // geometric sight-line hypotheses remain visible outside this renderer.
    let pose = display_order(candidates)[0];
    let mut coverage = vec![0_f64; width * height];
    for meridian in well_meridians(camera, origin, outer.ellipse, pupil.ellipse, pose) {
        for pair in well_profile(meridian).windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let (Some(a_px), Some(b_px)) = (camera.project(a.position), camera.project(b.position))
            else {
                continue;
            };
            let length = (b_px[0] - a_px[0]).hypot(b_px[1] - a_px[1]);
            let steps = (length * 3.).ceil().clamp(1., 2048.) as usize;
            for k in 0..=steps {
                let t = k as f64 / steps as f64;
                let q = [
                    a_px[0] + t * (b_px[0] - a_px[0]) - origin[0] as f64,
                    a_px[1] + t * (b_px[1] - a_px[1]) - origin[1] as f64,
                ];
                let opacity = a.opacity + t * (b.opacity - a.opacity);
                let (cx, cy) = (q[0].floor() as i32, q[1].floor() as i32);
                for y in cy - 1..=cy + 1 {
                    for x in cx - 1..=cx + 1 {
                        if x < 0
                            || y < 0
                            || x >= width as i32
                            || y >= height as i32
                            || !visible(x as usize, y as usize)
                        {
                            continue;
                        }
                        let pixel = [x as f64, y as f64];
                        // The front rim occludes the near wall; the far wall
                        // is visible only through the photographed aperture.
                        if ellipse_radius_squared(outer.ellipse, pixel) > 1.02
                            || (a.shaft
                                && b.shaft
                                && ellipse_radius_squared(pupil.ellipse, pixel) > 1.)
                        {
                            continue;
                        }
                        let alpha = opacity
                            * (1.0 - (q[0] - pixel[0]).hypot(q[1] - pixel[1])).clamp(0., 1.);
                        let sample = &mut coverage[y as usize * width + x as usize];
                        *sample = sample.max(alpha);
                    }
                }
            }
        }
    }
    // Composite once, so extra sampling/foreshortening cannot thicken or
    // brighten the stroke. Preserve the original RAW-derived pixels elsewhere.
    let color = 0x00ff60b5_u32;
    for (pixel, coverage) in pixels.iter_mut().zip(coverage) {
        let alpha = (255. * coverage).round() as u32;
        let channel = |shift: u32| {
            (((*pixel >> shift) & 255u32) * (255 - alpha)
                + ((color >> shift) & 255u32) * alpha
                + 127)
                / 255
        };
        *pixel = channel(16) << 16 | channel(8) << 8 | channel(0);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_pink_well(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    scale: usize,
    frame: &crate::EyeFrame,
    mode: crate::ViewMode,
) {
    let source = crate::student_preview::source(frame);
    let base = crate::student_preview::source_pixels(frame, mode);
    draw_review(
        pixels,
        width,
        height,
        x,
        y,
        scale,
        source.map(|p| p.as_ref()),
        true,
        base.as_deref().map(Vec::as_slice),
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    scale: usize,
    source: Option<&ProposalMasks>,
) {
    draw_review(
        pixels,
        width,
        height,
        x,
        y,
        scale,
        source,
        std::env::var("BUTTERCUP_VOID_PINK_WELL").as_deref() == Ok("1"),
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_review(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    scale: usize,
    source: Option<&ProposalMasks>,
    pink_well: bool,
    base: Option<&[u32]>,
) {
    let scale = scale.max(1);
    let Some(p) = source.filter(|p| {
        p.source_width > 0
            && p.source_height > 0
            && p.source_raw.len() == p.source_width * p.source_height
    }) else {
        crate::draw_text(
            pixels,
            width,
            height,
            x + 4,
            y + 8,
            "EYE AXIS: WAITING FOR SOURCE RAW",
            0x00ffc857,
        );
        return;
    };
    let mut preview = base.map(<[u32]>::to_vec).unwrap_or_else(|| {
        crate::color_preview(
            &p.source_raw,
            p.source_width,
            p.source_height,
            p.source_sensor_origin.0,
            p.source_sensor_origin.1,
            100,
            None,
        )
    });
    let camera = crate::joint_gaze_live::configured_camera().ok();
    if pink_well {
        if let (Some(camera), Some(pupil)) = (camera, p.inner_pupil_fit) {
            if let Some(candidates) = projections(
                camera,
                pupil.ellipse,
                [p.source_sensor_origin.0, p.source_sensor_origin.1],
            ) {
                draw_well(
                    &mut preview,
                    p.source_width,
                    p.source_height,
                    p,
                    camera,
                    candidates,
                );
            }
        }
    }
    for row in 0..p.source_height {
        for col in 0..p.source_width {
            crate::fill_rect(
                pixels,
                width,
                height,
                x + (col * scale) as i32,
                y + (row * scale) as i32,
                scale as i32,
                scale as i32,
                preview[row * p.source_width + col],
            );
        }
    }
    crate::fill_rect(
        pixels,
        width,
        height,
        x,
        y,
        (p.source_width * scale) as i32,
        18,
        0x00101010,
    );
    crate::draw_text(
        pixels,
        width,
        height,
        x + 4,
        y + 4,
        &format!(
            "{} AXIS ?  SOURCE {}",
            if p.inner_pupil_fit.is_some() {
                "PUPIL"
            } else {
                "IRIS"
            },
            p.source_sequence
        ),
        0x00ffffff,
    );
    // A missing inner aperture must not hide the outer circle's two poses.
    // These remain unsigned geometry hypotheses, never a substitute pupil or
    // a claim that image bounds/an operator range identify the true branch.
    let ellipse = p
        .inner_pupil_fit
        .map(|pupil| pupil.ellipse)
        .or_else(|| p.outer_fit.as_ref().map(|outer| outer.ellipse));
    let Some(ellipse) = ellipse else {
        crate::draw_text(
            pixels,
            width,
            height,
            x + 4,
            y + 20,
            "NO IRIS OR PUPIL FIT",
            0x00ffc857,
        );
        return;
    };
    if p.inner_pupil_fit.is_none() {
        crate::draw_text(
            pixels,
            width,
            height,
            x + 4,
            y + 20,
            "PUPIL OPENING: NO ACCEPTED FIT",
            0x00ffc857,
        );
    } else if pink_well && p.outer_fit.is_none() {
        crate::draw_text(
            pixels,
            width,
            height,
            x + 4,
            y + 20,
            "WELL: NO OUTER IRIS FIT",
            0x00ffc857,
        );
    }
    let Some(camera) = camera else {
        return;
    };
    let origin = [p.source_sensor_origin.0, p.source_sensor_origin.1];
    let Some(mut candidates) = projections(camera, ellipse, origin) else {
        return;
    };
    if pink_well {
        candidates = display_order(candidates);
    }
    let nested = if !pink_well && nested_enabled() {
        p.outer_fit
            .as_ref()
            .zip(p.inner_pupil_fit)
            .and_then(|(outer, pupil)| {
                crate::conic_solver::nested_pupil::compare(
                    camera,
                    outer.ellipse,
                    pupil.ellipse,
                    origin,
                )
            })
    } else {
        None
    };
    if let Some(result) = &nested {
        for (i, f) in result.fits.iter().enumerate() {
            let project = |v| {
                camera
                    .project(v)
                    .map(|q| [q[0] - origin[0] as f64, q[1] - origin[1] as f64])
            };
            if let (Some(center), Some(tip)) = (
                project(f.center),
                project(std::array::from_fn(|j| {
                    f.center[j] + 2. * f.radius * f.normal[j]
                })),
            ) {
                candidates[i] = Projection {
                    center,
                    tip,
                    normal: f.normal,
                    center_per_radius: f.center,
                };
            }
        }
    }
    let outline = ellipse.dense_points(128);
    let mut paint = |q: [f64; 2], color: u32, radius: i32| {
        if !q.iter().all(|v| v.is_finite()) {
            return;
        }
        let cx = (q[0] * scale as f64).round() as i32;
        let cy = (q[1] * scale as f64).round() as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let (lx, ly) = (cx + dx, cy + dy);
                let (px, py) = (x + lx, y + ly);
                if lx >= 0
                    && ly >= 0
                    && lx < (p.source_width * scale) as i32
                    && ly < (p.source_height * scale) as i32
                    && px >= 0
                    && py >= 0
                    && px < width as i32
                    && py < height as i32
                {
                    pixels[py as usize * width + px as usize] = color;
                }
            }
        }
    };
    if !pink_well || p.inner_pupil_fit.is_none() {
        for q in outline {
            paint([q.0, q.1], 0x00aaaaaa, 0);
        }
    }
    if let Some(result) = &nested {
        if let Some(outer) = &p.outer_fit {
            for q in outer.ellipse.dense_points(180) {
                paint([q.0, q.1], 0x00777777, 0);
            }
        }
        for (i, f) in result.fits.iter().enumerate() {
            for q in f.ellipse.dense_points(128) {
                paint([q.0, q.1], [0x00ff60b5, 0x0040dfff][i], 0);
            }
        }
    }
    for (i, c) in candidates.iter().enumerate() {
        let color = [0x00ff60b5, 0x0040dfff][i];
        let length = (c.tip[0] - c.center[0]).hypot(c.tip[1] - c.center[1]) * scale as f64;
        if !length.is_finite() {
            continue;
        }
        // A camera-directed normal can project to a point; do not invent a line.
        if length > 0.5 {
            let steps = (length.ceil() as usize).clamp(1, 4096);
            for k in 0..=steps {
                if ((k as f64 / steps as f64 * length) as usize / 5) % 2 == 0 {
                    let t = k as f64 / steps as f64;
                    paint(
                        std::array::from_fn(|j| c.center[j] + t * (c.tip[j] - c.center[j])),
                        color,
                        if pink_well {
                            0
                        } else {
                            (scale / 2).max(1) as i32
                        },
                    );
                }
            }
        }
        // Centers often overlap at this scale; retain both physical locations.
        paint(
            c.center,
            color,
            if pink_well {
                if i == 0 {
                    scale as i32
                } else {
                    0
                }
            } else if i == 0 {
                3 * scale as i32
            } else {
                2 * scale as i32
            },
        );
    }
    let bottom = y + (p.source_height * scale) as i32 - 34;
    crate::fill_rect(
        pixels,
        width,
        height,
        x,
        bottom - 2,
        (p.source_width * scale) as i32,
        36,
        0x00101010,
    );
    crate::draw_text(
        pixels,
        width,
        height,
        x + 4,
        bottom,
        &nested
            .as_ref()
            .map(|r| {
                format!(
                    "A {:.2}PX B {:.2}PX {}",
                    r.fits[0].rms_px,
                    r.fits[1].rms_px,
                    r.preference.map_or("AMBIGUOUS", |i| if i == 0 {
                        "A CONDITIONAL"
                    } else {
                        "B CONDITIONAL"
                    })
                )
            })
            .unwrap_or_else(|| {
                if pink_well {
                    "PINK UP/LEFT / CYAN DOWN/RIGHT".into()
                } else {
                    "PINK A / CYAN B: UNSIGNED".into()
                }
            }),
        0x00ffffff,
    );
    crate::draw_text(
        pixels,
        width,
        height,
        x + 4,
        bottom + 16,
        if pink_well {
            "WELL: ILLUSTRATIVE / UNSIGNED"
        } else if p.inner_pupil_fit.is_some() {
            "PUPIL CENTER / OUTWARD AXIS"
        } else {
            "IRIS DISK CENTER / OUTWARD AXIS"
        },
        0x00ffffff,
    );
}

/// Offline 2D inspection keeps observed contour samples distinct from fits.
/// It deliberately bypasses the 3D hypothesis renderer, including on rejection.
fn draw_pupil_fits(pixels: &mut [u32], p: &ProposalMasks, diagnostic: &serde_json::Value) {
    let (width, height) = (p.source_width * 2, p.source_height * 2);
    let preview = crate::color_preview(
        &p.source_raw,
        p.source_width,
        p.source_height,
        p.source_sensor_origin.0,
        p.source_sensor_origin.1,
        100,
        None,
    );
    for row in 0..p.source_height {
        for col in 0..p.source_width {
            crate::fill_rect(
                pixels,
                width,
                height,
                (col * 2) as i32,
                (row * 2) as i32,
                2,
                2,
                preview[row * p.source_width + col],
            );
        }
    }
    let mut point = |q: (f64, f64), color, size| {
        if q.0.is_finite() && q.1.is_finite() {
            crate::fill_rect(
                pixels,
                width,
                height,
                (q.0 * 2.).round() as i32,
                (q.1 * 2.).round() as i32,
                size,
                size,
                color,
            );
        }
    };
    if let Some(outer) = &p.outer_fit {
        for q in outer.ellipse.dense_points(512) {
            point(q, 0x005fd5ff, 2);
        }
    }
    for candidate in diagnostic["candidates"].as_array().into_iter().flatten() {
        for (key, color) in [
            ("flat_tire_points_native", 0x00ffb84d),
            ("retained_points_native", 0x008dff69),
        ] {
            for q in candidate[key].as_array().into_iter().flatten() {
                if let (Some(x), Some(y)) = (q[0].as_f64(), q[1].as_f64()) {
                    point((x, y), color, 3);
                }
            }
        }
    }
    if let Some(pupil) = p.inner_pupil_fit {
        for q in pupil.ellipse.dense_points(256) {
            point(q, 0x00ff64c8, 2);
        }
    }
    crate::fill_rect(pixels, width, height, 0, 0, width as i32, 19, 0x00101010);
    crate::draw_text(
        pixels,
        width,
        height,
        4,
        4,
        &format!(
            "SEQ {}  {}",
            p.source_sequence,
            if p.inner_pupil_fit.is_some() {
                "PUPIL FIT"
            } else if p.outer_fit.is_none() {
                "NO ADMITTED OUTER"
            } else {
                "PUPIL REJECTED"
            }
        ),
        0x00ffffff,
    );
    crate::fill_rect(
        pixels,
        width,
        height,
        0,
        height as i32 - 19,
        width as i32,
        19,
        0x00101010,
    );
    crate::draw_text(
        pixels,
        width,
        height,
        4,
        height as i32 - 15,
        "PINK PUPIL / CYAN OUTER / GREEN RETAINED / ORANGE EXCLUDED",
        0x00ffffff,
    );
}

/// Replay previously measured pupil fits against their exact native RAW sources.
/// This does not load a model, train on predictions, or promote a sign policy.
pub(crate) fn review(args: impl Iterator<Item = String>) -> Result<(), String> {
    use serde_json::{json, Value};
    use std::io::{Read, Seek, Write};
    let args: Vec<_> = args.collect();
    if args.len() != 2 {
        return Err("usage: --offline-void-sightline WORKER.jsonl OUTPUT_DIR".into());
    }
    let run = || -> Result<(), Box<dyn std::error::Error>> {
        let refit = std::env::var("BUTTERCUP_VOID_REFIT_PUPIL").as_deref() == Ok("1");
        let policy =
            std::env::var("BUTTERCUP_VOID_PUPIL_POLICY").unwrap_or_else(|_| "strict".into());
        let (constrain_arcs, position_support) = match policy.as_str() {
            "strict" => (true, false),
            "count-only" => (false, false),
            "tangent-position" => (true, true),
            "position" => (false, true),
            _ => {
                return Err(
                    "unknown pupil policy: strict/count-only/tangent-position/position".into(),
                )
            }
        };
        let fits_only = std::env::var("BUTTERCUP_VOID_PUPIL_FITS_ONLY").as_deref() == Ok("1");
        let pink_well =
            !fits_only && std::env::var("BUTTERCUP_VOID_PINK_WELL").as_deref() == Ok("1");
        let out = std::path::Path::new(&args[1]);
        if out.exists() {
            return Err("output already exists".into());
        }
        let parent = out.parent().ok_or("missing parent")?.canonicalize()?;
        if !parent.starts_with(std::path::Path::new("outputs").canonicalize()?) {
            return Err("output must be under checked outputs link".into());
        }
        let cases: Vec<Value> = std::fs::read_to_string(&args[0])?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        if cases.is_empty() {
            return Err("empty cache".into());
        }
        std::fs::create_dir(out)?;
        let camera = crate::joint_gaze_live::configured_camera()?;
        let shown: Vec<usize> = match std::env::var("BUTTERCUP_VOID_REVIEW_INDICES") {
            Ok(value) => serde_json::from_str(&value)?,
            Err(_) => (0..12.min(cases.len()))
                .map(|i| i * (cases.len() - 1) / 11)
                .collect(),
        };
        if shown.is_empty() || shown.len() > 12 || shown.iter().any(|&i| i >= cases.len()) {
            return Err("review indices must contain 1..12 valid source rows".into());
        }
        let mut records = vec![];
        let mut tiles = vec![];
        for (index, case) in cases.iter().enumerate() {
            let row = &case["input"];
            let meta = &row["frame"];
            let n = |key: &str| meta[key].as_u64().ok_or_else(|| format!("missing {key}"));
            if case["timestamp_ns"].as_u64() != Some(n("timestamp_ns")?)
                || case["sequence"].as_u64() != Some(n("sequence")?)
            {
                return Err("cache/source mismatch".into());
            }
            let origin = [n("sensor_x")? as u32, n("sensor_y")? as u32];
            let e = &case["pupil_void"]["ellipse"];
            let mut pupil = (|| {
                Some(Ellipse {
                    center: (e["center"][0].as_f64()?, e["center"][1].as_f64()?),
                    major_radius: e["major_radius"].as_f64()?,
                    minor_radius: e["minor_radius"].as_f64()?,
                    angle: e["angle"].as_f64()?,
                })
            })();
            let selected = case["candidates"].as_array().and_then(|items| {
                items.iter().find(|c| {
                    c["query"] == case["selected_query"] && c["baseline_raw_admitted"] == true
                })
            });
            let outer = selected.and_then(|c| {
                let e = &c["baseline_ellipse"];
                Some(Ellipse {
                    center: (e["center"][0].as_f64()?, e["center"][1].as_f64()?),
                    major_radius: e["major_radius"].as_f64()?,
                    minor_radius: e["minor_radius"].as_f64()?,
                    angle: e["angle"].as_f64()?,
                })
            });
            let mut refit_diagnostic = Value::Null;
            if refit {
                pupil = None;
                if let Some(o) = outer {
                    let (w, h) = (n("width")? as usize, n("height")? as usize);
                    let mut f =
                        std::fs::File::open(row["raw_file"].as_str().ok_or("missing raw")?)?;
                    f.seek(std::io::SeekFrom::Start(
                        row["raw_offset"].as_u64().ok_or("missing offset")?,
                    ))?;
                    let mut bytes =
                        vec![0; row["raw_length"].as_u64().ok_or("missing length")? as usize];
                    f.read_exact(&mut bytes)?;
                    let raw = crate::raw10::try_unpack_raw10(&bytes, w, h, n("stride")? as usize)?;
                    refit_diagnostic = crate::sam31_outer::inspect_pupil_fit_with_position_support(
                        std::sync::Arc::new(crate::sam31_outer::RawFrame {
                            eye_index: n("eye_id")? as usize - 1,
                            sequence: n("sequence")?,
                            timestamp_ns: n("timestamp_ns")?,
                            sensor_x: origin[0],
                            sensor_y: origin[1],
                            width: w,
                            height: h,
                            registration_anchor: None,
                            pupil_component_seed: None,
                            pixels: std::sync::Arc::new(raw),
                        }),
                        o,
                        true,
                        constrain_arcs,
                        true,
                        position_support,
                    );
                    let e = &refit_diagnostic["ellipse"];
                    pupil = (|| {
                        Some(Ellipse {
                            center: (e["center"][0].as_f64()?, e["center"][1].as_f64()?),
                            major_radius: e["major_radius"].as_f64()?,
                            minor_radius: e["minor_radius"].as_f64()?,
                            angle: e["angle"].as_f64()?,
                        })
                    })();
                }
            }
            let candidates = pupil.and_then(|e| projections(camera, e, origin));
            let mut record = json!({"index":index,"source":row,"pupil_ellipse":e,
                "status":if pupil.is_some(){"projection-unavailable"}else{"missing-pupil"},"signed":false});
            record["cached_pupil_ellipse"] = e.clone();
            record["pupil_ellipse"] = pupil
                .map(|p| {
                    json!({"center":p.center,"major_radius":p.major_radius,
                "minor_radius":p.minor_radius,"angle":p.angle})
                })
                .unwrap_or(Value::Null);
            record["pupil_source"] = json!(if refit {
                "fresh RAW component; search-domain censored, free shape, no temporal shape fallback"
            } else {
                "cached worker fit; may include outer-shape recovery"
            });
            record["raw_refit"] = refit_diagnostic;
            let outer_projections = outer.and_then(|e| projections(camera, e, origin));
            record["outer_candidates"] = outer_projections
                .map(|poses| {
                    json!(
                        poses.map(|p| json!({"center_roi":p.center,"normal":p.normal,
                    "tip_roi":p.tip,"center_per_iris_radius":p.center_per_radius}))
                    )
                })
                .unwrap_or(Value::Null);
            record["displayed_basis"] = json!(if candidates.is_some() {
                "pupil"
            } else if pupil.is_none() && outer_projections.is_some() {
                "outer-iris"
            } else {
                "unavailable"
            });
            if pink_well {
                let shown_poses = candidates.or_else(|| {
                    if pupil.is_none() {
                        outer_projections
                    } else {
                        None
                    }
                });
                record["display_color_candidate_indices"] = shown_poses
                    .map(display_indices)
                    .map_or(Value::Null, |indices| json!(indices));
                record["well_surface"] = match (outer, pupil, candidates) {
                    (Some(outer), Some(pupil), Some(poses)) => {
                        let arcs =
                            well_meridians(camera, origin, outer, pupil, display_order(poses)[0]);
                        json!({"meridians":arcs.len(),
                            "illustrative_radius_per_pupil_radius":arcs.first().map(|m| norm3(sub3(m.rim,m.sphere_center)))})
                    }
                    _ => Value::Null,
                };
            }
            record["nested_comparison"] = match (outer, pupil) {
                (Some(o), Some(p)) => {
                    crate::conic_solver::nested_pupil::compare(camera, o, p, origin)
                        .map(|c| c.report)
                        .unwrap_or(json!({"status":"invalid-geometry"}))
                }
                _ => json!({"status":"missing-admitted-pair"}),
            };
            if let (Some(e), Some(c)) = (pupil, candidates) {
                let mut max_residual = 0_f64;
                for p in c {
                    let projected = crate::conic_solver::joint::ProjectedCircle::project(
                        camera,
                        p.center_per_radius,
                        p.normal,
                        1.,
                        origin,
                    )
                    .ok_or("invalid reconstructed circle")?;
                    for q in e.dense_points(64) {
                        max_residual = max_residual.max(projected.residual_px(q).abs());
                    }
                }
                if max_residual > 1e-5 {
                    return Err(format!("circle roundtrip failed {max_residual}").into());
                }
                record["status"] = json!("two-unsigned-poses");
                record["max_reprojection_error_px"] = json!(max_residual);
                record["candidates"]=json!(c.map(|p|json!({"center_roi":p.center,"normal":p.normal,"tip_roi":p.tip,
                    "center_per_pupil_radius":p.center_per_radius,"ellipse_centroid_offset_px":(p.center[0]-e.center.0).hypot(p.center[1]-e.center.1)})));
            }
            if shown.contains(&index) {
                let (w, h) = (n("width")? as usize, n("height")? as usize);
                let mut f = std::fs::File::open(row["raw_file"].as_str().ok_or("missing raw")?)?;
                f.seek(std::io::SeekFrom::Start(
                    row["raw_offset"].as_u64().ok_or("missing offset")?,
                ))?;
                let mut bytes =
                    vec![0; row["raw_length"].as_u64().ok_or("missing length")? as usize];
                f.read_exact(&mut bytes)?;
                let raw = crate::raw10::try_unpack_raw10(&bytes, w, h, n("stride")? as usize)?;
                let p = ProposalMasks {
                    source_sequence: n("sequence")?,
                    source_timestamp_ns: n("timestamp_ns")?,
                    source_sensor_origin: (origin[0], origin[1]),
                    source_width: w,
                    source_height: h,
                    source_raw: std::sync::Arc::new(raw),
                    outer_fit: outer.map(|ellipse| crate::sam31_outer::OuterMaskFitReview {
                        ellipse,
                        source_component_area_px: 0.,
                        retained_points: Default::default(),
                        conic_segments: Default::default(),
                        flat_tire_points: Default::default(),
                        upper_flat_tire: false,
                        lower_flat_tire: false,
                    }),
                    inner_pupil_fit: pupil.map(|ellipse| crate::sam31_outer::PupilVoidFitReview {
                        ellipse,
                        raw_support: Default::default(),
                    }),
                    ..Default::default()
                };
                let mut pixels = vec![0; w * h * 4];
                if fits_only {
                    draw_pupil_fits(&mut pixels, &p, &record["raw_refit"]);
                } else {
                    draw(&mut pixels, w * 2, h * 2, 0, 0, 2, Some(&p));
                }
                tiles.push((w * 2, h * 2, pixels));
            }
            records.push(record);
        }
        let tw = tiles.iter().map(|t| t.0).max().unwrap();
        let th = tiles.iter().map(|t| t.1).max().unwrap();
        let (w, h) = (tw * 4, th * 3);
        let mut sheet = vec![0_u32; w * h];
        for (i, (width, height, tile)) in tiles.into_iter().enumerate() {
            for y in 0..height {
                let start = (i / 4 * th + y) * w + i % 4 * tw;
                sheet[start..start + width].copy_from_slice(&tile[y * width..(y + 1) * width]);
            }
        }
        let mut ppm = std::fs::File::create(out.join("review.ppm"))?;
        write!(ppm, "P6\n{w} {h}\n255\n")?;
        let rgb: Vec<u8> = sheet
            .iter()
            .flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8])
            .collect();
        ppm.write_all(&rgb)?;
        let available = records
            .iter()
            .filter(|r| r["status"] == "two-unsigned-poses")
            .count();
        let mut nested_counts = std::collections::BTreeMap::new();
        let mut displayed_basis_counts = std::collections::BTreeMap::new();
        for r in &records {
            *displayed_basis_counts
                .entry(r["displayed_basis"].as_str().unwrap())
                .or_insert(0usize) += 1;
            *nested_counts
                .entry(
                    r["nested_comparison"]["status"]
                        .as_str()
                        .unwrap_or("unavailable"),
                )
                .or_insert(0usize) += 1;
        }
        let report = json!({"input":args[0],"rows":records.len(),"available":available,"missing":records.len()-available,
            "nested_counts":nested_counts,
            "displayed_basis_counts":displayed_basis_counts,
            "raw_pupil_refit":refit,"pupil_policy":policy,"pupil_fits_only":fits_only,
            "pink_well":pink_well,
            "pink_well_surface":"illustrative shared sphere through the pupil rim; great-circle arcs to ray-matched outer boundary; sharp inward local-normal drop of 1.8 pupil radii, constant opacity 0.9. Curvature starts at 1.5 times the largest flat outer radius and may increase to retain visible ray intersections; not measured globe anatomy",
            "pink_well_color_order":"pink: smaller camera-normal Y (up), then X (left) within 1e-6 Y; cyan: other exact pose. Display permutation only, not binocular sign evidence",
            "review_indices":shown,"camera":{"focal_px":camera.focal_px,"principal_px":camera.principal_px},
            "baseline_candidate":"outer fits frozen; cached or explicitly fresh RAW pupil fits; no gaze publication or outer-area change",
            "limitations":"Rob-only recorded SAM/RAW pupil fits; no human pupil center or sign truth. No metric depth, no corneal refraction or visual-axis calibration. SN-FEIDA not re-estimated; no independent scale. Both branches retained, not sign evidence.","records":records});
        std::fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
        println!(
            "void review: {available}/{} frames have two projected poses; {}",
            cases.len(),
            out.display()
        );
        Ok(())
    };
    run().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic_solver::joint::ProjectedCircle;
    #[test]
    fn paired_source_colors_do_not_follow_opposite_local_eigenvector_orders() {
        // Source-matched Rob capture, both eyes at sequence 1734 / sensor
        // timestamp 1690021533827570807. Previously pink pointed down in
        // subject-right and up in subject-left despite the same exposure.
        let camera = PinholeCamera {
            focal_px: [4000.; 2],
            principal_px: [4000., 3000.],
        };
        let right = Ellipse {
            center: (221.05915057185965, 211.82818147419965),
            major_radius: 34.819517279042394,
            minor_radius: 27.27036877884551,
            angle: -0.3602348412144676,
        };
        let left = Ellipse {
            center: (182.009642530315, 132.98310500106834),
            major_radius: 28.27833922731508,
            minor_radius: 19.775123872878886,
            angle: -0.4688824940717224,
        };
        let raw_right = projections(camera, right, [3440, 906]).unwrap();
        let raw_left = projections(camera, left, [4512, 978]).unwrap();
        assert!(raw_right[0].normal[1] > 0. && raw_left[0].normal[1] < 0.);
        let fixed_right = display_order(raw_right);
        let fixed_left = display_order(raw_left);
        for fixed in [fixed_right, fixed_left] {
            assert!(fixed[0].normal[1] < 0. && fixed[1].normal[1] > 0.);
            assert!(fixed[0].tip[1] < fixed[0].center[1]);
            assert!(fixed[1].tip[1] > fixed[1].center[1]);
        }
        for (ellipse, origin) in [(right, [3440, 906]), (left, [4512, 978])] {
            let original = display_order(projections(camera, ellipse, origin).unwrap());
            for turn in [-1., 1., 2.] {
                let equivalent = Ellipse {
                    angle: ellipse.angle + turn * std::f64::consts::PI,
                    ..ellipse
                };
                let colored = display_order(projections(camera, equivalent, origin).unwrap());
                for (a, b) in original.iter().zip(colored) {
                    assert!(norm3(sub3(a.normal, b.normal)) < 1e-9);
                    assert!(norm3(sub3(a.center_per_radius, b.center_per_radius)) < 1e-7);
                }
            }
        }
    }

    #[test]
    fn spherical_meridians_keep_boundaries_and_turn_sharply_without_fading() {
        let camera = PinholeCamera {
            focal_px: [1200.; 2],
            principal_px: [400., 300.],
        };
        let normal = crate::geometry::normalized3([0.4, -0.3, 0.8]).unwrap();
        let center = [10., -8., -80.];
        let origin = [240, 100];
        let circle = |r| {
            ProjectedCircle::project(camera, center, normal, r, origin)
                .unwrap()
                .ellipse()
                .unwrap()
        };
        let pupil = circle(4.);
        let outer = circle(12.);
        for pose in projections(camera, pupil, origin).unwrap() {
            let meridians = well_meridians(camera, origin, outer, pupil, pose);
            assert_eq!(meridians.len(), 12);
            let sphere_center = meridians[0].sphere_center;
            let sphere_radius = norm3(sub3(meridians[0].rim, sphere_center));
            for m in meridians {
                assert_eq!(m.sphere_center, sphere_center);
                let drop = sub3(m.bottom, m.rim);
                let local_normal = normalized3(sub3(m.rim, sphere_center)).unwrap();
                assert!((dot3(drop, local_normal) + 1.8).abs() < 1e-10);
                assert!(norm3(crate::geometry::cross3(drop, local_normal)) < 1e-10);
                let profile = well_profile(m);
                assert!(norm3(sub3(profile[0].position, m.ledge)) < 1e-10);
                assert!(norm3(sub3(profile[48].position, m.rim)) < 1e-10);
                assert!(norm3(sub3(profile.last().unwrap().position, m.bottom)) < 1e-10);
                let meridian_plane = normalized3(crate::geometry::cross3(
                    sub3(m.ledge, sphere_center),
                    sub3(m.rim, sphere_center),
                ))
                .unwrap();
                for point in &profile {
                    assert!(
                        dot3(sub3(point.position, sphere_center), meridian_plane).abs() < 1e-9,
                        "a globe meridian must not spiral"
                    );
                    assert_eq!(point.opacity, 0.9, "neither arc nor drop may fade");
                }
                for point in &profile[..=48] {
                    assert!(
                        (norm3(sub3(point.position, sphere_center)) - sphere_radius).abs() < 1e-9
                    );
                }
                let chord_midpoint = scale3(add3(m.ledge, m.rim), 0.5);
                assert!(
                    norm3(sub3(profile[24].position, chord_midpoint)) > 1e-4,
                    "the surface stroke must curve, not interpolate a flat chord"
                );
                let approach =
                    normalized3(sub3(profile[48].position, profile[47].position)).unwrap();
                let descent =
                    normalized3(sub3(profile[49].position, profile[48].position)).unwrap();
                assert!(
                    dot3(approach, descent).abs() < 0.02,
                    "arc/drop tangents must turn 90 degrees, without a rounded transition"
                );
                for (endpoint, ellipse) in [(m.rim, pupil), (m.ledge, outer)] {
                    let q = camera.project(endpoint).unwrap();
                    assert!(
                        (ellipse_radius_squared(
                            ellipse,
                            [q[0] - origin[0] as f64, q[1] - origin[1] as f64]
                        ) - 1.)
                            .abs()
                            < 1e-9
                    );
                }
            }
        }
    }

    #[test]
    fn well_requires_both_fits_and_respects_occlusion_mask() {
        use std::sync::Arc;
        let camera = PinholeCamera {
            focal_px: [500.; 2],
            principal_px: [100., 100.],
        };
        let pupil = Ellipse {
            center: (100., 100.),
            major_radius: 20.,
            minor_radius: 15.,
            angle: 0.,
        };
        let candidates = projections(camera, pupil, [0, 0]).unwrap();
        let mut p = ProposalMasks {
            source_width: 200,
            source_height: 200,
            inner_pupil_fit: Some(crate::sam31_outer::PupilVoidFitReview {
                ellipse: pupil,
                raw_support: Default::default(),
            }),
            ..Default::default()
        };
        let base = vec![0x00202020; 200 * 200];
        let mut image = base.clone();
        draw_well(&mut image, 200, 200, &p, camera, candidates);
        assert_eq!(image, base, "missing outer must not invent a well");
        p.outer_fit = Some(crate::sam31_outer::OuterMaskFitReview {
            ellipse: Ellipse {
                major_radius: 60.,
                minor_radius: 45.,
                ..pupil
            },
            source_component_area_px: 0.,
            retained_points: Default::default(),
            conic_segments: Default::default(),
            flat_tire_points: Default::default(),
            upper_flat_tire: false,
            lower_flat_tire: false,
        });
        draw_well(&mut image, 200, 200, &p, camera, candidates);
        assert_ne!(image, base);
        p.semantic = Some(crate::sam31_outer::SemanticProposalMasks {
            prompt_index: 0,
            width: 20,
            height: 20,
            selected_query: Some(0),
            masks: vec![crate::sam31_outer::ProposalMask {
                query: 0,
                score: 1.,
                pixels: Arc::new(vec![0; 400]),
                boundary_pixels: Arc::default(),
            }],
        });
        image.clone_from(&base);
        draw_well(&mut image, 200, 200, &p, camera, candidates);
        assert_eq!(image, base, "occluded anatomy must not get well strokes");
        p.semantic = None;
        p.inner_pupil_fit = None;
        draw_well(&mut image, 200, 200, &p, camera, candidates);
        assert_eq!(image, base, "missing pupil must not invent a well");
    }

    #[test]
    fn perspective_centers_and_rays_recover_known_circle_without_centroid_substitution() {
        let camera = PinholeCamera {
            focal_px: [1200., 1250.],
            principal_px: [400., 300.],
        };
        let normal = crate::geometry::normalized3([0.4, -0.3, 0.8]).unwrap();
        let center = [10., -8., -80.];
        let radius = 4.;
        let origin = [240, 100];
        let ellipse = ProjectedCircle::project(camera, center, normal, radius, origin)
            .unwrap()
            .ellipse()
            .unwrap();
        let candidates = projections(camera, ellipse, origin).unwrap();
        let candidate = candidates
            .iter()
            .min_by(|a, b| {
                let d = |c: &Projection| {
                    c.normal
                        .iter()
                        .zip(normal)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                };
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        let expected = camera.project(center).unwrap();
        for j in 0..2 {
            assert!((candidate.center[j] + origin[j] as f64 - expected[j]).abs() < 1e-7);
        }
        assert!(
            (candidate.center[0] - ellipse.center.0).hypot(candidate.center[1] - ellipse.center.1)
                > 0.1
        );
        for c in candidates {
            let projected =
                ProjectedCircle::project(camera, c.center_per_radius, c.normal, 1., origin)
                    .unwrap();
            assert!(ellipse
                .dense_points(32)
                .iter()
                .all(|&q| projected.residual_px(q).abs() < 1e-7));
        }
        // Moving the ROI origin cannot change sensor-space geometry.
        let translated = Ellipse {
            center: (ellipse.center.0 + 20., ellipse.center.1 - 30.),
            ..ellipse
        };
        let other = projections(camera, translated, [220, 130]).unwrap();
        for (a, b) in candidates.iter().zip(other) {
            assert!((a.center[0] - b.center[0] + 20.).abs() < 1e-7);
            assert!((a.center[1] - b.center[1] - 30.).abs() < 1e-7);
        }
    }
    #[test]
    fn invalid_pupil_does_not_produce_an_axis() {
        let camera = PinholeCamera {
            focal_px: [1000.; 2],
            principal_px: [200.; 2],
        };
        let bad = Ellipse {
            center: (100., 100.),
            major_radius: 10.,
            minor_radius: 0.,
            angle: 0.,
        };
        assert!(projections(camera, bad, [0, 0]).is_none());
    }
}
