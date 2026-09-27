//! Read-only presentation of the existing SAM/Student joint-conic publication.
//! No inference, refitting, probability conversion, or new gaze authority lives here.
use super::*;
use crate::gaze_target_solver::joint_tracking::PublishedJoint;
use crate::roi_evidence::RoiId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Layer {
    #[default]
    Conics,
    Segments,
    Masks,
    Raw,
}

#[derive(Default)]
struct EyeEvidence {
    status: &'static str,
    source: Option<u64>,
    sequence: Option<u64>,
    lag_ns: Option<u64>,
    arrival_age: Option<Duration>,
    solver_status: Option<String>,
    retained: usize,
    excluded: usize,
    used: usize,
    rejected: usize,
    weight: f64,
    rms_px: Option<f64>,
    rejected_rms_px: Option<f64>,
    sigma_px: Option<f64>,
}

pub(super) struct Report {
    state: &'static str,
    reason: String,
    eyes: [EyeEvidence; 2],
    publication: Option<Arc<PublishedJoint>>,
    sign_resolved: bool,
    continuous_gaze_enabled: bool,
    skew_ns: Option<u64>,
}

/// Source pixels and all overlays must share the same proposal, including its
/// dimensions/origin. A newer transported ROI is never the overlay backdrop.
fn source(
    frame: &EyeFrame,
    method: SegmentationMode,
    eye: usize,
) -> Option<&sam31_outer::ProposalMasks> {
    let p = frame.sam31_proposal_masks.as_deref()?;
    (method.uses_mask_geometry()
        && frame.segmentation_mode == method
        && frame.eye_id == eye as u32 + 1
        && p.eye_index == eye
        && p.source_width > 0
        && p.source_height > 0
        && p.source_width.checked_mul(p.source_height) == Some(p.source_raw.len())
        && frame
            .gaze_authority_sam_prompt_generation
            .is_none_or(|g| g == p.prompt_generation))
    .then_some(p)
}

fn matches_source(
    p: &sam31_outer::ProposalMasks,
    publication: &PublishedJoint,
    eye: usize,
) -> bool {
    publication.exposures[eye].is_some_and(|e| {
        e.roi == RoiId(eye as u32 + 1)
            && e.sequence == p.source_sequence
            && e.timestamp_ns == p.source_timestamp_ns
    }) && publication.sensor_origins_px[eye]
        == Some([p.source_sensor_origin.0, p.source_sensor_origin.1])
        && publication.dimensions_px[eye] == Some([p.source_width as u32, p.source_height as u32])
}

/// Use one publication as a whole. Matching displayed RAW times alone cannot
/// join different solver generations, clocks, or independently fitted targets.
fn current_publication(snapshot: &Snapshot, publication: &PublishedJoint) -> bool {
    if !snapshot.stereo || !snapshot.second || snapshot.object_running || !snapshot.method.uses_mask_geometry() {
        return false;
    }
    let mut clock_time = None;
    for (i, exposure) in publication.exposures.iter().enumerate() {
        let Some(exposure) = exposure else {
            if publication.solution.contributing_eyes[i] {
                return false;
            }
            continue;
        };
        if clock_time.is_some_and(|key| key != (exposure.clock, exposure.timestamp_ns)) {
            return false;
        }
        clock_time = Some((exposure.clock, exposure.timestamp_ns));
        let Some(frame) = snapshot.eyes[i].as_ref() else {
            return false;
        };
        let Some(p) = source(frame, snapshot.method, i) else {
            return false;
        };
        if !matches_source(p, publication, i)
            || !snapshot.source_arrival_age[i]
                .is_some_and(|age| age.as_nanos() <= SAM31_RESULT_MAX_AGE_NS as u128)
            || !frame.joint_gaze_active
            || frame.gaze_policy_error.is_some()
            || frame.presentation_pivot_held
            || (publication.solution.contributing_eyes[i] && !snapshot.present[i])
            || !frame
                .timestamp_ns
                .checked_sub(p.source_timestamp_ns)
                .is_some_and(|lag| lag <= SAM31_RESULT_MAX_AGE_NS)
            || !frame.joint_conic.as_ref().is_some_and(|shown| {
                shown.source_generation == publication.source_generation
                    && shown.exposures == publication.exposures
                    && shown.solution.target_camera_mm == publication.solution.target_camera_mm
            })
        {
            return false;
        }
    }
    clock_time.is_some()
        && publication
            .solution
            .contributing_eyes
            .into_iter()
            .any(|used| used)
        && publication
            .solution
            .target_camera_mm
            .into_iter()
            .all(f64::is_finite)
}

pub(super) fn inspect(snapshot: &Snapshot) -> Report {
    let mut report = Report {
        state: "WAITING FOR SOLVE",
        reason: "Waiting for a solve aligned with the displayed mask sources.".into(),
        eyes: std::array::from_fn(|_| EyeEvidence::default()),
        publication: None,
        sign_resolved: false,
        continuous_gaze_enabled: snapshot.continuous_gaze_sign,
        skew_ns: None,
    };
    for (i, result) in report.eyes.iter_mut().enumerate() {
        result.status = "NO SOURCE FRAME";
        if i == 1 && !snapshot.second {
            result.status = "SECOND EYE DISABLED";
            continue;
        }
        let Some(frame) = snapshot.eyes[i].as_ref() else {
            continue;
        };
        if frame.segmentation_mode != snapshot.method {
            result.status = "MODEL SWITCH IN PROGRESS";
            continue;
        }
        let Some(p) = source(frame, snapshot.method, i) else {
            result.status = "SOURCE UNAVAILABLE";
            continue;
        };
        result.arrival_age = snapshot.source_arrival_age[i];
        result.solver_status = frame.joint_conic_status.clone();
        result.source = Some(p.source_timestamp_ns);
        result.sequence = Some(p.source_sequence);
        result.lag_ns = frame.timestamp_ns.checked_sub(p.source_timestamp_ns);
        if let Some(fit) = &p.outer_fit {
            result.retained = fit.retained_points.len();
            result.excluded = fit.flat_tire_points.len();
        }
        result.status = if frame.gaze_policy_error.is_some() {
            "SETTINGS CHANGED / WAITING"
        } else if result.lag_ns.is_none() {
            "FUTURE SOURCE / NOT USED"
        } else if result.arrival_age.is_none() {
            "SOURCE CLOCK UNAVAILABLE"
        } else if result
            .arrival_age
            .is_some_and(|age| age.as_nanos() > SAM31_RESULT_MAX_AGE_NS as u128)
        {
            "OLD SOURCE / NOT USED"
        } else if result
            .lag_ns
            .is_some_and(|lag| lag > SAM31_RESULT_MAX_AGE_NS)
        {
            "OLD SOURCE / NOT USED"
        } else if frame.presentation_pivot_held || !snapshot.present[i] {
            "HELD SOURCE / NOT NEW EVIDENCE"
        } else if p.outer_fit.is_none() {
            "NO LIMBUS CONIC"
        } else {
            "SOURCE CONICS AVAILABLE"
        };
    }
    report.skew_ns = report.eyes[0]
        .source
        .zip(report.eyes[1].source)
        .map(|(a, b)| a.abs_diff(b));
    if !snapshot.method.uses_mask_geometry() {
        report.state = "SELECT SAM OR STUDENT";
        report.reason = "Stereo conics use the SAM3.1 or Eye Student segmentation source.".into();
    } else if !snapshot.stereo || !snapshot.second {
        report.state = "STEREO OFF";
        report.reason = "Enable stereo (Shift+3) to use the joint solver with either mask detector and any view.".into();
    } else if snapshot.object_running {
        report.state = "EYE ANALYSIS PAUSED";
        report.reason =
            "Object search is using the sensor. Stop search to resume eye analysis.".into();
    } else {
        report.publication = snapshot
            .eyes
            .iter()
            .flatten()
            .filter_map(|f| f.joint_conic.as_ref())
            .filter(|p| current_publication(snapshot, p))
            .max_by_key(|p| p.exposures.iter().flatten().map(|e| e.timestamp_ns).max())
            .cloned();
        if let Some(publication) = &report.publication {
            let s = &publication.solution;
            report.state = if s.contributing_eyes == [true, true] {
                "BOTH EYES CONTRIBUTE"
            } else {
                "ONE EYE CONTRIBUTES"
            };
            report.reason = if s.contributing_eyes == [true, true] {
                "Joint target; conditional on camera and scale priors."
            } else {
                "The current solve has usable support from one eye only. Two visible images do not establish a binocular solve."
            }.into();
            if publication.exposures.iter().any(Option::is_none)
                && snapshot.eyes.iter().enumerate().any(|(i, frame)| {
                    publication.exposures[i].is_some()
                        && frame
                            .as_ref()
                            .and_then(|f| source(f, snapshot.method, i))
                            .is_some_and(|p| p.source_group_roi_count == 2)
                })
            {
                report.state = "PARTNER SOLVE PENDING";
                report.reason = "Provisional one-eye result; the second eye from this paired exposure has not completed.".into();
            }
            report.sign_resolved = snapshot
                .eyes
                .iter()
                .flatten()
                .filter(|f| {
                    f.joint_conic.as_ref().is_some_and(|p| {
                        p.source_generation == publication.source_generation
                            && p.exposures == publication.exposures
                    })
                })
                .filter_map(|f| joint_gaze_live::surface(f, true))
                .any(|s| s.sign_resolved);
            if publication.orientation_ready.is_some_and(|ready| !ready.into_iter().any(|v|v)) {
                report.state="SCREEN ORIENTATION NEEDED";
                report.reason="Look at the top plus during M calibration. Shift+M reorients using an existing display calibration.".into();
            }
            for (i, eye) in report.eyes.iter_mut().enumerate() {
                let mut used_square = 0.0;
                let mut rejected_square = 0.0;
                let mut rejected_weight = 0.0;
                let mut sigma_square = 0.0;
                // ArcSupport contains one selected alternative per correlation
                // group. Never turn retained point counts into observation votes.
                for arc in s
                    .arcs
                    .iter()
                    .filter(|a| Some(a.exposure) == publication.exposures[i])
                {
                    if arc.used {
                        eye.used += 1;
                    } else {
                        eye.rejected += 1;
                    }
                    if !arc.evidence_weight.is_finite()
                        || arc.evidence_weight <= 0.0
                        || !arc.rms_px.is_finite()
                        || !arc.sigma_px.is_finite()
                    {
                        continue;
                    }
                    if arc.used {
                        eye.weight += arc.evidence_weight;
                        used_square += arc.evidence_weight * arc.rms_px.powi(2);
                        sigma_square += arc.evidence_weight * arc.sigma_px.powi(2);
                    } else {
                        rejected_weight += arc.evidence_weight;
                        rejected_square += arc.evidence_weight * arc.rms_px.powi(2);
                    }
                }
                if eye.weight > 0.0 {
                    eye.rms_px = Some((used_square / eye.weight).sqrt());
                    eye.sigma_px = Some((sigma_square / eye.weight).sqrt());
                }
                if rejected_weight > 0.0 {
                    eye.rejected_rms_px = Some((rejected_square / rejected_weight).sqrt());
                }
            }
        } else if report.eyes.iter().any(|e| e.source.is_none()) {
            report.state = "WAITING FOR EYE EVIDENCE";
        } else if report
            .eyes
            .iter()
            .any(|e| e.status.contains("HELD") || e.status.contains("OLD"))
        {
            report.state = "HELD / WAITING FOR SOLVE";
            report.reason = "Source images remain inspectable; held or expired geometry is not presented as a new solve.".into();
        }
    }
    report
}

fn number(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map_or("--".into(), |v| format!("{v:.2}"))
}

impl Report {
    // Iris-center separation is an IPD proxy under the current metric priors,
    // not an independently measured distance between pupil centers.
    fn ipd_mm(&self) -> Option<f64> {
        let solution=&self.publication.as_ref()?.solution;
        if solution.contributing_eyes != [true,true] {return None;}
        let [Some(a),Some(b)]=solution.eye_centers_camera_mm else {return None;};
        let distance=(a[0]-b[0]).hypot(a[1]-b[1]).hypot(a[2]-b[2]);
        (distance.is_finite() && distance>0.0).then_some(distance)
    }
    fn ipd_label(&self) -> String {
        self.ipd_mm().map_or_else(||"IPD EST: --".into(),|v|format!("IPD EST: {v:.1} MM"))
    }

    pub(super) fn json(&self) -> serde_json::Value {
        let solution = self.publication.as_ref().map(|p| &p.solution);
        serde_json::json!({
            "state": self.state, "reason": self.reason,
            "ipd_estimate_mm": self.ipd_mm(),
            "ipd_basis": "solved iris-center separation; metric scale prior dependent",
            "target_camera_mm": solution.map(|s| s.target_camera_mm),
            "direction_sign_resolved": self.sign_resolved,
            "screen_orientation_ready": self.publication.as_ref().and_then(|p|p.orientation_ready),
            "continuous_gaze_sign": self.publication.as_ref().filter(|_|self.continuous_gaze_enabled).and_then(|p|p.continuous_gaze_sign.as_ref()).map(|r|r.json()),
            "probability": null, "target_covariance": null,
            "uncertainty_status": solution.and_then(|s|s.posterior.as_ref()).map_or("not estimated",|p|p.status),
            "local_uncertainty": solution.and_then(|s|s.local_uncertainty.as_ref()).map(|u|u.json()),
            "posterior": solution.and_then(|s|s.posterior.as_ref()).map(|p|p.json()),
            "source_skew_ns": self.skew_ns.map(|v| v.to_string()),
            "source_generation": self.publication.as_ref().map(|p| p.source_generation.to_string()),
            "fit_cost": solution.map(|s| s.robust_cost),
            "alternative_cost_gap": solution.and_then(|s| s.alternative_cost_margin),
            "eyes": self.eyes.iter().enumerate().map(|(i,e)| serde_json::json!({
                "status": e.status, "source_timestamp_ns": e.source.map(|v| v.to_string()),
                "sequence": e.sequence.map(|v| v.to_string()), "source_lag_ns": e.lag_ns.map(|v| v.to_string()),
                "host_arrival_age_ms": e.arrival_age.map(|age| age.as_secs_f64() * 1000.0),
                "solver_status": e.solver_status,
                "modeled": solution.map(|s| s.modeled_eyes[i]),
                "contributing": solution.map(|s| s.contributing_eyes[i]),
                "unlocalized_eye_cost": solution.map(|s| s.unlocalized_eye_cost[i]),
                "retained_points": e.retained, "excluded_points": e.excluded,
                "used_groups": solution.map(|_| e.used), "rejected_groups": solution.map(|_| e.rejected),
                "used_arc_weight": solution.map(|_| e.weight), "used_rms_px": e.rms_px,
                "rejected_rms_px": e.rejected_rms_px, "engineering_sigma_px": e.sigma_px,
            })).collect::<Vec<_>>()
        })
    }

    pub(super) fn rows(&self) -> Vec<String> {
        let mut rows = vec![
            self.state.into(),
            self.reason.clone(),
            self.ipd_label(),
            "IPD uses solved iris centers and uncertain metric scale priors.".into(),
            format!(
                "MASK SOURCE SKEW: {} MS",
                number(self.skew_ns.map(|v| v as f64 / 1e6))
            ),
            "Model spread describes ambiguity; it is not measured gaze accuracy.".into(),
        ];
        if let Some(p) = &self.publication {
            let s = &p.solution;
            if let Some(report)=p.continuous_gaze_sign.as_ref().filter(|_|self.continuous_gaze_enabled) {
                rows.push(format!("CONTINUOUS GAZE (EXPERIMENT): {}",report.status));
                rows.push("Trajectory score does not establish sign truth.".into());
            }
            if let Some(posterior) = &s.posterior {
                rows.push(format!("MODEL UNCERTAINTY: {}", posterior.status));
                rows.push(format!(
                    "{} SAMPLES / {:.0} EFFECTIVE",
                    posterior.samples, posterior.effective_samples
                ));
                if posterior.status=="estimated-conditional" {
                    rows.push("MODEL MASS WITHIN 15 DEG OF CHOSEN RAY:".into());
                    for eye in 0..2 {
                        if let Some(precision)=posterior.admission_numerics(eye) {
                            rows.push(format!("{}: {:.1}% +/- {:.1} POINTS (2 SE)",
                                if eye==0 {"RIGHT"} else {"LEFT"},100.0*precision.mass,200.0*precision.standard_error));
                        }
                    }
                    rows.push("Sampling error is approximate; it cannot rule out unseen directions.".into());
                }
                for (i, mode) in posterior.modes.iter().enumerate() {
                    if let Some(mass) = mode.model_mass {
                        rows.push(format!(
                            "DIRECTION REGION {}: {:.0}% MODEL MASS",
                            i + 1,
                            100.0 * mass
                        ));
                    }
                }
                rows.push("Conditional on current contours and camera/eye priors; unexplored directions may remain.".into());
            } else {
                rows.push("MODEL UNCERTAINTY: NOT ESTIMATED".into());
            }
            rows.push(format!(
                "DIRECTION SIGN: {}",
                if self.sign_resolved {
                    "RESOLVED"
                } else {
                    "UNRESOLVED"
                }
            ));
            rows.push(format!(
                "SHARED TARGET / CAMERA MM: X {:.1} Y {:.1} Z {:.1}",
                s.target_camera_mm[0], s.target_camera_mm[1], s.target_camera_mm[2]
            ));
            rows.push(format!(
                "FIT COST {} / {} STARTS",
                number(Some(s.robust_cost)),
                s.hypotheses_evaluated
            ));
            rows.push(format!(
                "COMPETING SOLVE COST GAP: {}",
                number(s.alternative_cost_margin)
            ));
            if s.alternative_cost_margin.is_none() {
                rows.push(
                    "No distinct alternative found; this does not establish certainty.".into(),
                );
            }
            if s.target_viewpoint_bounds_active.into_iter().any(|v| v) {
                rows.push("TARGET AT SEARCH BOUND".into());
            }
        }
        for (i, e) in self.eyes.iter().enumerate() {
            rows.push(format!("{}: {}", eye_name(i), e.status));
            if let Some(age) = e.arrival_age {
                rows.push(format!(
                    "SOURCE RECEIVED {:.0} MS AGO",
                    age.as_secs_f64() * 1000.0
                ));
            }
            if self.publication.is_none() {
                if let Some(status) = &e.solver_status {
                    rows.push(format!("SOLVER: {status}"));
                }
            }
            rows.push(format!(
                "POINTS {} RETAINED / {} EXCLUDED",
                e.retained, e.excluded
            ));
            if let Some(publication) = &self.publication {
                let solution = &publication.solution;
                if !solution.modeled_eyes[i] {
                    rows.push(format!(
                        "UNLOCALIZED / OMITTED COST {}",
                        number(Some(solution.unlocalized_eye_cost[i]))
                    ));
                } else if !solution.contributing_eyes[i] {
                    rows.push("NO ACCEPTED ARC GROUPS".into());
                }
                rows.push(format!("GROUPS {} USED / {} REJECTED", e.used, e.rejected));
                rows.push(format!(
                    "USED RMS {} PX / SIGMA {} PX",
                    number(e.rms_px),
                    number(e.sigma_px)
                ));
                rows.push(format!("REJECTED RMS {} PX", number(e.rejected_rms_px)));
                rows.push(format!(
                    "MODEL 90% GAZE RADIUS: {} DEG",
                    number(
                        solution
                            .posterior
                            .as_ref()
                            .and_then(|p| p.gaze_radius_90_degrees[i])
                    )
                ));
                rows.push(format!(
                    "LOCAL ANGULAR SIGMA: {} DEG",
                    number(
                        solution
                            .local_uncertainty
                            .as_ref()
                            .and_then(|u| u.worst_axis_sigma_degrees(i))
                    )
                ));
            }
        }
        rows.extend([
            "Local sigma covers one direction basin; the sampled model radius also includes competing directions.".into(),
            "Boundary SIGMA is an engineering allowance; arc weights are not probabilities.".into(),
            "GREEN retained / PINK excluded / CYAN source fit / WHITE joint fit".into(),
            "3 enable/disable stereo globally. F next linked view. V image appearance.".into(),
        ]);
        rows
    }
}

fn take_row(area: &mut Rect, height: usize) -> Rect {
    let row = Rect {
        h: height.min(area.h),
        ..*area
    };
    area.y += row.h;
    area.h -= row.h;
    row
}

fn conic(pixels: &mut [u32], w: usize, h: usize, ellipse: geometry::Ellipse) {
    if ![
        ellipse.center.0,
        ellipse.center.1,
        ellipse.major_radius,
        ellipse.minor_radius,
        ellipse.angle,
    ]
    .into_iter()
    .all(f64::is_finite)
        || ellipse.minor_radius <= 0.0
        || ellipse.major_radius <= 0.0
    {
        return;
    }
    let points = ellipse.dense_points(240);
    for (i, a) in points.iter().enumerate() {
        let b = points[(i + 1) % points.len()];
        // Dashed white distinguishes the shared reconstruction from the cyan
        // source fit; neither is substituted for observed contour samples.
        if i % 10 < 6 {
            draw_line_clipped(
                pixels,
                w,
                h,
                a.0.round() as i32,
                a.1.round() as i32,
                b.0.round() as i32,
                b.1.round() as i32,
                INK,
            );
        }
    }
}

const SEGMENT_LABEL_REFRESH: Duration = Duration::from_millis(250);
const SEGMENT_LABEL_HOLD: Duration = Duration::from_millis(350);

#[derive(Default)]
struct SegmentLabels {
    context: Option<(SegmentationMode,u64,usize,usize)>,
    clock: Option<crate::roi_evidence::SourceClock>,
    slots: Vec<SegmentLabel>,
}
struct SegmentLabel {
    key: (u8,u32),
    source: Option<crate::roi_evidence::ExposureKey>,
    seen: Instant,
    refreshed: Instant,
    value: String,
}
impl SegmentLabels {
    fn slot(&mut self, key:(u8,u32), source:crate::roi_evidence::ExposureKey,
        value:String, now:Instant) -> Option<usize> {
        let i=if let Some(i)=self.slots.iter().position(|s|s.key==key) {i} else {
            // Bounded presentation storage. Slots are never reused within a
            // detector/source-clock context, so a dropout cannot renumber peers.
            if self.slots.len()>=20 {return None;}
            self.slots.push(SegmentLabel {key,source:None,seen:now,refreshed:now,value:value.clone()});
            self.slots.len()-1
        };
        let slot=&mut self.slots[i];
        if slot.source.is_none_or(|old|source.timestamp_ns>old.timestamp_ns) {
            slot.source=Some(source);slot.seen=now;
            if now.saturating_duration_since(slot.refreshed)>=SEGMENT_LABEL_REFRESH {
                slot.value=value;slot.refreshed=now;
            }
        }
        Some(i)
    }
}
thread_local! {
    static SEGMENT_LABELS: std::cell::RefCell<[SegmentLabels;2]> =
        std::cell::RefCell::new(std::array::from_fn(|_|SegmentLabels::default()));
}

pub(crate) fn draw_stereo_segments(
    pixels: &mut [u32], width: usize, height: usize,
    frame: &EyeFrame, p: &sam31_outer::ProposalMasks,
) -> usize {
    draw_stereo_segments_at(pixels,width,height,frame,p,true,Instant::now())
}

/// Display sections of the already solved conic beside its actual observations.
/// RAW sample joins are not fitted conics: joining noisy radial peaks made the
/// pupil look concave. This does not refit, smooth, or replace solver evidence.
/// Each neighboring pair bounds a separate section. Invalid points and large
/// phase gaps stay disconnected; a missing observation cannot close a ring.
fn fitted_arc_sections(
    ellipse: geometry::Ellipse,
    observations: &[(f64, f64)],
) -> Vec<((f64, f64), (f64, f64))> {
    if ![ellipse.center.0, ellipse.center.1, ellipse.major_radius,
        ellipse.minor_radius, ellipse.angle].into_iter().all(f64::is_finite)
        || ellipse.minor_radius <= 0.0 || ellipse.major_radius < ellipse.minor_radius
    {
        return Vec::new();
    }
    let (s, c) = ellipse.angle.sin_cos();
    let phase = |p: (f64, f64)| {
        let x = p.0 - ellipse.center.0;
        let y = p.1 - ellipse.center.1;
        let u = (c * x + s * y) / ellipse.major_radius;
        let v = (-s * x + c * y) / ellipse.minor_radius;
        (u.is_finite() && v.is_finite() && u.hypot(v) > 1.0e-6).then(|| v.atan2(u))
    };
    let point = |phase: f64| {
        let x = ellipse.major_radius * phase.cos();
        let y = ellipse.minor_radius * phase.sin();
        (ellipse.center.0 + c * x - s * y, ellipse.center.1 + s * x + c * y)
    };
    let mut sections = Vec::new();
    for pair in observations.windows(2).take(63) {
        let Some((a, b)) = phase(pair[0]).zip(phase(pair[1])) else { continue; };
        let delta = (b - a).sin().atan2((b - a).cos());
        if delta.abs() > std::f64::consts::FRAC_PI_2 { continue; }
        let steps = (delta.abs() * ellipse.major_radius).ceil().clamp(1.0, 128.0) as usize;
        for i in 0..steps {
            sections.push((point(a + delta * i as f64 / steps as f64),
                point(a + delta * (i + 1) as f64 / steps as f64)));
        }
    }
    sections
}

fn draw_stereo_segments_at(
    pixels: &mut [u32], width: usize, height: usize,
    frame: &EyeFrame, p: &sam31_outer::ProposalMasks, allow_publication:bool, now:Instant,
) -> usize {
    let eye=p.eye_index;
    if eye>=2 {return 0;}
    let joint=frame.joint_conic.as_deref().filter(|j| allow_publication && frame.joint_gaze_active
        && matches_source(p,j,eye)
        && frame.timestamp_ns.saturating_sub(p.source_timestamp_ns)<=900_000_000);
    let arcs=joint.map(|j|j.solution.arcs.iter().filter(|a|a.exposure.roi==RoiId(eye as u32+1)).collect::<Vec<_>>()).unwrap_or_default();
    let total=joint.map_or(0.0,|j|j.solution.arcs.iter().filter(|a|a.used).map(|a|a.evidence_weight).sum::<f64>());
    let palette=[0x005b_def4,0x00ff_8fb8,0x00ff_d36b,0x00a8_ef90];
    SEGMENT_LABELS.with(|labels| {
        let mut labels=labels.borrow_mut();let labels=&mut labels[eye];
        let context=(frame.segmentation_mode,p.prompt_generation,width,height);
        let clock=joint.and_then(|j|j.exposures[eye]).map(|e|e.clock);
        if labels.context!=Some(context) || clock.is_some_and(|clock|labels.clock.is_some_and(|old|old!=clock)) {
            *labels=SegmentLabels {context:Some(context),clock,..Default::default()};
        }
        if clock.is_some() {labels.clock=clock;}
        let mut present=[false;20];
        let rows=((height.saturating_sub(72))/20).max(1);
        let position=|i:usize| {
            let x=if i%2==0 {4} else {width.saturating_sub(104) as i32};
            (x,(28+(i/2)*20) as i32)
        };
        for a in &arcs {
            let kind=match a.kind {
                crate::roi_evidence::BoundaryKind::OuterLimbus=>0,
                crate::roi_evidence::BoundaryKind::InnerLimbus=>1,
                crate::roi_evidence::BoundaryKind::PupillaryBoundary=>2,
                _=>3,
            };
            let value=if a.used && total>0.0 {format!("{:>5.1}%",100.0*a.evidence_weight/total)} else {"REJECT".into()};
            let Some(i)=labels.slot((kind,a.evidence_group),a.exposure,value,now) else {continue;};
            present[i]=true;
            let color=palette[i%palette.len()];
            let valid=|p:(f64,f64)|p.0.is_finite() && p.1.is_finite()
                && p.0>=0.0 && p.1>=0.0 && p.0<width as f64 && p.1<height as f64;
            if a.used {
                if let Some(ellipse)=joint.and_then(|j|j.solution.ellipses_roi_px[eye].get(kind as usize).copied().flatten()) {
                    for (a,b) in fitted_arc_sections(ellipse,&a.points_roi_px) {
                        draw_line_clipped(pixels,width,height,a.0.round() as i32,a.1.round() as i32,
                            b.0.round() as i32,b.1.round() as i32,color);
                    }
                }
            }
            // Keep every measured coordinate visible. Never draw point-to-point
            // chords as anatomical curves, or put rejected evidence on the fit.
            for &(x,y) in a.points_roi_px.iter().filter(|p|valid(**p)) {
                let (x,y)=(x.round() as i32,y.round() as i32);
                fill_rect(pixels,width,height,x-1,y-1,3,3,0x0011_1820);
                if a.used {
                    fill_rect(pixels,width,height,x,y,1,1,color);
                } else {
                    draw_line_clipped(pixels,width,height,x-1,y-1,x+1,y+1,color);
                    draw_line_clipped(pixels,width,height,x-1,y+1,x+1,y-1,color);
                }
            }
            if i/2>=rows {continue;}
            let Some(anchor)=a.points_roi_px.get(a.points_roi_px.len()/2).copied().filter(|p|valid(*p)) else {continue;};
            let (x,y)=position(i);let pin=if i%2==0 {x+94} else {x};
            draw_line_clipped(pixels,width,height,anchor.0 as i32,anchor.1 as i32,pin,y+6,color);
            fill_rect(pixels,width,height,anchor.0 as i32-2,anchor.1 as i32-2,5,5,color);
        }
        // Draw labels last, in permanent slots, so leader lines cannot erase
        // neighboring text. Retained labels never retain old contour geometry.
        for (i,slot) in labels.slots.iter().enumerate().filter(|(i,_)|i/2<rows) {
            let age=now.saturating_duration_since(slot.seen);
            let held=!present[i] && age<=SEGMENT_LABEL_HOLD;
            let value=if present[i] || held {slot.value.as_str()} else {"   -- "};
            let color=if present[i] || held {palette[i%palette.len()]} else {0x007e_8b98};
            let (x,y)=position(i);
            fill_rect(pixels,width,height,x,y,100,14,0x0011_1820);
            draw_text(pixels,width,height,x+2,y+2,&format!("{:>2} {}{}",i+1,value,if held {" H"} else {"  "}),color);
        }
    });
    if joint.is_none() {
        let message = if !frame.joint_gaze_active {
            "STEREO OFF - SHIFT+3 TO ENABLE"
        } else {
            "WAITING FOR MATCHING STEREO SEGMENTS"
        };
        draw_text(pixels,width,height,4,8,message,0x007e_8b98);
    }
    draw_text(pixels,width,height,4,height.saturating_sub(24) as i32,"CURVES: FIT / DOTS: MEASURED POINTS",0x00ff_ffff);
    draw_text(pixels,width,height,4,height.saturating_sub(12) as i32,"FIT WEIGHT / X=REJECT / H=HELD LABEL",0x00ff_ffff);
    arcs.len()
}

fn source_image(
    frame: &EyeFrame,
    p: &sam31_outer::ProposalMasks,
    mode: ViewMode,
    layer: Layer,
    publication: Option<&PublishedJoint>,
    eye: usize,
) -> Vec<u32> {
    let mut pixels = vec![BG; p.source_width * p.source_height];
    let overlay = match layer {
        Layer::Conics if p.outer_fit.is_some() => RoiOverlayMode::SamOuterIrisFit,
        Layer::Masks
            if p.semantic.as_ref().is_some_and(|s| {
                s.selected_query.is_some()
                    && s.masks.iter().any(|m| Some(m.query) == s.selected_query)
            }) =>
        {
            RoiOverlayMode::SamOuterIrisMasks
        }
        _ => RoiOverlayMode::StudentSourceRaw,
    };
    // Both detectors already publish the same ProposalMasks contract. Reuse
    // its cached native RAW renderer and existing fit/mask layers for both.
    student_preview::draw(
        &mut pixels,
        p.source_width,
        p.source_height,
        0,
        0,
        1,
        frame,
        mode,
        overlay,
    );
    if layer == Layer::Segments {
        let allowed=publication.is_some_and(|j|frame.joint_conic.as_deref().is_some_and(|f|std::ptr::eq(f,j)));
        draw_stereo_segments_at(&mut pixels,p.source_width,p.source_height,frame,p,allowed,Instant::now());
    }
    if layer == Layer::Conics {
        if let Some(joint) = publication.filter(|joint| matches_source(p, joint, eye)) {
            for ellipse in joint.solution.ellipses_roi_px[eye].iter().flatten() {
                conic(&mut pixels, p.source_width, p.source_height, *ellipse);
            }
        }
    }
    pixels
}

/// Compact source-matched scene beside normal ROI views. No solver work or
/// state mutation occurs here; the same publication drives the diagnostic page.
pub(super) fn render_scene(c: &mut Canvas, area: Rect, snapshot: &Snapshot) {
    c.fill(area, CARD);
    let mut body = area.inset(8);
    c.text(take_row(&mut body, 22), "STEREO SCENE", INK);
    let report = inspect(snapshot);
    c.text(take_row(&mut body, 20), if report.publication.is_some() && !report.sign_resolved { "SIGN UNRESOLVED" } else { report.state }, MUTED);
    c.text(take_row(&mut body, 20), &report.ipd_label(), INK);
    let Some(publication) = report.publication else { return; };
    if body.w < 16 || body.h < 16 { return; }
    let solution = &publication.solution;
    let target = solution.target_camera_mm;
    let gaze_ends: [Option<[f64;3]>;2] = std::array::from_fn(|eye| {
        let center=solution.eye_centers_camera_mm[eye]?;
        let direction=solution.eye_gaze_directions[eye]?;
        let distance=center.iter().zip(target).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        Some(std::array::from_fn(|i|center[i]+direction[i]*distance))
    });
    let normal_ends: [Option<[f64;3]>;2] = std::array::from_fn(|eye| {
        let center=solution.eye_centers_camera_mm[eye]?;
        let normal=solution.eye_normals[eye]?;
        Some(std::array::from_fn(|i|center[i]+normal[i]*20.0))
    });
    let mut points = vec![[0.0; 3], target];
    points.extend(gaze_ends.iter().flatten().copied());
    points.extend(normal_ends.iter().flatten().copied());
    points.extend(solution.eye_centers_camera_mm.iter().flatten().copied());
    if points.iter().flatten().any(|v| !v.is_finite()) { return; }
    // Fixed oblique camera-coordinate overview with a common metric scale.
    let view = |p: [f64;3]| [0.866*p[0]+0.5*p[2], p[1]+0.25*p[2]];
    let mut low = [f64::INFINITY;2];
    let mut high = [f64::NEG_INFINITY;2];
    for p in points { let p=view(p); for i in 0..2 {low[i]=low[i].min(p[i]);high[i]=high[i].max(p[i]);} }
    let scale = ((body.w.saturating_sub(24)) as f64/(high[0]-low[0]).max(1.0))
        .min((body.h.saturating_sub(24)) as f64/(high[1]-low[1]).max(1.0));
    let project = |p| { let p=view(p); ((body.w as f64/2.0+(p[0]-(low[0]+high[0])/2.0)*scale) as i32,
        (body.h as f64/2.0+(p[1]-(low[1]+high[1])/2.0)*scale) as i32) };
    let mut pixels=vec![CARD;body.w*body.h];
    let mut line=|a:(i32,i32),b:(i32,i32),color| draw_line_clipped(&mut pixels,body.w,body.h,a.0,a.1,b.0,b.1,color);
    let camera=project([0.0;3]);
    for (a,b) in [((-7,-5),(7,-5)),((7,-5),(7,5)),((7,5),(-7,5)),((-7,5),(-7,-5)),
        ((7,-4),(14,-8)),((14,-8),(14,8)),((14,8),(7,4))] {
        line((camera.0+a.0,camera.1+a.1),(camera.0+b.0,camera.1+b.1),MUTED);
    }
    for eye in 0..2 {
        let Some(center)=solution.eye_centers_camera_mm[eye] else {continue;};
        let origin=project(center);
        let color=if solution.contributing_eyes[eye] {if eye==0 {0x70c7ff} else {0xffbf7b}} else {MUTED};
        for step in 0..24 {
            let ring=|n:usize| {let a=n as f64*std::f64::consts::TAU/24.0;
                (origin.0+(6.0*a.cos()) as i32,origin.1+(6.0*a.sin()) as i32)};
            line(ring(step),ring(step+1),color);
        }
        if let Some(end)=gaze_ends[eye] {line(origin,project(end),color);}
        if let Some(end)=normal_ends[eye] {line(origin,project(end),0xcddc79);}

    }
    let target=project(target);
    line((target.0-4,target.1),(target.0+4,target.1),INK);
    line((target.0,target.1-4),(target.0,target.1+4),INK);
    c.image(body,&pixels,body.w,body.h);
}

pub(super) fn render(c: &mut Canvas, ui: &mut Workspace, mut area: Rect, snapshot: &Snapshot) {
    let report = inspect(snapshot);
    if area.h < 100 {
        // At the minimum window size the shell leaves less than two text
        // rows for imagery. Keep activation usable and explain the missing
        // canvas instead of clipping controls into the inspector below.
        let toggle = take_row(&mut area, 28);
        if toggle.h >= 24 {
            button(
                c,
                ui,
                toggle,
                if snapshot.stereo {
                    "SHIFT+3 STEREO OFF"
                } else {
                    "SHIFT+3 STEREO ON"
                },
                snapshot.stereo,
                Action::ToggleStereo,
            );
        }
        c.text(area, "ENLARGE FOR EYE IMAGES", MUTED);
        return;
    }
    let controls = take_row(&mut area, 32);
    let wide = controls.w >= 600;
    let toggle = Rect {
        w: if wide { 268 } else { controls.w },
        h: controls.h.saturating_sub(4),
        ..controls
    };
    if toggle.h > 0 {
        button(
            c,
            ui,
            toggle,
            if snapshot.stereo {
                "SHIFT+3 STEREO OFF"
            } else {
                "SHIFT+3 STEREO ON"
            },
            snapshot.stereo,
            Action::ToggleStereo,
        );
    }
    let layers = if wide {
        Rect {
            x: controls.x + 276,
            w: controls.w - 276,
            ..controls
        }
    } else {
        take_row(&mut area, 32)
    };
    for (i, (label, layer)) in [
        ("CONICS", Layer::Conics),
        ("SEGMENTS", Layer::Segments),
        ("MASKS", Layer::Masks),
        ("RAW", Layer::Raw),
    ]
    .into_iter()
    .enumerate()
    {
        let w = layers.w / 4;
        if layers.h > 0 {
            button(
                c,
                ui,
                Rect {
                    x: layers.x + i * w,
                    w: w.saturating_sub(4),
                    h: layers.h.saturating_sub(4),
                    ..layers
                },
                label,
                ui.stereo_layer == layer,
                Action::StereoLayer(layer),
            );
        }
    }
    let heading = take_row(&mut area, 24);
    c.text(
        heading,
        report.state,
        if report.publication.is_some() {
            ACCENT
        } else {
            MUTED
        },
    );
    c.text(take_row(&mut area, 22), &report.ipd_label(), INK);
    let info_h = if area.h >= 340 { 150 } else { 0 };
    let cards = split_pair(Rect {
        h: area.h.saturating_sub(info_h),
        ..area
    });
    for (i, card) in cards.into_iter().enumerate() {
        c.fill(card, CARD);
        let mut body = card.inset(if card.h < 180 { 4 } else { 10 });
        let title = take_row(&mut body, 24);
        c.text(title, eye_name(i), INK);
        let evidence = &report.eyes[i];
        if card.h >= 180 {
            let status = take_row(&mut body, 20);
            c.text(status, evidence.status, MUTED);
        }
        let footer = Rect {
            y: body.y + body.h.saturating_sub(20),
            h: 20.min(body.h),
            ..body
        };
        let image_area = Rect {
            h: body.h.saturating_sub(20),
            ..body
        };
        if i == 0 || snapshot.second {
            if let Some(frame) = snapshot.eyes[i].as_ref() {
                if let Some(p) = source(frame, snapshot.method, i) {
                    let pixels = source_image(
                        frame,
                        p,
                        ui.roi_view(i).pixels,
                        ui.stereo_layer,
                        report.publication.as_deref(),
                        i,
                    );
                    c.image(image_area, &pixels, p.source_width, p.source_height);
                }
            }
        }
        let lag = evidence
            .lag_ns
            .map_or("--".into(), |v| format!("{:.0}", v as f64 / 1e6));
        c.text(
            footer,
            &format!(
                "SOURCE {} / LAG {lag} MS",
                evidence.sequence.map_or("--".into(), |v| v.to_string())
            ),
            MUTED,
        );
    }
    if info_h == 0 {
        return;
    } // All details remain in the scrollable inspector.
    let mut info = Rect {
        y: area.y + area.h - info_h,
        h: info_h,
        ..area
    }
    .inset(8);
    c.text(
        take_row(&mut info, 24),
        "RELATIVE ARC SUPPORT / NOT PROBABILITY",
        INK,
    );
    let bar = take_row(&mut info, 12);
    c.fill(bar, CARD);
    let total = report.eyes.iter().map(|e| e.weight).sum::<f64>();
    if total > 0.0 {
        let right = (bar.w as f64 * report.eyes[0].weight / total).round() as usize;
        c.fill(Rect { w: right, ..bar }, ACCENT);
        c.fill(
            Rect {
                x: bar.x + right,
                w: bar.w - right,
                ..bar
            },
            0x00e7_b577,
        );
    }
    take_row(&mut info, 8);
    for (i, column) in split_pair(Rect {
        h: info.h.min(80),
        ..info
    })
    .into_iter()
    .enumerate()
    {
        let e = &report.eyes[i];
        let mut rows = vec![if total > 0.0 {
            format!(
                "{} {:.0}% ARC WEIGHT",
                if i == 0 { "RIGHT" } else { "LEFT" },
                100.0 * e.weight / total
            )
        } else {
            "WAITING FOR SOLVER SUPPORT".into()
        }];
        if report.publication.is_some() {
            rows.push(format!("{} USED / {} REJECTED GROUPS", e.used, e.rejected));
            rows.push(format!("RMS {} PX", number(e.rms_px)));
            let posterior = report
                .publication
                .as_ref()
                .and_then(|p| p.solution.posterior.as_ref());
            rows.push(
                posterior
                    .and_then(|p| p.gaze_radius_90_degrees[i])
                    .map_or_else(
                        || "GAZE SPREAD UNRESOLVED".into(),
                        |radius| format!("MODEL 90% RADIUS {radius:.1} DEG"),
                    ),
            );
        }
        text_rows(c, column, &rows, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic_solver::joint::{ArcSupport, JointConicSolution, PinholeCamera};
    use crate::eye_scene_model::binocular_pose::{approximate_scene, EyePoseInput};
    use crate::roi_evidence::{BoundaryKind, ExposureKey, SourceClock};

    // Deliberately synthetic UI evidence, with known asymmetric arc weights
    // and a rejected group. These numbers are not a corpus accuracy result.
    fn fixture(method: SegmentationMode) -> Snapshot {
        let mut snapshot = super::super::tests::example_snapshot();
        snapshot.method = method;
        snapshot.present = [true; 2];
        snapshot.recovery = None;
        let time = 1_000_000_000;
        let clock = SourceClock {
            domain: 1,
            epoch: 7,
        };
        let exposures = std::array::from_fn(|i| {
            Some(ExposureKey {
                roi: RoiId(i as u32 + 1),
                clock,
                sequence: if i == 0 { 118 } else { 519 },
                timestamp_ns: time,
            })
        });
        for (i, frame) in snapshot.eyes.iter_mut().enumerate() {
            let frame = frame.as_mut().unwrap();
            frame.eye_id = i as u32 + 1;
            frame.timestamp_ns = time + 12_000_000;
            frame.segmentation_mode = method;
            frame.joint_gaze_active = true;
            frame.gaze_policy_error = None;
            frame.gaze_authority_sam_prompt_generation = Some(0);
            let p = Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap());
            p.eye_index = i;
            p.source_sequence = exposures[i].unwrap().sequence;
            p.source_timestamp_ns = time;
            p.source_sensor_origin = (frame.sensor_x, frame.sensor_y);
            let fit = p.outer_fit.as_mut().unwrap();
            let points = fit.ellipse.dense_points(96);
            fit.retained_points = Arc::new(points[10..78].to_vec());
            fit.flat_tire_points = Arc::new(points[78..].to_vec());
            fit.conic_segments = Arc::new(vec![(0..68).collect()]);
            let mask = (0..48 * 32)
                .map(|i| {
                    let x = (i % 48) as f64 * 8.0 - 192.0;
                    let y = (i / 48) as f64 * 8.0 - 128.0;
                    u8::from(x * x / 6400.0 + y * y / 3600.0 < 1.0)
                })
                .collect::<Vec<_>>();
            p.semantic = Some(sam31_outer::SemanticProposalMasks {
                prompt_index: 0,
                width: 48,
                height: 32,
                selected_query: Some(0),
                masks: vec![sam31_outer::ProposalMask {
                    query: 0,
                    score: 0.9,
                    pixels: Arc::new(mask),
                    boundary_pixels: Arc::new(vec![]),
                }],
            });
        }
        let ellipse = snapshot.eyes[0]
            .as_ref()
            .unwrap()
            .sam31_proposal_masks
            .as_ref()
            .unwrap()
            .outer_fit
            .as_ref()
            .unwrap()
            .ellipse;
        let camera = PinholeCamera {
            focal_px: [4000.0; 2],
            principal_px: [4000.0, 3000.0],
        };
        let scene = approximate_scene(
            camera,
            [
                Some(EyePoseInput {
                    limbus_center_sensor_px: [3600.0, 3000.0],
                    pixels_per_10mm: None,
                }),
                Some(EyePoseInput {
                    limbus_center_sensor_px: [4400.0, 3000.0],
                    pixels_per_10mm: None,
                }),
            ],
        )
        .unwrap();
        let mut arcs = vec![];
        for (eye, used, weight, rms) in [
            (0, true, 3.0, 1.5),
            (1, true, 1.0, 3.0),
            (1, false, 2.0, 19.0),
        ] {
            arcs.push(ArcSupport {
                exposure: exposures[eye].unwrap(),
                evidence_group: arcs.len() as u32,
                arc_index: arcs.len(),
                points_roi_px: vec![],
                kind: BoundaryKind::OuterLimbus,
                rms_px: rms,
                sigma_px: 2.0,
                support_length_px: weight * 32.0,
                evidence_weight: weight,
                boundary_normal_samples: 0,
                boundary_normal_rms_radians: None,
                mask_level: None,
                used,
            });
        }
        let solution = JointConicSolution {
            factor_costs: None,
            local_uncertainty: None,
            posterior: None,
            mask_level_families: vec![],
            arc_alternative_marginals: vec![],
            target_camera_mm: [70.0, -120.0, 250.0],
            target_reference_camera_mm: [0.0, 0.0, -350.0],
            target_viewpoint_axial_distance_mm: 600.0,
            target_viewpoint_slopes: [0.1, -0.2],
            target_viewpoint_slope_limit: 1.5,
            target_viewpoint_bounds_active: [false; 2],
            eye_centers_camera_mm: [Some([-32.0, 0.0, -350.0]), Some([32.0, 0.0, -350.0])],
            eye_normals: [Some([0.1, -0.2, 0.97]); 2],
            eye_gaze_directions: [Some([0.1, -0.2, 0.97]); 2],
            surface_axis_alignment_radians: [Some([0.0; 2]); 2],
            effective_pivots_camera_mm: [None; 2],
            ellipses_roi_px: [[Some(ellipse), None, None]; 2],
            arcs,
            contributing_eyes: [true; 2],
            modeled_eyes: [true; 2],
            unlocalized_eye_cost: [0.0; 2],
            robust_cost: 4.25,
            alternative_cost_margin: Some(3.1),
            alternative_target_camera_mm: None,
            hypotheses_evaluated: 16,
            hypotheses_by_association: [16, 0, 0],
            refinement_steps: 12,
        };
        let publication = Arc::new(PublishedJoint {
            orientation_ready: None,
            continuous_gaze_sign: None,
            exposures,
            sensor_origins_px: std::array::from_fn(|i| {
                let p = snapshot.eyes[i]
                    .as_ref()
                    .unwrap()
                    .sam31_proposal_masks
                    .as_ref()
                    .unwrap();
                Some([p.source_sensor_origin.0, p.source_sensor_origin.1])
            }),
            dimensions_px: [Some([384, 256]); 2],
            scene,
            solution,
            diagnostic_hypotheses: vec![],
            source_generation: 5,
        });
        for frame in snapshot.eyes.iter_mut().flatten() {
            frame.joint_conic = Some(Arc::clone(&publication));
        }
        snapshot
    }

    #[test]
    fn stereo_ui_reports_real_support_without_fabricating_probability() {
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let snapshot = fixture(method);
            let report = inspect(&snapshot);
            assert_eq!(report.state, "BOTH EYES CONTRIBUTE");
            assert_eq!(report.ipd_mm(), Some(64.0));
            assert_eq!(
                report.skew_ns,
                Some(0),
                "ROI-local sequences may differ on the same exposure"
            );
            assert_eq!(report.eyes[0].weight, 3.0);
            assert_eq!(
                report.eyes[1].weight, 1.0,
                "rejected support cannot raise an eye's contribution"
            );
            assert_eq!(report.eyes[1].rejected, 1);
            assert_eq!(report.eyes[0].rms_px, Some(1.5));
            assert_eq!(report.eyes[1].rejected_rms_px, Some(19.0));
            assert!(report.json()["probability"].is_null());
            assert!(report.json()["target_covariance"].is_null());
            assert!(report.json()["target_camera_mm"].is_array());
        }
    }

    #[test]
    fn posterior_spread_is_displayed_only_with_its_exact_fresh_publication() {
        use crate::conic_solver::joint::posterior::{DirectionNumerics, ModelPosterior, PosteriorMode};
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let mut snapshot = fixture(method);
            let mut publication = (**snapshot.eyes[0]
                .as_ref()
                .unwrap()
                .joint_conic
                .as_ref()
                .unwrap())
            .clone();
            publication.solution.posterior = Some(ModelPosterior {
                direction_prior: Default::default(),
                camera_mount: crate::conic_solver::camera_mount::CameraMount::Flexible,
                status: "estimated-conditional",
                samples: 512,
                feasible_samples: 230,
                pilot_samples: 0,
                pilot_feasible_samples: 0,
                adapted_proposals: 0,
                global_proposals: 0,
                effective_samples: 95.0,
                maximum_sample_mass: Some(0.02),
                modeled_eyes: [true, true],
                modes: vec![PosteriorMode {
                    target_camera_mm: publication.solution.target_camera_mm,
                    model_mass: Some(1.0),
                }],
                target_mean_camera_mm: Some(publication.solution.target_camera_mm),
                target_covariance_mm2: Some([
                    [100.0, 0.0, 0.0],
                    [0.0, 80.0, 0.0],
                    [0.0, 0.0, 300.0],
                ]),
                gaze_radius_90_degrees: [Some(3.5), Some(4.2)],
                direction_numerics: [Some(DirectionNumerics {mass:0.97,standard_error:0.01}),
                    Some(DirectionNumerics {mass:0.93,standard_error:0.025})],
                require_numerical_margin: false,
                marginalized_inner_radii: 0,
                marginal_draw_policy: "unchanged-full",
                replicas: 1,
                replicate_direction_numerics: [None, None],
                mask_state_proposals: vec![],
                population_integration: None,
                supported_mode_selection: None,
            });
            let publication = Arc::new(publication);
            for frame in snapshot.eyes.iter_mut().flatten() {
                frame.joint_conic = Some(Arc::clone(&publication));
            }
            let report = inspect(&snapshot);
            assert_eq!(report.json()["posterior"]["gaze_radius_90_degrees"][0], 3.5);
            assert_eq!(report.json()["posterior"]["direction_numerics"][1]["mass_standard_error"],0.025);
            assert!(report.rows().iter().any(|r|r.contains("LEFT: 93.0% +/- 5.0 POINTS")));
            assert!(report.rows().iter().any(|row| row.contains("3.50 DEG")));
            assert!(report
                .rows()
                .iter()
                .any(|row| row.contains("not measured gaze accuracy")));
            if let Some(dir) = std::env::var_os("BUTTERCUP_UI_TEST_EXPORT") {
                let mut ui = Workspace {
                    scope: Scope::Linked,
                    linked: LinkedView::StereoSolver,
                    ..Default::default()
                };
                let mut pixels = vec![0; 1200 * 850];
                render_snapshot(&mut ui, &mut pixels, 1200, 850, &snapshot);
                export_eye_ppm(
                    &PathBuf::from(dir).join(format!("stereo-posterior-{}.ppm", method.label())),
                    &pixels,
                    1200,
                    850,
                )
                .unwrap();
            }
            snapshot.source_arrival_age[0] = Some(Duration::from_secs(2));
            let expired = inspect(&snapshot);
            assert!(expired.json()["posterior"].is_null());
            assert!(!expired.rows().iter().any(|row| row.contains("3.50 DEG")));
        }
    }

    #[test]
    fn stereo_ui_rejects_wrong_clock_generation_crop_source_and_stale_geometry() {
        for case in 0..14 {
            let mut snapshot = fixture(SegmentationMode::Sam31);
            let left = snapshot.eyes[1].as_mut().unwrap();
            match case {
                0 => {
                    Arc::make_mut(left.sam31_proposal_masks.as_mut().unwrap()).source_sequence += 1
                }
                1 => {
                    Arc::make_mut(left.sam31_proposal_masks.as_mut().unwrap())
                        .source_timestamp_ns += 1
                }
                2 => {
                    Arc::make_mut(left.sam31_proposal_masks.as_mut().unwrap())
                        .source_sensor_origin
                        .0 += 8
                }
                3 => Arc::make_mut(left.joint_conic.as_mut().unwrap()).source_generation += 1,
                4 => {
                    Arc::make_mut(left.joint_conic.as_mut().unwrap()).exposures[1]
                        .as_mut()
                        .unwrap()
                        .clock
                        .epoch += 1
                }
                5 => left.timestamp_ns += SAM31_RESULT_MAX_AGE_NS,
                6 => left.timestamp_ns = 999_999_999,
                7 => left.gaze_policy_error = Some("waiting for settings"),
                8 => left.presentation_pivot_held = true,
                9 => left.segmentation_mode = SegmentationMode::EyeStudent,
                10 => left.gaze_authority_sam_prompt_generation = Some(99),
                11 => snapshot.present[1] = false,
                12 => snapshot.source_arrival_age[1] = None,
                13 => snapshot.source_arrival_age[1] = Some(Duration::from_secs(2)),
                _ => unreachable!(),
            }
            let report = inspect(&snapshot);
            assert!(
                report.publication.is_none(),
                "case {case} cannot reuse the partner's older solve"
            );
            assert!(report.json()["target_camera_mm"].is_null());
        }
    }

    #[test]
    fn stereo_ui_distinguishes_single_eye_no_fit_disabled_and_paused() {
        let mut snapshot = fixture(SegmentationMode::EyeStudent);
        let mut publication = snapshot.eyes[0]
            .as_ref()
            .unwrap()
            .joint_conic
            .as_ref()
            .unwrap()
            .as_ref()
            .clone();
        publication.solution.contributing_eyes[1] = false;
        publication.solution.modeled_eyes[1] = false;
        publication.solution.ellipses_roi_px[1] = [None; 3];
        publication
            .solution
            .arcs
            .iter_mut()
            .filter(|a| a.exposure.roi == RoiId(2))
            .for_each(|a| a.used = false);
        let publication = Arc::new(publication);
        for frame in snapshot.eyes.iter_mut().flatten() {
            frame.joint_conic = Some(publication.clone());
        }
        assert_eq!(inspect(&snapshot).state, "ONE EYE CONTRIBUTES");
        assert_eq!(inspect(&snapshot).json()["eyes"][1]["modeled"], false);
        let mut pending = snapshot.clone();
        let right = pending.eyes[0].as_mut().unwrap();
        Arc::make_mut(right.joint_conic.as_mut().unwrap()).exposures[1] = None;
        Arc::make_mut(right.sam31_proposal_masks.as_mut().unwrap()).source_group_roi_count = 2;
        assert_eq!(inspect(&pending).state, "PARTNER SOLVE PENDING");
        snapshot.second = false;
        assert_eq!(inspect(&snapshot).state, "STEREO OFF");
        assert!(inspect(&snapshot).publication.is_none());
        snapshot.second = true;
        snapshot.object_running = true;
        assert_eq!(inspect(&snapshot).state, "EYE ANALYSIS PAUSED");
        assert!(inspect(&snapshot).publication.is_none());
        snapshot.object_running = false;
        for frame in snapshot.eyes.iter_mut().flatten() {
            frame.joint_conic = None;
            Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).outer_fit = None;
        }
        let report = inspect(&snapshot);
        assert_eq!(report.eyes[0].status, "NO LIMBUS CONIC");
        assert_eq!(report.eyes[0].used, 0);
        assert!(
            report.json()["eyes"][0]["used_groups"].is_null(),
            "not measured is not a measured zero"
        );
        assert!(report.publication.is_none());
    }

    #[test]
    fn stereo_ui_raw_and_conics_stay_on_their_own_source_after_roi_moves() {
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let snapshot = fixture(method);
            let mut frame = snapshot.eyes[0].clone().unwrap();
            let p = source(&frame, method, 0).unwrap();
            let original = source_image(&frame, p, ViewMode::QuadColor, Layer::Conics, None, 0);
            let raw = source_image(&frame, p, ViewMode::QuadColor, Layer::Raw, None, 0);
            assert_ne!(original, raw);
            let masks = source_image(&frame, p, ViewMode::QuadColor, Layer::Masks, None, 0);
            assert_ne!(
                masks, raw,
                "the existing semantic mask actually renders for either detector"
            );
            assert_ne!(
                masks, original,
                "mask and conic controls expose separate layers"
            );
            assert_eq!(
                raw,
                color_preview(
                    &p.source_raw,
                    p.source_width,
                    p.source_height,
                    p.source_sensor_origin.0,
                    p.source_sensor_origin.1,
                    100,
                    None
                )
            );
            frame.sensor_x += 320;
            frame.sensor_y += 96;
            frame.width = 420;
            frame.height = 280;
            frame.sequence += 10;
            frame.quad_color = Arc::new(vec![0x00ff_0000; 420 * 280]);
            let p = source(&frame, method, 0).unwrap();
            assert_eq!(
                source_image(&frame, p, ViewMode::QuadColor, Layer::Conics, None, 0),
                original
            );
            assert_eq!(
                source_image(&frame, p, ViewMode::QuadColor, Layer::Masks, None, 0),
                masks
            );
            Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).outer_fit = None;
            let p = source(&frame, method, 0).unwrap();
            assert_eq!(
                source_image(&frame, p, ViewMode::QuadColor, Layer::Conics, None, 0),
                raw,
                "missing tracing still displays the original RAW content"
            );
        }
    }

    #[test]
    fn fitted_sections_show_the_solved_conic_without_moving_noisy_observations() {
        let ellipse=geometry::Ellipse {center:(180.0,130.0),major_radius:36.0,minor_radius:18.0,angle:0.4};
        let (s,c)=ellipse.angle.sin_cos();
        let p=|phase:f64,noise:f64| {
            let x=(ellipse.major_radius+noise)*phase.cos();
            let y=(ellipse.minor_radius+noise)*phase.sin();
            (ellipse.center.0+c*x-s*y,ellipse.center.1+s*x+c*y)
        };
        // Noisy, non-convex sample joins across the +/-PI phase boundary.
        let samples=vec![p(3.02,2.0),p(3.10,-3.0),p(3.18,4.0),p(3.26,-2.0)];
        let saved=samples.clone();
        let sections=fitted_arc_sections(ellipse,&samples);
        assert!(!sections.is_empty());
        assert_eq!(samples,saved,"the display must not replace measured evidence");
        let length=sections.iter().map(|&(a,b)|(a.0-b.0).hypot(a.1-b.1)).sum::<f64>();
        assert!(length<20.0,"phase wrap must not draw the long way around");
        for &(a,b) in &sections {
            for (x,y) in [a,b] {
                let (x,y)=(x-ellipse.center.0,y-ellipse.center.1);
                let r=((c*x+s*y)/ellipse.major_radius).powi(2)+((-s*x+c*y)/ellipse.minor_radius).powi(2);
                assert!((r-1.0).abs()<1.0e-10,"curve endpoints belong to the solved ellipse");
            }
        }
        assert!(fitted_arc_sections(ellipse,&[p(0.0,0.0),p(2.0,0.0)]).is_empty());
        assert!(fitted_arc_sections(ellipse,&[p(0.0,0.0),(f64::NAN,0.0),p(0.2,0.0)]).is_empty());
        assert!(fitted_arc_sections(ellipse,&[p(0.0,0.0)]).is_empty());
    }

    #[test]
    fn segment_labels_reserve_slots_throttle_values_and_do_not_renew_held_sources() {
        let now=Instant::now();let mut labels=SegmentLabels::default();
        let source=crate::roi_evidence::ExposureKey {roi:RoiId(1),
            clock:crate::roi_evidence::SourceClock {domain:1,epoch:1},sequence:1,timestamp_ns:1};
        assert_eq!(labels.slot((0,0),source,"10.0%".into(),now),Some(0));
        assert_eq!(labels.slot((2,103),source,"20.0%".into(),now),Some(1));
        let next=crate::roi_evidence::ExposureKey {sequence:2,timestamp_ns:100_000_001,..source};
        // First arc missing, second remains in the same numbered/color slot.
        assert_eq!(labels.slot((2,103),next,"90.0%".into(),now+Duration::from_millis(100)),Some(1));
        assert_eq!(labels.slots[1].value,"20.0%");
        assert_eq!(labels.slot((0,7),next,"5.0%".into(),now+Duration::from_millis(100)),Some(2));
        // Repainting an old source must not keep it fresh or update its value.
        labels.slot((2,103),next,"99.0%".into(),now+Duration::from_secs(1));
        assert_eq!(labels.slots[1].seen,now+Duration::from_millis(100));
        assert_eq!(labels.slots[1].value,"20.0%");
        assert!(now.saturating_duration_since(labels.slots[0].seen)<=SEGMENT_LABEL_HOLD);
        assert!((now+Duration::from_secs(1)).saturating_duration_since(labels.slots[0].seen)>SEGMENT_LABEL_HOLD);
        let newer=crate::roi_evidence::ExposureKey {sequence:3,timestamp_ns:300_000_001,..source};
        assert_eq!(labels.slot((0,0),newer,"30.0%".into(),now+Duration::from_millis(300)),Some(0));
        assert_eq!(labels.slots[0].value,"30.0%");
        labels.slot((0,0),source,"1.0%".into(),now+Duration::from_secs(2));
        assert_eq!(labels.slots[0].value,"30.0%");
        for group in 0..100 {labels.slot((1,group),newer,"1.0%".into(),now);}
        assert_eq!(labels.slots.len(),20);
    }

    #[test]
    fn stereo_segment_pins_use_source_points_and_refuse_mismatched_pixels() {
        for method in [SegmentationMode::Sam31,SegmentationMode::EyeStudent] {
            let mut snapshot=fixture(method);
            let frame=snapshot.eyes[1].as_mut().unwrap();
            let publication=Arc::make_mut(frame.joint_conic.as_mut().unwrap());
            for (i,a) in publication.solution.arcs.iter_mut().filter(|a|a.exposure.roi==RoiId(2)).enumerate() {
                a.points_roi_px=vec![(150.0,110.0+i as f64*20.0),(160.0,115.0+i as f64*20.0),(170.0,110.0+i as f64*20.0)];
            }
            let p=source(frame,method,1).unwrap();
            let mut pixels=vec![0;420*280];
            assert_eq!(draw_stereo_segments(&mut pixels,420,280,frame,p),2);
            assert!(pixels.contains(&0x005b_def4));assert!(pixels.contains(&0x00ff_8fb8));
            let before=joint_gaze_live::json(frame);
            assert_eq!(draw_stereo_segments(&mut pixels,420,280,frame,p),2);
            assert_eq!(joint_gaze_live::json(frame),before);
            Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).source_sequence+=1;
            let p=source(frame,method,1).unwrap();pixels.fill(0);
            assert_eq!(draw_stereo_segments(&mut pixels,420,280,frame,p),0);
            assert_eq!(pixels[110*420+150],0,"a retained label must not retain stale segment geometry");
        }
    }

    #[test]
    fn stereo_scene_follows_solver_selection_without_changing_roi_views() {
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let mut snapshot=fixture(method);
            let before=joint_gaze_live::json(snapshot.eyes[0].as_ref().unwrap());
            for scope in [Scope::Roi,Scope::Linked,Scope::Global] {
                for enabled in [false,true] {
                    snapshot.stereo=enabled;
                    let mut ui=Workspace {scope,linked:LinkedView::Compare,..Default::default()};
                    let overlays=[ui.roi_view(0).overlay,ui.roi_view(1).overlay];
                    let mut pixels=vec![0;1200*850];
                    render_snapshot(&mut ui,&mut pixels,1200,850,&snapshot);
                    let canvas=Layout::new(1200,850).canvas;
                    let scene=ui.hits.iter().any(|(r,a)| matches!(a,Action::StereoSolver)
                        && r.y>=canvas.y && r.y<canvas.y+canvas.h);
                    assert_eq!(scene,enabled && scope!=Scope::Global);
                    assert_eq!(ui.scope,scope);
                    assert_eq!([ui.roi_view(0).overlay,ui.roi_view(1).overlay],overlays);
                    if enabled && scope==Scope::Roi && method==SegmentationMode::EyeStudent {
                        if let Ok(path)=std::env::var("BUTTERCUP_STEREO_SCENE_PPM") {
                            let mut ppm=b"P6\n1200 850\n255\n".to_vec();
                            for pixel in &pixels {ppm.extend_from_slice(&[(pixel>>16) as u8,(pixel>>8) as u8,*pixel as u8]);}
                            std::fs::write(path,ppm).unwrap();
                        }
                    }
                }
            }
            assert_eq!(joint_gaze_live::json(snapshot.eyes[0].as_ref().unwrap()),before);
        }
    }

    #[test]
    fn stereo_ui_layout_layers_and_pending_states_render_without_changing_analysis() {
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let snapshot = fixture(method);
            let original = joint_gaze_live::json(snapshot.eyes[0].as_ref().unwrap());
            assert_eq!(LinkedView::Compare.next(method), LinkedView::StereoSolver);
            for (w, h) in [
                (320, 240),
                (640, 480),
                (1200, 850),
                (800, 1200),
                (904, 2048),
            ] {
                for layer in [Layer::Conics, Layer::Segments, Layer::Masks, Layer::Raw] {
                    let mut ui = Workspace {
                        scope: Scope::Linked,
                        linked: LinkedView::StereoSolver,
                        stereo_layer: layer,
                        ..Default::default()
                    };
                    let mut pixels = vec![0; w * h];
                    render_snapshot(&mut ui, &mut pixels, w, h, &snapshot);
                    for (rect, _) in &ui.hits {
                        assert!(
                            rect.x + rect.w <= w && rect.y + rect.h <= h,
                            "{w}x{h}: {rect:?}"
                        );
                    }
                    assert_eq!(ui.preview_defaults, RoiView::default());
                    assert!(ui
                        .hits
                        .iter()
                        .any(|(_, a)| matches!(a, Action::ToggleStereo)));
                    if let Some(dir) = std::env::var_os("BUTTERCUP_UI_TEST_EXPORT") {
                        if layer == Layer::Conics {
                            export_eye_ppm(
                                &PathBuf::from(dir)
                                    .join(format!("stereo-{}-{w}x{h}.ppm", method.label())),
                                &pixels,
                                w,
                                h,
                            )
                            .unwrap();
                        }
                    }
                }
            }
            assert_eq!(
                joint_gaze_live::json(snapshot.eyes[0].as_ref().unwrap()),
                original
            );
            for state in 0..3 {
                let mut missing = snapshot.clone();
                match state {
                    0 => missing.second = false,
                    1 => missing.eyes = [None, None],
                    _ => missing.eyes[1].as_mut().unwrap().presentation_pivot_held = true,
                }
                let mut ui = Workspace {
                    scope: Scope::Linked,
                    linked: LinkedView::StereoSolver,
                    ..Default::default()
                };
                let mut pixels = vec![0; 1200 * 850];
                render_snapshot(&mut ui, &mut pixels, 1200, 850, &missing);
                if let Some(dir) = std::env::var_os("BUTTERCUP_UI_TEST_EXPORT") {
                    export_eye_ppm(
                        &PathBuf::from(dir).join(format!("stereo-pending-{state}.ppm")),
                        &pixels,
                        1200,
                        850,
                    )
                    .unwrap();
                }
            }
        }
        assert!(
            !LinkedView::available(SegmentationMode::Native).contains(&LinkedView::StereoSolver)
        );
    }
}
