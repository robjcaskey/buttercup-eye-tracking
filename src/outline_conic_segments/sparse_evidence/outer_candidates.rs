//! Offline fallback for a selected mask that failed its existing RAW gate.
//! Expanded guides only locate RAW searches. Every emitted point is a measured
//! positive edge maximum from the shared extractor, never a completed ellipse.
use super::*;
use std::f64::consts::TAU;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct Report {
    pub(crate) guides: usize,
    pub(crate) raw_runs: usize,
    pub(crate) emitted_arcs: usize,
    pub(crate) emitted_points: usize,
    pub(crate) sector_alternatives: [usize; 8],
    pub(crate) offsets_px: Vec<f64>,
}

fn length(points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|p| (p[1].0 - p[0].0).hypot(p[1].1 - p[0].1))
        .sum()
}

/// Four bounded neighboring guides, eight FIXED image-angle groups, at most
/// four alternatives per group. Queries/guide copies do not create new votes.
/// Does not edit existing pupil arcs, ROI reliability or source identity.
pub(crate) fn append(
    packet: &mut OwnedRoiEvidence,
    raw: &[u16],
    guide: Ellipse,
    group_base: u32,
) -> Report {
    let short = packet.dimensions_px[0].min(packet.dimensions_px[1]) as f64;
    let guides = [-0.08, 0.0, 0.08, 0.16].map(|fraction| {
        let offset = fraction * short;
        (
            Ellipse {
                major_radius: guide.major_radius + offset,
                minor_radius: guide.minor_radius + offset,
                ..guide
            },
            offset,
        )
    });
    append_guides(packet, raw, guide, &guides, group_base)
}

fn append_guides(
    packet: &mut OwnedRoiEvidence,
    raw: &[u16],
    center_guide: Ellipse,
    guides: &[(Ellipse, f64)],
    group_base: u32,
) -> Report {
    let mut report = Report::default();
    if group_base.checked_add(7).is_none()
        || !center_guide.center.0.is_finite()
        || !center_guide.center.1.is_finite()
    {
        return report;
    }
    let center = center_guide.center;
    let sector = |p: (f64, f64)| {
        (((p.1 - center.1).atan2(p.0 - center.0).rem_euclid(TAU) / TAU * 8.0) as usize).min(7)
    };
    let mut alternatives: [Vec<OwnedBoundaryArc>; 8] = std::array::from_fn(|_| Vec::new());
    for &(guide, offset) in guides.iter().take(4) {
        let mut temporary = OwnedRoiEvidence {
            arcs: Vec::new(),
            conics: Vec::new(),
            ..packet.clone()
        };
        append_raw_ring_arcs_with_cohesion(
            &mut temporary,
            raw,
            guide,
            BoundaryKind::OuterLimbus,
            0,
            RawArcConfig {
                radial_search_px: 12.0,
                ..Default::default()
            },
            true,
        );
        report.guides += 1;
        report.offsets_px.push(offset);
        report.raw_runs += temporary.arcs.len();
        for arc in temporary.arcs {
            // Split at a fixed image-angle boundary. Ellipse phase depends on
            // the guide's axes and is not a common correlation partition.
            let mut start = 0;
            while start < arc.points_roi_px.len() {
                let group = sector(arc.points_roi_px[start]);
                let mut end = start + 1;
                while end < arc.points_roi_px.len() && sector(arc.points_roi_px[end]) == group {
                    end += 1;
                }
                if end - start >= 3 {
                    let points = arc.points_roi_px[start..end].to_vec();
                    if length(&points) >= 6.0 {
                        alternatives[group].push(OwnedBoundaryArc {
                            evidence_group: group_base + group as u32,
                            points_roi_px: points,
                            ..arc.clone()
                        });
                    }
                }
                start = end;
            }
        }
    }
    let start = packet.arcs.len();
    for (group, arcs) in alternatives.iter_mut().enumerate() {
        // Rank measured support, never agreement with the candidate joint fit.
        arcs.sort_by(|a, b| length(&b.points_roi_px).total_cmp(&length(&a.points_roi_px)));
        let mut kept: Vec<OwnedBoundaryArc> = Vec::new();
        for arc in arcs.drain(..) {
            // Exact repeated guides cannot crowd out another edge hypothesis.
            if kept.iter().any(|a| a.points_roi_px == arc.points_roi_px) {
                continue;
            }
            kept.push(arc);
            if kept.len() == 4 {
                break;
            }
        }
        report.sector_alternatives[group] = kept.len();
        report.emitted_points += kept.iter().map(|a| a.points_roi_px.len()).sum::<usize>();
        packet.arcs.extend(kept);
    }
    report.emitted_arcs = packet.arcs.len() - start;
    if report.emitted_arcs > 0 {
        packet.conics.push(OwnedConicHint {
            kind: BoundaryKind::OuterLimbus,
            ellipse_roi_px: center_guide,
            supporting_arc_indices: (start..packet.arcs.len()).collect(),
        });
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi_evidence::{RoiId, SourceClock};

    fn packet() -> OwnedRoiEvidence {
        OwnedRoiEvidence {
            exposure: ExposureKey {
                roi: RoiId(1),
                clock: SourceClock {
                    domain: 1,
                    epoch: 1,
                },
                sequence: 7,
                timestamp_ns: 123,
            },
            sensor_origin_px: [2500, 900],
            dimensions_px: [256, 192],
            arcs: Vec::new(),
            conics: Vec::new(),
            detail_reliability: Some(0.8),
        }
    }
    fn guide() -> Ellipse {
        Ellipse {
            center: (128.0, 96.0),
            major_radius: 40.0,
            minor_radius: 40.0,
            angle: 0.0,
        }
    }
    fn image() -> Vec<u16> {
        (0..192)
            .flat_map(|y| {
                (0..256).map(move |x| {
                    let radius = (x as f64 - 128.0).hypot(y as f64 - 96.0);
                    (100.0 + 60.0 / (1.0 + (-(radius - 72.0) / 1.3).exp())) as u16
                })
            })
            .collect()
    }
    #[test]
    fn rejected_mask_raw_fallback_finds_measured_rim_beyond_original_window() {
        let raw = image();
        let mut baseline = packet();
        append_raw_ring_arcs_with_cohesion(
            &mut baseline,
            &raw,
            guide(),
            BoundaryKind::OuterLimbus,
            0,
            RawArcConfig::default(),
            true,
        );
        assert!(
            baseline.arcs.is_empty(),
            "true RAW rim lies outside the original bounded guide search"
        );
        let mut p = packet();
        let source = p.exposure;
        let report = append(&mut p, &raw, guide(), 0);
        assert!(
            report.emitted_arcs >= 6 && report.emitted_arcs <= 32,
            "{report:?}"
        );
        for arc in &p.arcs {
            for &(x, y) in &arc.points_roi_px {
                assert!(
                    ((x - 128.0).hypot(y - 96.0) - 72.0).abs() < 3.0,
                    "observed edge, not expanded guide: {x} {y}"
                );
            }
        }
        assert_eq!(p.exposure, source);
        assert_eq!(p.sensor_origin_px, [2500, 900]);
        assert_eq!(p.detail_reliability, Some(0.8));
    }
    #[test]
    fn rejected_mask_raw_fallback_keeps_flat_and_saturated_sources_unavailable() {
        for value in [0, 140, 1023] {
            let mut p = packet();
            let report = append(&mut p, &vec![value; 256 * 192], guide(), 0);
            assert_eq!(report.emitted_arcs, 0);
            assert!(p.arcs.is_empty() && p.conics.is_empty());
        }
    }
    #[test]
    fn repeated_raw_guides_share_evidence_groups_and_do_not_duplicate_arcs() {
        let raw = image();
        let g = Ellipse {
            major_radius: 72.0,
            minor_radius: 72.0,
            ..guide()
        };
        let mut a = packet();
        let mut b = packet();
        append_guides(&mut a, &raw, g, &[(g, 0.0)], 200);
        append_guides(&mut b, &raw, g, &[(g, 0.0); 4], 200);
        assert!(!a.arcs.is_empty());
        assert_eq!(a.arcs.len(), b.arcs.len());
        for (a, b) in a.arcs.iter().zip(&b.arcs) {
            assert_eq!(a.evidence_group, b.evidence_group);
            assert_eq!(a.points_roi_px, b.points_roi_px);
            assert!((200..208).contains(&a.evidence_group));
        }
    }
}
