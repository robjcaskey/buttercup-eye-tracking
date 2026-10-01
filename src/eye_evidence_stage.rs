//! Shared source-local evidence finalization for mask detector adapters.
//!
//! Inference and publication transport stay with the caller. Geometry conversion,
//! refinement and admission run once here; rejected geometry retains its exact
//! source packet for inspection and bounded joint evidence. This stage neither
//! chooses a gaze sign nor uses presentation state.
use super::*;

pub(super) struct DetectorEvidence {
    pub(super) semantic: SemanticProposalMasks,
    pub(super) outer_fit: Option<OuterMaskFitReview>,
    pub(super) outer_logits: Option<boundary_logits::Plane>,
    pub(super) outer_support: RawRingSupport,
    pub(super) pupil_fit: Option<PupilVoidFitReview>,
}

pub(super) struct FinalizedEvidence {
    pub(super) proposal: Arc<ProposalMasks>,
    pub(super) admitted: Result<OuterResult, String>,
}

pub(super) fn finalize(
    batch: &Batch,
    current_luma: Option<&FloatImage>,
    video_outer: Option<DetectorEvidence>,
    refine: impl FnOnce(
        &RawFrame,
        &FloatImage,
        &mut OuterMaskFitReview,
        &mut RawRingSupport,
        Option<PupilVoidFitReview>,
    ) -> Option<limbus_refinement::Attempt>,
) -> Result<FinalizedEvidence, String> {
    let source = batch
        .frames
        .last()
        .ok_or_else(|| "SAM31 video tracker received no source frame".to_string())?;
    let DetectorEvidence {
        semantic,
        outer_fit,
        outer_logits,
        outer_support,
        pupil_fit,
    } = video_outer
        .ok_or_else(|| "SAM31 video tracker produced no current-frame mask".to_string())?;
    if semantic.prompt_index != OUTER_IRIS_PROMPT {
        return Err("SAM31 video geometry requires the mandatory outer-iris prompt".into());
    }
    let mut outer_fit = outer_fit.map(|fit| model_review_in_source(fit, source.width));
    let mut outer_support = outer_support;
    let limbus_refinement = outer_fit
        .as_mut()
        .zip(current_luma)
        .and_then(|(review, image)| refine(source, image, review, &mut outer_support, pupil_fit));
    let outer_boundary_logits = outer_logits
        .as_ref()
        .zip(outer_fit.as_ref())
        .zip(semantic.selected_query)
        .map(|((plane, fit), query)| {
            boundary_logits::measure(
                plane,
                boundary_logits::Source {
                    eye_index: source.eye_index,
                    sequence: source.sequence,
                    timestamp_ns: source.timestamp_ns.to_string(),
                    tracking_epoch: batch.tracking_epoch,
                    prompt_generation: batch.prompt_generation,
                    sensor_origin: (source.sensor_x, source.sensor_y),
                    width: source.width,
                    height: source.height,
                },
                semantic.prompt_index,
                query,
                limbus_refinement
                    .as_ref()
                    .is_some_and(|attempt| attempt.applied),
                &fit.retained_points,
                &fit.conic_segments,
            )
            .map(Arc::new)
        })
        .transpose()?;
    let quality = semantic
        .selected_query
        .and_then(|selected| semantic.masks.iter().find(|mask| mask.query == selected))
        .map(|mask| f64::from(mask.score))
        .unwrap_or_default();
    let outer_ellipse = outer_fit.as_ref().map(|fit| fit.ellipse);
    // Always retain the same-exposure RAW pupil void alongside an outer
    // proposal.  Virtual contact needs this private cue to choose between
    // the two antipodal surface normals even when the operator has not
    // selected SAM as the public rough-center provider.  Target selection
    // below still controls whether the pupil is published as a normal Y
    // product; this review-only fit cannot silently change that mode.
    let proposal_pupil_fit = pupil_fit;
    let proposal_masks = Arc::new(ProposalMasks {
        tracking_epoch: batch.tracking_epoch,
        prompt_generation: batch.prompt_generation,
        eye_index: batch.eye_index,
        source_sequence: source.sequence,
        source_timestamp_ns: source.timestamp_ns,
        source_group_roi_count: if batch.source_group_claimed.is_some() {
            2
        } else {
            1
        },
        source_sensor_origin: (source.sensor_x, source.sensor_y),
        source_width: source.width,
        source_height: source.height,
        source_raw: Arc::clone(&source.pixels),
        semantic: Some(semantic),
        outer_fit,
        outer_boundary_logits,
        limbus_refinement,
        inner_pupil_fit: proposal_pupil_fit,
        pupil_occlusion: None,
        adapters: Vec::new(),
    });
    let admitted = (|| {
        let outer_ellipse=outer_ellipse
            .ok_or_else(|| "SAM31 video tracker mask had no plausible limbus fit; current source proposal published without conditioning memory".to_string())?;
        if current_luma.is_none() {
            return Err("SAM31 video tracker could not construct current RAW luma".to_string());
        }
        if !live_detector_raw_gate_passes(outer_support) {
            return Err(format!(
                "SAM31 video outer RAW ring support {:.3} from {} samples and {} strong sectors is below {:.3}",
                outer_support.score,
                outer_support.points,
                outer_support.strong_sectors,
                MIN_RAW_RING_SUPPORT_SCORE,
            ));
        }
        let pupil_fit = matches!(
            batch.target,
            Target::InnerPupilVoid | Target::OuterLimbusAndInnerPupilVoid
        )
        .then(|| proposal_pupil_fit.map(|review| (review.ellipse, review.raw_support)))
        .flatten();
        let (target_ellipse, target_support, pupil_ellipse) =
            select_target_products(batch.target, outer_ellipse, outer_support, pupil_fit)?;
        let to_sensor = |mut ellipse: Ellipse| {
            ellipse.center.0 += source.sensor_x as f64;
            ellipse.center.1 += source.sensor_y as f64;
            ellipse
        };
        let source_registration_anchor_sensor = source.registration_anchor.map(|center| {
            (
                center.0 + source.sensor_x as f64,
                center.1 + source.sensor_y as f64,
            )
        });
        Ok(OuterResult {
            tracking_epoch: batch.tracking_epoch,
            prompt_generation: batch.prompt_generation,
            target: batch.target,
            eye_index: batch.eye_index,
            source_sequence: source.sequence,
            source_timestamp_ns: source.timestamp_ns,
            source_sensor_origin: (source.sensor_x, source.sensor_y),
            source_registration_anchor_sensor,
            sensor_ellipse: to_sensor(target_ellipse),
            sensor_outer_ellipse: to_sensor(outer_ellipse),
            sensor_pupil_ellipse: pupil_ellipse.map(to_sensor),
            agreeing_adapters: 0,
            quality,
            raw_ring_support_score: outer_support.score,
            raw_ring_support_points: outer_support.points,
            raw_ring_positive_fraction: outer_support.positive_fraction,
            raw_ring_strong_sectors: outer_support.strong_sectors,
            raw_target_support_score: target_support.score,
            raw_target_support_points: target_support.points,
            raw_target_positive_fraction: target_support.positive_fraction,
            raw_target_strong_sectors: target_support.strong_sectors,
            elapsed_ms: 0,
            video_tracked: true,
            proposal_masks: Arc::clone(&proposal_masks),
        })
    })();
    Ok(FinalizedEvidence {
        proposal: proposal_masks,
        admitted,
    })
}

/// Adapt the finalized source packet to the joint solver's existing evidence
/// contract. Clock identity is supplied by the source coordinator, never UI
/// completion time. Weak arcs survive without gaining monocular authority.
pub(crate) fn joint_evidence(
    proposal: &ProposalMasks,
    exposure: crate::roi_evidence::ExposureKey,
) -> Option<crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence> {
    use crate::outline_conic_segments::sparse_evidence::{
        append_occluded_raw_ring_arcs, append_raw_ring_arcs, append_retained_sam_arcs,
        OwnedRoiEvidence, RawArcConfig,
    };
    use crate::roi_evidence::BoundaryKind;
    if proposal.eye_index.checked_add(1)? != exposure.roi.0 as usize
        || proposal.source_sequence != exposure.sequence
        || proposal.source_timestamp_ns != exposure.timestamp_ns
        || proposal.source_width.checked_mul(proposal.source_height)? != proposal.source_raw.len()
    {
        return None;
    }
    let mut packet = OwnedRoiEvidence {
        exposure,
        sensor_origin_px: [
            proposal.source_sensor_origin.0,
            proposal.source_sensor_origin.1,
        ],
        dimensions_px: [proposal.source_width as u32, proposal.source_height as u32],
        arcs: Vec::new(),
        conics: Vec::new(),
        detail_reliability: None,
    };
    if let Some(review) = &proposal.outer_fit {
        append_retained_sam_arcs(&mut packet, review, 0);
        if !crate::sam31_outer::proposal_raw_outer_admitted(proposal) {
            for arc in &mut packet.arcs {
                arc.normal_band_half_width_px = 5.0;
            }
        }
    }
    if let Some((pupil, config)) =
        proposal
            .inner_pupil_fit
            .zip(proposal.outer_fit.as_ref().and_then(|outer| {
                RawArcConfig::for_pupil(
                    &proposal.source_raw,
                    proposal.source_width,
                    proposal.source_height,
                    outer.ellipse,
                )
            }))
    {
        if let Some(occlusion) = &proposal.pupil_occlusion {
            // A diagnostic occluder must bind to this exact source; a mismatch
            // is an error in the caller, not permission to use the baseline.
            assert!(occlusion.binds_to(&packet), "pupil occlusion bound to another source");
            append_occluded_raw_ring_arcs(
                &mut packet,
                &proposal.source_raw,
                pupil.ellipse,
                BoundaryKind::PupillaryBoundary,
                100,
                config,
                occlusion,
            );
        } else {
            append_raw_ring_arcs(
                &mut packet,
                &proposal.source_raw,
                pupil.ellipse,
                BoundaryKind::PupillaryBoundary,
                100,
                config,
            );
        }
    }
    Some(packet)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi_evidence::{ExposureKey, RoiId, SourceClock};

    #[test]
    fn joint_packet_preserves_weak_source_arcs_and_refuses_mismatched_identity() {
        let (source, _, review, _) = limbus_refinement::tests::fixture();
        let proposal = ProposalMasks {
            eye_index: source.eye_index,
            source_sequence: source.sequence,
            source_timestamp_ns: source.timestamp_ns,
            source_sensor_origin: (source.sensor_x, source.sensor_y),
            source_width: source.width,
            source_height: source.height,
            // Flat RAW deliberately fails the independent detector gate.
            source_raw: Arc::new(vec![100; source.width * source.height]),
            outer_fit: Some(review),
            ..Default::default()
        };
        let exposure = ExposureKey {
            roi: RoiId(source.eye_index as u32 + 1),
            clock: SourceClock {
                domain: 42,
                epoch: 7,
            },
            sequence: source.sequence,
            timestamp_ns: source.timestamp_ns,
        };
        assert!(!proposal_raw_outer_admitted(&proposal));
        let packet = joint_evidence(&proposal, exposure).unwrap();
        assert_eq!(packet.exposure, exposure);
        assert_eq!(packet.sensor_origin_px, [source.sensor_x, source.sensor_y]);
        assert!(!packet.arcs.is_empty());
        assert!(packet
            .arcs
            .iter()
            .all(|arc| arc.normal_band_half_width_px == 5.0));
        for wrong in [
            ExposureKey {
                sequence: exposure.sequence + 1,
                ..exposure
            },
            ExposureKey {
                timestamp_ns: exposure.timestamp_ns + 1,
                ..exposure
            },
            ExposureKey {
                roi: RoiId(exposure.roi.0 + 1),
                ..exposure
            },
        ] {
            assert!(joint_evidence(&proposal, wrong).is_none());
        }
        let mut malformed = proposal;
        malformed.source_width += 1;
        assert!(joint_evidence(&malformed, exposure).is_none());
    }
}
