//! Shared limbus projection reference for pupil-center and pupil-size reasoning.
//!
//! Centers are native ROI-local pixels; canonical points are dimensionless in
//! the fronto-parallel limbus plane. Projection provenance does not itself
//! authorize an anatomical observation. No temporal state or rendering lives here.

use super::pupil_radius_units::FrontoParallelCircleRadiusPx;
use crate::raw_iris_focus;

/// Provenance of the limbus projection used to express pupil size in
/// fronto-parallel radius space.  The pupil-size posterior is independent of
/// the selected rough-center method; this enum records only the current scale
/// and affine projection reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PupilProjectionSource {
    #[default]
    SelectedIris,
    CensoredLimbus,
    RawEyeAnatomy,
}

impl PupilProjectionSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SelectedIris => "selected-iris",
            Self::CensoredLimbus => "censored-limbus",
            Self::RawEyeAnatomy => "raw-eye-anatomy",
        }
    }
}

/// Weak-perspective projection of a physical circular limbus.  Its larger
/// semi-axis is the fronto-parallel radius; `minor_to_major` carries the
/// affine foreshortening needed to draw a meaningful pupil-size reticle back
/// in the untouched RAW ROI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PupilProjectionReference {
    pub(crate) center: (f64, f64),
    pub(crate) fronto_parallel_limbus_radius_px: FrontoParallelCircleRadiusPx,
    pub(crate) equivalent_limbus_radius_px: f64,
    pub(crate) minor_to_major: f64,
    pub(crate) angle: f64,
    pub(crate) source: PupilProjectionSource,
}

impl PupilProjectionReference {
    pub(crate) fn from_axes(
        center: (f64, f64),
        mut major_radius: f64,
        mut minor_radius: f64,
        mut angle: f64,
        source: PupilProjectionSource,
    ) -> Option<Self> {
        if !center.0.is_finite()
            || !center.1.is_finite()
            || !major_radius.is_finite()
            || !minor_radius.is_finite()
            || !angle.is_finite()
            || major_radius <= 4.0
            || minor_radius <= 2.0
        {
            return None;
        }
        if major_radius < minor_radius {
            std::mem::swap(&mut major_radius, &mut minor_radius);
            angle += std::f64::consts::FRAC_PI_2;
        }
        let assessment =
            crate::conic_solver::assess_projected_circular_limbus_axes(major_radius, minor_radius)?;
        if assessment.minor_to_major + 1.0e-12 < assessment.minimum_minor_to_major {
            return None;
        }
        let minor_to_major = assessment.minor_to_major;
        let fronto_parallel_limbus_radius_px =
            FrontoParallelCircleRadiusPx::from_projected_circular_limbus_axes(
                major_radius,
                minor_radius,
            )?;
        Some(Self {
            center,
            fronto_parallel_limbus_radius_px,
            equivalent_limbus_radius_px: (major_radius * minor_radius).sqrt(),
            minor_to_major,
            angle,
            source,
        })
    }

    pub(crate) fn from_outer(
        boundary: &raw_iris_focus::OuterIrisBoundary,
        source: PupilProjectionSource,
    ) -> Option<Self> {
        (!boundary.points.is_empty())
            .then(|| {
                Self::from_axes(
                    boundary.center,
                    boundary.major_radius,
                    boundary.minor_radius,
                    boundary.angle,
                    source,
                )
            })
            .flatten()
    }

    pub(crate) fn from_censored(
        observation: raw_iris_focus::RoiTruncatedLimbusObservation,
    ) -> Option<Self> {
        (observation.confidence >= 0.45)
            .then(|| {
                Self::from_axes(
                    observation.center,
                    observation.major_radius,
                    observation.minor_radius,
                    observation.angle,
                    PupilProjectionSource::CensoredLimbus,
                )
            })
            .flatten()
    }

    pub(crate) fn from_raw_focus(focus: &raw_iris_focus::BorderFocus) -> Option<Self> {
        if !focus.radius.is_finite() || focus.radius <= 4.0 || !focus.axis_angle.is_finite() {
            return None;
        }
        let mut ratio = focus.axis_ratio.abs();
        let mut angle = focus.axis_angle;
        if !ratio.is_finite() || ratio <= 0.0 {
            ratio = 1.0;
        }
        if ratio < 1.0 {
            ratio = 1.0 / ratio;
            angle += std::f64::consts::FRAC_PI_2;
        }
        let ratio_root = ratio.clamp(1.0, 2.5).sqrt();
        Self::from_axes(
            focus.center,
            focus.radius * ratio_root,
            focus.radius / ratio_root,
            angle,
            PupilProjectionSource::RawEyeAnatomy,
        )
    }
}

pub(crate) fn pupil_projection_canonical_point(
    projection: PupilProjectionReference,
    point: (f64, f64),
) -> Option<(f64, f64)> {
    let major = projection.fronto_parallel_limbus_radius_px.value();
    let minor = major * projection.minor_to_major;
    if !major.is_finite() || !minor.is_finite() || major <= 1.0 || minor <= 1.0 {
        return None;
    }
    let delta = (point.0 - projection.center.0, point.1 - projection.center.1);
    let (sine, cosine) = projection.angle.sin_cos();
    let canonical = (
        (cosine * delta.0 + sine * delta.1) / major,
        (-sine * delta.0 + cosine * delta.1) / minor,
    );
    (canonical.0.is_finite() && canonical.1.is_finite()).then_some(canonical)
}

pub(crate) fn pupil_projection_image_point(
    projection: PupilProjectionReference,
    canonical: (f64, f64),
) -> Option<(f64, f64)> {
    let major = projection.fronto_parallel_limbus_radius_px.value();
    let minor = major * projection.minor_to_major;
    if !major.is_finite()
        || !minor.is_finite()
        || major <= 1.0
        || minor <= 1.0
        || !canonical.0.is_finite()
        || !canonical.1.is_finite()
    {
        return None;
    }
    let local = (canonical.0 * major, canonical.1 * minor);
    let (sine, cosine) = projection.angle.sin_cos();
    Some((
        projection.center.0 + cosine * local.0 - sine * local.1,
        projection.center.1 + sine * local.0 + cosine * local.1,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapped_axes_preserve_the_same_rectification_geometry() {
        let reference = PupilProjectionReference::from_axes(
            (190.25, 128.75),
            100.0,
            80.0,
            0.35,
            PupilProjectionSource::SelectedIris,
        )
        .unwrap();
        let swapped = PupilProjectionReference::from_axes(
            reference.center,
            80.0,
            100.0,
            0.35 - std::f64::consts::FRAC_PI_2,
            PupilProjectionSource::SelectedIris,
        )
        .unwrap();
        assert_eq!(
            reference.fronto_parallel_limbus_radius_px,
            swapped.fronto_parallel_limbus_radius_px
        );
        assert_eq!(reference.minor_to_major, swapped.minor_to_major);
        for point in [(190.25, 128.75), (207.375, 143.625), (135.5, 97.25)] {
            let expected = pupil_projection_canonical_point(reference, point).unwrap();
            let actual = pupil_projection_canonical_point(swapped, point).unwrap();
            assert!((actual.0 - expected.0).hypot(actual.1 - expected.1) < 1.0e-12);
            let restored = pupil_projection_image_point(swapped, actual).unwrap();
            assert!((restored.0 - point.0).hypot(restored.1 - point.1) < 1.0e-10);
        }
    }
}
