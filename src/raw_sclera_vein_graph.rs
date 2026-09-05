//! Blue-channel scleral vessel graph and temporal relocation diagnostics.
//!
//! This observation layer is shared by Clusters and Vessel Features. It
//! extracts thin dark ridges from physical Quad-Bayer blue cells, crawls their
//! 8-connected topology, describes landmarks by distance to a branch and the
//! shared main-feature orientation bucket, and associates those descriptors
//! in absolute sensor coordinates. The optional surface coordinates are a
//! presentation/audit lift onto the globe implied by the current limbus; they
//! do not publish gaze or anatomy.

use crate::raw_motion_octrees::{FeatureOrientationBucket, IrisEllipseSeed};
use std::collections::VecDeque;
use std::time::Instant;

const CELL: usize = 4;
const MAX_LANDMARKS: usize = 128;
const MAX_MISSES: u8 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VeinSegment {
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub strength: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VeinLandmark {
    pub id: u64,
    pub point: [f32; 2],
    pub degree: u8,
    pub distance_to_branch_px: f32,
    pub tangent_rad: f32,
    pub strength: f32,
    pub matched: bool,
    pub motion_px: [f32; 2],
    /// Eye-centered unit-sphere coordinate; +Z faces the camera.
    pub surface: Option<[f32; 3]>,
    pub surface_delta: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScleraVeinGraphOverlay {
    pub timestamp_ns: u64,
    pub segments: Vec<VeinSegment>,
    pub landmarks: Vec<VeinLandmark>,
    pub ridge_cells: usize,
    pub branch_points: usize,
    pub matched_landmarks: usize,
    pub stable_landmarks: usize,
    pub threshold: f32,
    pub elapsed_us: u64,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    sensor: [f32; 2],
    local: [f32; 2],
    degree: u8,
    distance_to_branch_px: f32,
    tangent_rad: f32,
    strength: f32,
    surface: Option<[f32; 3]>,
}

#[derive(Clone, Copy, Debug)]
struct Track {
    id: u64,
    sensor: [f32; 2],
    degree: u8,
    distance_to_branch_px: f32,
    tangent_rad: f32,
    surface: Option<[f32; 3]>,
    hits: u16,
    missed: u8,
}

#[derive(Default)]
pub struct ScleraVeinGraphTracker {
    tracks: Vec<Track>,
    next_id: u64,
    last_timestamp_ns: Option<u64>,
}

fn quantile(mut values: Vec<f32>, fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f32 * fraction.clamp(0.0, 1.0)).round() as usize;
    let (_, value, _) = values.select_nth_unstable_by(index, f32::total_cmp);
    *value
}

fn angle_distance_mod_pi(first: f32, second: f32) -> f32 {
    let mut delta = (first - second).abs().rem_euclid(std::f32::consts::PI);
    if delta > std::f32::consts::FRAC_PI_2 {
        delta = std::f32::consts::PI - delta;
    }
    delta
}

fn aligned_start(sensor: u32) -> usize {
    (CELL - sensor as usize % CELL) % CELL
}

fn quad_blue_luma(raw: &[u16], width: usize, x: usize, y: usize) -> (f32, f32) {
    let average = |offset_x: usize, offset_y: usize| {
        let i = (y + offset_y) * width + x + offset_x;
        (f32::from(raw[i])
            + f32::from(raw[i + 1])
            + f32::from(raw[i + width])
            + f32::from(raw[i + width + 1]))
            * 0.25
    };
    let red = average(0, 0);
    let green = (average(2, 0) + average(0, 2)) * 0.5;
    let blue = average(2, 2);
    (blue, (red + 2.0 * green + blue) * 0.25)
}

fn neighbours(index: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let x = index % width;
    let y = index / width;
    (-1isize..=1).flat_map(move |dy| {
        (-1isize..=1).filter_map(move |dx| {
            if dx == 0 && dy == 0 {
                return None;
            }
            let nx = x.checked_add_signed(dx)?;
            let ny = y.checked_add_signed(dy)?;
            (nx < width && ny < height).then_some(ny * width + nx)
        })
    })
}

fn surface_point(point: [f32; 2], iris: Option<IrisEllipseSeed>) -> Option<[f32; 3]> {
    let iris = iris?;
    let globe_radius = (iris.major_radius.max(iris.minor_radius) * 1.83) as f32;
    if !globe_radius.is_finite() || globe_radius < 8.0 {
        return None;
    }
    let dx = point[0] - iris.center.0 as f32;
    let dy = point[1] - iris.center.1 as f32;
    let radial_squared = dx * dx + dy * dy;
    if radial_squared >= globe_radius * globe_radius {
        return None;
    }
    Some([
        dx / globe_radius,
        dy / globe_radius,
        (1.0 - radial_squared / (globe_radius * globe_radius)).sqrt(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn detect(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    iris: Option<IrisEllipseSeed>,
) -> (Vec<VeinSegment>, Vec<Candidate>, usize, usize, f32) {
    let start_x = aligned_start(sensor_x);
    let start_y = aligned_start(sensor_y);
    let grid_width = width.saturating_sub(start_x) / CELL;
    let grid_height = height.saturating_sub(start_y) / CELL;
    if grid_width < 12
        || grid_height < 10
        || raw.len() < width.saturating_mul(height)
        || start_x + (grid_width - 1) * CELL + 3 >= width
        || start_y + (grid_height - 1) * CELL + 3 >= height
    {
        return (Vec::new(), Vec::new(), 0, 0, 0.0);
    }
    let mut blue = vec![0.0f32; grid_width * grid_height];
    let mut luma = vec![0.0f32; blue.len()];
    for y in 0..grid_height {
        for x in 0..grid_width {
            (blue[y * grid_width + x], luma[y * grid_width + x]) =
                quad_blue_luma(raw, width, start_x + x * CELL, start_y + y * CELL);
        }
    }
    let bright_floor = quantile(luma.clone(), 0.48);
    let mut response = vec![0.0f32; blue.len()];
    let mut eligible_responses = Vec::new();
    for y in 2..grid_height - 2 {
        for x in 2..grid_width - 2 {
            let i = y * grid_width + x;
            let ring = [
                blue[i - 2],
                blue[i + 2],
                blue[i - 2 * grid_width],
                blue[i + 2 * grid_width],
                blue[i - 2 * grid_width - 2],
                blue[i - 2 * grid_width + 2],
                blue[i + 2 * grid_width - 2],
                blue[i + 2 * grid_width + 2],
            ];
            let surround = ring.iter().sum::<f32>() / ring.len() as f32;
            let local = [
                luma[i - 1],
                luma[i + 1],
                luma[i - grid_width],
                luma[i + grid_width],
            ];
            let bright_neighbours = local
                .iter()
                .filter(|value| **value >= bright_floor * 0.82)
                .count();
            if bright_neighbours < 3 || surround >= 1018.0 {
                continue;
            }
            let local_point = [
                (start_x + x * CELL) as f32 + 1.5,
                (start_y + y * CELL) as f32 + 1.5,
            ];
            if iris.is_some_and(|ellipse| {
                let (sine, cosine) = ellipse.angle.sin_cos();
                let dx = local_point[0] as f64 - ellipse.center.0;
                let dy = local_point[1] as f64 - ellipse.center.1;
                let ex = cosine * dx + sine * dy;
                let ey = -sine * dx + cosine * dy;
                (ex / ellipse.major_radius.max(1.0)).hypot(ey / ellipse.minor_radius.max(1.0))
                    < 1.03
            }) {
                continue;
            }
            response[i] = (surround - blue[i]).max(0.0);
            if response[i] > 0.0 {
                eligible_responses.push(response[i]);
            }
        }
    }
    if eligible_responses.len() < 16 {
        return (Vec::new(), Vec::new(), 0, 0, 0.0);
    }
    // Keep the crawl permissive. Topological/temporal consistency, rather
    // than a single aggressive photometric cutoff, decides which landmarks
    // become stable relocation evidence.
    let threshold = quantile(eligible_responses, 0.60).max(2.0);
    let mut ridge = vec![false; response.len()];
    for y in 2..grid_height - 2 {
        for x in 2..grid_width - 2 {
            let i = y * grid_width + x;
            if response[i] < threshold {
                continue;
            }
            let horizontal_peak = response[i] >= response[i - 1] && response[i] >= response[i + 1];
            let vertical_peak =
                response[i] >= response[i - grid_width] && response[i] >= response[i + grid_width];
            let diagonal_a = response[i] >= response[i - grid_width - 1]
                && response[i] >= response[i + grid_width + 1];
            let diagonal_b = response[i] >= response[i - grid_width + 1]
                && response[i] >= response[i + grid_width - 1];
            ridge[i] = horizontal_peak || vertical_peak || diagonal_a || diagonal_b;
        }
    }
    let degree = (0..ridge.len())
        .map(|i| {
            neighbours(i, grid_width, grid_height)
                .filter(|other| ridge[*other])
                .count() as u8
        })
        .collect::<Vec<_>>();
    let branch_points = degree.iter().filter(|value| **value >= 3).count();
    let mut branch_distance = vec![u16::MAX; ridge.len()];
    let mut queue = VecDeque::new();
    for i in 0..ridge.len() {
        if ridge[i] && degree[i] >= 3 {
            branch_distance[i] = 0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let next_distance = branch_distance[i].saturating_add(1);
        for other in neighbours(i, grid_width, grid_height) {
            if ridge[other] && next_distance < branch_distance[other] {
                branch_distance[other] = next_distance;
                queue.push_back(other);
            }
        }
    }
    let center = |i: usize| {
        [
            (start_x + (i % grid_width) * CELL) as f32 + 1.5,
            (start_y + (i / grid_width) * CELL) as f32 + 1.5,
        ]
    };
    let mut segments = Vec::new();
    for i in 0..ridge.len() {
        if !ridge[i] {
            continue;
        }
        for other in neighbours(i, grid_width, grid_height).filter(|other| *other > i) {
            if ridge[other] {
                segments.push(VeinSegment {
                    start: center(i),
                    end: center(other),
                    strength: ((response[i] + response[other]) / (2.0 * threshold)).clamp(0.0, 4.0),
                });
            }
        }
    }
    let mut ranked = (0..ridge.len())
        .filter(|i| ridge[*i] && (degree[*i] != 2 || branch_distance[*i] % 4 == 0))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        degree[*right]
            .cmp(&degree[*left])
            .then_with(|| response[*right].total_cmp(&response[*left]))
    });
    let mut candidates: Vec<Candidate> = Vec::new();
    for i in ranked {
        let local = center(i);
        if candidates.iter().any(|candidate| {
            (candidate.local[0] - local[0]).hypot(candidate.local[1] - local[1]) < 5.0
        }) {
            continue;
        }
        let linked = neighbours(i, grid_width, grid_height)
            .filter(|other| ridge[*other])
            .map(center)
            .collect::<Vec<_>>();
        let direction = linked.iter().fold([0.0f32; 2], |sum, point| {
            [
                sum[0] + (point[0] - local[0]).abs(),
                sum[1] + (point[1] - local[1]).abs(),
            ]
        });
        let tangent_rad = direction[1]
            .atan2(direction[0])
            .rem_euclid(std::f32::consts::PI);
        let sensor = [sensor_x as f32 + local[0], sensor_y as f32 + local[1]];
        candidates.push(Candidate {
            sensor,
            local,
            degree: degree[i],
            distance_to_branch_px: if branch_distance[i] == u16::MAX {
                999.0
            } else {
                f32::from(branch_distance[i]) * CELL as f32
            },
            tangent_rad,
            strength: (response[i] / threshold).clamp(0.0, 4.0),
            surface: surface_point(local, iris),
        });
        if candidates.len() >= MAX_LANDMARKS {
            break;
        }
    }
    (
        segments,
        candidates,
        ridge.iter().filter(|value| **value).count(),
        branch_points,
        threshold,
    )
}

impl ScleraVeinGraphTracker {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        raw: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        timestamp_ns: u64,
        iris: Option<IrisEllipseSeed>,
    ) -> ScleraVeinGraphOverlay {
        let started = Instant::now();
        if self
            .last_timestamp_ns
            .is_some_and(|last| timestamp_ns <= last || timestamp_ns - last > 500_000_000)
        {
            self.clear();
        }
        let (segments, candidates, ridge_cells, branch_points, threshold) =
            detect(raw, width, height, sensor_x, sensor_y, iris);
        let mut associations = Vec::new();
        for (track_index, track) in self.tracks.iter().enumerate() {
            for (candidate_index, candidate) in candidates.iter().enumerate() {
                let distance = (track.sensor[0] - candidate.sensor[0])
                    .hypot(track.sensor[1] - candidate.sensor[1]);
                if distance > 34.0 || track.degree.abs_diff(candidate.degree) > 1 {
                    continue;
                }
                let orientation_compatible =
                    FeatureOrientationBucket::from_angle_radians(track.tangent_rad)
                        .zip(FeatureOrientationBucket::from_angle_radians(
                            candidate.tangent_rad,
                        ))
                        .is_some_and(|(track, candidate)| track.compatible(candidate));
                if !orientation_compatible {
                    continue;
                }
                let branch_distance = (track.distance_to_branch_px
                    - candidate.distance_to_branch_px)
                    .abs()
                    .min(40.0);
                let angle = angle_distance_mod_pi(track.tangent_rad, candidate.tangent_rad);
                associations.push((
                    distance
                        + branch_distance * 0.45
                        + angle * 7.0
                        + f32::from(track.degree.abs_diff(candidate.degree)) * 9.0,
                    track_index,
                    candidate_index,
                ));
            }
        }
        associations.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut track_match = vec![None; self.tracks.len()];
        let mut candidate_match = vec![None; candidates.len()];
        for (_, track, candidate) in associations {
            if track_match[track].is_none() && candidate_match[candidate].is_none() {
                track_match[track] = Some(candidate);
                candidate_match[candidate] = Some(track);
            }
        }
        let mut output = Vec::with_capacity(candidates.len());
        for (track_index, track) in self.tracks.iter_mut().enumerate() {
            if let Some(candidate_index) = track_match[track_index] {
                let candidate = candidates[candidate_index];
                let previous_sensor = track.sensor;
                let previous_surface = track.surface;
                track.sensor = candidate.sensor;
                track.degree = candidate.degree;
                track.distance_to_branch_px = candidate.distance_to_branch_px;
                track.tangent_rad = candidate.tangent_rad;
                track.surface = candidate.surface;
                track.hits = track.hits.saturating_add(1);
                track.missed = 0;
                output.push(VeinLandmark {
                    id: track.id,
                    point: candidate.local,
                    degree: candidate.degree,
                    distance_to_branch_px: candidate.distance_to_branch_px,
                    tangent_rad: candidate.tangent_rad,
                    strength: candidate.strength,
                    matched: true,
                    motion_px: [
                        candidate.sensor[0] - previous_sensor[0],
                        candidate.sensor[1] - previous_sensor[1],
                    ],
                    surface: candidate.surface,
                    surface_delta: previous_surface.zip(candidate.surface).map(
                        |(before, after)| {
                            [
                                after[0] - before[0],
                                after[1] - before[1],
                                after[2] - before[2],
                            ]
                        },
                    ),
                });
            } else {
                track.missed = track.missed.saturating_add(1);
            }
        }
        self.tracks.retain(|track| track.missed <= MAX_MISSES);
        for (candidate_index, candidate) in candidates.iter().copied().enumerate() {
            if candidate_match[candidate_index].is_some() {
                continue;
            }
            self.next_id = self.next_id.wrapping_add(1).max(1);
            self.tracks.push(Track {
                id: self.next_id,
                sensor: candidate.sensor,
                degree: candidate.degree,
                distance_to_branch_px: candidate.distance_to_branch_px,
                tangent_rad: candidate.tangent_rad,
                surface: candidate.surface,
                hits: 1,
                missed: 0,
            });
            output.push(VeinLandmark {
                id: self.next_id,
                point: candidate.local,
                degree: candidate.degree,
                distance_to_branch_px: candidate.distance_to_branch_px,
                tangent_rad: candidate.tangent_rad,
                strength: candidate.strength,
                surface: candidate.surface,
                ..VeinLandmark::default()
            });
        }
        self.last_timestamp_ns = Some(timestamp_ns);
        let matched_landmarks = output.iter().filter(|landmark| landmark.matched).count();
        let stable_landmarks = output
            .iter()
            .filter(|landmark| {
                self.tracks
                    .iter()
                    .any(|track| track.id == landmark.id && track.hits >= 4 && track.missed == 0)
            })
            .count();
        ScleraVeinGraphOverlay {
            timestamp_ns,
            segments,
            landmarks: output,
            ridge_cells,
            branch_points,
            matched_landmarks,
            stable_landmarks,
            threshold,
            elapsed_us: started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_cell(raw: &mut [u16], width: usize, x: usize, y: usize, blue: u16) {
        let px = x * CELL;
        let py = y * CELL;
        for dy in 0..2 {
            for dx in 0..2 {
                raw[(py + dy) * width + px + dx] = 760;
                raw[(py + dy) * width + px + 2 + dx] = 760;
                raw[(py + 2 + dy) * width + px + dx] = 760;
                raw[(py + 2 + dy) * width + px + 2 + dx] = blue;
            }
        }
    }

    fn branching_frame(width: usize, height: usize, shift: usize) -> Vec<u16> {
        let mut raw = vec![760u16; width * height];
        for y in 0..height / CELL {
            for x in 0..width / CELL {
                write_cell(&mut raw, width, x, y, 760);
            }
        }
        for y in 5..height / CELL - 5 {
            let center = 13 + shift;
            write_cell(&mut raw, width, center, y, 520);
            if y >= 12 {
                let arm = (y - 12) / 2;
                if center + arm < width / CELL - 3 {
                    write_cell(&mut raw, width, center + arm, y, 520);
                }
            }
        }
        raw
    }

    #[test]
    fn blue_ridges_form_branch_descriptors_and_relocate() {
        let (width, height) = (128, 96);
        let mut tracker = ScleraVeinGraphTracker::default();
        let first = tracker.observe(
            &branching_frame(width, height, 0),
            width,
            height,
            100,
            200,
            1_000_000,
            None,
        );
        assert!(
            first.ridge_cells > 8,
            "ridge={} branch={} threshold={}",
            first.ridge_cells,
            first.branch_points,
            first.threshold,
        );
        assert!(first.branch_points > 0);
        let second = tracker.observe(
            &branching_frame(width, height, 1),
            width,
            height,
            100,
            200,
            20_000_000,
            None,
        );
        assert!(second.matched_landmarks > 0);
        assert!(second.landmarks.iter().any(|landmark| landmark.matched));
    }

    #[test]
    fn limbus_geometry_lifts_visible_vessel_to_unit_globe() {
        let point = surface_point(
            [70.0, 48.0],
            Some(IrisEllipseSeed::circle((64.0, 48.0), 20.0)),
        )
        .unwrap();
        let length = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
        assert!((length - 1.0).abs() < 1.0e-5);
    }
}
