//! Native-RAW red-opponent Canny lines and short-lived temporal tracks.
//!
//! This module deliberately does not estimate an iris, pupil, globe, or gaze
//! ray. It borrows the current unpacked RAW10 slice, evaluates the physical
//! Quad-Bayer colour cells in place, and tracks only thin red-opponent lines
//! embedded in a bright surround. The derived response grid is compact; the
//! source image is neither copied nor resized.

use std::collections::VecDeque;
use std::time::Instant;

const QUAD_CELL: usize = 4;
const MAX_ANCHORS: usize = 96;
const MAX_TRACKS: usize = 144;
const MAX_TRAIL_POINTS: usize = 14;
const MIN_PERSISTENT_HITS: u16 = 3;
const MIN_STABLE_HITS: u16 = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RedCannySegment {
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub strength: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RedCannyAnchor {
    pub id: u64,
    pub point: [f32; 2],
    pub strength: f32,
    pub persistent: bool,
    pub stable: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RedCannyTrail {
    pub id: u64,
    pub points: Vec<[f32; 2]>,
    pub stable: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScleraRedCannyOverlay {
    pub timestamp_ns: u64,
    pub segments: Vec<RedCannySegment>,
    pub anchors: Vec<RedCannyAnchor>,
    pub trails: Vec<RedCannyTrail>,
    pub accepted_edge_cells: usize,
    pub persistent_tracks: usize,
    pub stable_tracks: usize,
    pub high_threshold: f32,
    pub bright_floor: f32,
    pub elapsed_us: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    luma: f32,
    red_opponent: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    local: [f32; 2],
    sensor: [f32; 2],
    strength: f32,
    pigment: f32,
    tangent: f32,
}

#[derive(Clone, Debug)]
struct Track {
    id: u64,
    sensor: [f32; 2],
    velocity_px_s: [f32; 2],
    strength: f32,
    pigment: f32,
    tangent: f32,
    hits: u16,
    age: u16,
    missed: u8,
    history_sensor: VecDeque<[f32; 2]>,
}

#[derive(Clone, Copy, Debug)]
struct Association {
    track: usize,
    candidate: usize,
    cost: f32,
}

#[derive(Default)]
pub struct ScleraRedCannyTracker {
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
    let mut difference = (first - second).abs() % std::f32::consts::PI;
    if difference > std::f32::consts::FRAC_PI_2 {
        difference = std::f32::consts::PI - difference;
    }
    difference
}

fn aligned_start(sensor: u32) -> usize {
    (QUAD_CELL - sensor as usize % QUAD_CELL) % QUAD_CELL
}

fn quad_cell(raw: &[u16], width: usize, x: usize, y: usize) -> Cell {
    let average_2x2 = |offset_x: usize, offset_y: usize| {
        let index = (y + offset_y) * width + x + offset_x;
        (f32::from(raw[index])
            + f32::from(raw[index + 1])
            + f32::from(raw[index + width])
            + f32::from(raw[index + width + 1]))
            * 0.25
    };
    // The physical 4x4 Quad-Bayer cell is RG/GB, with each entry represented
    // by one 2x2 same-colour block.
    let red = average_2x2(0, 0);
    let green = (average_2x2(2, 0) + average_2x2(0, 2)) * 0.5;
    let blue = average_2x2(2, 2);
    let mean = (red + green + blue) / 3.0;
    let stabilizer = (mean * 0.02).max(1.0);
    Cell {
        luma: (red + 2.0 * green + blue) * 0.25,
        red_opponent: ((red + stabilizer) / (green + stabilizer))
            .ln()
            .clamp(-2.0, 2.0),
    }
}

fn smooth_three_tap(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0; input.len()];
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let left = input[y * width + x.saturating_sub(1)];
            let center = input[y * width + x];
            let right = input[y * width + (x + 1).min(width - 1)];
            horizontal[y * width + x] = (left + 2.0 * center + right) * 0.25;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let top = horizontal[y.saturating_sub(1) * width + x];
            let center = horizontal[y * width + x];
            let bottom = horizontal[(y + 1).min(height - 1) * width + x];
            output[y * width + x] = (top + 2.0 * center + bottom) * 0.25;
        }
    }
    output
}

fn nms_neighbours(angle: f32) -> [(isize, isize); 2] {
    let degrees = angle.to_degrees().rem_euclid(180.0);
    if !(22.5..157.5).contains(&degrees) {
        [(-1, 0), (1, 0)]
    } else if degrees < 67.5 {
        [(-1, -1), (1, 1)]
    } else if degrees < 112.5 {
        [(0, -1), (0, 1)]
    } else {
        [(-1, 1), (1, -1)]
    }
}

fn neighbour_index(
    x: usize,
    y: usize,
    offset: (isize, isize),
    width: usize,
    height: usize,
) -> Option<usize> {
    let nx = x.checked_add_signed(offset.0)?;
    let ny = y.checked_add_signed(offset.1)?;
    (nx < width && ny < height).then_some(ny * width + nx)
}

fn cell_center(start_x: usize, start_y: usize, x: usize, y: usize) -> [f32; 2] {
    [
        (start_x + x * QUAD_CELL) as f32 + 1.5,
        (start_y + y * QUAD_CELL) as f32 + 1.5,
    ]
}

#[allow(clippy::too_many_arguments)]
fn detect(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
) -> (Vec<RedCannySegment>, Vec<Candidate>, usize, f32, f32) {
    let start_x = aligned_start(sensor_x);
    let start_y = aligned_start(sensor_y);
    let grid_width = width.saturating_sub(start_x) / QUAD_CELL;
    let grid_height = height.saturating_sub(start_y) / QUAD_CELL;
    if grid_width < 10
        || grid_height < 8
        || raw.len() < width.saturating_mul(height)
        || start_x + (grid_width - 1) * QUAD_CELL + 3 >= width
        || start_y + (grid_height - 1) * QUAD_CELL + 3 >= height
    {
        return (Vec::new(), Vec::new(), 0, 0.0, 0.0);
    }

    let mut cells = Vec::with_capacity(grid_width * grid_height);
    for cell_y in 0..grid_height {
        for cell_x in 0..grid_width {
            cells.push(quad_cell(
                raw,
                width,
                start_x + cell_x * QUAD_CELL,
                start_y + cell_y * QUAD_CELL,
            ));
        }
    }
    let bright_floor = quantile(cells.iter().map(|cell| cell.luma).collect(), 0.58);
    let red = cells
        .iter()
        .map(|cell| cell.red_opponent)
        .collect::<Vec<_>>();
    let smooth = smooth_three_tap(&red, grid_width, grid_height);
    let mut gradient_x = vec![0.0f32; cells.len()];
    let mut gradient_y = vec![0.0f32; cells.len()];
    let mut magnitude = vec![0.0f32; cells.len()];
    let mut pigment = vec![0.0f32; cells.len()];
    let mut eligible = vec![false; cells.len()];

    for y in 2..grid_height - 2 {
        for x in 2..grid_width - 2 {
            let index = y * grid_width + x;
            let at = |dx: isize, dy: isize| {
                smooth[(y.checked_add_signed(dy).unwrap()) * grid_width
                    + x.checked_add_signed(dx).unwrap()]
            };
            let gx =
                (at(1, -1) + 2.0 * at(1, 0) + at(1, 1) - at(-1, -1) - 2.0 * at(-1, 0) - at(-1, 1))
                    * 0.125;
            let gy =
                (at(-1, 1) + 2.0 * at(0, 1) + at(1, 1) - at(-1, -1) - 2.0 * at(0, -1) - at(1, -1))
                    * 0.125;
            gradient_x[index] = gx;
            gradient_y[index] = gy;
            magnitude[index] = gx.hypot(gy);

            let mut local_peak = f32::NEG_INFINITY;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    local_peak = local_peak.max(at(dx, dy));
                }
            }
            let ring = [
                at(-2, 0),
                at(2, 0),
                at(0, -2),
                at(0, 2),
                at(-2, -2),
                at(2, -2),
                at(-2, 2),
                at(2, 2),
            ];
            let ring_red = ring.iter().sum::<f32>() / ring.len() as f32;
            pigment[index] = (local_peak - ring_red).max(0.0);

            let surround_indices = [
                (x - 2, y),
                (x + 2, y),
                (x, y - 2),
                (x, y + 2),
                (x - 2, y - 2),
                (x + 2, y - 2),
                (x - 2, y + 2),
                (x + 2, y + 2),
            ];
            let bright_count = surround_indices
                .iter()
                .filter(|(sample_x, sample_y)| {
                    cells[*sample_y * grid_width + *sample_x].luma >= bright_floor * 0.90
                })
                .count();
            let surround_luma = surround_indices
                .iter()
                .map(|(sample_x, sample_y)| cells[sample_y * grid_width + sample_x].luma)
                .sum::<f32>()
                / surround_indices.len() as f32;
            eligible[index] =
                bright_count >= 5 && surround_luma >= bright_floor * 0.94 && surround_luma < 1018.0;
        }
    }

    let eligible_magnitudes = magnitude
        .iter()
        .zip(eligible.iter())
        .filter_map(|(value, accepted)| (*accepted && *value > 0.0).then_some(*value))
        .collect::<Vec<_>>();
    let eligible_pigment = pigment
        .iter()
        .zip(eligible.iter())
        .filter_map(|(value, accepted)| (*accepted && *value > 0.0).then_some(*value))
        .collect::<Vec<_>>();
    if eligible_magnitudes.len() < 12 || eligible_pigment.len() < 8 {
        return (Vec::new(), Vec::new(), 0, 0.0, bright_floor);
    }
    let high_threshold = quantile(eligible_magnitudes, 0.82).max(0.0045);
    let low_threshold = high_threshold * 0.42;
    let pigment_high = quantile(eligible_pigment, 0.58).max(0.0035);
    let pigment_low = (pigment_high * 0.35).max(0.0015);

    let mut nms = vec![0.0f32; cells.len()];
    for y in 2..grid_height - 2 {
        for x in 2..grid_width - 2 {
            let index = y * grid_width + x;
            if !eligible[index] || magnitude[index] < low_threshold {
                continue;
            }
            let neighbours = nms_neighbours(gradient_y[index].atan2(gradient_x[index]));
            let before =
                magnitude[neighbour_index(x, y, neighbours[0], grid_width, grid_height).unwrap()];
            let after =
                magnitude[neighbour_index(x, y, neighbours[1], grid_width, grid_height).unwrap()];
            if magnitude[index] >= before && magnitude[index] >= after {
                nms[index] = magnitude[index];
            }
        }
    }

    let mut accepted = vec![false; cells.len()];
    let mut queue = VecDeque::new();
    for y in 2..grid_height - 2 {
        for x in 2..grid_width - 2 {
            let index = y * grid_width + x;
            if nms[index] >= high_threshold && pigment[index] >= pigment_high {
                accepted[index] = true;
                queue.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let Some(index) = neighbour_index(x, y, (dx, dy), grid_width, grid_height) else {
                    continue;
                };
                if !accepted[index]
                    && eligible[index]
                    && nms[index] >= low_threshold
                    && pigment[index] >= pigment_low
                {
                    accepted[index] = true;
                    queue.push_back((
                        x.checked_add_signed(dx).unwrap(),
                        y.checked_add_signed(dy).unwrap(),
                    ));
                }
            }
        }
    }

    let accepted_edge_cells = accepted.iter().filter(|value| **value).count();
    let mut positions = vec![None; cells.len()];
    for y in 1..grid_height - 1 {
        for x in 1..grid_width - 1 {
            let index = y * grid_width + x;
            if !accepted[index] {
                continue;
            }
            let angle = gradient_y[index].atan2(gradient_x[index]);
            let neighbours = nms_neighbours(angle);
            let before =
                magnitude[neighbour_index(x, y, neighbours[0], grid_width, grid_height).unwrap()];
            let after =
                magnitude[neighbour_index(x, y, neighbours[1], grid_width, grid_height).unwrap()];
            let denominator = before - 2.0 * magnitude[index] + after;
            let offset = if denominator.abs() > 1.0e-6 {
                (0.5 * (before - after) / denominator).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let unit = if magnitude[index] > 1.0e-8 {
                [
                    gradient_x[index] / magnitude[index],
                    gradient_y[index] / magnitude[index],
                ]
            } else {
                [0.0, 0.0]
            };
            let center = cell_center(start_x, start_y, x, y);
            positions[index] = Some([
                center[0] + unit[0] * offset * QUAD_CELL as f32,
                center[1] + unit[1] * offset * QUAD_CELL as f32,
            ]);
        }
    }

    let mut segments = Vec::new();
    let forward = [(1isize, 0isize), (0, 1), (1, 1), (-1, 1)];
    for y in 1..grid_height - 1 {
        for x in 1..grid_width - 1 {
            let index = y * grid_width + x;
            let Some(start) = positions[index] else {
                continue;
            };
            let tangent = gradient_y[index].atan2(gradient_x[index]) + std::f32::consts::FRAC_PI_2;
            for offset in forward {
                let Some(other_index) = neighbour_index(x, y, offset, grid_width, grid_height)
                else {
                    continue;
                };
                let Some(end) = positions[other_index] else {
                    continue;
                };
                let other_tangent = gradient_y[other_index].atan2(gradient_x[other_index])
                    + std::f32::consts::FRAC_PI_2;
                if angle_distance_mod_pi(tangent, other_tangent) <= 0.70 {
                    segments.push(RedCannySegment {
                        start,
                        end,
                        strength: ((nms[index] + nms[other_index])
                            / (2.0 * high_threshold.max(1.0e-6)))
                        .clamp(0.0, 3.0),
                    });
                }
            }
        }
    }

    let mut ranked = positions
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            point.map(|point| {
                let score = nms[index] * (1.0 + pigment[index] * 5.0);
                (index, point, score)
            })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.2.total_cmp(&left.2));
    let mut candidates: Vec<Candidate> = Vec::new();
    let exclusion_radius_squared = (QUAD_CELL as f32 * 2.25).powi(2);
    for (index, local, _) in ranked {
        if candidates.iter().any(|candidate| {
            (candidate.local[0] - local[0]).powi(2) + (candidate.local[1] - local[1]).powi(2)
                < exclusion_radius_squared
        }) {
            continue;
        }
        let tangent = (gradient_y[index].atan2(gradient_x[index]) + std::f32::consts::FRAC_PI_2)
            .rem_euclid(std::f32::consts::PI);
        candidates.push(Candidate {
            local,
            sensor: [sensor_x as f32 + local[0], sensor_y as f32 + local[1]],
            strength: (nms[index] / high_threshold.max(1.0e-6)).clamp(0.0, 4.0),
            pigment: pigment[index],
            tangent,
        });
        if candidates.len() >= MAX_ANCHORS {
            break;
        }
    }
    (
        segments,
        candidates,
        accepted_edge_cells,
        high_threshold,
        bright_floor,
    )
}

impl ScleraRedCannyTracker {
    pub fn clear(&mut self) {
        self.tracks.clear();
        self.last_timestamp_ns = None;
    }

    pub fn observe(
        &mut self,
        raw: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        timestamp_ns: u64,
    ) -> ScleraRedCannyOverlay {
        let started = Instant::now();
        let (segments, candidates, accepted_edge_cells, high_threshold, bright_floor) =
            detect(raw, width, height, sensor_x, sensor_y);
        let dt = self.last_timestamp_ns.and_then(|previous| {
            (timestamp_ns > previous)
                .then_some((timestamp_ns - previous) as f32 * 1.0e-9)
                .filter(|seconds| *seconds <= 0.50)
        });
        if self.last_timestamp_ns.is_some() && dt.is_none() {
            self.clear();
        }
        let seconds = dt.unwrap_or(1.0 / 60.0).clamp(1.0 / 240.0, 0.50);

        let mut associations = Vec::new();
        let maximum_distance = (7.0 + seconds * 260.0).clamp(9.0, 36.0);
        for (track_index, track) in self.tracks.iter().enumerate() {
            let predicted = [
                track.sensor[0] + track.velocity_px_s[0] * seconds,
                track.sensor[1] + track.velocity_px_s[1] * seconds,
            ];
            for (candidate_index, candidate) in candidates.iter().enumerate() {
                let distance =
                    (candidate.sensor[0] - predicted[0]).hypot(candidate.sensor[1] - predicted[1]);
                if distance > maximum_distance {
                    continue;
                }
                let angle = angle_distance_mod_pi(track.tangent, candidate.tangent);
                if angle > 0.90 {
                    continue;
                }
                let pigment = (track.pigment - candidate.pigment).abs();
                let strength = (track.strength - candidate.strength).abs();
                associations.push(Association {
                    track: track_index,
                    candidate: candidate_index,
                    cost: distance + angle * 5.0 + pigment * 18.0 + strength * 0.6,
                });
            }
        }
        associations.sort_by(|left, right| left.cost.total_cmp(&right.cost));
        let mut track_assignment = vec![None; self.tracks.len()];
        let mut candidate_assignment = vec![None; candidates.len()];
        for association in associations {
            if track_assignment[association.track].is_none()
                && candidate_assignment[association.candidate].is_none()
            {
                track_assignment[association.track] = Some(association.candidate);
                candidate_assignment[association.candidate] = Some(association.track);
            }
        }

        for (track_index, track) in self.tracks.iter_mut().enumerate() {
            track.age = track.age.saturating_add(1);
            if let Some(candidate_index) = track_assignment[track_index] {
                let candidate = candidates[candidate_index];
                let measured_velocity = [
                    (candidate.sensor[0] - track.sensor[0]) / seconds,
                    (candidate.sensor[1] - track.sensor[1]) / seconds,
                ];
                track.velocity_px_s = [
                    track.velocity_px_s[0] * 0.55 + measured_velocity[0] * 0.45,
                    track.velocity_px_s[1] * 0.55 + measured_velocity[1] * 0.45,
                ];
                track.sensor = candidate.sensor;
                track.strength = track.strength * 0.6 + candidate.strength * 0.4;
                track.pigment = track.pigment * 0.6 + candidate.pigment * 0.4;
                track.tangent = candidate.tangent;
                track.hits = track.hits.saturating_add(1);
                track.missed = 0;
                track.history_sensor.push_back(candidate.sensor);
                while track.history_sensor.len() > MAX_TRAIL_POINTS {
                    track.history_sensor.pop_front();
                }
            } else {
                track.missed = track.missed.saturating_add(1);
            }
        }
        self.tracks.retain(|track| track.missed <= 2);

        // Rebuild the candidate-to-track map after retention; the small
        // bounded population keeps this explicit lookup cheaper and clearer
        // than carrying unstable vector indices through `retain`.
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let already_matched = self.tracks.iter().any(|track| {
                track.missed == 0
                    && (track.sensor[0] - candidate.sensor[0])
                        .hypot(track.sensor[1] - candidate.sensor[1])
                        < 0.25
            });
            if already_matched {
                continue;
            }
            self.next_id = self.next_id.wrapping_add(1).max(1);
            let mut history_sensor = VecDeque::new();
            history_sensor.push_back(candidate.sensor);
            self.tracks.push(Track {
                id: self.next_id,
                sensor: candidate.sensor,
                velocity_px_s: [0.0, 0.0],
                strength: candidate.strength,
                pigment: candidate.pigment,
                tangent: candidate.tangent,
                hits: 1,
                age: 1,
                missed: 0,
                history_sensor,
            });
            let _ = candidate_index;
        }
        if self.tracks.len() > MAX_TRACKS {
            self.tracks.sort_by(|left, right| {
                right
                    .hits
                    .cmp(&left.hits)
                    .then_with(|| left.missed.cmp(&right.missed))
                    .then_with(|| right.strength.total_cmp(&left.strength))
            });
            self.tracks.truncate(MAX_TRACKS);
        }
        self.last_timestamp_ns = Some(timestamp_ns);

        let mut anchors = Vec::new();
        let mut trails = Vec::new();
        let mut persistent_tracks = 0usize;
        let mut stable_tracks = 0usize;
        for track in self.tracks.iter().filter(|track| track.missed == 0) {
            let local = [
                track.sensor[0] - sensor_x as f32,
                track.sensor[1] - sensor_y as f32,
            ];
            if local[0] < 0.0
                || local[1] < 0.0
                || local[0] >= width as f32
                || local[1] >= height as f32
            {
                continue;
            }
            let persistent = track.hits >= MIN_PERSISTENT_HITS;
            let stable = track.hits >= MIN_STABLE_HITS;
            persistent_tracks += usize::from(persistent);
            stable_tracks += usize::from(stable);
            anchors.push(RedCannyAnchor {
                id: track.id,
                point: local,
                strength: track.strength,
                persistent,
                stable,
            });
            if persistent {
                let points = track
                    .history_sensor
                    .iter()
                    .map(|point| [point[0] - sensor_x as f32, point[1] - sensor_y as f32])
                    .filter(|point| {
                        point[0] >= 0.0
                            && point[1] >= 0.0
                            && point[0] < width as f32
                            && point[1] < height as f32
                    })
                    .collect::<Vec<_>>();
                if points.len() >= 2 {
                    trails.push(RedCannyTrail {
                        id: track.id,
                        points,
                        stable,
                    });
                }
            }
        }

        ScleraRedCannyOverlay {
            timestamp_ns,
            segments,
            anchors,
            trails,
            accepted_edge_cells,
            persistent_tracks,
            stable_tracks,
            high_threshold,
            bright_floor,
            elapsed_us: started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_cell(
        raw: &mut [u16],
        width: usize,
        cell_x: usize,
        cell_y: usize,
        red: u16,
        green: u16,
        blue: u16,
    ) {
        let x = cell_x * 4;
        let y = cell_y * 4;
        for dy in 0..2 {
            for dx in 0..2 {
                raw[(y + dy) * width + x + dx] = red;
                raw[(y + dy) * width + x + 2 + dx] = green;
                raw[(y + 2 + dy) * width + x + dx] = green;
                raw[(y + 2 + dy) * width + x + 2 + dx] = blue;
            }
        }
    }

    fn synthetic_vessel(width: usize, height: usize, vessel_cell_x: usize) -> Vec<u16> {
        let mut raw = vec![700u16; width * height];
        for cell_y in 0..height / 4 {
            for cell_x in 0..width / 4 {
                let distance = cell_x.abs_diff(vessel_cell_x);
                let (red, green, blue) = if distance == 0 {
                    (720, 470, 560)
                } else if distance == 1 {
                    (710, 610, 650)
                } else {
                    (700, 700, 700)
                };
                write_cell(&mut raw, width, cell_x, cell_y, red, green, blue);
            }
        }
        raw
    }

    #[test]
    fn red_opponent_canny_finds_a_vessel_in_bright_sclera() {
        let (width, height) = (128, 96);
        let raw = synthetic_vessel(width, height, 16);
        let (segments, candidates, accepted, threshold, _) = detect(&raw, width, height, 0, 0);
        assert!(accepted >= 12, "accepted={accepted}");
        assert!(!segments.is_empty());
        assert!(!candidates.is_empty());
        assert!(threshold > 0.0);
        assert!(candidates
            .iter()
            .all(|candidate| (candidate.local[0] - 64.0).abs() < 18.0));
    }

    #[test]
    fn achromatic_dark_line_is_not_a_red_sclera_vessel() {
        let (width, height) = (128, 96);
        let mut raw = vec![700u16; width * height];
        for cell_y in 0..height / 4 {
            for cell_x in 0..width / 4 {
                let value = if cell_x.abs_diff(16) <= 1 { 480 } else { 700 };
                write_cell(&mut raw, width, cell_x, cell_y, value, value, value);
            }
        }
        let (segments, candidates, accepted, _, _) = detect(&raw, width, height, 0, 0);
        assert_eq!(accepted, 0);
        assert!(segments.is_empty());
        assert!(candidates.is_empty());
    }

    #[test]
    fn repeated_exact_frames_promote_red_line_tracks() {
        let (width, height) = (128, 96);
        let raw = synthetic_vessel(width, height, 16);
        let mut tracker = ScleraRedCannyTracker::default();
        let mut overlay = ScleraRedCannyOverlay::default();
        for frame in 0..7u64 {
            overlay = tracker.observe(
                &raw,
                width,
                height,
                3_000,
                1_500,
                1_000_000_000 + frame * 16_000_000,
            );
        }
        assert_eq!(overlay.timestamp_ns, 1_096_000_000);
        assert!(overlay.persistent_tracks > 0);
        assert!(overlay.stable_tracks > 0);
        assert!(overlay.anchors.iter().any(|anchor| anchor.stable));
        assert!(!overlay.trails.is_empty());
    }

    #[test]
    fn absolute_sensor_coordinates_survive_a_matching_roi_shift() {
        let (width, height) = (128, 96);
        let raw = synthetic_vessel(width, height, 16);
        let shifted = synthetic_vessel(width, height, 15);
        let mut tracker = ScleraRedCannyTracker::default();
        for frame in 0..3u64 {
            let _ = tracker.observe(
                &raw,
                width,
                height,
                3_000,
                1_500,
                2_000_000_000 + frame * 16_000_000,
            );
        }
        // Moving the ROI four sensor pixels right moves the same physical
        // vessel one Quad-Bayer cell left in local coordinates.
        let overlay = tracker.observe(&shifted, width, height, 3_004, 1_500, 2_048_000_000);
        assert!(overlay.persistent_tracks > 0);
    }
}
