//! Re-express the unchanged COLMAP map in each acquired camera coordinate frame.
//! This is rigid pose playback, not a measurement of independent point motion.
use super::{hash, json, num, quantiles, Result, Value};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Copy)]
struct Pose {
    q: [f64; 4],
    t: [f64; 3],
}
impl Pose {
    fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let [w, x, y, z] = self.q;
        [
            (1. - 2. * y * y - 2. * z * z) * p[0]
                + (2. * x * y - 2. * z * w) * p[1]
                + (2. * x * z + 2. * y * w) * p[2]
                + self.t[0],
            (2. * x * y + 2. * z * w) * p[0]
                + (1. - 2. * x * x - 2. * z * z) * p[1]
                + (2. * y * z - 2. * x * w) * p[2]
                + self.t[1],
            (2. * x * z - 2. * y * w) * p[0]
                + (2. * y * z + 2. * x * w) * p[1]
                + (1. - 2. * x * x - 2. * y * y) * p[2]
                + self.t[2],
        ]
    }
}
fn project(p: [f64; 3], k: [f64; 4], offset: [f64; 2]) -> Option<[f64; 2]> {
    (p[2] > 0. && p.iter().all(|v| v.is_finite())).then(|| {
        // COLMAP pixel centers start at 0.5. Native RAW centers start at 0.
        [
            k[0] * p[0] / p[2] + k[2] - offset[0] - 0.5,
            k[1] * p[1] / p[2] + k[3] - offset[1] - 0.5,
        ]
    })
}
fn numbers<const N: usize>(a: &[&str]) -> Result<[f64; N]> {
    let mut result = [0.0_f64; N];
    if a.len() != N {
        return Err("COLMAP numeric field count".into());
    }
    for (v, s) in result.iter_mut().zip(a) {
        *v = s.parse::<f64>()?;
        if !v.is_finite() {
            return Err("non-finite COLMAP value".into());
        }
    }
    Ok(result)
}

pub fn enrich(model: &mut Value, inventory: &Value) -> Result<()> {
    let images = inventory["images"].as_array().ok_or("RAW inventory")?;
    let xyz = model["points"]
        .as_array()
        .ok_or("model points")?
        .iter()
        .map(|p| {
            [
                p[0].as_f64().unwrap(),
                p[1].as_f64().unwrap(),
                p[2].as_f64().unwrap(),
            ]
        })
        .collect::<Vec<_>>();
    if images.is_empty() || images.len() > 512 || xyz.len() * images.len() > 5_000_000 {
        return Err("playback exceeds bounded frame/point budget".into());
    }
    let first_time = num(&images[0]["source"]["timestamp_ns"]);
    let mut previous_time = None;
    let mut frames = Vec::new();
    let mut names = BTreeMap::new();
    for (index, image) in images.iter().enumerate() {
        let time = num(&image["source"]["timestamp_ns"]);
        if previous_time.is_some_and(|last| time <= last) {
            return Err("playback requires strictly increasing acquisition timestamps".into());
        }
        previous_time = Some(time);
        let name = image["name"].as_str().ok_or("image name")?;
        if names.insert(name.to_string(), index).is_some() {
            return Err("duplicate source image".into());
        }
        frames.push(json!({"index":index,"name":name,
            "time_s":(time-first_time) as f64/1e9,
            "timestamp_ns":time.to_string(),"sequence":image["source"]["sequence"],
            "offset":image["offset"],"pose":null,"observations":[]}));
    }
    let mut point_ids = Vec::new();
    let mut point_lookup = BTreeMap::new();
    let mut tracks = vec![Vec::<usize>::new(); xyz.len()];
    let mut errors = Vec::new();
    let mut per_point_errors = vec![Vec::<f64>::new(); xyz.len()];
    let mut behind = 0;
    let mut poses = Vec::new();
    let mut model_hashes = json!({});
    if let Some(path) = model["component"]["path"].as_str() {
        let path = Path::new(path);
        for name in ["images.txt", "cameras.txt", "points3D.txt"] {
            model_hashes[name] = json!(hash(&fs::read(path.join(name))?));
        }
        for (index, line) in fs::read_to_string(path.join("points3D.txt"))?
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .enumerate()
        {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() < 8 || index >= xyz.len() || numbers::<3>(&fields[1..4])? != xyz[index]
            {
                return Err("point order differs from the displayed COLMAP map".into());
            }
            let id: u64 = fields[0].parse()?;
            if point_lookup.insert(id, index).is_some() {
                return Err("duplicate COLMAP point ID".into());
            }
            point_ids.push(id.to_string());
        }
        if point_ids.len() != xyz.len() {
            return Err("COLMAP point count changed".into());
        }
        let mut cameras = BTreeMap::new();
        for line in fs::read_to_string(path.join("cameras.txt"))?
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let a = line.split_whitespace().collect::<Vec<_>>();
            if a.len() != 8 || a[1] != "PINHOLE" {
                return Err("playback currently requires exported PINHOLE cameras".into());
            }
            let k = numbers::<4>(&a[4..8])?;
            if k[0] <= 0. || k[1] <= 0. {
                return Err("invalid focal length".into());
            }
            cameras.insert(a[0].parse::<u64>()?, k);
        }
        let text = fs::read_to_string(path.join("images.txt"))?;
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        while let Some(line) = lines.next() {
            let a = line.split_whitespace().collect::<Vec<_>>();
            if a.len() != 10 {
                return Err("invalid COLMAP image pose row".into());
            }
            let index = *names.get(a[9]).ok_or("pose missing from RAW inventory")?;
            if !frames[index]["pose"].is_null() {
                return Err("duplicate COLMAP image pose".into());
            }
            let pose = Pose {
                q: numbers(&a[1..5])?,
                t: numbers(&a[5..8])?,
            };
            if (pose.q.iter().map(|v| v * v).sum::<f64>() - 1.).abs() > 1e-6 {
                return Err("COLMAP quaternion is not unit length".into());
            }
            let k = *cameras.get(&a[8].parse::<u64>()?).ok_or("camera ID")?;
            let offset = [
                num(&images[index]["offset"][0]) as f64,
                num(&images[index]["offset"][1]) as f64,
            ];
            frames[index]["pose"] = json!({"q":pose.q,"t":pose.t,"intrinsics":k});
            poses.push(pose);
            let data = lines
                .next()
                .ok_or("missing image observations")?
                .split_whitespace()
                .collect::<Vec<_>>();
            if data.len() % 3 != 0 {
                return Err("invalid observation fields".into());
            }
            let mut observations = Vec::new();
            for a in data.chunks_exact(3) {
                if a[2] == "-1" {
                    continue;
                }
                let point = *point_lookup
                    .get(&a[2].parse::<u64>()?)
                    .ok_or("observation point ID")?;
                let uv = numbers::<2>(&a[0..2])?;
                let measured = [uv[0] - offset[0] - 0.5, uv[1] - offset[1] - 0.5];
                if let Some(predicted) = project(pose.apply(xyz[point]), k, offset) {
                    let error = (predicted[0] - measured[0]).hypot(predicted[1] - measured[1]);
                    errors.push(error);
                    per_point_errors[point].push(error);
                } else {
                    behind += 1;
                }
                tracks[point].push(index);
                observations.push(json!([point, measured[0], measured[1]]));
            }
            frames[index]["observations"] = json!(observations);
        }
    }
    for track in &mut tracks {
        track.sort_unstable();
        track.dedup();
    }
    let mut error_deltas = Vec::new();
    let mut track_mismatches = 0;
    for (i, errors) in per_point_errors.iter().enumerate() {
        if !errors.is_empty() {
            error_deltas.push(
                (errors.iter().sum::<f64>() / errors.len() as f64
                    - model["points"][i][6].as_f64().unwrap())
                .abs(),
            );
        }
        if tracks[i].len() as f64 != model["points"][i][7].as_f64().unwrap() {
            track_mismatches += 1;
        }
    }
    let mut coordinates: [Vec<f64>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(poses.len() * xyz.len()));
    for pose in &poses {
        for &p in &xyz {
            let p = pose.apply(p);
            for k in 0..3 {
                coordinates[k].push(p[k]);
            }
        }
    }
    let bounds = |tail: f64, axes: &[Vec<f64>; 3]| -> Value {
        let lo: [f64; 3] = std::array::from_fn(|k| {
            if axes[k].is_empty() {
                0.
            } else {
                axes[k][((axes[k].len() - 1) as f64 * tail).floor() as usize]
            }
        });
        let hi: [f64; 3] = std::array::from_fn(|k| {
            if axes[k].is_empty() {
                1.
            } else {
                axes[k][((axes[k].len() - 1) as f64 * (1. - tail)).ceil() as usize]
            }
        });
        json!({"lo":lo,"hi":hi})
    };
    for axis in &mut coordinates {
        axis.sort_by(f64::total_cmp);
    }
    model["camera_bounds"] = json!({"all":bounds(0.,&coordinates),"core":bounds(0.01,&coordinates),"basis":"fixed extent over all points and all registered poses; no framewise centering or scaling"});
    model["playback_validation"] = json!({"registered_frames":poses.len(),"missing_pose_frames":frames.len()-poses.len(),
        "reprojection_error_px":quantiles(errors),"stored_point_mean_error_delta_px":quantiles(error_deltas),
        "observations_behind_camera":behind,"point_track_count_mismatches":track_mismatches,
        "transform":"camera = R(q_world_to_camera) * world + t; fixed camera at origin; no interpolation across missing poses",
        "model_text_sha256":model_hashes});
    model["frames"] = json!(frames);
    model["point_ids"] = json!(point_ids);
    model["point_tracks"] = json!(tracks);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn world_to_camera_pose_has_correct_direction_and_keeps_rigid_distances() {
        let h = 0.5_f64.sqrt();
        let pose = Pose {
            q: [h, 0., 0., h],
            t: [4., 5., 6.],
        };
        let p = pose.apply([1., 2., 3.]);
        for (a, b) in p.into_iter().zip([2., 6., 9.]) {
            assert!((a - b).abs() < 1e-12);
        }
        let center = pose.apply([-5., 4., -6.]);
        assert!(center.into_iter().all(|v| v.abs() < 1e-12));
        let other = pose.apply([2., 4., 5.]);
        assert!(((0..3).map(|k| (other[k] - p[k]).powi(2)).sum::<f64>() - 9.).abs() < 1e-12);
    }
    #[test]
    fn native_reprojection_preserves_crop_and_half_pixel_conventions() {
        let k = [4000., 4000., 148.5, 548.5];
        let p = [0.1, -0.05, 2.];
        assert_eq!(project(p, k, [92., 40.]), Some([256., 408.]));
        assert_eq!(project(p, k, [124., 40.]), Some([224., 408.]));
        assert_eq!(project([1., 2., -1.], k, [0., 0.]), None);
    }
}
