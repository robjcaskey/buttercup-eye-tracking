//! Reusable timed target recipes: no gaze admission, fitting or camera control.
use serde_json::{json, Value};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub(crate) struct Step {
    pub location_index: usize,
    pub anchor_uv: [f64; 2],
    pub offset_px: [f64; 2],
    pub hold: Duration,
}
pub(crate) struct Session {
    pub recipe: &'static str,
    pub steps: Vec<Step>,
    pub size: Option<(usize, usize)>,
    frame_inset: Option<usize>,
    pub started: Option<Instant>,
    pub finished: Option<&'static str>,
    last_tick: Option<Instant>,
}
impl Session {
    pub fn micro_motion() -> Self {
        let mut steps = Vec::new();
        // Opposite directions and zero-motion controls; deterministic schedule.
        // Amplitudes refer to physical framebuffer pixels, not eye displacement.
        for (location_index, anchor_uv) in [
            [0.5, 0.5],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ]
        .into_iter()
        .enumerate()
        {
            steps.push(Step {
                location_index,
                anchor_uv,
                offset_px: [0., 0.],
                hold: Duration::from_secs(2),
            });
            for amplitude in [1., 2., 3., 4., 5., 0., 0.5, 0.25, 1., 0.25, 5., 0.5] {
                for direction in [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]] {
                    steps.push(Step {
                        location_index,
                        anchor_uv,
                        offset_px: [amplitude * direction[0], amplitude * direction[1]],
                        hold: Duration::from_millis(200),
                    });
                    steps.push(Step {
                        location_index,
                        anchor_uv,
                        offset_px: [0., 0.],
                        hold: Duration::from_millis(200),
                    });
                }
            }
        }
        Self {
            recipe: "micro-motion-five-locations-v2",
            steps,
            size: None,
            frame_inset: None,
            started: None,
            finished: None,
            last_tick: None,
        }
    }
    pub fn abort(&mut self, reason: &'static str) {
        if self.finished.is_none() {
            self.finished = Some(reason);
        }
    }
    pub fn duration(&self) -> Duration {
        self.steps.iter().map(|s| s.hold).sum()
    }
    pub fn recording_limit(&self) -> Duration {
        self.duration() + Duration::from_secs(20)
    }
    pub fn tick(
        &mut self,
        now: Instant,
        size: (usize, usize),
        recording: bool,
        frame_inset: usize,
    ) {
        if self.finished.is_some() {
            return;
        }
        if !recording {
            return;
        }
        if size.0 < 32
            || size.1 < 32
            || size.0.min(size.1) <= frame_inset.saturating_mul(2).saturating_add(16)
        {
            self.abort("DISPLAY TOO SMALL");
            return;
        }
        if self.size.is_some_and(|s| s != size) {
            self.abort("DISPLAY RESIZED");
            return;
        }
        if self.frame_inset.is_some_and(|inset| inset != frame_inset) {
            self.abort("LIGHT FRAME CHANGED");
            return;
        }
        if self
            .last_tick
            .is_some_and(|t| now.saturating_duration_since(t) > Duration::from_millis(750))
        {
            self.abort("PRESENTATION INTERRUPTED");
            return;
        }
        self.size = Some(size);
        self.frame_inset = Some(frame_inset);
        self.started.get_or_insert(now);
        self.last_tick = Some(now);
        if self.index(now).is_none() {
            self.abort("COMPLETE");
        }
    }
    pub fn index(&self, now: Instant) -> Option<usize> {
        if self.finished.is_some() {
            return None;
        }
        let mut elapsed = now.saturating_duration_since(self.started?);
        for (i, step) in self.steps.iter().enumerate() {
            if elapsed < step.hold {
                return Some(i);
            }
            elapsed -= step.hold;
        }
        None
    }
    pub fn point(&self, index: usize) -> Option<(f64, f64)> {
        let (w, h) = self.size?;
        let s = self.steps.get(index)?;
        // Keep the full five-pixel motion plus disk/antialias footprint clear
        // of the current light frame. Anchors are relative to that interior.
        let margin = self.frame_inset? as f64 + 8.;
        Some((
            margin + (w as f64 - 2. * margin) * s.anchor_uv[0] + s.offset_px[0],
            margin + (h as f64 - 2. * margin) * s.anchor_uv[1] + s.offset_px[1],
        ))
    }
    pub fn tail_start(&self, index: usize) -> usize {
        let mut start = index.saturating_sub(3);
        while start < index && self.steps[start].location_index != self.steps[index].location_index
        {
            start += 1;
        }
        start
    }
    pub fn metadata(&self, now: Instant) -> Value {
        json!({"purpose":"recorded-stimulus","recipe":self.recipe,"phase":self.finished.unwrap_or(if self.started.is_some(){"presenting"}else{"arming-raw"}),
            "step_index":self.index(now),"elapsed_ns":self.started.map(|s|now.saturating_duration_since(s).as_nanos().to_string()),
            "framebuffer_size":self.size,"changes_monitor_calibration":false,
            "location_index":self.index(now).map(|i| self.steps[i].location_index),"location_count":5,
            "anchor_uv":self.index(now).map(|i| self.steps[i].anchor_uv),"frame_inset_px":self.frame_inset,
            "target_semantics":"commanded stimulus, NOT measured fixation, eye displacement or training ground truth",
            "rendering":"achromatic 8x8 area-sampled disk; subpixel coordinates via grayscale coverage, not RGB subpixel driving",
            "timing":"host presentation submissions, not measured scanout/photon/exposure timing"})
    }
    pub fn recipe_json(&self) -> Value {
        json!({"recipe":self.recipe,"units":"physical framebuffer pixels",
            "duration_ms":self.duration().as_millis(),"location_count":5,
            "anchor_semantics":"normalized within the current light-frame interior with an eight-pixel motion/target margin; center then four inset corners",
            "steps":self.steps.iter().map(|s|json!({"location_index":s.location_index,"anchor_uv":s.anchor_uv,"offset_px":s.offset_px,"hold_ns":s.hold.as_nanos().to_string()})).collect::<Vec<_>>(),
            "tail":"previous three commands within the current location only; separately recorded as tail, not fixation targets"})
    }
}

pub(crate) fn disk(
    pixels: &mut [u32],
    w: usize,
    h: usize,
    point: (f64, f64),
    radius: f64,
    brightness: f64,
) {
    let (cx, cy) = point;
    for y in ((cy - radius - 1.).floor().max(0.) as usize)
        ..((cy + radius + 1.).ceil().max(0.) as usize).min(h)
    {
        for x in ((cx - radius - 1.).floor().max(0.) as usize)
            ..((cx + radius + 1.).ceil().max(0.) as usize).min(w)
        {
            let mut count = 0.;
            for sy in 0..8 {
                for sx in 0..8 {
                    let dx = x as f64 + (sx as f64 + 0.5) / 8. - cx;
                    let dy = y as f64 + (sy as f64 + 0.5) / 8. - cy;
                    if dx * dx + dy * dy <= radius * radius {
                        count += 1.;
                    }
                }
            }
            let value = (255. * brightness * count / 64.).round() as u32;
            let old = pixels[y * w + x] & 255;
            let v = old.max(value);
            pixels[y * w + x] = v * 0x010101;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn waits_for_recording_and_runs_without_gaze() {
        let t = Instant::now();
        let mut s = Session::micro_motion();
        s.tick(t, (100, 100), false, 0);
        assert!(s.index(t).is_none());
        s.tick(t, (100, 100), true, 0);
        assert_eq!(s.index(t), Some(0));
        s.tick(t + Duration::from_millis(200), (100, 100), true, 0);
        assert!(s.finished.is_none());
    }
    #[test]
    fn finite_complete_schedule_and_subpixel_steps() {
        let s = Session::micro_motion();
        assert!(s.steps.iter().any(|s| s.offset_px == [0.25, 0.]));
        assert!(s
            .steps
            .iter()
            .all(|s| s.offset_px.iter().all(|v| v.abs() <= 5.)));
        assert_eq!(
            s.steps.iter().map(|s| s.hold).sum::<Duration>(),
            Duration::from_millis(106000)
        );
    }
    #[test]
    fn fractional_target_changes_pixels_without_color_fringes() {
        let mut a = vec![0; 400];
        let mut b = a.clone();
        disk(&mut a, 20, 20, (10., 10.), 2., 1.);
        disk(&mut b, 20, 20, (10.25, 10.), 2., 1.);
        assert_ne!(a, b);
        assert!(b.iter().all(|v| (v & 255) == ((v >> 8) & 255)));
    }
    #[test]
    fn resize_and_presentation_gaps_abort() {
        for resize in [false, true] {
            let mut s = Session::micro_motion();
            let t = Instant::now();
            s.tick(t, (100, 100), true, 0);
            s.tick(
                t + Duration::from_secs(1),
                if resize { (200, 100) } else { (100, 100) },
                true,
                0,
            );
            assert!(s.finished.is_some());
        }
    }
    #[test]
    fn completes_and_removes_target_without_eye_samples() {
        let t = Instant::now();
        let mut s = Session::micro_motion();
        for i in 0..=2120 {
            s.tick(t + Duration::from_millis(i * 50), (1920, 1080), true, 0);
        }
        assert_eq!(s.finished, Some("COMPLETE"));
        assert!(s.index(t + Duration::from_secs(106)).is_none());
        assert_eq!(s.metadata(t)["changes_monitor_calibration"], false);
    }

    #[test]
    fn repeats_identical_motion_at_five_distinct_interior_positions() {
        let t = Instant::now();
        for size in [(1920, 1080), (1080, 1920), (800, 800)] {
            let mut s = Session::micro_motion();
            let inset = (size.0.min(size.1) as f64 * 0.44).round() as usize;
            s.tick(t, size, true, inset);
            let per_location = s.steps.len() / 5;
            let mut anchors = Vec::new();
            for location in 0..5 {
                let first = location * per_location;
                anchors.push(s.point(first).unwrap());
                assert_eq!(s.steps[first].hold, Duration::from_secs(2));
                assert_eq!(
                    s.tail_start(first),
                    first,
                    "old location must not leave a tail"
                );
                for i in 0..per_location {
                    assert_eq!(s.steps[first + i].offset_px, s.steps[i].offset_px);
                    assert_eq!(s.steps[first + i].hold, s.steps[i].hold);
                    let (x, y) = s.point(first + i).unwrap();
                    assert!(x - 3. >= inset as f64 && x + 3. < (size.0 - inset) as f64);
                    assert!(y - 3. >= inset as f64 && y + 3. < (size.1 - inset) as f64);
                }
            }
            assert_eq!(anchors[0], (size.0 as f64 / 2., size.1 as f64 / 2.));
            for i in 0..5 {
                for j in 0..i {
                    assert_ne!(anchors[i], anchors[j]);
                }
            }
            assert!(s.recording_limit() > s.duration() + Duration::from_secs(10));
        }
    }

    #[test]
    fn frame_changes_abort_instead_of_moving_the_origin_mid_measurement() {
        let t = Instant::now();
        let mut s = Session::micro_motion();
        s.tick(t, (1920, 1080), true, 100);
        s.tick(t + Duration::from_millis(20), (1920, 1080), true, 110);
        assert_eq!(s.finished, Some("LIGHT FRAME CHANGED"));
        let mut small = Session::micro_motion();
        small.tick(t, (32, 32), true, 14);
        assert_eq!(small.finished, Some("DISPLAY TOO SMALL"));
    }

    #[test]
    fn location_boundary_and_recorded_recipe_are_explicit() {
        let t = Instant::now();
        let mut s = Session::micro_motion();
        s.tick(t, (1920, 1080), true, 100);
        let next = t + Duration::from_millis(21200);
        assert_eq!(s.metadata(next)["location_index"], 1);
        assert_eq!(s.metadata(next)["anchor_uv"], json!([0.25, 0.25]));
        assert_eq!(s.recipe_json()["duration_ms"], 106000);
        assert_eq!(s.recipe_json()["location_count"], 5);
    }
}
