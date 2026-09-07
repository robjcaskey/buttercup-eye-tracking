use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

fn source(t: u64) -> Source {
    Source {
        eye: 0,
        authority: 1,
        sign_epoch: 1,
        timestamp_ns: t * 200_000_000,
    }
}
fn target(id: u64) -> Target {
    Target {
        id,
        rect: Rect {
            x: 0.0,
            y: 0.0,
            w: 500.0,
            h: 500.0,
        },
    }
}
fn hit(id: u64) -> Hit {
    Hit {
        workspace: 1,
        focused: 1,
        target: Some(target(id)),
    }
}
fn sample(t: u64) -> Sample {
    Sample {
        source: source(t),
        age: Duration::ZERO,
        target: (0.7, 0.5),
    }
}

#[test]
fn dwell_needs_three_fresh_samples_and_elapsed_time() {
    let now = Instant::now();
    let mut d = Dwell::default();
    assert!(d.observe(now, source(1), hit(2)).is_none());
    assert!(d
        .observe(now + Duration::from_millis(100), source(2), hit(2))
        .is_none());
    assert!(d
        .observe(now + Duration::from_millis(200), source(3), hit(2))
        .is_none());
    assert_eq!(d.observe(now + DWELL, source(4), hit(2)), Some(target(2)));
    let mut d = Dwell::default();
    d.observe(now, source(1), hit(2));
    assert!(d.observe(now + DWELL, source(2), hit(2)).is_none());
}

#[test]
fn window_workspace_manual_focus_and_sign_changes_reset_dwell() {
    let now = Instant::now();
    for change in 0..5 {
        let mut d = Dwell::default();
        d.observe(now, source(1), hit(2));
        d.observe(now + Duration::from_millis(200), source(2), hit(2));
        let mut next = hit(2);
        let mut src = source(3);
        match change {
            0 => next.target = Some(target(3)),
            1 => next.workspace = 2,
            2 => next.focused = 4,
            3 => src.sign_epoch = 2,
            _ => next.target.as_mut().unwrap().rect.w = 450.0,
        }
        assert!(d
            .observe(now + Duration::from_millis(400), src, next)
            .is_none());
    }
}

#[test]
fn gaps_and_no_target_reset_dwell_and_current_focus_needs_no_command() {
    let now = Instant::now();
    let mut d = Dwell::default();
    d.observe(now, source(1), hit(2));
    d.observe(now + Duration::from_millis(200), source(2), hit(2));
    assert!(d
        .observe(now + Duration::from_secs(1), source(3), hit(2))
        .is_none());
    assert!(d
        .observe(now + Duration::from_secs(1), source(4), hit(1))
        .is_none());
    assert!(d.pending.is_none());
    assert!(d
        .observe(
            now,
            source(5),
            Hit {
                target: None,
                ..hit(2)
            }
        )
        .is_none());
}

#[test]
fn freshness_is_not_refreshed_by_repeated_publications_or_slow_ipc() {
    let now = Instant::now();
    let enabled = now - Duration::from_secs(2);
    let mut input = Input {
        at: now,
        sample: Ok(sample(1)),
    };
    assert!(input.fresh(now, enabled).is_ok());
    assert!(input
        .fresh(now + MAX_SOURCE_AGE + Duration::from_millis(1), enabled)
        .is_err());
    input.sample.as_mut().unwrap().age = MAX_SOURCE_AGE + Duration::from_millis(1);
    assert!(input.fresh(now, enabled).is_err());
    input.sample = Ok(sample(2));
    input.sample.as_mut().unwrap().age = Duration::from_millis(1);
    assert!(input.fresh(now, now).is_err());
    input.sample = Ok(sample(3));
    input.sample.as_mut().unwrap().target = (1.1, 0.5);
    assert!(input.fresh(now, enabled).is_err());
    input.sample = Err("paused: calibration or accuracy screen");
    assert_eq!(
        input.fresh(now, enabled).err(),
        Some("paused: calibration or accuracy screen")
    );
}

#[test]
fn repeat_and_out_of_order_sources_do_not_advance_dwell() {
    assert!(!source_advanced(Some(source(10)), source(10)));
    assert!(!source_advanced(Some(source(10)), source(9)));
    assert!(source_advanced(Some(source(10)), source(11)));
}

#[test]
fn delayed_short_source_sequence_is_not_a_long_fixation() {
    let now = Instant::now();
    let mut d = Dwell::default();
    let mut src = source(1);
    d.observe(now, src, hit(2));
    src.timestamp_ns += 10_000_000;
    d.observe(now + Duration::from_millis(200), src, hit(2));
    src.timestamp_ns += 10_000_000;
    assert!(d
        .observe(now + Duration::from_millis(400), src, hit(2))
        .is_none());
}

struct Fake {
    focuses: Arc<Mutex<Vec<u64>>>,
    looked: Arc<AtomicBool>,
}
impl Backend for Fake {
    fn hit(&mut self, _: (f64, f64)) -> Result<Hit, String> {
        self.looked.store(true, Ordering::SeqCst);
        Ok(hit(2))
    }
    fn focus(&mut self, target: Target) -> Result<(), String> {
        self.focuses.lock().unwrap().push(target.id);
        Ok(())
    }
}

#[test]
fn worker_starts_off_and_off_invalidates_old_publications() {
    let mut c = Controller::default();
    assert_eq!(c.snapshot()["enabled"], false);
    let focuses = Arc::new(Mutex::new(vec![]));
    let looked = Arc::new(AtomicBool::new(false));
    c.start(
        Fake {
            focuses: focuses.clone(),
            looked: looked.clone(),
        },
        "TEST".into(),
    )
    .unwrap();
    let generation = c.enabled_generation().unwrap();
    c.publish(generation, Instant::now(), Ok(sample(1)));
    let deadline = Instant::now() + Duration::from_secs(1);
    while !looked.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(looked.load(Ordering::SeqCst));
    c.disable();
    for timestamp in 2..6 {
        c.publish(generation, Instant::now(), Ok(sample(timestamp)));
    }
    assert_eq!(c.snapshot()["enabled"], false);
    assert!(focuses.lock().unwrap().is_empty());
    c.start(
        Fake {
            focuses: focuses.clone(),
            looked,
        },
        "TEST".into(),
    )
    .unwrap();
    c.publish(generation, Instant::now(), Ok(sample(20)));
    assert!(c.channel.state.lock().unwrap().input.is_none());
    c.disable();
}

#[test]
fn worker_delivers_one_focus_after_dwell_but_never_for_repeated_frames() {
    let mut c = Controller::default();
    let focuses = Arc::new(Mutex::new(vec![]));
    let looked = Arc::new(AtomicBool::new(false));
    c.start(
        Fake {
            focuses: focuses.clone(),
            looked: looked.clone(),
        },
        "TEST".into(),
    )
    .unwrap();
    let generation = c.enabled_generation().unwrap();
    for t in 1..=3 {
        if t > 1 {
            std::thread::sleep(Duration::from_millis(210));
        }
        looked.store(false, Ordering::SeqCst);
        c.publish(generation, Instant::now(), Ok(sample(t)));
        let deadline = Instant::now() + Duration::from_secs(1);
        while !looked.load(Ordering::SeqCst) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(looked.load(Ordering::SeqCst));
    }
    let deadline = Instant::now() + Duration::from_secs(1);
    while focuses.lock().unwrap().is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(*focuses.lock().unwrap(), [2]);
    for _ in 0..4 {
        c.publish(generation, Instant::now(), Ok(sample(3)));
    }
    c.disable();
    assert_eq!(*focuses.lock().unwrap(), [2]);
}

#[test]
fn focus_controls_work_without_uinput_or_camera_lease() {
    let shared = Arc::new(Mutex::new(crate::SharedState::default()));
    crate::handle_control_command("LEASE CLAIM focus-mode-test 5000", &shared);
    for command in ["GAZE FOCUS STATUS", "GAZE FOCUS OFF"] {
        let reply: Value =
            serde_json::from_str(&crate::handle_control_command(command, &shared)).unwrap();
        assert_eq!(reply["ok"], true);
        assert_eq!(reply["gaze_focus"]["enabled"], false);
        assert_eq!(reply["gaze_focus"]["moves_pointer"], false);
    }
    let s = shared.lock().unwrap();
    assert!(s.control_lease.is_some());
    assert!(s.mouse_output.enabled_generation().is_none());
}
