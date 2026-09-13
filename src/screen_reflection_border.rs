//! Checked optical clock on the shared presentation border. No new window,
//! camera control, inference dependency, or separate recording format.
//!
//! Render intent is not a receipt: publish only after a successful buffer
//! submission. All modes feed the same bounded RAW recovery worker. Neither
//! an emitted symbol nor a held recovery proves scanout/exposure timing.
use crate::screen_reflection_code::{OpticalCodeScheme, SpatialCodeLayout};
use crate::screen_reflection_stimulus::{ClockSessionSnapshot, RecoveryProgress, RecoveryReport};
use crate::screen_reflection_temporal as temporal;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const STALE: Duration = Duration::from_millis(750);
const TRANSITIONS: usize = 96;

// Short, opt-in optics experiment. Reuse the checked spatial code before
// spending effort on a larger payload. No change to normal clock behavior.
pub(crate) fn spatial_trial() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("BUTTERCUP_SPATIAL_BORDER_TRIAL").as_deref() == Ok("1"))
}

#[derive(Clone)]
pub(crate) struct Epoch {
    pub id: String,
    pub tag: u8,
    started: Instant,
}

impl Epoch {
    pub fn new(started: Instant) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self {
            id: format!("border-{}-{time:x}-{serial:x}", std::process::id()),
            tag: (serial % 16) as u8,
            started,
        }
    }
    pub fn code_at(&self, now: Instant) -> u64 {
        (now.saturating_duration_since(self.started).as_nanos() / 200_000_000) as u64
    }
    pub fn pixel_at(&self, now: Instant) -> u32 {
        temporal::symbol_pixel(self.code_at(now), self.tag, temporal::DEFAULT_AMPLITUDE)
    }
    pub fn metadata(&self, now: Instant) -> Value {
        let code = self.code_at(now);
        if spatial_trial() {
            let phase = now.saturating_duration_since(self.started).as_secs() / 15;
            let variant = ["mono", "blue", "opponent"][phase as usize % 3];
            return json!({"scheme":"reed-muller-session-v3",
                "epoch":format!("{}-spatial-{variant}-{phase}",self.id),"session_tag":self.tag,
                "code_index":code.to_string(),"code_mod":code%32,"code_hz":5.0,
                "spatial_trial":true,"variant":variant,"phase_seconds":15,
                "logical_cells":[8,4],"counter_combinations":32,"maximum_arrival_skew_ms":3000,
                "check":"existing RM(1,4) distance-8 code; initial optics pilot, not final 14-bit format",
                "region":"repeated complete tiles with dark gutters on four border strips",
                "timing":"successful host submission is not scanout; code can span target updates"});
        }
        json!({"scheme":temporal::SCHEME,"epoch":self.id,"session_tag":self.tag,
            "code_index":code.to_string(),"code_mod":code % temporal::PERIOD as u64,
            "symbol":temporal::sign(code,self.tag),"rgb_u24":self.pixel_at(now),
            "code_hz":temporal::DEFAULT_CODE_HZ,"amplitude":temporal::DEFAULT_AMPLITUDE,
            "word_symbols":temporal::WORD_BITS,"minimum_warmup_seconds":12.8,
            "region":"four disjoint perimeter strips; central target unchanged",
            "timing":"render intent until successful host submission; not measured photons",
            "session_identity":"phase offset only; not independent optical authentication"})
    }

    pub(crate) fn draw_spatial(&self, pixels:&mut[u32], width:usize, height:usize, band:usize, now:Instant) {
        if pixels.len()<width.saturating_mul(height) || band==0 {return;}
        let variant=(now.saturating_duration_since(self.started).as_secs()/15)%3;
        let signs=crate::screen_reflection_code::FrameCode::new(self.code_at(now),self.tag)
            .physical_signs_for(OpticalCodeScheme::ReedMullerV3);
        let colors=match variant {
            0=>[0x303030,0xdcdcdc],
            1=>[0x909030,0x9090f0],
            _=>[0x408e30,0x865bf0],
        };
        for (left,top,right,bottom) in [(0,0,width,band),(0,height-band,width,height),
            (0,band,band,height-band),(width-band,band,width,height-band)] {
            if left>=right || top>=bottom {continue;}
            for y in top..bottom {pixels[y*width+left..y*width+right].fill(0x101010);}
            let pitch=((right-left)/10).min((bottom-top)/6).max(1);
            let (tw,th)=(10*pitch,6*pitch);
            for ty in (top..bottom).step_by(th) {for tx in (left..right).step_by(tw) {
                if tx+tw>right || ty+th>bottom {continue;}
                for row in 0..4 {for col in 0..8 {
                    let color=colors[usize::from(signs[row*8+col]>0)];
                    for y in ty+(row+1)*pitch..ty+(row+2)*pitch {
                        pixels[y*width+tx+(col+1)*pitch..y*width+tx+(col+2)*pitch].fill(color);
                    }
                }}
            }}
        }
    }
}

#[derive(Clone)]
struct Transition {
    code: u64,
    begin_ns: u64,
    end_ns: u64,
}

#[derive(Default)]
struct State {
    epoch: String,
    generation: u64,
    size: [usize; 2],
    width_fraction: f64,
    snapshot: Option<ClockSessionSnapshot>,
    last_present: Option<Instant>,
    transitions: VecDeque<Transition>,
    progress: Value,
    recovery: Value,
    recovery_at: Option<Instant>,
}

impl State {
    fn fresh(&self, now: Instant) -> bool {
        self.last_present
            .is_some_and(|t| now.saturating_duration_since(t) < STALE)
            && self.snapshot.is_some()
    }

    fn presented(
        &mut self,
        lightbox: &Value,
        size: [usize; 2],
        begin_ns: u64,
        end_ns: u64,
        now: Instant,
    ) -> Value {
        let clock = &lightbox["optical_clock"];
        let parsed = (|| {
            if lightbox["enabled"] != true || (clock["scheme"] != temporal::SCHEME
                && !(clock["spatial_trial"]==true && clock["scheme"]=="reed-muller-session-v3")) {
                return None;
            }
            let epoch = clock["epoch"].as_str()?;
            let tag = u8::try_from(clock["session_tag"].as_u64()?).ok()?;
            let code = clock["code_index"].as_str()?.parse::<u64>().ok()?;
            let width = lightbox["width_fraction"].as_f64()?;
            if size.iter().any(|s| *s < 3) || !(0.0..0.5).contains(&width) || end_ns < begin_ns {
                return None;
            }
            Some((epoch, tag, code, width))
        })();
        let Some((epoch, tag, code, width)) = parsed else {
            self.snapshot = None;
            self.last_present = None;
            return Value::Null;
        };
        let discontinuity = self
            .snapshot
            .as_ref()
            .is_some_and(|s| code < s.code_index || code > s.code_index.saturating_add(1));
        if self.epoch != epoch
            || !self.fresh(now)
            || self.size != size
            || self.width_fraction != width
            || discontinuity
        {
            self.generation = self.generation.saturating_add(1);
            self.epoch = epoch.to_owned();
            self.transitions.clear();
            self.progress = Value::Null;
            self.recovery = Value::Null;
            self.recovery_at = None;
        }
        self.size = size;
        self.width_fraction = width;
        if self.transitions.back().is_none_or(|t| t.code != code) {
            self.transitions.push_back(Transition {
                code,
                begin_ns,
                end_ns,
            });
            while self.transitions.len() > TRANSITIONS {
                self.transitions.pop_front();
            }
        }
        self.snapshot = Some(ClockSessionSnapshot {
            code_layout: SpatialCodeLayout::LEGACY,
            temporal_code: clock["spatial_trial"]!=true,
            session_id: format!("{}-segment-{}", self.epoch, self.generation),
            session_tag: tag,
            code_hz: temporal::DEFAULT_CODE_HZ,
            display_refresh_hz: 0.,
            presentation_index: 0,
            code_index: code,
            present_commit_unix_ns: begin_ns,
            // Legacy enum is ignored by the temporal branch.
            code_scheme: OpticalCodeScheme::ReedMullerV3,
        });
        self.last_present = Some(now);
        json!({"emitted":clock,"decoder_session":self.snapshot.as_ref().unwrap().session_id,
            "border_width_fraction":width,"viewport_px":size,
            "progress":self.progress,"latest_recovery":self.recovery,
            "recovery_fresh":self.recovery_at.is_some_and(|t|now.saturating_duration_since(t)<Duration::from_millis(900)),
            "recovery_is_new_camera_observation":false,
            "timing":"host submissions only; optical recovery still required"})
    }

    fn report(&mut self, report: &RecoveryReport<'_>, now: Instant) -> Result<(), String> {
        if !self.fresh(now)
            || self
                .snapshot
                .as_ref()
                .is_none_or(|s| s.session_id != report.session_id)
        {
            return Err("border clock stale or wrong-session".into());
        }
        let t = self
            .transitions
            .iter()
            .find(|t| t.code == report.recovered_code_index)
            .ok_or("border code-outside-transition-window")?;
        if !report.verified
            || !report.score.is_finite()
            || !report.confidence_margin.is_finite()
            || report.host_arrival_unix_ns < t.begin_ns
            || report.host_arrival_unix_ns.saturating_sub(t.end_ns) > 3_000_000_000
        {
            return Err("border clock rejected invalid recovery".into());
        }
        self.recovery = json!({"sequence":report.sequence.to_string(),
            "recovered_code_index":report.recovered_code_index.to_string(),
            "camera_host_arrival_unix_ns":report.host_arrival_unix_ns.to_string(),
            "first_submit_begin_unix_ns":t.begin_ns.to_string(),"first_submit_end_unix_ns":t.end_ns.to_string(),
            "arrival_minus_first_submit_ms":(report.host_arrival_unix_ns-t.begin_ns) as f64/1e6,
            "correlation":report.score,"margin":report.confidence_margin,
            "verified_temporal_word":self.snapshot.as_ref().is_some_and(|s|s.temporal_code),
            "verified_spatial_code":self.snapshot.as_ref().is_some_and(|s|!s.temporal_code),
            "timing":"includes held-symbol age; not exposure latency"});
        self.recovery_at = Some(now);
        Ok(())
    }
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}

/// Called from the existing presentation journal, AFTER present() succeeds.
pub(crate) fn presented(lightbox: &Value, size: [usize; 2], begin: &str, end: &str) -> Value {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    match (begin.parse(), end.parse()) {
        (Ok(begin), Ok(end)) => state.presented(lightbox, size, begin, end, Instant::now()),
        _ => {
            state.snapshot = None;
            Value::Null
        }
    }
}

pub(crate) fn snapshot() -> Option<ClockSessionSnapshot> {
    let state = state().lock().ok()?;
    state
        .fresh(Instant::now())
        .then(|| state.snapshot.clone())
        .flatten()
}

pub(crate) fn status_for(epoch: &str) -> String {
    let Ok(s) = state().lock() else {
        return "UNAVAILABLE".into();
    };
    let now = Instant::now();
    if s.epoch != epoch || !s.fresh(now) {
        return "WAITING FOR DISPLAY".into();
    }
    if s.recovery_at
        .is_some_and(|t| now.saturating_duration_since(t) < Duration::from_millis(900))
    {
        return "CHECKED".into();
    }
    let symbols = s.progress["valid_symbols"].as_u64().unwrap_or(0);
    if symbols < temporal::WORD_BITS as u64 {
        format!("WARMING {symbols}/{}", temporal::WORD_BITS)
    } else {
        "SEARCHING".into()
    }
}

pub(crate) fn report(report: &RecoveryReport<'_>) -> Option<Result<(), String>> {
    if !report.session_id.starts_with("border-") {
        return None;
    }
    Some(
        state()
            .lock()
            .map_err(|_| "border clock lock poisoned".to_owned())
            .and_then(|mut s| s.report(report, Instant::now())),
    )
}

pub(crate) fn progress(progress: &RecoveryProgress<'_>) -> Option<Result<(), String>> {
    if !progress.session_id.starts_with("border-") {
        return None;
    }
    Some(state().lock().map_err(|_|"border clock lock poisoned".to_owned()).and_then(|mut s| {
        if !s.fresh(Instant::now()) || s.snapshot.as_ref().is_none_or(|v|v.session_id!=progress.session_id) {
            return Err("border clock stale or wrong-session".into());
        }
        s.progress=json!({"phase":format!("{:?}",progress.phase),"valid_symbols":progress.valid_frames,
            "required_symbols":progress.required_frames});
        Ok(())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata(epoch: &Epoch, now: Instant) -> Value {
        json!({"enabled":true,"width_fraction":0.16,"optical_clock":epoch.metadata(now)})
    }
    #[test]
    fn clock_uses_exact_shared_symbols_across_wrap_and_clones() {
        let now = Instant::now();
        let epoch = Epoch::new(now);
        for code in [0, 1, 62, 63, 1022, 1023, 2046] {
            let time = now + Duration::from_millis(code * 200 + 99);
            assert_eq!(epoch.code_at(time), code);
            assert_eq!(
                epoch.clone().pixel_at(time),
                temporal::symbol_pixel(code, epoch.tag, 0.12)
            );
            assert_eq!(
                epoch.metadata(time)["symbol"],
                temporal::sign(code, epoch.tag)
            );
        }
    }
    #[test]
    fn unpublished_intent_cannot_start_recovery_and_off_clears_it() {
        let now = Instant::now();
        let epoch = Epoch::new(now);
        let mut s = State::default();
        let mut m = metadata(&epoch, now);
        assert!(!s.fresh(now));
        assert!(!s.presented(&m, [640, 480], 100, 110, now).is_null());
        assert!(s.fresh(now));
        assert!(!s.fresh(now + STALE));
        m["enabled"] = json!(false);
        assert!(s.presented(&m, [640, 480], 120, 130, now).is_null());
        assert!(!s.fresh(now));
    }
    #[test]
    fn missing_symbols_resize_and_epoch_changes_reset_decoder() {
        let now = Instant::now();
        let epoch = Epoch::new(now);
        let mut s = State::default();
        s.presented(&metadata(&epoch, now), [640, 480], 100, 110, now);
        let original = s.snapshot.clone().unwrap().session_id;
        let time = now + Duration::from_millis(10);
        s.presented(&metadata(&epoch, time), [640, 480], 120, 130, time);
        assert_eq!(s.snapshot.as_ref().unwrap().session_id, original);
        let time = now + Duration::from_millis(410);
        s.presented(&metadata(&epoch, time), [640, 480], 140, 150, time);
        assert_ne!(s.snapshot.as_ref().unwrap().session_id, original);
        let id = s.snapshot.as_ref().unwrap().session_id.clone();
        s.presented(&metadata(&epoch, time), [640, 481], 160, 170, time);
        assert_ne!(s.snapshot.as_ref().unwrap().session_id, id);
        let id = s.snapshot.as_ref().unwrap().session_id.clone();
        s.presented(
            &metadata(&Epoch::new(time), time),
            [640, 481],
            180,
            190,
            time,
        );
        assert_ne!(s.snapshot.as_ref().unwrap().session_id, id);
    }
    #[test]
    fn recovery_joins_first_actual_submit_and_never_claims_photon_latency() {
        let now = Instant::now();
        let epoch = Epoch::new(now);
        let mut s = State::default();
        s.presented(&metadata(&epoch, now), [640, 480], 100, 110, now);
        s.presented(&metadata(&epoch, now), [640, 480], 120, 130, now);
        let id = s.snapshot.as_ref().unwrap().session_id.clone();
        let mut report = RecoveryReport {
            session_id: &id,
            sequence: 7,
            host_arrival_unix_ns: 150,
            recovered_code_index: 0,
            score: 0.99,
            confidence_margin: 0.4,
            verified: true,
        };
        s.report(&report, now).unwrap();
        assert_eq!(s.recovery["first_submit_begin_unix_ns"], "100");
        report.recovered_code_index = 1;
        assert!(s.report(&report, now).is_err());
        report.recovered_code_index = 0;
        report.host_arrival_unix_ns = 99;
        assert!(s.report(&report, now).is_err());
        assert!(s.report(&report, now + STALE).is_err());
    }
}
