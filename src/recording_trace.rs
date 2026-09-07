//! Sidecar evidence for RAW bundles: submitted UI targets/gaze and native thumbnails.
//! Neither host clock below is a sensor exposure clock or a measured scan-out time.
use crate::raw_eye_model_protocol::{CameraThumbnailFrame, ModelStreamFrame, ThumbnailKind};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) mod scene;
pub(crate) const SCHEMA: &str = "buttercup-viewer-event-v2";
pub(crate) const SCENE_SCHEMA: &str = "buttercup-scene-v1";
pub(crate) const METADATA_FILE: &str = "metadata.oim1";
const MAX_PENDING_BATCHES: usize = 256;
const MAX_PENDING_NATIVE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Target {
    pub id: String,
    pub role: &'static str,
    pub normalized: (f64, f64),
    pub appearance: Value,
}

impl Target {
    fn json(&self, size: [usize; 2]) -> Value {
        let p = pixels(self.normalized, size);
        json!({"id": self.id, "role": self.role, "normalized": self.normalized,
            "visible": true, "center_px": [p[0].round(), p[1].round()],
            "appearance": self.appearance})
    }
}

fn pixels(point: (f64, f64), size: [usize; 2]) -> [f64; 2] {
    [
        point.0 * size[0].saturating_sub(1) as f64,
        point.1 * size[1].saturating_sub(1) as f64,
    ]
}

/// Mapping input and drawn output are deliberately separate: clamping must not
/// make an off-screen prediction look accurate, and J does not disable evidence.
pub(crate) struct Gaze {
    pub predicted: Option<(f64, f64)>,
    pub drawn: Option<(f64, f64)>,
    pub drawn_source_timestamp_ns: Option<u64>,
    pub source_timestamp_ns: Option<u64>,
    pub source_basis: Value,
    pub roi_frame: Value,
    pub mapping: Value,
    pub held_geometry: bool,
    pub status: &'static str,
}

pub(crate) struct Presentation {
    pub mode: &'static str,
    pub size: [usize; 2],
    pub targets: Vec<Target>,
    pub gaze: Gaze,
    pub display: Value,
    pub scene: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct Stamp {
    unix_ns: String,
    monotonic_ns: String,
}

#[derive(Clone)]
pub(crate) struct Hub(Arc<Mutex<Journal>>);

impl Default for Hub {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Journal {
            origin: Instant::now(),
            session: format!(
                "{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            stream_epoch: 0,
            sources: VecDeque::new(),
            ingress: [Value::Null, Value::Null],
            global_capture: false,
            region: Value::Null,
            config_revision: 0,
            configuration: Value::Null,
            scene_revision: 0,
            scene: Value::Null,
            roi_states: Value::Null,
            metadata_sequence: 0,
            last_live_checkpoint: Instant::now(),
            publishers: Arc::new(Vec::new()),
            next_presentation: 0,
            targets: vec![],
            last_presentation: None,
            last_gaze_source: None,
            thumbnails: [None, None],
            subscribers: vec![],
        })))
    }
}

struct Journal {
    origin: Instant,
    session: String,
    stream_epoch: u64,
    sources: VecDeque<(u32, u64, Value)>,
    ingress: [Value; 2],
    global_capture: bool,
    region: Value,
    config_revision: u64,
    configuration: Value,
    scene_revision: u64,
    scene: Value,
    roi_states: Value,
    metadata_sequence: u64,
    last_live_checkpoint: Instant,
    publishers: Arc<Vec<crate::RawModelPublisher>>,
    next_presentation: u64,
    targets: Vec<Value>,
    last_presentation: Option<Value>,
    last_gaze_source: Option<(Value, u64)>,
    thumbnails: [Option<Arc<CameraThumbnailFrame>>; 2],
    subscribers: Vec<Weak<Mutex<Pending>>>,
}

impl Journal {
    fn checkpoint(&self, event: &str) -> Value {
        let stamp = self.stamp();
        json!({"schema": SCENE_SCHEMA, "event": event, "viewer_session_id": self.session,
            "host_unix_ns": stamp.unix_ns, "host_monotonic_ns": stamp.monotonic_ns,
            "configuration_revision": self.config_revision.to_string(),
            "scene_revision": self.scene_revision.to_string(),
            "configuration": self.configuration, "scene": self.scene,
            "region": self.region, "global_capture_active": self.global_capture,
            "last_presentation_snapshot": self.last_presentation,
            "snapshot_is_new_observation": false})
    }
    fn scene_event(&mut self, event: &str, data: Value) {
        let row = self.scene_row(event, data);
        self.publish(Batch::Scene(Arc::new(vec![row])));
    }
    fn scene_row(&self, event: &str, data: Value) -> Value {
        let stamp = self.stamp();
        json!({"schema": SCENE_SCHEMA,
            "event": event, "viewer_session_id": self.session,
            "host_unix_ns": stamp.unix_ns, "host_monotonic_ns": stamp.monotonic_ns,
            "configuration_revision": self.config_revision.to_string(), "data": data})
    }
    fn stamp(&self) -> Stamp {
        Stamp {
            unix_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .to_string(),
            monotonic_ns: self.origin.elapsed().as_nanos().to_string(),
        }
    }

    fn publish(&mut self, mut batch: Batch) {
        if let Batch::Rows(rows) | Batch::Scene(rows) = &mut batch {
            for row in Arc::make_mut(rows) {
                row["metadata_sequence"] = json!(self.metadata_sequence.to_string());
                self.metadata_sequence += 1;
            }
        }
        if let Batch::Rows(rows) | Batch::Scene(rows) = &batch {
            for row in rows.iter() {
                let metadata = Arc::new(row.clone());
                for publisher in self.publishers.iter() {
                    publisher.submit_packet(ModelStreamFrame::Metadata(Arc::clone(&metadata)));
                }
            }
        }
        // Overflow discards the pending interval as a whole, then restates
        // matching revisions. Never leave retained events pointing at a lost
        // configuration or attach a newer configuration to older samples.
        for weak in std::mem::take(&mut self.subscribers) {
            let Some(queue) = weak.upgrade() else {
                continue;
            };
            let mut pending = queue.lock().unwrap_or_else(|e| e.into_inner());
            if pending.closed {
                continue;
            }
            if pending.push(batch.clone()) {
                pending.push(Batch::Scene(Arc::new(vec![
                    self.checkpoint("queue_recovery_snapshot"),
                ])));
            }
            self.subscribers.push(weak);
        }
    }
}

impl Hub {
    pub fn attach_publishers(&self, publishers: Arc<Vec<crate::RawModelPublisher>>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).publishers = publishers;
    }
    pub fn stamp_json(&self) -> Value {
        let journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let stamp = journal.stamp();
        json!({"viewer_session_id": journal.session, "host_unix_ns": stamp.unix_ns,
            "host_monotonic_ns": stamp.monotonic_ns})
    }

    pub fn start_stream(&self) -> String {
        let mut journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        journal.stream_epoch += 1;
        let epoch = format!("{}:{}", journal.session, journal.stream_epoch);
        journal.region = Value::Null;
        journal.ingress = [Value::Null, Value::Null];
        journal.scene_event(
            "stream_started",
            json!({"stream_epoch": epoch,
            "camera_clock_continuity_across_reconnect": "unknown"}),
        );
        epoch
    }

    pub fn raw_arrived(&self, mut key: Value, arrived: Instant, unix_ns: u64) -> Value {
        let mut journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        key["viewer_session_id"] = json!(journal.session);
        let clock = json!({"source_key": key,
            "host_arrival_unix_ns": unix_ns.to_string(),
            "host_arrival_monotonic_ns": arrived.saturating_duration_since(journal.origin).as_nanos().to_string()});
        if let (Some(eye), Some(source)) = (
            key["roi_id"].as_u64(),
            key["sensor_timestamp_ns"]
                .as_str()
                .and_then(|s| s.parse::<u64>().ok()),
        ) {
            if !journal
                .sources
                .iter()
                .any(|(_, _, previous)| previous["source_key"] == key)
            {
                journal
                    .sources
                    .push_back((eye as u32, source, clock.clone()));
                while journal.sources.len() > 2048 {
                    journal.sources.pop_front();
                }
            }
            if (1..=2).contains(&eye) {
                journal.ingress[eye as usize - 1] = clock.clone();
            }
        }
        clock
    }

    pub fn source_reference(&self, eye: u32, timestamp_ns: Option<u64>) -> Value {
        let Some(timestamp_ns) = timestamp_ns else {
            return json!({"key": null, "status": "source-time-unavailable"});
        };
        let journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut matches = journal
            .sources
            .iter()
            .filter(|(roi, time, _)| *roi == eye && *time == timestamp_ns);
        let first = matches.next();
        if matches.next().is_some() {
            return json!({"key": null, "roi_id": eye,
            "sensor_timestamp_ns": timestamp_ns.to_string(), "status": "ambiguous-clock-epoch"});
        }
        match first {
            Some((_, _, clock)) => {
                json!({"key": clock["source_key"], "clock": clock, "status": "exact-source-key",
                "archive_scope": "resolve-in-frames-index-or-pre-recording-history"})
            }
            None => {
                json!({"key": null, "roi_id": eye, "sensor_timestamp_ns": timestamp_ns.to_string(), "status": "outside-bounded-source-history"})
            }
        }
    }

    /// Liveness of an exact RAW source for desktop mouse output, even outside
    /// recording. Do not substitute the newest frame or the solve's ready time.
    pub fn source_arrival_age(&self, eye: u32, timestamp_ns: u64) -> Option<Duration> {
        let journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut matches = journal.sources.iter()
            .filter(|(roi, time, _)| *roi == eye && *time == timestamp_ns);
        let (_, _, clock) = matches.next()?;
        if matches.next().is_some() { return None; }
        let elapsed_ns: u64 = clock["host_arrival_monotonic_ns"].as_str()?.parse().ok()?;
        journal.origin.elapsed().checked_sub(Duration::from_nanos(elapsed_ns))
    }

    pub fn region(&self, region: Value) {
        let mut journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if journal.region != region {
            journal.region = region.clone();
            journal.scene_event("region_changed", region);
        }
    }

    pub fn dropped_source(&self, clock: Value, reason: &str) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .scene_event(
                "source_dropped",
                json!({"clock": clock, "reason": reason, "archived": false}),
            );
    }

    pub fn capture_scope(&self) -> CaptureScope {
        let mut journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        journal.global_capture = true;
        journal.scene_event(
            "capture_phase_changed",
            json!({"global_capture_active": true,
            "reason": "global-thumbnail-camera-transaction"}),
        );
        CaptureScope(self.clone())
    }

    pub fn transport_state(&self) -> Value {
        let journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        json!({"global_capture_active": journal.global_capture, "region": journal.region,
            "latest_ingress": journal.ingress})
    }

    pub fn stamp(&self) -> Stamp {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).stamp()
    }

    /// Call only after buffer.present() succeeds. The two stamps bound the
    /// host submission call, not compositor presentation or physical photons.
    pub fn presented(&self, presentation: Presentation, before_submit: Stamp) {
        let mut journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let after = journal.stamp();
        let id = journal.next_presentation;
        let mut scene_changes = vec![];
        journal.next_presentation = journal.next_presentation.saturating_add(1);
        let configuration = json!({"display": presentation.display, "mapping": presentation.gaze.mapping,
            "gaze_basis": presentation.gaze.source_basis,
            "geometry": presentation.scene["geometry"], "calibration": presentation.scene["calibration"]});
        if configuration != journal.configuration {
            journal.config_revision += 1;
            journal.configuration = configuration.clone();
            scene_changes.push(journal.scene_row("configuration_changed", configuration));
        }
        let roi_states = presentation.scene["roi_states"].clone();
        if roi_states != journal.roi_states {
            let previous = std::mem::replace(&mut journal.roi_states, roi_states.clone());
            scene_changes.push(journal.scene_row(
                "roi_state_changed",
                json!({"previous": previous, "current": roi_states,
                "timing": "host-observed-state-transition; previous-interval-ends-here"}),
            ));
        }
        let scene = json!({"eyes": presentation.scene["eyes"], "roi_states": roi_states,
            "fused": presentation.scene.get("fused").cloned().unwrap_or_else(||json!({"target":null,"status":"joint-conics-unavailable"})),
            "configuration_revision": journal.config_revision.to_string()});
        if scene != journal.scene {
            journal.scene_revision += 1;
            journal.scene = scene.clone();
            let revision = journal.scene_revision.to_string();
            scene_changes.push(journal.scene_row(
                "scene_sample",
                json!({"scene_revision": revision, "sample": scene}),
            ));
        }
        let targets: Vec<_> = presentation
            .targets
            .iter()
            .map(|t| t.json(presentation.size))
            .collect();
        let gaze = presentation.gaze;
        let finite = |p: &(f64, f64)| p.0.is_finite() && p.1.is_finite();
        let predicted = gaze.predicted.filter(finite);
        let drawn = gaze
            .drawn
            .filter(finite)
            .map(|p| (p.0.clamp(0.0, 1.0), p.1.clamp(0.0, 1.0)));
        let mut source_advanced = false;
        if !gaze.held_geometry && predicted.is_some() {
            if let Some(source) = gaze.source_timestamp_ns {
                source_advanced = journal
                    .last_gaze_source
                    .as_ref()
                    .is_none_or(|(basis, last)| basis != &gaze.source_basis || source > *last);
                if source_advanced {
                    journal.last_gaze_source = Some((gaze.source_basis.clone(), source));
                }
            }
        }
        let base = json!({"schema": SCHEMA, "viewer_session_id": journal.session, "presentation_id": id.to_string(),
            "configuration_revision": journal.config_revision.to_string(), "scene_revision": journal.scene_revision.to_string(),
            "host_submit_begin_unix_ns": before_submit.unix_ns,
            "host_submit_begin_monotonic_ns": before_submit.monotonic_ns,
            "host_submit_end_unix_ns": after.unix_ns,
            "host_submit_end_monotonic_ns": after.monotonic_ns,
            "timing_semantics": "successful-host-buffer-submit-not-scanout",
            "mode": presentation.mode, "viewport_px": presentation.size});
        let mut rows = vec![];
        // Same ID at a different pixel position (including resize) is a remove
        // followed by an add. Spinner animation alone is not a target change.
        for target in &journal.targets {
            if !targets
                .iter()
                .any(|other| same_target_placement(target, other))
            {
                let mut row = base.clone();
                row["event"] = json!("target_removed");
                row["target"] = target.clone();
                row["target"]["visible"] = json!(false);
                rows.push(row);
            }
        }
        for target in &targets {
            if !journal
                .targets
                .iter()
                .any(|other| same_target_placement(target, other))
            {
                let mut row = base.clone();
                row["event"] = json!("target_added");
                row["target"] = target.clone();
                rows.push(row);
            }
        }
        let mut row = base;
        row["event"] = json!("presentation");
        row["active_targets"] = json!(targets);
        row["gaze"] = json!({
            "predicted_normalized": predicted,
            "predicted_px": predicted.map(|p| pixels(p, presentation.size)),
            "drawn_normalized": drawn,
            "drawn_center_px": drawn.map(|p| pixels(p, presentation.size).map(f64::round)),
            "cursor_drawn": drawn.is_some(),
            "drawn_source_timestamp_ns": drawn.and(gaze.drawn_source_timestamp_ns).map(|v| v.to_string()),
            "prediction_offscreen": predicted.map(|p| p.0 < 0.0 || p.0 > 1.0 || p.1 < 0.0 || p.1 > 1.0),
            "source_timestamp_ns": gaze.source_timestamp_ns.map(|v| v.to_string()),
            "source_advanced_for_basis": source_advanced,
            "held_geometry": gaze.held_geometry,
            "status": gaze.status, "source_basis": gaze.source_basis,
            "roi_frame_key": gaze.roi_frame, "mapping_revision": journal.config_revision.to_string(),
            "output_kind": "selected-eye-cursor-not-fused",
        });
        journal.targets = targets;
        journal.last_presentation = Some(row.clone());
        rows.push(row);
        // Commit all state before publishing: overflow during a configuration
        // change must checkpoint matching configuration, scene and target IDs.
        if !scene_changes.is_empty() {
            journal.publish(Batch::Scene(Arc::new(scene_changes)));
        }
        journal.publish(Batch::Rows(Arc::new(rows)));
        // A late/reconnecting live consumer recovers within one second.
        // Until then, unresolved revision references stay unavailable. This
        // checkpoint is explicitly a restatement, never a fresh observation.
        if !journal.publishers.is_empty()
            && journal.last_live_checkpoint.elapsed().as_secs_f64() >= 1.0
        {
            journal.last_live_checkpoint = Instant::now();
            let checkpoint =
                ModelStreamFrame::Metadata(Arc::new(journal.checkpoint("live_checkpoint")));
            for publisher in journal.publishers.iter() {
                publisher.submit_packet(checkpoint.clone());
            }
        }
    }

    pub fn thumbnail(&self, frame: Arc<CameraThumbnailFrame>) {
        let mut journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let index = usize::from(frame.kind == ThumbnailKind::SensorBand);
        journal.thumbnails[index] = Some(Arc::clone(&frame));
        journal.publish(Batch::Thumbnail(frame, false));
    }

    fn subscribe(&self) -> Subscription {
        let mut journal = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let stamp = journal.stamp();
        let mut pending = Pending::default();
        pending.push(Batch::Rows(Arc::new(vec![json!({
            "schema": SCHEMA, "event": "recording_started",
            "host_unix_ns": stamp.unix_ns, "host_monotonic_ns": stamp.monotonic_ns,
            "active_targets": journal.targets,
            "last_presentation_snapshot": journal.last_presentation,
            "snapshot_is_new_observation": false,
            "viewer_session_id": journal.session,
        })])));
        pending.push(Batch::Scene(Arc::new(vec![
            journal.checkpoint("recording_start_snapshot"),
        ])));
        for frame in journal.thumbnails.iter().flatten() {
            pending.push(Batch::Thumbnail(Arc::clone(frame), true));
        }
        let queue = Arc::new(Mutex::new(pending));
        journal.subscribers.retain(|weak| weak.strong_count() != 0);
        journal.subscribers.push(Arc::downgrade(&queue));
        Subscription {
            queue,
            hub: self.clone(),
        }
    }
}

#[derive(Clone)]
enum Batch {
    Rows(Arc<Vec<Value>>),
    Scene(Arc<Vec<Value>>),
    Thumbnail(Arc<CameraThumbnailFrame>, bool),
}
impl Batch {
    fn native_bytes(&self) -> usize {
        match self {
            Self::Thumbnail(frame, _) => frame.payload.len(),
            Self::Rows(_) | Self::Scene(_) => 0,
        }
    }
}

#[derive(Default)]
struct Pending {
    batches: VecDeque<Batch>,
    native_bytes: usize,
    dropped_metadata_batches: u64,
    dropped_thumbnails: u64,
    closed: bool,
}
impl Pending {
    fn push(&mut self, batch: Batch) -> bool {
        self.native_bytes += batch.native_bytes();
        self.batches.push_back(batch);
        if self.batches.len() > MAX_PENDING_BATCHES || self.native_bytes > MAX_PENDING_NATIVE_BYTES
        {
            for removed in self.batches.drain(..) {
                match removed {
                    Batch::Rows(_) | Batch::Scene(_) => self.dropped_metadata_batches += 1,
                    Batch::Thumbnail(..) => self.dropped_thumbnails += 1,
                }
            }
            self.native_bytes = 0;
            return true;
        }
        false
    }
}

struct Subscription {
    queue: Arc<Mutex<Pending>>,
    hub: Hub,
}
impl Subscription {
    fn take(&self, close: bool) -> Pending {
        let mut pending = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let result = std::mem::take(&mut *pending);
        pending.closed = result.closed || close;
        result
    }
}

/// The receiver owns disk I/O; UI and capture workers only enqueue bounded
/// in-memory batches. Every recorder subscribes independently and detaches
/// atomically before final draining, including error/shutdown finalization.
pub(crate) struct BundleTrace {
    subscription: Subscription,
    metadata: BufWriter<File>,
    thumbnail_index: BufWriter<File>,
    thumbnails: File,
}
impl BundleTrace {
    pub fn new(directory: &Path, hub: &Hub) -> Result<Self, String> {
        let open = |name| File::create(directory.join(name)).map_err(|e| e.to_string());
        let mut trace = Self {
            metadata: BufWriter::new(open(METADATA_FILE)?),
            thumbnail_index: BufWriter::new(open("thumbnails.jsonl")?),
            thumbnails: open("thumbnails.oic1")?,
            subscription: hub.subscribe(),
        };
        // Preserve the true recording boundary before any possible queue
        // overflow; a later recovery checkpoint cannot recreate that time.
        trace.drain(false)?;
        Ok(trace)
    }

    pub fn drain(&mut self, finish: bool) -> Result<(), String> {
        self.drain_with_reason(finish.then_some("requested-stop"))
    }

    pub fn drain_with_reason(&mut self, stop_reason: Option<&str>) -> Result<(), String> {
        let finish = stop_reason.is_some();
        let pending = self.subscription.take(finish);
        if pending.closed {
            return Ok(());
        }
        let stop_stamp = finish.then(|| self.subscription.hub.stamp());
        if pending.dropped_metadata_batches != 0 || pending.dropped_thumbnails != 0 {
            write_metadata(
                &mut self.metadata,
                &json!({"schema": SCHEMA, "event": "queue_gap",
                "dropped_metadata_batches": pending.dropped_metadata_batches,
                "dropped_thumbnails": pending.dropped_thumbnails,
                "policy": "pending-interval-dropped; queue_recovery_snapshot-restores-matching-revisions"}),
            )?;
        }
        for batch in pending.batches {
            match batch {
                Batch::Rows(rows) | Batch::Scene(rows) => {
                    for row in rows.iter() {
                        write_metadata(&mut self.metadata, row)?;
                    }
                }
                Batch::Thumbnail(frame, snapshot) => {
                    let offset = self
                        .thumbnails
                        .stream_position()
                        .map_err(|e| e.to_string())?;
                    frame.write_to(&mut self.thumbnails)?;
                    let end = self
                        .thumbnails
                        .stream_position()
                        .map_err(|e| e.to_string())?;
                    write_row(
                        &mut self.thumbnail_index,
                        &json!({
                            "schema": "buttercup-recorded-thumbnail-v1", "stream": "thumbnails.oic1",
                            "offset": offset, "length": end - offset,
                            "recording_start_snapshot": snapshot,
                            "camera": frame.metadata(),
                        }),
                    )?;
                }
            }
        }
        if let Some(stamp) = stop_stamp {
            write_metadata(
                &mut self.metadata,
                &json!({"schema": SCHEMA, "event": "recording_stopped", "sidecars_finalized": true,
                "stop_reason": stop_reason, "interrupted": stop_reason == Some("recorder-dropped-before-requested-stop"),
                "host_unix_ns": stamp.unix_ns, "host_monotonic_ns": stamp.monotonic_ns,
                "target_visibility_after_stop": "not-observed"}),
            )?;
            self.metadata
                .flush()
                .and_then(|()| self.thumbnail_index.flush())
                .and_then(|()| self.thumbnails.flush())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

fn same_target_placement(a: &Value, b: &Value) -> bool {
    ["id", "role", "normalized", "center_px", "visible"]
        .iter()
        .all(|key| a[*key] == b[*key])
}

pub(crate) struct CaptureScope(Hub);
impl Drop for CaptureScope {
    fn drop(&mut self) {
        let mut journal = self.0.0.lock().unwrap_or_else(|e| e.into_inner());
        journal.global_capture = false;
        journal.scene_event(
            "capture_phase_changed",
            json!({"global_capture_active": false,
            "reason": "global-thumbnail-camera-transaction-ended"}),
        );
    }
}

fn write_row(output: &mut impl Write, row: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *output, row).map_err(|e| e.to_string())?;
    output.write_all(b"\n").map_err(|e| e.to_string())
}

fn write_metadata(output: &mut impl Write, row: &Value) -> Result<(), String> {
    ModelStreamFrame::Metadata(Arc::new(row.clone())).write_to(output)
}

#[cfg(test)]
pub(crate) fn decode_metadata(mut bytes: &[u8]) -> Vec<Value> {
    let mut rows = vec![];
    while !bytes.is_empty() {
        let ModelStreamFrame::Metadata(value) = ModelStreamFrame::read_from(&mut bytes).unwrap()
        else {
            panic!("expected OIM1");
        };
        rows.push((*value).clone());
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_eye_model_protocol::ModelStreamFrame;

    fn presentation(target: Option<usize>, source: u64) -> Presentation {
        Presentation {
            mode: "test",
            size: [1001, 501],
            targets: target
                .map(|i| Target {
                    id: format!("target-{i}"),
                    role: "accuracy",
                    normalized: (0.4 + i as f64 * 0.1, 0.6),
                    appearance: json!({"style":"test"}),
                })
                .into_iter()
                .collect(),
            gaze: Gaze {
                predicted: Some((0.3, 1.2)),
                drawn: Some((0.3, 1.2)),
                drawn_source_timestamp_ns: Some(source),
                source_timestamp_ns: Some(source),
                source_basis: json!({"eye": 1, "authority": "SAM", "epoch": 1}),
                roi_frame: json!({"roi_id": 1, "sequence": "42", "sensor_timestamp_ns": "900000000"}),
                mapping: json!({"kind": "calibrated-affine-3d-gate"}),
                held_geometry: false,
                status: "predicted",
            },
            display: json!({"id":"test"}),
            scene: json!({"geometry":{"units":"inches"},"eyes":[],"roi_states":[]}),
        }
    }

    fn rows(subscription: &Subscription, close: bool) -> Vec<Value> {
        subscription
            .take(close)
            .batches
            .into_iter()
            .flat_map(|b| match b {
                Batch::Rows(rows) => rows.as_ref().clone(),
                _ => vec![],
            })
            .collect()
    }

    #[test]
    fn source_references_are_exact_per_roi_and_never_nearest_future_or_cross_epoch() {
        let hub = Hub::default();
        let register = |roi, time: u64, epoch| {
            hub.raw_arrived(
                json!({"roi_id":roi,
            "sensor_timestamp_ns":time.to_string(), "sequence":"7", "stream_epoch":epoch}),
                Instant::now(),
                12345,
            )
        };
        let time = 1u64 << 60;
        let first = register(1, time, "a");
        register(2, time, "a");
        register(1, time + 10, "a");
        assert_eq!(
            hub.source_reference(1, Some(time))["key"],
            first["source_key"]
        );
        assert_eq!(hub.source_reference(2, Some(time))["key"]["roi_id"], 2);
        assert_eq!(
            hub.source_reference(1, Some(time + 1))["status"],
            "outside-bounded-source-history"
        );
        register(1, time, "a"); // duplicate same buffer is not a second exposure
        assert_eq!(
            hub.source_reference(1, Some(time))["status"],
            "exact-source-key"
        );
        register(1, time, "b"); // equal clock after reconnect is NOT proven identical
        assert_eq!(
            hub.source_reference(1, Some(time))["status"],
            "ambiguous-clock-epoch"
        );
        for offset in 20..2080 {
            register(1, time + offset, "b");
        }
        assert_eq!(
            hub.source_reference(1, Some(time + 10))["status"],
            "outside-bounded-source-history"
        );
        assert_eq!(hub.0.lock().unwrap().sources.len(), 2048);
    }

    #[test]
    fn configuration_and_scene_are_change_only_and_spinner_is_not_a_new_target() {
        let hub = Hub::default();
        let subscription = hub.subscribe();
        subscription.take(false);
        let mut first = presentation(Some(0), 100);
        first.scene["eyes"] =
            json!([{"roi_id":1,"analysis_source":{"key":{"sensor_timestamp_ns":"100"}}}]);
        hub.presented(first, hub.stamp());
        let mut again = presentation(Some(0), 100);
        again.scene["eyes"] =
            json!([{"roi_id":1,"analysis_source":{"key":{"sensor_timestamp_ns":"100"}}}]);
        again.targets[0].appearance["phase_rad"] = json!(0.7);
        hub.presented(again, hub.stamp());
        let events: Vec<Value> = subscription
            .take(false)
            .batches
            .into_iter()
            .flat_map(|b| match b {
                Batch::Scene(v) | Batch::Rows(v) => v.as_ref().clone(),
                _ => vec![],
            })
            .collect();
        for event in ["configuration_changed", "scene_sample", "target_added"] {
            assert_eq!(
                events.iter().filter(|e| e["event"] == event).count(),
                1,
                "{event}"
            );
        }
        let presentations: Vec<_> = events
            .iter()
            .filter(|e| e["event"] == "presentation")
            .collect();
        assert_eq!(presentations.len(), 2);
        assert_eq!(
            presentations[1]["active_targets"][0]["appearance"]["phase_rad"],
            0.7
        );
        assert_eq!(
            presentations[0]["configuration_revision"],
            presentations[1]["configuration_revision"]
        );
        assert_eq!(
            presentations[0]["scene_revision"],
            presentations[1]["scene_revision"]
        );
        assert_eq!(presentations[1]["gaze"]["source_advanced_for_basis"], false);
        let snapshot = hub.0.lock().unwrap().checkpoint("test");
        assert_eq!(
            snapshot["scene_revision"],
            presentations[1]["scene_revision"]
        );
        assert_eq!(snapshot["snapshot_is_new_observation"], false);
    }

    #[test]
    fn targets_are_added_removed_once_per_actual_set_change_and_restart_is_a_snapshot() {
        let hub = Hub::default();
        hub.presented(presentation(Some(0), 100), hub.stamp());
        let first = hub.subscribe();
        let initial = rows(&first, false);
        assert_eq!(initial[0]["event"], "recording_started");
        assert_eq!(initial[0]["active_targets"][0]["id"], "target-0");
        assert_eq!(initial[0]["snapshot_is_new_observation"], false);
        hub.presented(presentation(Some(0), 101), hub.stamp());
        hub.presented(presentation(Some(1), 102), hub.stamp());
        hub.presented(presentation(None, 103), hub.stamp());
        let events = rows(&first, true);
        let kinds: Vec<_> = events
            .iter()
            .map(|v| v["event"].as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            [
                "presentation",
                "target_removed",
                "target_added",
                "presentation",
                "target_removed",
                "presentation"
            ]
        );
        assert_eq!(events[1]["target"]["center_px"], json!([400.0, 300.0]));
        assert_eq!(events[2]["target"]["center_px"], json!([500.0, 300.0]));
        assert_eq!(events[1]["presentation_id"], events[2]["presentation_id"]);
        hub.presented(presentation(Some(0), 104), hub.stamp());
        assert!(
            rows(&first, false).is_empty(),
            "closed recording must not admit later events"
        );
        let second = hub.subscribe();
        let restart = rows(&second, false);
        assert_eq!(restart.len(), 1);
        assert_eq!(restart[0]["active_targets"][0]["id"], "target-0");
    }

    #[test]
    fn source_time_is_not_redraw_time_and_offscreen_predictions_are_not_clamped() {
        let hub = Hub::default();
        let subscription = hub.subscribe();
        rows(&subscription, false);
        let source = (1u64 << 55) + 123;
        for (i, clock) in [source, source, source - 1, source + 1]
            .into_iter()
            .enumerate()
        {
            let mut p = presentation(None, clock);
            if i == 3 {
                p.gaze.held_geometry = true;
            }
            hub.presented(p, hub.stamp());
        }
        let events = rows(&subscription, false);
        assert_eq!(
            events
                .iter()
                .map(|v| v["gaze"]["source_advanced_for_basis"].as_bool().unwrap())
                .collect::<Vec<_>>(),
            [true, false, false, false]
        );
        assert_eq!(events[0]["gaze"]["source_timestamp_ns"], source.to_string());
        assert_eq!(events[0]["gaze"]["predicted_px"], json!([300.0, 600.0]));
        assert_eq!(events[0]["gaze"]["drawn_center_px"], json!([300.0, 500.0]));
        assert_eq!(events[0]["gaze"]["prediction_offscreen"], true);
        for event in &events {
            let ns = |key| event[key].as_str().unwrap().parse::<u128>().unwrap();
            assert!(ns("host_submit_end_monotonic_ns") >= ns("host_submit_begin_monotonic_ns"));
            assert_ne!(
                event["host_submit_end_unix_ns"],
                event["gaze"]["source_timestamp_ns"]
            );
        }
        let mut hidden = presentation(None, source + 2);
        hidden.gaze.drawn = None;
        hidden.gaze.drawn_source_timestamp_ns = None;
        hub.presented(hidden, hub.stamp());
        let event = rows(&subscription, false).pop().unwrap();
        assert_eq!(event["gaze"]["cursor_drawn"], false);
        assert_eq!(event["gaze"]["predicted_normalized"], json!([0.3, 1.2]));
    }

    #[test]
    fn resize_changes_pixel_targets_and_missing_gaze_cannot_relabel_a_repeat_fresh() {
        let hub = Hub::default();
        let subscription = hub.subscribe();
        rows(&subscription, false);
        hub.presented(presentation(Some(0), 44), hub.stamp());
        rows(&subscription, false);
        let mut p = presentation(Some(0), 44);
        p.size = [2001, 1001];
        p.gaze.predicted = None;
        p.gaze.drawn = None;
        hub.presented(p, hub.stamp());
        let changed = rows(&subscription, false);
        assert_eq!(changed.len(), 3);
        assert_eq!(changed[0]["event"], "target_removed");
        assert_eq!(changed[1]["target"]["center_px"], json!([800.0, 600.0]));
        assert!(changed[2]["gaze"]["predicted_normalized"].is_null());
        hub.presented(presentation(Some(0), 44), hub.stamp());
        let again = rows(&subscription, false).pop().unwrap();
        assert_eq!(again["gaze"]["source_advanced_for_basis"], false);
    }

    fn thumbnail(kind: ThumbnailKind, clock: u64) -> Arc<CameraThumbnailFrame> {
        let mut header = [0u8; 64];
        header[..4].copy_from_slice(b"OTH1");
        header[4..6].copy_from_slice(&1u16.to_le_bytes());
        header[6..8].copy_from_slice(&64u16.to_le_bytes());
        for (offset, value) in [(12, 3u32), (40, 4), (44, 1), (48, 8), (52, 8), (56, 16)] {
            header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        header[16..24].copy_from_slice(&2u64.to_le_bytes());
        header[24..32].copy_from_slice(&clock.to_le_bytes());
        Arc::new(CameraThumbnailFrame {
            kind,
            sensor_size_px: [8000, 6000],
            sensor_rect_px: [
                0,
                0,
                8000,
                if kind == ThumbnailKind::GlobalSensor {
                    6000
                } else {
                    576
                },
            ],
            host_received_unix_ns: 2000000,
            region_session: None,
            region_generation: None,
            camera_header: header,
            payload: Arc::new(vec![0, 1, 3, 2, 5, 4, 7, 6]),
        })
    }

    #[test]
    fn disk_trace_keeps_native_payloads_clocks_snapshot_provenance_and_final_pending_events() {
        let hub = Hub::default();
        let global = thumbnail(ThumbnailKind::GlobalSensor, 100);
        hub.thumbnail(Arc::clone(&global));
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::path::PathBuf::from(format!(
            "outputs/recording-trace-tests/{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let mut trace = BundleTrace::new(&directory, &hub).unwrap();
        let band = thumbnail(ThumbnailKind::SensorBand, 150);
        hub.thumbnail(Arc::clone(&band));
        hub.presented(presentation(Some(0), 150), hub.stamp());
        trace.drain(false).unwrap();
        hub.presented(presentation(None, 150), hub.stamp());
        trace.drain(true).unwrap();
        hub.presented(presentation(Some(1), 200), hub.stamp());
        trace.drain(true).unwrap();
        let read_rows = |name| {
            std::fs::read_to_string(directory.join(name))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>()
        };
        let index = read_rows("thumbnails.jsonl");
        assert_eq!(index.len(), 2);
        assert_eq!(index[0]["recording_start_snapshot"], true);
        assert_eq!(index[1]["recording_start_snapshot"], false);
        let bytes = std::fs::read(directory.join("thumbnails.oic1")).unwrap();
        for (entry, expected) in index.iter().zip([global, band]) {
            let start = entry["offset"].as_u64().unwrap() as usize;
            let length = entry["length"].as_u64().unwrap() as usize;
            let mut input = &bytes[start..start + length];
            let ModelStreamFrame::Thumbnail(frame) =
                ModelStreamFrame::read_from(&mut input).unwrap()
            else {
                panic!("native thumbnail");
            };
            assert!(input.is_empty());
            assert_eq!(frame.camera_header, expected.camera_header);
            assert_eq!(frame.payload, expected.payload);
            assert_eq!(frame.metadata(), entry["camera"]);
        }
        let bytes = std::fs::read(directory.join(METADATA_FILE)).unwrap();
        let events = decode_metadata(&bytes);
        assert!(!directory.join("viewer-events.jsonl").exists());
        assert_eq!(events.last().unwrap()["event"], "recording_stopped");
        assert_eq!(
            events
                .iter()
                .filter(|v| v["event"] == "target_removed")
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|v| v["event"] == "target_added")
                .count(),
            1
        );
        drop(trace);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn overflow_during_mapping_change_checkpoints_one_consistent_revision() {
        let hub = Hub::default();
        let subscription = hub.subscribe();
        rows(&subscription, false);
        hub.presented(presentation(Some(0), 1), hub.stamp());
        rows(&subscription, false);
        for source in 2..MAX_PENDING_BATCHES + 2 {
            hub.presented(presentation(Some(0), source as u64), hub.stamp());
        }
        let mut changed = presentation(Some(1), 900);
        changed.gaze.mapping = json!({"kind":"replacement-plane"});
        hub.presented(changed, hub.stamp());
        let pending = subscription.take(false);
        let Batch::Scene(events) = pending.batches.front().unwrap() else {
            panic!();
        };
        let snapshot = &events[0];
        assert_eq!(snapshot["event"], "queue_recovery_snapshot");
        assert_eq!(snapshot["configuration_revision"], "2");
        assert_eq!(
            snapshot["scene"]["configuration_revision"],
            snapshot["configuration_revision"]
        );
        assert_eq!(
            snapshot["last_presentation_snapshot"]["configuration_revision"],
            snapshot["configuration_revision"]
        );
        assert_eq!(
            snapshot["last_presentation_snapshot"]["scene_revision"],
            snapshot["scene_revision"]
        );
        assert_eq!(
            snapshot["last_presentation_snapshot"]["active_targets"][0]["id"],
            "target-1"
        );
    }

    #[test]
    fn mouse_output_age_uses_exact_source_and_rejects_ambiguous_stream_epochs() {
        let hub = Hub::default();
        let origin = Instant::now() - Duration::from_secs(10);
        hub.0.lock().unwrap().origin = origin;
        let key = |roi, source, epoch| json!({"roi_id":roi, "sensor_timestamp_ns":source,
            "stream_epoch":epoch});
        hub.raw_arrived(key(1, "100", "a"), origin + Duration::from_secs(3), 1);
        hub.raw_arrived(key(1, "200", "a"), origin + Duration::from_secs(9), 2);
        let age = hub.source_arrival_age(1, 100).unwrap();
        assert!(age >= Duration::from_secs(7));
        assert!(age < Duration::from_secs(8));
        assert!(hub.source_arrival_age(1, 200).unwrap() < Duration::from_secs(2));
        assert!(hub.source_arrival_age(2, 100).is_none());
        assert!(hub.source_arrival_age(1, 300).is_none());
        // Re-publication does not refresh the source's arrival clock.
        hub.raw_arrived(key(1, "100", "a"), origin + Duration::from_secs(9), 3);
        assert!(hub.source_arrival_age(1, 100).unwrap() >= Duration::from_secs(7));
        hub.raw_arrived(key(1, "100", "b"), origin + Duration::from_secs(9), 4);
        assert!(hub.source_arrival_age(1, 100).is_none());
    }

    #[test]
    fn overload_is_bounded_and_counted_with_latest_state_retained() {
        let hub = Hub::default();
        let subscription = hub.subscribe();
        rows(&subscription, false);
        for source in 0..(MAX_PENDING_BATCHES + 7) {
            hub.presented(presentation(Some(source % 2), source as u64), hub.stamp());
        }
        let pending = subscription.take(false);
        assert!(pending.batches.len() <= MAX_PENDING_BATCHES);
        assert!(pending.dropped_metadata_batches >= MAX_PENDING_BATCHES as u64);
        let Batch::Scene(checkpoint) = pending.batches.front().unwrap() else {
            panic!();
        };
        assert_eq!(checkpoint[0]["event"], "queue_recovery_snapshot");
        assert_eq!(checkpoint[0]["configuration_revision"], "1");
        assert_eq!(checkpoint[0]["snapshot_is_new_observation"], false);
        let Batch::Rows(rows) = pending.batches.back().unwrap() else {
            panic!();
        };
        assert_eq!(
            rows.last().unwrap()["gaze"]["source_timestamp_ns"],
            (MAX_PENDING_BATCHES + 6).to_string()
        );
        let mut pending = Pending::default();
        let mut frame = (*thumbnail(ThumbnailKind::GlobalSensor, 1)).clone();
        frame.payload = Arc::new(vec![0; MAX_PENDING_NATIVE_BYTES / 2 + 1]);
        let frame = Arc::new(frame);
        pending.push(Batch::Thumbnail(Arc::clone(&frame), false));
        pending.push(Batch::Thumbnail(frame, false));
        assert!(pending.native_bytes <= MAX_PENDING_NATIVE_BYTES);
        assert_eq!(pending.dropped_thumbnails, 2);
    }
}
