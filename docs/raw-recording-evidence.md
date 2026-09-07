# RAW recording evidence

`S` toggles a RAW bundle (`H` is an alias), including during the 20-target
accuracy check. Timed and automatically armed calibration captures use the same
writer. No new JSONL stream or compression is added for scene/coordinate data.

| Member | Contents |
| --- | --- |
| `metadata.oim1` | Framed target/gaze events and change-only scene/configuration records |
| `thumbnails.oic1` | Original global/sensor-band camera packets, without re-encoding |
| `thumbnails.jsonl` | Existing native-thumbnail byte index and snapshot provenance |

`manifest.json` references these members and schemas. Existing ROI files remain
`subject-right.raw10`, `subject-left.raw10`, `frames.jsonl`, `predictions.jsonl`
and `recovery.jsonl`. The newly introduced v1 `viewer-events.jsonl` is replaced
by OIM1 for future recordings; old archives are not retrofitted. Temporary bundle
staging also lives under the checked `outputs` link.

## Binary framing

The socket `--model-stream PATH` multiplexes three record types. ROI `OIR1`
retains its byte-for-byte 96-byte header and native payload. `OIC1` and `OIM1`
share this little-endian envelope:

| Offset | Type | OIC1 | OIM1 |
| --- | --- | --- | --- |
| 0 | 4 bytes | `OIC1` | `OIM1` |
| 4 | u16 | Version 1 | Version 1 |
| 6 | u16 | Prefix length 24 | Prefix length 24 |
| 8 | u32 | JSON length, 1–16 KiB | JSON object length, 1–1 MiB |
| 12 | u32 | Native header length, 64 | Zero |
| 16 | u32 | Native payload length, 1–64 MiB | Zero |
| 20 | u32 | Reserved, zero | Reserved, zero |

Follow the prefix with UTF-8 JSON, native header and native payload, in that
order. OIM1 contains only the length-delimited JSON object: no newline, delimiter,
gzip or image copy. Concatenating the exact OIM1 wire records gives
`metadata.oim1`. Use `ModelStreamFrame::read_from` for typed decoding; the updated
eye-only `RawModelFrame::read_from` skips other types. Already-compiled external
OIR1-only readers must be upgraded to decode/skip OIC1 and OIM1.

## Targets and gaze: buttercup-viewer-event-v2

Each successful framebuffer submission produces a `presentation`: mode,
viewport size, complete active targets, gaze, host-submit begin/end clocks,
configuration and scene revisions. Target transitions belong to that successful
submission, not a planned future step. Failed submissions produce no success
event. Automatic calibration stops recording after its final target removal.

Events:

- `recording_started`: current target set and last original presentation.
  Explicitly a snapshot, not a new observation.
- `target_added` / `target_removed`: stable ID, role, visible state, normalized
  position, rounded pixel center and spinner/crosshair appearance parameters.
  Every position change, including resize, is a remove/add pair. Spinner
  animation alone is not a new target; its appearance accompanies presentations.
- `presentation`: complete active targets, including an empty set when none
  are drawn. Arming, failure, completion and ordinary viewing are distinguishable.
- `queue_gap`: lost metadata-batch/thumbnail counts. Recovery uses a matching
  configuration/scene snapshot, never a silently substituted newer mapping.
- `recording_stopped`: sidecars finalized, stop reason and interruption flag.
  Recorder destruction before a requested stop is marked interrupted. Missing
  stop markers mean incomplete evidence; stop does not mean targets disappeared.

Gaze fields:

- `predicted_normalized` / `predicted_px`: current selected-eye mapped target,
  **unclamped**, or null. Normal viewing records predictions even with J off.
- `drawn_normalized` / `drawn_center_px`, `cursor_drawn`: actual drawn/clamped
  cursor. The accuracy view hides the cursor without hiding its prediction.
- `source_timestamp_ns`, `drawn_source_timestamp_ns`: original surface clocks,
  separate from the newer displayed RAW exposure. Held reticles retain their key.
- `source_basis`, `roi_frame_key`: ROI identity, raw sequence/time/epoch,
  provider and prompt generations, sign epoch and resolution.
- `source_advanced_for_basis`, `held_geometry`: redraw/duplicate/late-source and
  held-state evidence; neither proves a fit is accurate or a transported fit fresh.
- `mapping_revision`, `status`, `prediction_offscreen`: effective configuration
  reference and availability. A rough preset is not labeled a calibrated fit.

The cursor is selected-eye output, **not fused**, and is never copied to both
eye streams. UV runs top-left [0,0] to bottom-right [1,1]; framebuffer pixels are
UV times [width-1,height-1]. Only drawn centers are rounded/clamped. Display
identity, monitor/viewport origins where available, pixel dimensions and scaling
are in the configuration. Unknown compositor rotation stays null.

## Scene export: buttercup-scene-v1

`configuration_changed` carries the configuration in `data`: display, mapping,
gaze basis, calibration state and geometry. `scene_sample` carries
`data.scene_revision` and `data.sample` with independent eyes, ROI states,
configuration revision and an explicitly unavailable fused target.
Configurations and samples are emitted only on change; redraws reference them.

Start/recovery/live checkpoints contain `configuration`, `scene`, revisions
and the last original presentation at top level. They are explicitly
`snapshot_is_new_observation: false`. Revision IDs are scoped by
`viewer_session_id`. Never resolve missing revisions to the nearest or newest.

Other events are `stream_started`, `region_changed`, `capture_phase_changed`,
`source_dropped` and `roi_state_changed`. A state transition ends the previous
host-observed interval. Known reasons include intentional disable, global sensor
capture, temporary eviction, absent analysis, no new exposure for 750ms, not
detected, missing surface and unresolved sign.
`newer_ingress_than_displayed_analysis` reports processing lag, not proof of a
new SAM answer.

### Coordinate and geometry contract

The shared frame `viewer-eye-reference-inches-v1` uses **inches**, camera-right,
camera-down, toward-camera, **left handed**. Its origin is the assumed fixed
virtual ray origin of the selected reference eye, not a measured camera-space
pivot. A reference-eye change is a configuration change. Matrices are row-major,
multiplying column vectors. Reflect Y for a right-handed right/up/toward-camera
view; multiply lengths by 25.4 for millimeters.

`geometry.monitor` contains the actual current center, right/down axes, physical
size, corners and `uv_to_scene_3x3`. Multiplying by [u,v,1] yields an eye-relative
point in inches. It is the saved/estimated/preset plane, **not proof that its pose
is physically correct** or level in the room.

Each eye retains clocks, sign/authority epochs, candidate and admitted contact
axes, repeat/held state, scale hints and existing local pixel-space rotation
center/radius estimates. Pixel-space anatomy is not a metric eye position.
The scale hint uses MediaPipe projection plus an assumed 12mm limbus and
heuristic bounds, not calibrated probabilities. Contact axes are conic/surface
normals, not kappa-calibrated visual axes. Missing/unresolved admitted rays remain
null even when a provisional screen cursor is drawn.

Only the reference eye has the assumed [0,0,0] origin and identity coordinate
transform. The other eye's metric origin/transform and both anatomical radii
stay null. Camera optical center and pose are unsolved/null. Available
checkerboard intrinsics may be exported, explicitly **not used by this gaze
solve**. Metric head translation is not estimated. Both `visual_axis` fields
stay null; the binocular solver remains NotImplemented. A cosmetic camera or
second eyeball must be labeled schematic.

`monitor_intersection` requires a signed usable ray and known/assumed origin.
Statuses preserve on-screen, off-screen, behind-eye, parallel,
direction-unavailable, origin-unavailable and invalid cases. Hits retain
unclamped UV, scene point and range in inches. Compare intersection UV with
screen prediction only for a plane-only mapping and the same admitted source;
an affine correction is not a solved 3D visual ray. Exporting these records does
not change geometry, calibration, sign tracking or monitor defaults.

## Timing, sources and bounded delivery

New nanosecond, sequence, generation and epoch identifiers are decimal strings.
Legacy indexes retain numeric keys; the added `source_clock` carries exact
string keys so JavaScript need not round the old numbers.

`source_clock` preserves ROI, sequence, sensor timestamp, viewer process session,
host stream/reconnect epoch, camera region session/generation and host arrival.
`prediction_ready` is host EyeFrame completion, not CUDA completion.
`analysis_source` looks up the original surface exposure **exactly** in a bounded
2,048-key history, never nearest-future. Ambiguous clocks across epochs and
missing history remain explicit/unresolved. Resolve against `frames.jsonl`
first; references can predate recording, retaining their earlier host-arrival
time. In-recording dropped sources have `source_dropped` events. Separate
eye samples remain asynchronous; a containing newer RAW frame must never be
relabeled as the surface source.

Host Unix/process-monotonic submit begin/end clocks bound the successful
`buffer.present()` call, **not scanout or photons**. No sensor-to-host offset,
drift or scanout uncertainty bounds are available in this export. Target/exposure
alignment is host-arrival/submission approximate. Equal arrivals do not establish
equal exposure. Unix time may adjust; monotonic ordering is process-local.

UI/capture workers enqueue bounded batches; the receiver writes files. Queues
hold at most 256 batches / 16 MiB native payloads. Overflow discards the pending
interval, records counts and restates matching revisions. The latest global,
band and presentation snapshots bootstrap a recording, retaining original
clocks; cached thumbnails have `recording_start_snapshot: true`. Closing
detaches before final draining, preventing later-session leakage.

Live publishing remains bounded/non-blocking. `metadata_sequence` gaps expose
lost journal events; a live-only checkpoint is attempted every second for
late/reconnecting readers. Withhold dependent rendering until matching revisions
arrive; do not apply a later checkpoint retroactively to unmatched old samples.
These live-only checkpoints do not inflate archives.

## Native thumbnails and limits

Global and sensor-band packets are preserved before display/semantic acceptance.
No extra camera capture cadence is introduced. A new global appears only when
actually acquired; the cached/composed UI backdrop is not invented as a native
exposure. Current camera paths return packed RAW10_LE40 (ORT1) or linear
GRAY16LE (OTH1), not JPEG. The original 64-byte header and payload are unchanged.

OIC1 metadata includes encoding, dimensions/stride, sensor coverage, native
format/flags, source sequence/time, host receive time and known region IDs.
Global sequence is a capture request ID; band sequence is a fine-stream
acquisition sequence. Same-exposure-as-ROI and cross-connection hardware-clock
uncertainty are not invented.

This is not desktop recording or proof of physical visibility/fixation. Targets
from other processes are not captured. The user must actually fixate a target
for its position to be useful gaze ground truth.
