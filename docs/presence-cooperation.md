# Presence and gaze cooperation boundary

Experimental eye tracking has priority. A cooperating local client may consume
evidence already available to Buttercup. It does not acquire ownership of the
camera, tracking mode, exposure, focus, focus lock, calibration, or either ROI.
This internal contract defines telemetry semantics; it does not define or claim
support for a presence/gaze peer encoding. Camera session ownership is handled
separately below.

## Throne camera sessions

`camera_cooperation` implements camera ownership from UPC draft
`upc-fs-0.1-20260806-b`, including the `tcp` / `podbay-raw-v1` extension.
There is no runtime dependency on the Throne source checkout.

- When the configured UPC runtime root is absent, the viewer runs standalone
  without creating a runtime or requiring a session service.
- When it exists, a unique fresh, present descriptor must match both configured
  capture and control endpoints. The viewer requests control, renews its FIFO
  request, retains the exclusive lock, revalidates the attachment and publishes
  ownership before any camera access. It never steals a held lock.
- A stopped daemon, stale descriptor, malformed runtime or missing camera is
  not the same as an absent runtime. Startup refuses access rather than silently
  falling back. An actual camera-access ownership violation is process-fatal.
- Runtime appearance during standalone capture, attachment changes, replaced
  lock inodes and lost ownership invalidate camera access. Both camera ports
  share the same retained lease. Normal shutdown joins camera workers before
  releasing it; an error exits with the lease retained until process teardown.

Violations emit `FATAL ASSERTION: UPC camera ownership invariant violated` and
abort the entire process with SIGABRT; a worker cannot swallow the failure and
leave other camera threads alive. Core dumps are disabled on this fatal path.
A watchdog checks existing sessions every 100 ms, including runtime appearance
during standalone operation. These checks supplement validation at every camera
connection boundary. Runtime directories/files require exact 0700/0600 modes,
owned pinned no-symlink paths, and local-filesystem locks.

`--camera-cooperation-check` exercises the startup/ownership path without sending
camera commands. Isolated native fixtures test absence, acquisition, contention,
renewal, release and invalidation. This does not imply physical-camera readiness.
This adapter offers no unsupported presence signals and does not change ROI,
focus, exposure or tracking modes to satisfy another camera client.

## Presence and availability

A fresh direct eye observation in the selected ROI supports `Present` for that
eye observation. This is scoped evidence, not a global person-presence detector.
No eye in a selected ROI means `Unknown`, with an explanation such as outside
ROI, occlusion, insufficient evidence, or unavailable analysis when known. It
cannot establish that the person is absent. Retained identity and held geometry
must remain distinguishable from direct observations. The current typed helper
has no `Absent` value because its ROI-only inputs cannot establish absence.

The receiver's `active_mode_direct_present` aggregate is not a typed direct
observation: some modes include temporal or Driving retention, and accepted SAM
evidence may belong to an older source. Neither that aggregate, status strings,
nor `eye_identity_present` may construct `EyeEvidence::Direct` by themselves.
Integration needs the original per-mode direct observation and its exact source;
otherwise it must publish retained or unavailable evidence.

Process liveness, protocol availability, camera arrival, eye evidence, and gaze
availability are separate facts. A responsive service must not imply an eye is
present; a stopped viewer must not imply a person is absent. Any eventual
heartbeat must be independent of expensive segmentation. Fresh eye evidence may
coexist with unavailable or unsigned gaze.

## Source and gaze authority

Every observation carries its own eye, exact source epoch and timestamp,
authority and generation, and source-arrival age. Gaze additionally records its
sign epoch. The age is measured from that exact source's host arrival, not the
latest camera frame, solve completion, snapshot publication, or heartbeat.
Host-arrival age is not a measured sensor-to-host latency. Missing or ambiguous
clock correspondence yields unknown freshness. Reading a cached snapshot adds
elapsed time; it cannot renew the underlying observation.

The caller passes the existing `mouse_output::MAX_SOURCE_AGE` bound to the typed
helper. This avoids a separate freshness policy. The eventual integration must
obtain source key and age consistently from the same source receipt; separate
lookups must not race across a clock epoch. `Hub::source_receipt` captures both
under the journal lock and rejects ambiguous or missing source matches. Its
read-only key retains the complete native identity. `SourceReceipt::age_at`
normalizes each product's age to the common snapshot instant, including assembly
time; backwards time or overflow stays unknown. Repeated delivery never refreshes
the original arrival. Each product keeps its own age.

Gaze consumers share `desktop_gaze::authorized_gaze` after preparing the frame
under the current global gaze policy. Existing method/settings, eye, prompt,
source, resolved-sign and held-contact gates remain authoritative. The returned
authorization includes this exact source receipt, not a separate epoch lookup.
The telemetry
helper does not solve sign or select a competing gaze source. Rejection reasons
are preserved. `Snapshot::gaze_status` applies service availability and source
freshness before returning an observation.

Per-frame authorization is not sufficient for publication: the caller must
revalidate the current global method, settings and reference eye after projection
or snapshot assembly, using or extending the desktop `Selection` fence. A
concurrent selection/settings change invalidates pending and cached telemetry.
`Source::authority_generation` is not necessarily the global gaze settings
generation and must not substitute for it. The eventual wire integration must
reuse this publication fence rather than introducing a weaker parallel policy.

`SignedDirection` validates a finite camera-facing unit vector in right, down,
toward-camera coordinates. Representation validation is not evidence of resolved
sign; the caller must first pass the shared authority gates. A direction is not
a screen point, a calibration claim, or cross-user readiness.

## Optional requests and negotiation

Capability negotiation must distinguish supported read-only evidence from
temporarily unavailable evidence and unsupported operations. Missing evidence
returns an explicit reason. Requests needing global sensor scans, automatic
reacquisition, incompatible capture modes, ROI interruption, focus/exposure
changes, a second ROI, or calibration changes are declined under this service's
priority contract. A peer's higher requested priority does not override it.
No clever mode-switching fallback is permitted.

Already available passive overview or coarse observations may be reported with
their actual scope, authority and original source clock when the peer contract
supports them. This does not authorize a new global check. A current ROI arrival
must never make an older overview observation appear fresh.

Cooperation is local. This work grants no raw-frame export or new external
destination. The adapter must not execute arbitrary peer commands or interpret
mailbox contents as permission to alter these constraints.

The original note watched `/tmp/o.txt`; the authorized coordination channel is
`/tmp/o.xt`. Camera ownership now implements UPC as described above. The typed
snapshot helper and shared gaze-authority seam are implemented and tested, but
live eye/gaze telemetry publication is not yet claimed. While tracking owns the
camera, Throne yields and its camera observations become unknown until it
reacquires control. This must not be represented as absence or reading evidence.
