# Standalone optical clock bring-up

Status: **checked temporal optical lock demonstrated on the fixed hair-dryer
reflector and subsequently in a live eye ROI**, with native RAW replay controls
for the reflector. Corneal/pupil-only support, eye/glasses robustness and precise
transition-time/clock-rate recovery remain open. Keep this standalone: no integration into recorded-stimulus presentation
recipes or gaze training yet. Display commands alone are not exposure labels.

## September 12 investigation

The default stimulus is V3 RM(1,4): 16 logical symbols, five counter bits,
minimum same-session codeword distance eight. It can correct up to three logical
symbol errors under its code/geometry assumptions. The counter repeats every
32 code ticks; it is not a globally unique frame ID. Full session/manifest and a
bounded phase window are necessary to reject aliases. A checked word is not, by
itself, a verified temporal lock or proof of physical photon/exposure timing.

Concrete defects identified:

1. Camera RAW submissions to the clock worker were downstream of eye-tracking
   early returns. Display-only recovery could stream and record native frames
   while feeding **none** to the clock worker. Submission now occurs at RAW
   ingress, independently of focus, context and gaze admission.
2. Saturating subtraction when unwrapping a counter could manufacture code zero
   from an impossible early modulo code. Checked arithmetic preserves congruence.
3. A geometric size preference was also being used as the optical-contrast gate.
   An ideal 64x32 reflection had contrast 1.50 but ranked score 1.41 and was
   rejected. Ranking preference is now separate from the unchanged contrast
   threshold. This is not permission to accept unverified codes.
4. The local snapshot client performed one Unix-stream read. A live response
   ended after `code_index:0`, before the rest of the JSON arrived. Both ends now
   require a complete, bounded newline-framed message. A fragmented-reader test
   reproduces this failure without a GUI.
5. Sway's window-visible flag was mistaken for evidence of a powered display.
   The monitor had entered DPMS power save: `active:true`, `power:false`.
   Wayland appropriately stopped callbacks after frame zero while the duration
   timer and control socket continued running. The manifests for
   `hairdryer-large-1`, `hairdryer-large-2`, `hairdryer-temporal-1`, and the
   eight-second visibility check each contain **one** presentation. The latter
   was reported visible, focused and fullscreen, despite the powered-off output.
   Thus those
   trials are powered-off negative controls, not evidence that a changing signal
   was too weak. Startup now checks the actual output's power and availability.
   A bounded callback watchdog rechecks visibility/power before submitting a real
   frame, and exits explicitly if the output is unavailable. Each presentation
   records its trigger. Once the monitor was woken, normal callbacks resumed at
   approximately 60 Hz; the positive trials did not need watchdog commits.
6. The fixed reflector's native opponent response was inverted. The initial
   positive-polarity-only temporal decoder rejected it. The decoder now searches
   both polarities in the same nearest-word competition. Exhaustive minimum
   distance remains 17; the cross-polarity distance is 24. Error limits were not
   relaxed. The measured response to the displayed sequence had correlation
   -0.982 near 130 ms packet lag in a separate diagnostic alignment check.

The worker now keys history to the camera stream epoch, rejects repeated/late
sensor timestamps, resets after long gaps, and requires an independent checked
V3 decode to agree with a temporal projection before publishing that exposure as
verified. Projected clock values through unreadable frames are not measurements.
Rejected latency reports must not switch the readout to LOCKED.

## Diagnostic recipe

With the camera viewer running and the eye or reflector in its RAW crop, start a
bounded RAW recording and launch the same viewer executable with:

```
--screen-clock-stimulus --large-cells --fixed-target --code-hz 5
--amplitude 0.12 --duration-seconds 30 --output outputs/UNIQUE-clock.jsonl
```

This retains the same code but uses one 8x4 tile rather than four spatial copies;
each cell has four times the screen area. The target stays centered. The slower,
stronger chromatic diagnostic is explicitly selected, not a new default. Existing
four-copy recordings remain supported. Layout is propagated through the local
snapshot, manifest, RAW witness extraction and offline decoder. The clock window
requests fullscreen through Winit; it no longer moves/focuses Sway containers.

Inspect `CLOCK_RAW_DIAGNOSTIC` in the camera viewer log. A moving display alone
does not establish recovery. Keep RAW, the exact manifest, code scheme/layout,
source epoch/sequence/timestamp, reject counts, and receipt logs together.

## Evidence and limitations

- Native decoder baseline: 41 tests passed. Candidate: 42, including all 32
  checked words decoded from a synthetic native RAW large-cell reflection.
- Matched historical baseline: 432 RAW frames; weak fit score 0.0769 versus
  strongest control 0.0748 remains. Candidate direct witnesses in the fitted
  region change 41 to 42; this does not demonstrate a reliable causal live lock.
- Historical V2: 580 RAW frames. Both baseline and candidate fail to establish
  sustained optical onset. Candidate whole-stream witness counts are seven left,
  four right; baseline six left, four right. Failure is retained.
- First new trial: `outputs/clock-large-cells-live-1.tar`, 2,094 RAW frames and
  `outputs/screen-reflection-calibration/large-cells-live-1.jsonl`. Zero usable
  witnesses in either eye. The midpoint has no eye; the user subsequently
  clarified that the object is a reflective black hair dryer. The live worker
  also received no admitted frames before the ingress
  repair. This is **not** a successful positive-control capture.
- Logs: `outputs/clock-baseline-check.log`, `clock-candidate-check.log`,
  `clock-ingress-check.log`, `clock-*-replay.log`, and
  `live-clock-diagnostic.log`. Runtime artifacts are not source dependencies.

The current lag is arrival minus the first host submit of the recovered code.
At 5 Hz it also includes an unknown position within a 200 ms held-code interval,
plus compositor/scanout, exposure and transport. It is not exact display-to-sensor
latency or a sensor/host clock transform. Recover and bound actual optical
transitions before using it to align 200 ms target movements. Do not bridge
stream restarts or call held/interpolated predictions fresh optical observations.

## Spatially unresolved temporal diagnostic

`--temporal-code --fixed-target --code-hz 5 --amplitude 0.12` uses whole-field
chromatic modulation. This is an explicitly selected standalone diagnostic, not
a new presentation recipe. The viewer's existing **Z** standalone-clock launcher
now selects this diagnostic; Z/Esc closes it. It needs approximately 13 seconds of uninterrupted
signal. A 10-bit maximal-length sequence has period 1,023; all 63-symbol windows
have minimum Hamming distance 17 (exhaustively tested). Acceptance requires at
most six errors, a runner-up margin, analog agreement and a compatible current
exposure. The session tag is a phase shift, **not independent authentication**.

The fit uses native RAW10 Quad-Bayer plane means and relative sensor timestamps,
not host arrival or target timing to choose a code. Only after optical matching
does the receiver use the session log to unwrap a period and measure arrival
lag. The nominal symbol rate is assumed; clock drift is not yet estimated.
Per-frame phase support widths are heuristic support bands, not calibrated
confidence intervals. Reversed-time and constant-signal controls run in the
native offline decoder. Offline histories reset at stream, geometry and clock
boundaries; the live fixed-ROI path resets on stream changes and long gaps.

The first temporal recording contains 2,230 RAW frames at the exact held crops
(3428,3328) and (4472,3468), 420x280, focus 534, sensor band (0,3250,8000,576).
Both 1,115-frame eye streams produced zero optical words and zero control fits.
Its manifest proves the display stayed on one symbol, so this is an appropriate
rejection, not a positive validation. Keep that failure visible when comparing
the display-loop repair.

## Positive verification and controls

All tests retained the same two 420x280 crops, sensor band and focus 534; no
anatomical reacquisition or ROI motion was used. RAW was decoded independently
for each ROI, including the one not admitted by the eye tracker.

| Recording | Display evidence | Native right/left in-session matches | Controls |
| --- | --- | --- | --- |
| `hairdryer-clock-temporal-1.tar` | Powered off; one submit | 0 / 0 | Constant/reversed both 0 |
| `hairdryer-clock-temporal-2.tar` | Powered off; watchdog submits are not photons | 0 right; left not rerun | Constant/reversed both 0 |
| `hairdryer-clock-temporal-3.tar` | 2,402 submits, 200 codes, 40 s | 57 / 55 | Constant/reversed both 0, each ROI |
| `hairdryer-clock-temporal-4.tar` | 2,103 submits, 176 codes, 35 s | 51 / 51 | Constant/reversed both 0, each ROI |

The third capture has 2,422 RAW frames. The same right-ROI data yielded zero
matches with the positive-polarity baseline. With polarity recovery, 52/57 right
and 50/55 left in-session matches have zero errors; median analog agreement is
0.99996 and 0.99994. These overlapping 63-symbol windows are **not independent
trials**. The small number of extrapolated word fits after the stimulus closes
are rejected by the session-log join; never count them as measured display codes.

The fourth test demonstrated the online round trip: first checked receipt at
code 64 (~13 s warmup), then **41 acknowledged checked recoveries** through code
171. Packet-arrival minus first submit of the identified code had median
220.6 ms, range 130.9–313.0 ms. These are held-code lag measurements; subtracting
an assumed within-symbol age to claim precise camera latency is not justified.
The 5 Hz symbol itself spans 200 ms, and display submit is not measured scanout.

Artifacts are under `outputs/`: `live-clock-temporal-verified.log`,
`screen-reflection-calibration/hairdryer-temporal-{3,4}.jsonl`, matching RAW tars,
and `hairdryer-clock-temporal-*-{right,left}*.jsonl` native replays. Build/test
logs: `clock-polarity-check.log`, `clock-final-viewer-check.log`,
`clock-handoff-check.log`. Native decoder: 45 tests; viewer clock filters: 53
tests. None of this establishes performance on an actual eye or precision for
subpixel movement training. Next: isolate optical transition bounds and clock
rate/drift, then validate the eye reflection before integrating presentation.

## Returned-user eye trial and Z launcher repair

The user's `Z` key was reaching the viewer. Launch failed because a background
rebuild had replaced the running executable: Linux `current_exe()` returned a
nonexistent path ending in ` (deleted)`. Linux child launch now uses the retained
`/proc/self/exe` image, rather than stripping that suffix and possibly launching
a different build. The regression test copies the real test executable, runs it,
unlinks it while running, and successfully launches a child through the helper.
All 54 viewer clock tests pass, including that subprocess test.

Before restarting, the retained running image was launched directly for a
35-second real-eye test, leaving the user's active eye-student, ROI following
and manual focus setup in place. `outputs/eye-clock-first-live.tar` contains
932 native RAW frames; its companion manifest is
`outputs/screen-reflection-calibration/eye-first-live.jsonl`. The display logged
2,102 presentations and **35 accepted checked recoveries**, from code 64 through
173. Recent recovered words had zero bit errors and analog agreement around
0.99. This is evidence of optical timing recovery **in the eye ROI**, not proof
that only the pupil/cornea supplied the light signal. No glasses label was
established. The previous reflector negative controls do not replace independent
eye-specific controls. Held-code lag retains the timing limitations above.

The subsequent viewer restart initially hit the existing fatal RAW backlog
guard (712 ms versus its 350 ms limit). That guard was not weakened as part of
the launcher fix; the viewer was retried with the user's tracking mode and
manual focus restored.

## Shared presentation border clock

The same 5 Hz, 63-symbol LFSR10 temporal carrier can now illuminate only the
existing light-frame perimeter. `B` enables the light frame; `N` cycles white,
clock, automatic colors, then the individual color patterns. Clock is deliberately
excluded from the four-second auto rotation: its minimum warmup is 12.8 seconds.
It is available during normal viewing, mouse calibration, gaze accuracy checks,
and the Shift+backslash five-location micro-motion recording. Target recipes,
subpixel antialiasing, gaze calibration, camera ownership and ROI capture are
unchanged. No extra stimulus window or camera connection is opened.

`screen_reflection_temporal::symbol_rgb` owns the common carrier; the border
uses the same base RGB, opponent axis, amplitude 0.12 and session phase rule as
the successful whole-field trial. All four perimeter strips show the same
symbol, leaving the interior intact. Pattern selection/toggle creates an epoch;
cloning the light frame into calibration preserves that epoch and phase.

`screen_reflection_border` accepts only successful presentation journal receipts,
not render intent. Its bounded in-process bridge supplies session snapshots to
the existing native-RAW worker and receives checked-word reports. Disable,
resize/width changes, missing emitted code indices, reversed indices, or a
750 ms presentation gap reset/invalidate the decoder segment. The display
indicator expires checked status after 900 ms without another recovery. There
is no fallback from failed optical recovery to host-arrival clock guessing.

The existing `metadata.oim1` presentation record carries `optical_clock`:
emitted epoch/tag/index/symbol/RGB/recipe, decoder segment, border size,
warmup progress and latest checked camera receipt. Existing submit begin/end,
target geometry and selected gaze remain in that same record. Repeated recovery
metadata is explicitly not a new camera observation. The sequence identifies
the recovered RAW packet; first-submit timestamps join the actual emitted code,
not the most recent repeat of that code. Recovery includes held-symbol age and
must not be interpreted as exact monitor-to-exposure latency, drift recovery,
or subpixel eye-motion ground truth. No new JSONL stream or gzip is introduced.

Validation includes shared code/RGB equality across wraps, reset/staleness and
invalid-report rejection, preservation of calibration and micro-motion target
pixels, and OIM1 round-trip retention of clock, targets and gaze in both modes.
The whole-field eye trial above is NOT a border-only validation: perimeter-only
illumination has weaker/different reflection support. A live border recording
and eye-specific negative controls remain needed before claiming its accuracy.

Separately, standalone `Z` fullscreen approval still requires a producer-facing
pre-window grant from the attention manager. A request for that protocol was
sent through the shared coordination mailbox. File/URI queued-open approval is
not treated as permission to create an unrelated fullscreen window, and no
client-side focus/placement or self-approval is added by the border mode.
