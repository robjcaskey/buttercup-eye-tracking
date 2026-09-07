# Direct gaze and concurrent eye inference

## Behavior

Main-view and completed-calibration cursors use absolute placement. The surface
tracker now publishes the current exposure's selected, signed normal directly,
not the previous 35% exponential average. Its floating points still provide
continuity evidence for sign selection; scale admission, sustained sign votes,
source-time deduplication, camera-facing/convexity checks and epoch binding remain.
The published camera-near point likewise belongs to the current exposure. This
removes output easing, not SAM computation time, rejection gaps or sensor delay.
It can expose more measurement jitter; no new gaze-accuracy claim is made.

SAM uses two independent rendezvous worker lanes by default, routed by eye.
Each owns its model, prompts, staging tensor, CUDA stream, mask memory and
photometric state. The second lane loads tensors only when used; second-ROI
analysis still starts **off** (3 toggles it). Successive exposures for one eye
remain serial. Busy frames are dropped without a RAW FIFO, and result/proposal
channels are separate so one eye cannot fill the other's publication capacity.
Disabled/missing eyes do not make their sibling wait for a pair.

Prompt reloads update one revision under a mutex; each submitted batch retains
its immutable prompt reference and generation. Both lanes load the new revision
before their next batch and clear their own temporal memory. In-flight old
results retain their old generation and remain subject to host admission gates.
Global object inspection uses the primary lane and does not write eye memory.
Each lane closes its request channel and joins before LibTorch teardown.

The small C++ bridge uses a LibTorch stream guard for the worker's lifetime.
No CUDA tensors cross lanes. The bridge needs C++17, LibTorch headers and CUDA
headers (`CUDA_PATH`, default `/usr/local/cuda`); it dispatches synchronization
through LibTorch, not an independently linked system CUDA runtime. This follows
[PyTorch's stream ordering requirements](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-streams).
`BUTTERCUP_SAM31_PARALLEL_EYES=0` is an offline comparison/troubleshooting escape
hatch, not a UI or firmware option. `SECOND ROI ON|OFF|STATUS` is available on
the viewer control socket for scoped live verification.
`IRIS BOUNDS SET MINIMUM MAXIMUM` restores operator limits in native pixels
without requiring a fresh detection; it uses the same finite/range/gap limits
as the manual controls and does not invent observed size evidence.

## Matched RAW comparison, September 6

Capture: `outputs/iris-arc-compatibility-20260905/capture-1788594056`, source
sequences 918–941, 24 synchronized pairs (48 native 420x280 RAW10 images).
Identical inputs, preprocessing and fitter settings were submitted in the same
per-eye source order to one serial lane and two concurrent lanes. The first
four pairs of each run were excluded from latency summaries, not fit coverage.

| Diagnostic | Serial | Concurrent |
| --- | ---: | ---: |
| Median both-eye completion | 297.60 ms | 246.23 ms |
| P95 both-eye completion | 574.45 ms | 390.74 ms |
| Median right-eye delivery | 154.66 ms | 239.52 ms |
| Median left-eye delivery | 295.52 ms | 226.73 ms |
| Admitted right-eye fits | 24/24 | 24/24 |
| Admitted left-eye fits | 9/24 | 9/24 |

All fitted ellipse parameters and admission decisions matched exactly. Frontal
disk area therefore also matched exactly. The left-eye 15/24 dropouts remain;
concurrency is not a detection fix. On these outputs the old filtered versus
current projected gaze differed by median 0.121 and maximum 0.425 in normalized
direction components—not degrees, pixels, or error against known gaze.

The GPU was an RTX 5080; the viewer was stopped. Resource claim
`gpu=0;gpu-memory=0;cpu=*;memory-bandwidth=*;block=*`, token
`l18d2e26178b35bfa-61ecf`, soft-exclusive **honored=false** because another
owner was running a CPU/memory/QEMU test. Thus timing is provisional and the
right-eye latency regression is explicit: concurrency improves paired delivery
and left-eye waiting here, not every individual inference. No kernel-level
profiler trace or uncontended 2x-speedup claim is made.

This subset has no attached human limbus/gaze truth or independent scale chain
in this test. `frontal_disk_area_px2` is not independently scale-normalized
SN-FEIDA. Area parity checks scheduling equivalence, not anatomical correctness.
See [SN-FEIDA's definition and limits](flat-tire-area-and-motion.md).

Reports: `outputs/sam-parallel-corpus-report.json` and
`outputs/sam-parallel-corpus-run.log`. The ignored
`parallel_eye_corpus_latency_and_geometry` test replays this comparison using
`BUTTERCUP_PARALLEL_TEST_CAPTURE` and `BUTTERCUP_PARALLEL_TEST_REPORT`.
Synthetic tests cover immediate signed-direction steps, constant frontal disk
area, duplicate-source handling, sustained sign changes, camera-facing
convexity, independent admission under a blocked sibling, prompt revision
binding, and second-ROI disable/enable behavior. These complement the RAW run;
they are not a calibrated fidelity or full-corpus accuracy evaluation.

Final live build: `outputs/sam-parallel-live-build.log`. Both lanes processed
camera frames on distinct CUDA streams (IDs 3 and 35); see
`outputs/viewer-parallel-sam-final-20260906.log`. Focus 531 and exact manual
iris limits 77.917887–116.906301 px were restored, with second-ROI analysis and
J enabled. The monitor entered DPMS sleep, so final UI redraw/subjective cursor
feel awaits waking the display; source-side inference remained active. The
final no-SAM-feature unit suite for `sam31_outer::tests` passed 63 tests with
one external-data test ignored (`outputs/sam-parallel-all-sam-tests.log`).
