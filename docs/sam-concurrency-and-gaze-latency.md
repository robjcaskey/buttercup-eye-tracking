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

SAM uses two independent eye lanes by default. Within each lane, image
preprocessing, encoding and semantic detection of exposure N+1 can overlap
ordered video-memory tracking and RAW/conic fitting of exposure N. The image
stage owns its model, prompts, pinned staging and photometric state. The tracking
stage alone owns the video memory, pupil history and geometry decisions. Each
stage has its own CUDA stream. The second eye loads tensors only when used;
second-ROI analysis still starts **off** (3 toggles it). An actually evicted eye
does not make its sibling wait for a pair. In joint SAM mode with both ROIs
resident, input is submitted as one attested same-read group; an incomplete
nominal stereo read is dropped instead of borrowing the other eye's old RAW.

There is one replaceable waiting RAW exposure per eye, one image-stage exposure,
and one tracking-stage exposure. A rendezvous between stages prevents a FIFO of
encoded frames. Joint-mode waiting pairs are replaced atomically; after one
lane claims its half, the other half is protected until claimed too. Outside
joint mode each lane retains its independent latest-frame policy. As load
increases, newer RAW replaces older waiting RAW before
expensive processing starts; as load falls, fewer or no frames are replaced.
No manually chosen inference FPS is required. Already-processing work is not
repeatedly discarded just because another camera exposure arrived (which could
starve publication). The result and proposal channels remain separate per eye.

Drops are scheduling events, not tracker misses, new observations or synthetic
negative masks. Retained frames keep their exact exposure clocks, sensor origins,
motion snapshots, immutable prompts and epochs. Memory commits remain ordered
per eye, with the existing duplicate/source-gap/session guards unchanged. The
optional pupil prompt reuses the current image features when supported; its
mask is selected/fitted only in the ordered tracking stage. Optional inference
failure cannot authorize a different cold RAW pupil acquisition.

Host-side admission matters before these GPU lanes: beginning a nearly stale
first ROI can otherwise spend the second ROI's remaining 200 ms queue budget.
The host now anticipates that paired work using bounded measured CPU timings;
it can shed an aging read early but cannot grant stale-packet exceptions.
Active recording preserves native ingress before analysis shedding, and
`source_dropped` metadata distinguishes saved-but-unanalyzed RAW from discarded
payloads. The dark recent calibration contained 191 incomplete archived reads,
all with exact host partner-drop receipts. See the measured limits and
recovery tradeoffs in [the host-queue audit](joint-conic-solver.md#the-host-queue-was-selectively-discarding-the-second-eye).

Prompt reloads update one revision under a mutex; each submitted batch retains
its immutable prompt reference and generation. Both lanes load the new revision
before their next batch and clear their own temporal memory. In-flight old
results retain their old generation and remain subject to host admission gates.
Global object inspection uses the primary image stage and does not write eye
memory. It can replace waiting RAW, but RAW cannot evict a waiting explicit
scene request. Each lane closes its input, joins both stages (including on an
image-stage panic), and disposes of all tensors before LibTorch teardown.

The small C++ bridge uses a LibTorch stream guard for each stage's lifetime.
Before handing off a frame, the producer synchronizes its stream; the consumer
registers every transferred CUDA tensor's storage with the caching allocator
on its own stream. This covers logits, all feature-pyramid levels, decoder
queries, optional pupil outputs and features retained in history. Readiness
alone would not prevent premature allocation reuse. No CUDA tensors cross eyes.
The bridge needs C++17, LibTorch headers and CUDA
headers (`CUDA_PATH`, default `/usr/local/cuda`); it dispatches synchronization
through LibTorch, not an independently linked system CUDA runtime. This follows
[PyTorch's stream ordering requirements](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-streams)
and [allocator lifetime requirements](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.record_stream.html).
`BUTTERCUP_SAM31_PARALLEL_EYES=0` is an offline comparison/troubleshooting escape
hatch, not a UI or firmware option. `SECOND ROI ON|OFF|STATUS` is available on
the viewer control socket for scoped live verification.
`IRIS BOUNDS SET MINIMUM MAXIMUM` restores operator limits in native pixels
without requiring a fresh detection; it uses the same finite/range/gap limits
as the manual controls and does not invent observed size evidence.

`BUTTERCUP_SAM31_FRAME_PIPELINE=0` restores rendezvous/busy-drop input and an
acknowledgment between stages for a no-overlap comparison. It retains identical
inference and fitting code. The default is enabled. `StatusSnapshot` separates
`replaced_batches` (a subset of total scheduling drops) from completions; query
diagnostics report encoding, tracking/fitting, waiting, and total host
submission-to-publication milliseconds. These host durations never replace
sensor timestamps or pretend to measure physical exposure-to-screen latency.

## Matched RAW comparison, September 6

This first comparison predates the within-eye pipeline; the additional pipeline
validation and its latency/geometry limitations are recorded below.

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

## Within-eye pipeline validation, September 6

The no-drop comparison again used the same capture, sequences 918–941, 24
pairs of native 420x280 RAW10. Serial, two-eye/no-frame-overlap, and two-eye
frame-pipeline runs produced **exactly identical outer ellipses and admission
decisions**, asserted in the test. Right fits: 24/24; left: 9/24 in each run.
This test deliberately waits between pairs, so its 251.25 versus 253.65 ms
paired medians do not measure within-eye throughput. Report:
`outputs/sam-frame-pipeline-parity.json`.

The offered-load test uses all 35 pairs, sequences 918–952. Four pairs warm
each eye; the remaining 31 pairs (922–952) arrive every 100 ms, with combined
outer-limbus/optional-pupil requests and original source clocks preserved.
An additional 30 ms offered cadence stresses replacement; it does not change
the recorded motion's source timestamps. Baseline and candidate run separately,
not concurrently on the same GPU. Results include time to drain bounded work.

| 100 ms offered cadence | No frame overlap | Frame pipeline |
| --- | ---: | ---: |
| Completed frames, right / left | 8 / 9 | 15 / 16 |
| Completed frames/sec, right / left | 2.65 / 2.98 | 4.24 / 4.52 |
| Admitted limbus fits, right / left | 8 / 4 | 15 / 6 |
| Median submission-to-publication, right / left | 347 / 300 ms | 415 / 381.5 ms |
| Median waiting, right / left | 1 / 1 ms | 53 / 51.5 ms |
| Waiting frames superseded, right / left | 0 / 0 | 16 / 15 |

The pipeline improves update throughput here, **not individual result age**.
Do not describe the roughly 70–80 ms median age regression as a latency win.
At the 30 ms overload cadence, completions increase from 3 to 5 per eye, with
26 waiting frames per eye superseded. No unbounded RAW or encoded FIFO is
allowed. Tests assert source-ordered commits, at most three outstanding stages
per eye, and eventual publication/processed rejection of the newest input.
Reports: `outputs/sam-frame-pipeline-load.json` and
`outputs/sam-frame-pipeline-eval.log`. The first load report's baseline
`dropped` counter includes retry attempts during warmup; derive offered drops
as 31 minus submitted there. The harness now subtracts warmup counters.

Dropping different sources changes temporal history; geometry under overload
is **not** guaranteed identical. Five fitted source/eye observations were common
to the 100 ms runs. The largest difference was a 12.387 px center shift and a
7.418 px reduction in major radius (frontal disk-area ratio 0.8595). Without
labels, neither answer is established as better. The left eye remains unreliable:
only 6/16 processed frames yielded admitted fits in the pipelined arm. No
optional pupil ellipse was admitted in either arm of this subset. These tests
therefore do not demonstrate improved pupil accuracy or gaze accuracy.

This subset has no human localization labels or independent scale chain in
the harness. Pixel frontal-disk area is **not SN-FEIDA**; an unchanged area is
not an accuracy certificate. The no-drop parity and overload differences are
scheduling/temporal-history diagnostics, not calibrated fidelity estimates or
a full-corpus geometry evaluation. Native source clocks/origins are retained;
host offer cadence and host completion times are measured separately.

Resource claim: RTX 5080, predicate
`gpu=0;gpu-memory=0;cpu=*;memory-bandwidth=*;block=*`, token
`l18d2e46b8cb24d82-6b957`, soft-exclusive **honored=true**. No live viewer ran
during the comparison. One hundred SAM/photometry/queue tests passed, with
three external-data tests ignored; the parity and load tests then ran explicitly.

### Does the GPU genuinely overlap frames?

A separate Nsight Systems CUDA trace enabled **one eye only**, excluding
cross-eye concurrency as an explanation. It recorded kernels on image stream
13 and tracking stream 17, with **75.482 ms of simultaneous execution** across
the short replay. One captured interval overlaps an image `Kernel2` with a
tracking `vectorized_elementwise_kernel`. There were 101,015 kernel records,
including warmup; 2,850.431 ms had at least one kernel active. This is evidence
of real cross-frame GPU execution, not an occupancy measurement or a claim
that every operation benefits. CPU fitting overlap and reduced idle gaps also
contribute. Profiling timings are not substituted for the unprofiled table.

Artifacts: `outputs/sam-frame-pipeline-singleeye.nsys-rep`, its `.sqlite`
export, and `outputs/sam-frame-pipeline-profile.log`. Claim token
`l18d2e4b88377419c-6c395`, same predicate, **honored=true**. The profiler used
`--trace=cuda --sample=none --cpuctxsw=none` and the `frame_pipeline_offered_load`
test with `BUTTERCUP_PIPELINE_TEST_EYES=1`,
`BUTTERCUP_PIPELINE_TEST_PERIOD_MS=100`, and
`BUTTERCUP_PIPELINE_TEST_ONLY=true`.

Final no-feature regressions passed: 87 SAM/photometry/queue tests (one ignored),
six ROI-continuity tests, and 29 eye-scene tests. The live SAM build succeeded;
see `outputs/sam-frame-pipeline-final-build.log`. The restarted viewer logged
both image/tracking stream pairs (3/35 and 67/99), each with `pipeline=true`.
Manual focus 531, iris bounds 77.917887–116.906301 px, second-ROI analysis and
J were restored for this session. No recording was started. The display was
in DPMS sleep; UI redraw and subjective gaze behavior were not verified.
Live log: `outputs/viewer-frame-pipeline-20260906.log`. Current frames were
being processed, but no healthy eye lock was established during this check;
the existing bounded global reacquisition remained active.
