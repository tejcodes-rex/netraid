# NetraID, Measured Benchmarks

> **Every number here was measured on real models and real faces on this machine**, not
> estimated. Reproduce with the scripts in `ml/scripts/` (`01_download_models.py` →
> `02_convert_to_tflite.py` → `03_benchmark.py`). Faces: LFW, aligned with the ArcFace
> 5-point pipeline (`insightface` SCRFD detector + `norm_crop`). 1200 faces aligned with
> **0 detection failures**.

## 1. Recognition model, accuracy vs compression (apples-to-apples)

MobileFaceNet `w600k_mbf` (insightface, MIT), 512-d ArcFace embedding. Evaluated on **300
aligned faces / ~58 identities**, same faces and same verification protocol for both rows.

| Variant | Size | Verify accuracy | TAR@FAR=1e-3 | Embedding fidelity vs FP32 |
|---|---:|---:|---:|---:|
| FP32 | 13.62 MB | **99.76%** | 97.80% | 1.0000 (reference) |
| **INT8** | **3.51 MB** | **99.75%** | 97.80% | 0.9741 |

**int8 costs only 0.01 percentage points of accuracy for a 3.9× size reduction** in the lab.
However, on the target mid-range phone the int8 `.tflite` emitted NaN and the fp16 stalled at
load, so NetraID **deploys the FP32 model on-device** (see §4). The FP32 numbers above are the
shipped operating point.

On a larger 600-face / 62-identity run the FP32 model scored **99.76% accuracy, 98.86%
TAR@FAR=1e-3**, consistent. **On-device** (Vivo V2246), a genuine match scored cosine **≈ 0.84**
and a different person **≈ 0.03** at acceptance threshold **0.38** (`ml/scripts/check_separation.py`
reproduces the same clean separation on LFW: same-person ≈ 0.62 median, different-person < 0.18).

## 2. TFLite conversion, sizes (all real files in `ml/models/`)

ONNX → TFLite via `onnx2tf` (compiler-free flatbuffer path).

| TFLite variant | Size | Notes |
|---|---:|---|
| float32 | 13.63 MB | **Cosine fidelity vs ONNX = 1.00000** (conversion is exact) |
| float16 | 6.83 MB | Near-lossless; runs via GPU/Core-ML delegate on device |
| **int8 (per-channel)** | **3.64 MB** | Deploys via NNAPI (Android) / Core ML (iOS) |
| dynamic-range | 3.47 MB | int8 weights, float activations |

> Deployment note: per-channel int8 / dynamic-range `.tflite` hang under desktop x86 LiteRT,
> and on the **target mid-range phone** the int8 model emitted NaN and the fp16 model stalled
> the interpreter at load. NetraID therefore **deploys the float32 `.tflite` (13.63 MB)** on
> device. The int8/fp16 variants remain in `ml/models/` as the compression study above; the
> shipped artifact is float32 (see §4).

## 3. Latency

**Measured end-to-end on a mid-range Vivo V2246:** the multi-frame recognize + liveness
verdict (3 quality-gated frames, each embedded twice for flip-TTA = 6 embedder passes, plus
matching and the margin check) ran in **371-457 ms**, comfortably under the **< 1 s**
requirement. Read from the on-device result card and the `[verify]` timing log.

| Measurement | Runtime | Time |
|---|---|---:|
| Verify verdict: 3-frame aggregate, flip-TTA, match + margin | TFLite float32, Vivo V2246 CPU | **371-457 ms** |
| Full single-face pipeline: detect → landmarks → align → TTA embed (Pipeline Demo screen) | same device | **avg 284 ms** (detect 29-46, landmarks 119-166, align 11-16, embed 103-113) |
| MobileFaceNet FP32 | ONNX Runtime, desktop CPU | ~7 ms |
| MobileFaceNet FP32 | TFLite float32, desktop CPU | ~80 ms |

The full on-device path (detect → landmarks/liveness → align → embed → match) is pipelined in the
vision-camera frame processor; detection and landmarks run per frame on the worklet thread, and
the embedding + match run on the JS thread at capture time.

## 3b. On-device accuracy engine (measured 24 Jul 2026, Vivo V2246)

The recognition layer aggregates several defenses; all figures below are read from the
device's diagnostic logs during live runs:

- **Multi-shot enrollment** (6 candidates, sharpest kept, embedding outliers dropped):
  per-shot agreement with the stored template measured **0.965-0.994** cosine.
- **Multi-frame verification** (3 frames, majority person, median score): a genuine match
  scored **0.884-0.905** per frame; the aggregate verdict landed at **0.89-0.90** against the
  **0.38** acceptance threshold.
- **Flip-TTA embedding**: every crop is embedded twice (original + mirror) and averaged.
- **Margin rule**: the winner must beat the best *different* person by **≥ 0.08** cosine.
  In live testing this correctly refused to decide when the same face was enrolled under two
  IDs (both scored ~0.9); enrollment now blocks that case outright (duplicate-face guard,
  threshold 0.6).
- **Pipeline Demo screen** (bundled LFW frames through the identical code path): genuine pair
  **0.669**, all impostor pairs **≤ 0.054**, 6/6 verdicts correct (`docs/assets/`).
- **Quality gates**: Laplacian-variance sharpness floor, exposure bounds (mean 40-235),
  frontal-pose gate, and a 45-frame camera warm-up so cold-start auto-exposure frames can
  never enter a template (a failure observed and reproduced on-device before the gate).

## 3c. Anti-spoof calibration (measured 24 Jul 2026, Vivo V2246)

Attack protocol: the enrolled user's own photo displayed on a laptop screen, actively
swung/tilted in front of the camera for tens of seconds per attempt, across multiple attempts.

- **Active layer**: blink is mandatory in every challenge set; blink evidence is
  depth-weighted (a shallow single-frame EAR dip, the landmark-noise signature of a moving
  photo, is rejected) and pauses (without losing accumulated evidence) while the head moves
  faster than 12% of the inter-ocular distance per processed frame, which real hand tremor
  never reaches and photo-swings always exceed. Smile/turn must hold 2 consecutive frames. Gestures
  carry a 10 s wall-clock budget (timeout swaps the stuck gesture, progress kept; a stuck
  blink keeps its slot); the capture phase carries 8 s (timeout restarts the attempt).
- **Passive layer (MiniFASNet float32)**: input convention matters, BGR channel order with
  RAW 0-255 floats; fed [0,1] RGB the model degenerates to a constant verdict (measured on
  desktop and device). With correct input: live-face captures scored P(real) **0.12-0.74**,
  the laptop-screen replay scored **≤ 0.04** (P(screen-fake) up to 0.99). The gate accepts at
  **max P(real) ≥ 0.08** across the verify captures, 2x above the worst spoof reading and
  2.8x below the weakest live reading.
- **Outcome**: a different person's photo is rejected by recognition (cosine ≈ 0.05-0.07);
  the enrolled user's own screen replay is rejected by liveness. Both verified live on the
  target device.
- **Stability**: continuous max-rate frame processing fragmented the worklet heap into an
  `OutOfMemoryError` after ~1 min of sustained attack; the pipeline now processes alternate
  frames (it is CPU-bound below camera rate regardless), which removed the failure.

## 4. On-device size budget (≤ 20 MB requirement)

| Component | Model | Size |
|---|---|---:|
| Face detection | MediaPipe BlazeFace (short-range) | 0.23 MB |
| Landmarks (liveness) | MediaPipe FaceLandmarker | 2.55 MB |
| Recognition | **MobileFaceNet float32** | **13.63 MB** |
| Passive anti-spoof | MiniFASNet (1 scale, float32) | 1.68 MB |
| **Total** | | **≈ 17.3 MB** |

**≈ 17.3 MB, under the 20 MB cap**, while exceeding the > 95% accuracy and < 1 s targets.

**Why float32 for recognition (not int8/fp16):** the quantized variants are smaller on paper,
but on the **target mid-range hardware (Vivo V2246)** the int8 model emitted **NaN** embeddings
and the fp16 model **stalled the TFLite interpreter at load**. The float32 model is robust on
the CPU delegate, loads reliably, and still leaves ~3 MB of headroom under the cap. The detection,
landmark, and anti-spoof models stay quantized/compact. This is a real on-device finding, not a
desktop assumption.

## 5. How to reproduce

```bash
cd ml && pip install -r requirements.txt
python scripts/01_download_models.py     # open-source weights (MIT/Apache-2.0)
python scripts/02_convert_to_tflite.py   # align real faces, calibrate, convert int8
python scripts/03_benchmark.py           # size / accuracy / fidelity / latency
python scripts/_onnx_int8.py             # int8 accuracy via ONNX Runtime (§1)
```

## 6. Limitations
- LFW is the standard academic benchmark but is not India-specific; for deployment we
  recommend a small **India-representative pilot set** to re-calibrate the acceptance
  threshold (currently **0.38** on-device) and the liveness thresholds.
- Passive RGB anti-spoofing generalizes imperfectly across camera hardware; this is exactly
  why NetraID **pairs it with an active challenge** (uncorrelated failure modes). See
  `LIVENESS.md`.
