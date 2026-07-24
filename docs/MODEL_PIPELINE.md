# NetraID, Model Selection & Conversion Pipeline

End-to-end, reproducible path from open-source weights to the on-device `.tflite` files.
Run order: `01_download_models.py → 02_convert_to_tflite.py → 03_benchmark.py →
04_export_app_models.py` (all in `ml/scripts/`).

## 1. Model selection (and why)
| Role | Model | License | On-device |
|---|---|---|---|
| Recognition | **MobileFaceNet** `w600k_mbf` (insightface) | MIT | int8 **3.5 MB**, 512-d |
| Detection | MediaPipe **BlazeFace** short-range | Apache-2.0 | 0.23 MB |
| Landmarks | MediaPipe **FaceLandmarker** (468-pt) | Apache-2.0 | 2.55 MB |
| Passive liveness | **MiniFASNet** (Silent-Face) | Apache-2.0 | 0.75 MB |

**Rejected: EdgeFace**, better accuracy/param but **non-commercial weights**, which violates
the hackathon's "open-source only, no additional licenses" rule. We will not ship it.

## 2. Conversion: ONNX → TFLite int8
We use **onnx2tf** (compiler-free flatbuffer path), it correctly handles the NCHW→NHWC
transpose that older `onnx-tf` got wrong.

Critical choices learned during build:
1. **Per-channel** quantization, not per-tensor. Per-tensor dropped accuracy to ~80%;
   per-channel preserves it. (Conv weights vary a lot across channels.)
2. **Real calibration data.** We align **256 real LFW faces** with the exact ArcFace
   5-point transform used at inference and feed them as the representative dataset. Random
   calibration collapses int8 accuracy (saturated activations → NaN cosine).
3. **Match preprocessing exactly**: RGB, 112×112, `(x-127.5)/127.5`. The #1 cause of
   "accuracy collapses after quantization" is a normalization mismatch between calibration
   and runtime.
4. **Keep embedding output in float** so cosine discrimination survives.

## 3. Validation (see `BENCHMARKS.md` for numbers)
- **TFLite float32 vs ONNX**: cosine fidelity **1.00000** → conversion is exact.
- **int8 accuracy**: measured via ONNX Runtime dynamic quantization (a reliable CPU path) →
  **99.75%** vs FP32 **99.76%** (−0.01 pp) at **3.5 MB**.
- Deployment note: on the target mid-range phone the int8 model emitted NaN embeddings and the
  fp16 model stalled the TFLite interpreter at load, so the **float32 `.tflite` (13.6 MB) is the
  deployed artifact**. The int8/fp16 variants remain as the compression study; float32 still
  fits the 20 MB budget (≈ 17 MB full stack).

## 4. Reproduce
```bash
cd ml && pip install -r requirements.txt
python scripts/01_download_models.py
python scripts/02_convert_to_tflite.py
python scripts/03_benchmark.py
python scripts/_onnx_int8.py      # int8 accuracy via ONNX Runtime
python scripts/04_export_app_models.py
```
