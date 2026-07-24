# NetraID, Roadmap & Status

## Done
- [x] **ML pipeline**, download (MIT/Apache-2.0 models) → ArcFace align (1200 LFW, 0 fails)
      → conversion → benchmarks. **99.76% FP32 on LFW**; ship float32 on-device (int8 NaN /
      fp16 stall on mid-range HW), full stack ≈ 17 MB.
- [x] **React Native app**, vision-camera frame processor, fast-tflite recognition,
      active+passive liveness, encrypted store, sync/purge, Home/Enroll/Verify screens.
- [x] **Design system**, shared "Highway Control Room" language across app + web.
- [x] **Web Command Center** (`web/index.html`), live telemetry dashboard.
- [x] **Backend**, SAM template + idempotent sync Lambda + DynamoDB (ap-south-1).
- [x] **Docs**, Architecture, Model pipeline, Liveness, Benchmarks, Integration, Security.
- [x] **Pitch deck** (`deck/index.html`).
- [x] **MiniFASNet** passive anti-spoof: .pth → ONNX → TFLite int8 (0.75 MB,
      `ml/scripts/05_convert_minifasnet.py`), staged into the app and wired into the worklet.
- [x] **BlazeFace anchor decode + NMS**, 5-point Umeyama alignment + bilinear warp, and
      FaceMesh ROI handling, all implemented in the worklet (`blazeface.ts`, `align.ts`,
      `frameProcessor.ts`). No stubs.

## Next (to harden for the demo / pilot)
- [ ] **On-device latency benchmark** on a real mid-range Android (int8 NNAPI) to publish the
      end-to-end <1 s number measured on hardware.
- [ ] **India-representative calibration set** to re-tune `matchThreshold` + liveness
      thresholds for local demographics and harsh-light conditions.
- [ ] **Cognito user pool** + device enrollment flow; signed sync requests end-to-end.
- [ ] CI: `tsc`/lint for the app, `pytest` smoke for the ML scripts.

## Stretch
- [ ] GhostFaceNet-S variant (MIT) as an accuracy-max option; A/B vs MobileFaceNet.
- [ ] sqlite-vec ANN index if a device ever holds thousands of templates.
- [ ] Background sync via WorkManager / BGTaskScheduler.
