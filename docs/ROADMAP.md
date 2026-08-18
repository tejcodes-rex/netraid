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

## Model refinement plan (what "we would improve the model" actually means)

The open-source weights are the **starting point, not the product**. They were trained on
public benchmarks that do not look like an NHAI worksite, and the accuracy that matters is
accuracy on this deployment, not on LFW. Four concrete pieces of work, in priority order:

1. **Fine-tune recognition on field data (highest value).** MobileFaceNet's w600k_mbf
   backbone is trained on WebFace600K, which under-represents Indian faces in NHAI field
   conditions: hard-hat shadow across the brow, sunglasses pushed up, dust, sweat, glare at
   9 a.m. and 4 p.m. on an open carriageway. Collect a consented pilot set on the target
   handsets across a full working day, then fine-tune the last blocks with **ArcFace loss**
   and hard-negative mining on the pairs the current model actually confuses. Report the
   gain as ROC/TAR-at-fixed-FAR on a held-out set from a *different* site, not as LFW.
2. **Retrain the passive anti-spoof, which is the weakest component.** MiniFASNet is trained
   on CelebA-Spoof/SiW-style data, mostly non-Indian subjects and attack media that are not
   what will be used here. Capture the real attack set (prints from a local shop, the phones
   staff actually own, tablets, laptops) on the real handsets and fine-tune on it. This is
   where the largest single improvement is available, precisely because it is the barrier
   with the weakest transfer.
3. **Calibrate rather than assume, per deployment.** `matchThreshold`, `matchMargin`,
   `passiveThreshold`, `screenSpoofMax` and `chromaThreshold` are all deployment constants
   that should be *measured* on a validation set from the deployment, then re-measured
   whenever the handset fleet changes. Publish the operating point and the FAR/FRR at it.
4. **Quantisation-aware training.** int8 currently emits NaN on the target hardware, which
   is why float32 (13.6 MB) ships. QAT would recover a ~4x size and latency reduction
   without the accuracy loss that naive post-training quantisation caused.

Two things worth being explicit about, because they are the usual objection to building on
open source:
- **What is ours is the system, not the weights.** The fusion logic across three
  uncorrelated barriers, the reaction-timed challenge FSM, the chroma challenge, the
  aggregation statistics, the on-device engineering (channel-order and alignment bugs that
  silently collapse accuracy), and the calibration methodology. Swapping the backbone for a
  better one is a one-file change precisely because of that separation.
- **Licences permit it.** Every shipped model is MIT or Apache-2.0, so fine-tuned
  derivatives can be deployed by NHAI without a licensing question.

## Stretch
- [ ] GhostFaceNet-S variant (MIT) as an accuracy-max option; A/B vs MobileFaceNet.
- [ ] sqlite-vec ANN index if a device ever holds thousands of templates.
- [ ] Background sync via WorkManager / BGTaskScheduler.
