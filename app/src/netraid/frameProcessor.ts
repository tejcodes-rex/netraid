// Real-time camera frame processor (vision-camera worklet). Runs on a dedicated
// thread for every frame. Pipeline:
//   detect (BlazeFace) -> ROI crop -> landmarks (FaceMesh) -> active liveness FSM
//   -> passive anti-spoof (MiniFASNet) -> emit aligned 112x112 crop to JS.
// Heavy alignment (umeyama + warp) and recognition run on the JS thread.

import { useEffect } from 'react';
import { useFrameProcessor } from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { Worklets, useSharedValue } from 'react-native-worklets-core';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { Platform } from 'react-native';
import { decode, nms, pickPrimary } from './blazeface';
import { ear, interOcular, mouthWidthRatio, yawRatio, TH } from './liveness';
import { umeyama, warpAffine, REF5 } from './align';
import { assessCrop, MIN_SHARPNESS } from './quality';
import type { Challenge } from './types';
import type { Pt } from './liveness';

export interface FrameCallbacks {
  onChallengeProgress: (done: Challenge[]) => void;
  onCapture: (rgb: Uint8Array, passiveScore: number, sharpness: number) => void;
  onNoFace: () => void;
  /** The current challenge was not completed within the time budget. The
   * screen should issue a fresh random challenge set (anti-replay). */
  onChallengeTimeout?: () => void;
  onDebug?: (msg: string) => void;
}

const ROI_SRC = 256; // resolution of the ROI buffer we warp the aligned crop from
// FaceLandmarker face-presence logit gate. The model returns a strongly negative
// value when no face is in the square, positive when a face is well-framed.
const FACE_PRESENCE_MIN = -11; // landmark-validity gate (BlazeFace already gates the
// background out; this just skips frames where the mesh is too uncertain to use).

/**
 * @param resetKey bump to fully reset the liveness FSM (new attempt). The
 * challenges array alone can change WITHOUT resetting progress: on a gesture
 * timeout the screen swaps just the stuck challenge, and completed steps keep
 * their state.
 */
export function useNetraFrameProcessor(
  challenges: Challenge[], cb: FrameCallbacks, resetKey = 0, paused = false,
) {
  const detector = useTensorflowModel(require('../../assets/models/blazeface_short_range.tflite'));
  const landmarker = useTensorflowModel(require('../../assets/models/face_landmarker.tflite'));
  // float32 weights: the int8 MiniFASNet emits saturated one-hots and NaN on
  // both desktop LiteRT and this device class (same failure family as the int8
  // recognition model), so it cannot discriminate. The fp32 graph is 1.7 MB
  // and produces real probability distributions.
  const antispoof = useTensorflowModel(
    require('../../assets/models/minifasnet_fp32.tflite'),
  );
  const { resize } = useResizePlugin();

  // Report model tensor shapes to the UI (debug aid).
  useEffect(() => {
    const lm: any = landmarker.model;
    if (lm) {
      cb.onDebug?.('MESH in=' + JSON.stringify(lm.inputs?.[0]?.shape) + ' out=' + JSON.stringify((lm.outputs ?? []).map((o: any) => o.shape)));
    } else {
      cb.onDebug?.(`load det=${detector.state} mesh=${landmarker.state} spoof=${antispoof.state}`);
    }
  }, [detector.state, landmarker.state, antispoof.state, landmarker.model]);

  const step = useSharedValue(0);
  const blinkClosed = useSharedValue(0);
  const baseMouth = useSharedValue(0);
  const frameN = useSharedValue(0);
  const lastEmit = useSharedValue(0);
  const stepStart = useSharedValue(0);
  const firstTs = useSharedValue(0);
  const prevNoseX = useSharedValue(0);
  const prevNoseY = useSharedValue(0);
  const holdN = useSharedValue(0);
  const lastGesture = useSharedValue('');
  const blinkOpen = useSharedValue(0);

  // A new ATTEMPT (resetKey change) restarts the FSM: shared values outlive
  // the worklet rebuild, so clear them explicitly. All *Start/*Ts values are
  // wall-clock ms: the worklet's effective frame rate varies wildly with
  // device load, so budgets counted in frames are meaningless.
  useEffect(() => {
    step.value = 0;
    blinkClosed.value = 0;
    baseMouth.value = 0;
    lastEmit.value = 0;
    stepStart.value = 0;
    holdN.value = 0;
    lastGesture.value = '';
    blinkOpen.value = 0;
  }, [resetKey, step, blinkClosed, baseMouth, lastEmit, stepStart, holdN, lastGesture, blinkOpen]);

  const onProgress = Worklets.createRunOnJS(cb.onChallengeProgress);
  const onNoFace = Worklets.createRunOnJS(cb.onNoFace);
  const onTimeout = Worklets.createRunOnJS(cb.onChallengeTimeout ?? (() => {}));
  const onDebug = Worklets.createRunOnJS(cb.onDebug ?? (() => {}));
  // Align + warp on the JS thread, then hand the finished crop to the caller.
  const onReady = Worklets.createRunOnJS(
    (src: number[], size: number, mesh5: number[][], passive: number, gain: number) => {
      try {
        // src is RGBA (size*size*4): on this device the plugin's 3-channel 'rgb'
        // output is alpha-first (ARGB-truncated), which red-casts the crop and
        // collapses embeddings. We request 'rgba' and strip alpha to true RGB here.
        // The dim-light gain measured on the detector view is applied so the
        // recognition crop matches what detection and landmarks saw.
        const n = size * size;
        const rgb3 = new Uint8Array(n * 3);
        for (let p = 0, q = 0; p < n; p++, q += 3) {
          const o = p * 4;
          let r = src[o] * gain, g = src[o + 1] * gain, b = src[o + 2] * gain;
          rgb3[q] = r > 255 ? 255 : r;
          rgb3[q + 1] = g > 255 ? 255 : g;
          rgb3[q + 2] = b > 255 ? 255 : b;
        }
        const M = umeyama(mesh5, REF5);
        const rgb = warpAffine(rgb3, size, size, M);
        // Severe motion blur or a badly exposed crop poisons both enrollment
        // templates and verify probes; drop such crops here so every consumer
        // gets usable input.
        const q = assessCrop(rgb);
        if (q.sharpness < MIN_SHARPNESS) return;
        if (q.brightness < 40 || q.brightness > 235) return;
        cb.onCapture(rgb, passive, q.sharpness);
      } catch {
        // a malformed frame: skip this capture, the next frame will retry
      }
    },
  );

  return useFrameProcessor((frame) => {
    'worklet';
    const det = detector.model, mesh = landmarker.model, spoof = antispoof.model;
    frameN.value += 1;
    if (det == null || mesh == null || spoof == null) return;
    // Paused: the camera stays live (reactivating a stopped camera strands the
    // session on this device) but there is nothing to capture right now, a
    // result card is up, or the shot quota is met. Skip all heavy work.
    if (paused) return;
    // Process alternate frames only. The pipeline is CPU-bound well below the
    // camera rate anyway, and every processed frame allocates several large
    // resize buffers; running flat-out fragments the worklet heap until the
    // frame processor throws OutOfMemoryError (observed after ~1 min of
    // continuous face processing on the target device).
    if (frameN.value % 2 === 1) return;

    // Camera warm-up: the first ~1.2 s arrive while auto-exposure and white
    // balance are still ramping. A template built from cold-start frames will
    // not match a settled-sensor probe, so nothing is processed until the
    // sensor stabilizes. Wall-clock, not frames: worklet fps varies with load.
    const nowMs = Date.now();
    if (firstTs.value === 0) firstTs.value = nowMs;
    if (nowMs - firstTs.value < 1200) return;

    const W = frame.width, H = frame.height;
    const side = Math.min(W, H);
    const offX = (W - side) / 2, offY = (H - side) / 2;

    // 1) DETECT: rotate the centered square upright (the sensor frame is
    //    landscape) and run BlazeFace to locate the face.
    const detU = resize(frame, {
      crop: { x: offX, y: offY, width: side, height: side },
      scale: { width: 128, height: 128 }, rotation: '270deg', pixelFormat: 'rgb', dataType: 'uint8',
    });
    // Adaptive dim-light gain: outdoor evening / shadow frames sit far below the
    // brightness the detector was trained on and its scores collapse (~0.3).
    // Normalize the mean toward ~110 (clamped) before every model input.
    let sum = 0;
    for (let i = 0; i < detU.length; i++) sum += detU[i];
    const meanBright = sum / detU.length;
    const gain = meanBright < 105 ? Math.min(2.8, 110 / (meanBright + 1)) : 1;
    const detIn = new Float32Array(detU.length);
    for (let i = 0; i < detU.length; i++) {
      const v = detU[i] * gain;
      detIn[i] = (v > 255 ? 255 : v) / 255;
    }
    const detOut = det.runSync([detIn]);
    const boxesT = (detOut[0].length > detOut[1].length ? detOut[0] : detOut[1]) as Float32Array;
    const scoresT = (detOut[0].length > detOut[1].length ? detOut[1] : detOut[0]) as Float32Array;
    const sq = pickPrimary(nms(decode(boxesT, scoresT, 0.18)));
    if (!sq) {
      if (frameN.value % 20 === 0) {
        let mx = -1e9;
        for (let i = 0; i < scoresT.length; i++) if (scoresT[i] > mx) mx = scoresT[i];
        onDebug('FIT 0 bright=' + meanBright.toFixed(0) + ' gain=' + gain.toFixed(1) + ' best=' + (1 / (1 + Math.exp(-mx))).toFixed(2));
      }
      onNoFace();
      return;
    }

    // 2) Map the upright-128 box to a TIGHT square crop in original frame pixels.
    //    The detect view was rotated 90deg CW, so a point (u,v) in the upright
    //    view came from (v, 1-u) in the pre-rotation square.
    const bcu = (sq.xmin + sq.xmax) / 2, bcv = (sq.ymin + sq.ymax) / 2;
    const bw = sq.xmax - sq.xmin, bh = sq.ymax - sq.ymin;
    const faceFrac = Math.max(bw, bh); // fraction of the square the face fills
    const fitPct = Math.round(Math.min(1, faceFrac / 0.62) * 100);

    // The user centers their face in the on-screen circle (which is the frame
    // center), so crop a CENTERED square sized to the detected face and rotate it
    // upright for the landmark model. Mapping BlazeFace's rotated box back to frame
    // pixels is error-prone; centering is robust given the guide circle.
    let fsize = Math.max(bw, bh) * side * 1.25;
    if (fsize > side) fsize = side; else if (fsize < side * 0.3) fsize = side * 0.3;
    const roiCrop = { x: (W - fsize) / 2, y: (H - fsize) / 2, width: fsize, height: fsize };
    // Directional guidance from the box position in the upright (preview) image.
    if (frameN.value % 8 === 0) {
      let guide = 'ok';
      if (fitPct < 55) guide = 'closer';
      else if (bcu < 0.38) guide = 'left';
      else if (bcu > 0.62) guide = 'right';
      else if (bcv < 0.34) guide = 'down';
      else if (bcv > 0.66) guide = 'up';
      onDebug('FIT ' + fitPct + ' ' + guide);
    }

    // 3) LANDMARKS on the tight, upright ROI. FaceLandmarker input is 256x256.
    const meshU = resize(frame, {
      crop: roiCrop, scale: { width: 256, height: 256 }, rotation: '270deg', pixelFormat: 'rgb', dataType: 'uint8',
    });
    const meshIn = new Float32Array(meshU.length);
    for (let i = 0; i < meshU.length; i++) {
      const v = meshU[i] * gain;
      meshIn[i] = (v > 255 ? 255 : v) / 255;
    }
    const meshOut = mesh.runSync([meshIn]);
    const presence = (meshOut[1] as Float32Array)[0];
    const lm = toPointsNorm(landmarkTensor(meshOut)); // ROI-normalized [0,1], upright
    if (frameN.value % 10 === 0) onDebug('pres=' + presence.toFixed(1) + ' fit=' + fitPct);
    // Low presence = this frame's landmarks are unreliable; skip the liveness
    // update but DO NOT flash "no face" (BlazeFace already confirmed a face).
    if (presence < -13.5) return; // landmark-validity gate (background sits ~ -14.1)

    // Dev-only calibration probe: sample the passive anti-spoof score for
    // whatever is in front of the camera (live face or spoof medium). Metro
    // inlines __DEV__, so release builds compile this block out entirely.
    if (__DEV__ && frameN.value % 60 === 0) {
      const pU = resize(frame, {
        crop: roiCrop, scale: { width: 80, height: 80 }, rotation: '270deg', pixelFormat: 'rgba', dataType: 'uint8',
      });
      const pN = 80 * 80;
      const pIn = new Float32Array(pN * 3);
      for (let p = 0, q = 0; p < pN; p++, q += 3) {
        const o = p * 4;
        // BGR, raw 0-255 (see step 5)
        pIn[q] = pU[o + 2]; pIn[q + 1] = pU[o + 1]; pIn[q + 2] = pU[o];
      }
      const pr = spoof.runSync([pIn])[0] as Float32Array;
      // 3 classes (Silent-Face): [print-fake, real, screen-fake]
      let probs = '';
      for (let i = 0; i < pr.length; i++) probs += (i ? ',' : '') + pr[i].toFixed(3);
      onDebug('PASV [' + probs + '] conf=' + sq.score.toFixed(2));
    }

    // 4) ACTIVE liveness FSM. Each challenge has a 10 s wall-clock budget; on
    // timeout the screen swaps the stuck gesture for a fresh random one
    // (completed progress is kept), so a spoof can never park the flow and a
    // replayed gesture recording goes stale, without punishing a slow human.
    const s = step.value;
    if (s < challenges.length) {
      if (stepStart.value === 0) stepStart.value = nowMs;
      if (nowMs - stepStart.value > 10000) {
        stepStart.value = 0;
        onTimeout();
        return;
      }
      // Fresh tracking state whenever the CURRENT gesture changes, whether by
      // normal advance or by a timeout swap: a stale smile baseline or blink
      // counter from a previous gesture must never leak into this one.
      if (lastGesture.value !== challenges[s]) {
        lastGesture.value = challenges[s];
        blinkClosed.value = 0;
        baseMouth.value = 0;
        holdN.value = 0;
        blinkOpen.value = 0;
      }
      if (frameN.value % 12 === 0) {
        onDebug(challenges[s] + ' ear=' + ear(lm).toFixed(2) + ' yaw=' + yawRatio(lm).toFixed(2) + ' mouth=' + mouthWidthRatio(lm).toFixed(2));
      }
      // Motion-stability gate for blink: swinging a photo in front of the
      // camera blurs the eye landmarks into a fake "closure". A real blink
      // happens on a stable-ish head, so closure evidence PAUSES (not erased,
      // hand tremor must not punish real users) while the nose moves more
      // than 12% of the inter-ocular distance per processed frame. Photo
      // swings sit far above this.
      const nose = lm[1];
      const moved = Math.hypot(nose.x - prevNoseX.value, nose.y - prevNoseY.value) /
        (interOcular(lm) + 1e-6);
      prevNoseX.value = nose.x; prevNoseY.value = nose.y;
      if (challenges[s] === 'blink' && moved > 0.12) {
        return;
      }
      // Smile / head-turn must HOLD for 2 consecutive processed frames: a
      // single-frame landmark flicker cannot complete a gesture. Blink is
      // transient by nature (the reopen instant IS the event), so it fires
      // directly.
      const ok = satisfied(challenges[s], lm, blinkClosed, baseMouth, blinkOpen, frameN.value);
      const isBlink = challenges[s] === 'blink';
      if (ok) holdN.value += 1;
      else if (!isBlink) holdN.value = 0;
      if (ok && (isBlink || holdN.value >= 2)) {
        step.value = s + 1;
        stepStart.value = 0;
        holdN.value = 0;
        onProgress(challenges.slice(0, s + 1));
        blinkClosed.value = 0; baseMouth.value = 0;
      }
      return;
    }

    // 4b) Capture-phase timeout: liveness passed but a clean frontal capture is
    // not arriving (user drifted off-angle, walked away, camera covered). After
    // 8 s hand control back to the screen, which restarts with a fresh
    // challenge set instead of stalling forever.
    if (challenges.length > 0) {
      if (stepStart.value === 0) stepStart.value = nowMs;
      else if (nowMs - stepStart.value > 8000) {
        stepStart.value = 0;
        onTimeout();
        return;
      }
    }

    // 4c) QUALITY GATE: the recognition embedding must come from a FRONTAL, well
    // framed face. The active challenges often end with the head turned, and
    // embedding an off-angle / loosely framed face collapses accuracy. Wait until
    // the user faces the camera squarely before capturing.
    const yr = yawRatio(lm);
    if (yr < 0.78 || yr > 1.28 || fitPct < 48) {
      if (frameN.value % 10 === 0) onDebug('FRONT hold yaw=' + yr.toFixed(2) + ' fit=' + fitPct);
      return;
    }

    // 4d) CONFIDENCE GATE for emission: tracking runs on lenient thresholds so
    // dim scenes still get guidance, but an EMITTED crop must be a confident
    // face. Rejects face-like texture (fabric folds, furniture) that clears the
    // lenient detector score or sits near the landmark-presence noise floor.
    if (sq.score < 0.28 || presence < -8) {
      if (frameN.value % 10 === 0) onDebug('WEAK face conf=' + sq.score.toFixed(2) + ' pres=' + presence.toFixed(1));
      return;
    }

    // 5) PASSIVE anti-spoof on the tight ROI (80x80). Use rgba + strip alpha
    // (the plugin's 3-channel 'rgb' is alpha-first on this device).
    // MiniFASNet input convention (Silent-Face): BGR channel order, RAW 0-255
    // floats, NO /255 normalization. Feeding [0,1] RGB makes it emit a
    // constant "screen-fake" for everything (measured on desktop and device).
    // Deliberately NOT gain-boosted: it judges capture authenticity and should
    // see the sensor's real response, not a synthetically brightened one.
    const spU = resize(frame, {
      crop: roiCrop, scale: { width: 80, height: 80 }, rotation: '270deg', pixelFormat: 'rgba', dataType: 'uint8',
    });
    const sp80 = 80 * 80;
    const spIn = new Float32Array(sp80 * 3);
    for (let p = 0, q = 0; p < sp80; p++, q += 3) {
      const o = p * 4;
      spIn[q] = spU[o + 2]; spIn[q + 1] = spU[o + 1]; spIn[q + 2] = spU[o];
    }
    const prob = spoof.runSync([spIn])[0] as Float32Array;
    const passive = prob.length > 1 ? prob[1] : prob[0]; // P(live)

    // 6) Emit aligned crop: 5 landmarks (256 ROI space) -> umeyama -> warp.
    // 256 inlined (module consts are not reliably captured into the worklet).
    // The FSM stays in the completed state and keeps emitting quality-gated
    // crops (throttled to ~120 ms apart for temporal diversity); the screen
    // collects as many as it needs and then deactivates the camera.
    if (nowMs - lastEmit.value < 120) return;
    lastEmit.value = nowMs;
    const srcBuf = resize(frame, {
      crop: roiCrop, scale: { width: 256, height: 256 }, rotation: '270deg', pixelFormat: 'rgba', dataType: 'uint8',
    });
    const m5 = mesh5(lm, 256);
    onReady(Array.from(srcBuf as Uint8Array), 256, m5, passive, gain);
  }, [challenges, detector.model, landmarker.model, antispoof.model, paused]);
}

// ---- worklet helpers --------------------------------------------------------
function norm(buf: Float32Array, lo: number) {
  'worklet';
  // 0..255 -> [lo,1]; lo=0 gives [0,1], lo=-1 gives [-1,1].
  const a = (1 - lo) / 255;
  for (let i = 0; i < buf.length; i++) buf[i] = buf[i] * a + lo;
}

function clamp01(v: number): number { 'worklet'; return v < 0 ? 0 : v > 1 ? 1 : v; }

function landmarkTensor(out: any[]): Float32Array {
  'worklet';
  // FaceLandmarker emits a 1404-length landmark tensor (+ a presence score).
  return (out[0].length >= 1404 ? out[0] : out[1]) as Float32Array;
}

function toPointsNorm(raw: Float32Array): Pt[] {
  'worklet';
  // FaceLandmarker emits coordinates in its 256x256 input pixel space.
  const pts: Pt[] = [];
  for (let i = 0; i + 2 < raw.length; i += 3) pts.push({ x: raw[i] / 256, y: raw[i + 1] / 256 });
  return pts;
}

function mesh5(lm: Pt[], size: number): number[][] {
  'worklet';
  const m = (a: number, b: number) => [((lm[a].x + lm[b].x) / 2) * size, ((lm[a].y + lm[b].y) / 2) * size];
  return [
    m(33, 133), m(362, 263),
    [lm[1].x * size, lm[1].y * size],
    [lm[61].x * size, lm[61].y * size],
    [lm[291].x * size, lm[291].y * size],
  ];
}

function satisfied(
  c: Challenge, lm: Pt[],
  armed: { value: number }, base: { value: number }, earOpen: { value: number }, n: number,
): boolean {
  'worklet';
  if (c === 'blink') {
    // Adaptive-but-robust blink. We learn this eye's open baseline (running
    // max) so glasses-limited closures still count, but CLAMP the baseline
    // below 0.42: FaceLandmarker throws EAR spikes up to ~0.75 during head
    // motion, and an unclamped max would poison the reopen threshold so the
    // eyes could never read "open" again (permanent-armed stall). Closure
    // fires below the smaller of an absolute 0.24 and 72% of the baseline;
    // reopen above 82% of the baseline. A static photo's EAR is constant, so
    // it never dips and never arms.
    const e = ear(lm);
    if (e > earOpen.value && e < 0.42) earOpen.value = e;
    const base = earOpen.value > 0.24 ? earOpen.value : 0.34;
    const closeAt = 0.24 < base * 0.72 ? 0.24 : base * 0.72;
    if (e < closeAt) armed.value = 1;
    else if (armed.value >= 1 && e > base * 0.82) return true;
    return false;
  }
  if (c === 'smile') {
    const w = mouthWidthRatio(lm);
    // Track the NEUTRAL (narrowest) mouth as the baseline; a smile widens it.
    // 8% gain: measured natural smiles widen the mouth 6-12% over neutral.
    if (base.value === 0 || w < base.value) base.value = w;
    return w > base.value * 1.08;
  }
  // Front camera is mirrored: turning the head to the user's left lowers the yaw
  // ratio, so the conditions are flipped relative to a non-mirrored feed.
  if (c === 'turn-left') return yawRatio(lm) < 1 / TH.yawTurn;
  return yawRatio(lm) > TH.yawTurn; // turn-right
}
