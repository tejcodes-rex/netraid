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
import { ear, interOcular, mouthWidthRatio, yawRatio, gestureSatisfied } from './liveness';
import { umeyama, warpAffine, REF5 } from './align';
import { assessCrop, MIN_SHARPNESS } from './quality';
import { DEFAULT_CONFIG, type Challenge } from './types';
import type { Pt } from './liveness';

export interface FrameCallbacks {
  onChallengeProgress: (done: Challenge[]) => void;
  onCapture: (
    rgb: Uint8Array,
    /** MiniFASNet P(real) for this capture. */
    passiveScore: number,
    /** MiniFASNet P(screen-replay) for this capture. */
    passiveScreen: number,
    sharpness: number,
  ) => void;
  onNoFace: () => void;
  /** Every processed frame's passive reading, not just the three that become
   * captures. A threshold cannot be chosen from three samples; it needs the
   * distribution, and the distribution is what this carries. */
  onPassiveSample?: (real: number, screen: number) => void;
  /** The tracked face was lost or replaced between proving liveness and being
   * photographed for recognition, so the two are not about the same subject. */
  onContinuityBreak?: (reason: 'lost' | 'jump') => void;
  /** One face-chromaticity measurement during flash slot `slot`. Fires only
   * while the screen is driving the chroma challenge (see chroma.ts). */
  onChromaSample?: (
    slot: number, r: number, g: number, b: number,
    bgR: number, bgG: number, bgB: number,
    cR: number, cG: number, cB: number,
    pR: number, pG: number, pB: number,
    cY: number, pY: number,
  ) => void;
  /** A post-gesture frame that reads as a presentation attack. Reported as it
   * happens so the UI can say so immediately, rather than showing a green
   * "liveness passed" and contradicting itself at the final verdict. */
  onSpoofEvidence?: (screen: number, real: number) => void;
  /** The current challenge was not completed within its reaction window. The
   * screen must restart the whole attempt with a fresh random set: keeping
   * partial progress and re-rolling only the stuck gesture would hand a
   * replayed video unlimited attempts at the gestures it happens to contain. */
  onChallengeTimeout?: (reason: 'gesture' | 'capture') => void;
  onDebug?: (msg: string) => void;
}

const ROI_SRC = 256; // resolution of the ROI buffer we warp the aligned crop from
// FaceLandmarker face-presence logit gate. The model returns a strongly negative
// value when no face is in the square, positive when a face is well-framed.
const FACE_PRESENCE_MIN = -11; // landmark-validity gate (BlazeFace already gates the
// background out; this just skips frames where the mesh is too uncertain to use).

/**
 * @param resetKey bump to fully reset the liveness FSM (new attempt). Every
 * challenge timeout bumps it: a stalled step discards all progress and the
 * screen issues a brand-new random set, so no attempt can be stretched out.
 */
export function useNetraFrameProcessor(
  challenges: Challenge[], cb: FrameCallbacks, resetKey = 0, paused = false,
  /** Drives the chroma challenge. The screen owns the flash sequence and hands
   * the worklet the slot it is currently displaying; -1 means no flash is up,
   * which is also the state the sampler stays silent in. `settleUntil` is the
   * wall-clock instant the post-flash colour cast has cleared: until then the
   * sensor is still re-converging and no frame is evidence of anything. */
  flash?: { slot: { value: number }; since: { value: number }; settleUntil: { value: number } },
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
  // Post-prompt neutral-frame counter (see gestureSatisfied). Shared values are
  // the only thing reliably captured into the worklet, so the per-step budget
  // travels the same way instead of as a captured module constant.
  const neutralN = useSharedValue(0);
  const stepBudgetMs = useSharedValue(DEFAULT_CONFIG.challengeTimeoutMs);
  // Disarmed unless the screen-replay gate is enforced: probabilities never
  // exceed 1, so 2 makes the in-worklet early abort unreachable while the gate
  // is only being measured. Keeping the threshold in a shared value (rather
  // than branching in the worklet) keeps the hot path free of extra captures.
  const screenMax = useSharedValue(
    DEFAULT_CONFIG.screenSpoofMode === 'enforce' ? DEFAULT_CONFIG.screenSpoofMax : 2,
  );
  const spoofN = useSharedValue(0);
  // Fallbacks so the worklet always closes over a concrete object: a screen
  // that does not drive the chroma challenge simply leaves the slot at -1.
  const idleSlot = useSharedValue(-1);
  const idleSince = useSharedValue(0);
  const idleSettle = useSharedValue(0);
  // Last known face region, in frame pixels. The chroma challenge measures
  // through it while the screen is a coloured lamp, because the detector cannot
  // be relied on to find a face under saturated light.
  const roiX = useSharedValue(0);
  const roiY = useSharedValue(0);
  const roiS = useSharedValue(0);
  // CONTINUITY BINDING state. A liveness proof is only worth anything if it is
  // bound to the frames the identity is read from; see the check below.
  const contArmed = useSharedValue(0);
  const contCx = useSharedValue(0);
  const contCy = useSharedValue(0);
  const contSize = useSharedValue(0);
  const contTs = useSharedValue(0);
  // Last ROI colour balance, for the post-flash colour-recovery gate.
  const prevChR = useSharedValue(0);
  const prevChG = useSharedValue(0);
  const prevChB = useSharedValue(0);
  const flashSlot = flash ? flash.slot : idleSlot;
  const flashSince = flash ? flash.since : idleSince;
  const flashSettleUntil = flash ? flash.settleUntil : idleSettle;

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
    neutralN.value = 0;
    spoofN.value = 0;
    flashSettleUntil.value = 0;
    roiS.value = 0;
    contArmed.value = 0;
    contSize.value = 0;
  }, [resetKey, step, blinkClosed, baseMouth, lastEmit, stepStart, holdN, lastGesture, blinkOpen, neutralN, spoofN, flashSettleUntil, roiS, contArmed, contSize]);

  const onProgress = Worklets.createRunOnJS(cb.onChallengeProgress);
  const onNoFace = Worklets.createRunOnJS(cb.onNoFace);
  const onTimeout = Worklets.createRunOnJS(cb.onChallengeTimeout ?? (() => {}));
  const onDebug = Worklets.createRunOnJS(cb.onDebug ?? (() => {}));
  const onChroma = Worklets.createRunOnJS(cb.onChromaSample ?? (() => {}));
  const onSpoof = Worklets.createRunOnJS(cb.onSpoofEvidence ?? (() => {}));
  const onBreak = Worklets.createRunOnJS(cb.onContinuityBreak ?? (() => {}));
  const onPassive = Worklets.createRunOnJS(cb.onPassiveSample ?? (() => {}));
  // Align + warp on the JS thread, then hand the finished crop to the caller.
  const onReady = Worklets.createRunOnJS(
    (src: number[], size: number, mesh5: number[][], passive: number, screen: number, gain: number) => {
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
        cb.onCapture(rgb, passive, screen, q.sharpness);
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

    // 1a) CHROMA CHALLENGE sampling, BEFORE detection and on the last known face
    // region.
    //
    // This used to sit after detection and after the landmark-validity gate,
    // and so it never ran: under a saturated red, green or blue flash BlazeFace
    // stops finding the face, the frame returns early, and the challenge
    // measured nothing. On the target handset every attempt reported
    // `usable: false` with zero samples, which is why the barrier meant to stop
    // a recorded replay was silently absent while a replay was passing.
    //
    // The subject is asked to hold still for the ~1.8 s the flash owns the
    // screen, so the region located in the frames immediately before it is the
    // right one to measure through.
    const slotNow = flashSlot.value;
    if (slotNow >= 0) {
      // Skip the first 150 ms of a slot: a frame caught while the display was
      // still switching belongs to the previous colour.
      if (roiS.value > 0 && nowMs - flashSince.value > 150) {
        const rc = { x: roiX.value, y: roiY.value, width: roiS.value, height: roiS.value };
        const cU = resize(frame, {
          crop: rc, scale: { width: 32, height: 32 }, rotation: '270deg',
          pixelFormat: 'rgba', dataType: 'uint8',
        });
        let sr = 0, sg = 0, sb = 0;
        // CENTRE vs RIM of the face region, the two halves of the verdict.
        // Skin is close to Lambertian, so a patch returns our light in
        // proportion to the cosine of the angle between its surface normal and
        // the phone. Nose and forehead face the screen head-on; cheeks and
        // jawline curve away at a grazing angle. A photograph or a display is
        // flat, so every part of it shares one normal and responds uniformly,
        // whatever its content or brightness. That difference is geometry, and
        // no recording can carry it.
        let cr = 0, cg = 0, cb = 0, cn = 0, cy = 0;
        let pr = 0, pg = 0, pb = 0, pn = 0, py = 0;
        for (let y = 0; y < 32; y++) {
          for (let x = 0; x < 32; x++) {
            const o = (y * 32 + x) * 4;
            const R = cU[o], G = cU[o + 1], B = cU[o + 2];
            sr += R; sg += G; sb += B;
            const dx = x - 15.5, dy = y - 15.5;
            const r2 = dx * dx + dy * dy;
            // Inner disc: nose, mouth and forehead, facing the phone.
            // Rec.601 luma. The SHAPE verdict is taken on this, not on colour:
            // Lambertian falloff changes how much light a patch returns, and
            // chromaticity divides intensity out by construction.
            const Y = 0.299 * R + 0.587 * G + 0.114 * B;
            if (r2 <= 42) { cr += R; cg += G; cb += B; cy += Y; cn++; }
            // Outer annulus, kept inside the crop so it stays on the face
            // rather than sampling whatever is beside it.
            else if (r2 >= 110 && r2 <= 210) { pr += R; pg += G; pb += B; py += Y; pn++; }
          }
        }
        const tot = sr + sg + sb;
        const ctot = cr + cg + cb;
        const ptot = pr + pg + pb;
        // The SAME measurement on the frame border, away from the face. The
        // sensor's auto-white-balance corrects the whole image, so it moves
        // both regions together and cancels in the difference; our flash does
        // not, because it falls off with distance and only reaches the near
        // subject. See the ChromaSample docs in chroma.ts.
        const bU = resize(frame, {
          scale: { width: 24, height: 24 }, rotation: '270deg',
          pixelFormat: 'rgba', dataType: 'uint8',
        });
        let br = 0, bg = 0, bb = 0;
        for (let y = 0; y < 24; y++) {
          for (let x = 0; x < 24; x++) {
            if (x > 3 && x < 20 && y > 3 && y < 20) continue; // border ring only
            const o = (y * 24 + x) * 4;
            br += bU[o]; bg += bU[o + 1]; bb += bU[o + 2];
          }
        }
        const btot = br + bg + bb;
        if (tot > 0 && btot > 0 && ctot > 0 && ptot > 0 && cn > 0 && pn > 0) {
          onChroma(
            slotNow,
            sr / tot, sg / tot, sb / tot,
            br / btot, bg / btot, bb / btot,
            cr / ctot, cg / ctot, cb / ctot,
            pr / ptot, pg / ptot, pb / ptot,
            cy / cn, py / pn,
          );
        }
      }
      // Hold the capture-phase budget: no recognition frame may be emitted while
      // the flash is up, so charging that time to the allowance spends a quarter
      // of it before the first capture is even permitted.
      stepStart.value = 0;
      // The detector is blind under this light, so no "face missing" time may
      // accrue here. The POSITION check deliberately still spans the flash: the
      // face found after it must be where the face before it was.
      contTs.value = nowMs;
      // Nothing else runs while the screen is a coloured lamp: the gesture FSM
      // would read distorted landmarks, MiniFASNet would judge a colour-cast
      // frame, and any capture emitted here would carry the cast into the
      // embedding. The flash phase measures, and only measures.
      return;
    }

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
      // CONTINUITY: once a live subject has proved a gesture, they must stay in
      // frame until the identity capture. Vanishing long enough for a phone or
      // a print to be raised into position voids the attempt.
      if (contArmed.value === 1 && nowMs - contTs.value > 600) {
        contArmed.value = 0;
        onBreak('lost');
        return;
      }
      onNoFace();
      return;
    }

    // 2) Map the upright-128 box to a TIGHT square crop in original frame pixels.
    //    The detect view was rotated 90deg CW, so a point (u,v) in the upright
    //    view came from (v, 1-u) in the pre-rotation square.
    const bcu = (sq.xmin + sq.xmax) / 2, bcv = (sq.ymin + sq.ymax) / 2;
    const bw = sq.xmax - sq.xmin, bh = sq.ymax - sq.ymin;

    // 2b) CONTINUITY BINDING.
    //
    // Passing the gestures proves that A live human was in front of this camera.
    // It says nothing about WHO is photographed for recognition a moment later.
    // Without binding the two, the cheapest attack in the world defeats every
    // barrier above: perform the blinks and turns with your own face, then raise
    // a photograph or a phone into frame for the capture. Liveness passes on the
    // attacker, identity passes on the victim, and neither check is wrong on its
    // own.
    //
    // So from the first satisfied gesture until the capture completes, the
    // subject must remain the SAME tracked face: continuously present, and
    // moving continuously. Substituting a print or a display for a head means
    // either a gap in frame or a discontinuous jump in where the face is and how
    // big it is, because a hand cannot put a phone exactly where a head was
    // inside one frame interval.
    //
    // Geometric and model-free, so unlike the CNN it does not vary with
    // illumination or sensor response: it holds in harsh sun and in near dark.
    const bsize = Math.max(bw, bh);
    if (contArmed.value === 1 && contSize.value > 0) {
      const dx = bcu - contCx.value, dy = bcv - contCy.value;
      const jump = Math.sqrt(dx * dx + dy * dy);
      const ratio = bsize / contSize.value;
      // Generous: a real user leaning in, or a hand tremor, must never trip it.
      // A swap does not look like either.
      if (jump > 0.22 || ratio > 1.7 || ratio < 0.59) {
        contArmed.value = 0;
        onBreak('jump');
        return;
      }
    }
    contCx.value = bcu; contCy.value = bcv; contSize.value = bsize; contTs.value = nowMs;
    // Armed once the LAST gesture is satisfied, not the first.
    //
    // Arming at the first one made this a false-positive machine: six of eight
    // genuine attempts were voided, because "turn your head left" is a large,
    // deliberate movement of the face box and the check cannot tell it from a
    // substitution. The subject is being ASKED to move during the gestures.
    //
    // The window that actually needs binding is the one after the proof exists
    // and before the identity is read: gestures complete, then flash, then
    // capture. That is where a photograph would be raised into frame, and
    // nothing legitimate moves much there because the user is being told to
    // hold still.
    if (step.value >= challenges.length) contArmed.value = 1;
    const faceFrac = Math.max(bw, bh); // fraction of the square the face fills
    const fitPct = Math.round(Math.min(1, faceFrac / 0.62) * 100);

    // The user centers their face in the on-screen circle (which is the frame
    // center), so crop a CENTERED square sized to the detected face and rotate it
    // upright for the landmark model. Mapping BlazeFace's rotated box back to frame
    // pixels is error-prone; centering is robust given the guide circle.
    let fsize = Math.max(bw, bh) * side * 1.25;
    if (fsize > side) fsize = side; else if (fsize < side * 0.3) fsize = side * 0.3;
    const roiCrop = { x: (W - fsize) / 2, y: (H - fsize) / 2, width: fsize, height: fsize };
    // Remembered for the chroma challenge, which measures through it while the
    // detector is blinded by the flash (see 1a).
    roiX.value = roiCrop.x; roiY.value = roiCrop.y; roiS.value = fsize;
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

    // 4) ACTIVE liveness FSM, run as a REACTION TEST rather than a gesture
    // search. Each step opens a window when its prompt is issued; the gesture
    // counts only if it starts from a confirmed-neutral face inside that window
    // and no earlier than the human reaction floor (see gestureSatisfied). A
    // recording plays on its own clock, so its gestures land at an arbitrary
    // phase relative to the prompt rather than just after it.
    const s = step.value;
    if (s < challenges.length) {
      if (stepStart.value === 0) stepStart.value = nowMs;
      const elapsed = nowMs - stepStart.value;
      if (elapsed > stepBudgetMs.value) {
        // Window closed. Hand back to the screen, which restarts the ENTIRE
        // attempt with a fresh random set. Re-rolling only the stuck gesture
        // (the previous behaviour) let an attacker keep the flow alive until
        // the gestures their recording happens to contain came up.
        stepStart.value = 0;
        onTimeout('gesture');
        return;
      }
      // Fresh tracking state whenever the CURRENT gesture changes, whether by
      // normal advance or by a restart: a stale smile baseline, blink counter
      // or neutral count from a previous gesture must never leak into this one.
      if (lastGesture.value !== challenges[s]) {
        lastGesture.value = challenges[s];
        blinkClosed.value = 0;
        baseMouth.value = 0;
        holdN.value = 0;
        blinkOpen.value = 0;
        neutralN.value = 0;
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
      const ok = gestureSatisfied(
        challenges[s], lm,
        { armed: blinkClosed, base: baseMouth, earOpen: blinkOpen, neutral: neutralN },
        elapsed,
      );
      const isBlink = challenges[s] === 'blink';
      if (ok) holdN.value += 1;
      else if (!isBlink) holdN.value = 0;
      if (ok && (isBlink || holdN.value >= 2)) {
        step.value = s + 1;
        stepStart.value = 0;
        holdN.value = 0;
        onProgress(challenges.slice(0, s + 1));
        blinkClosed.value = 0; baseMouth.value = 0; neutralN.value = 0;
      }
      return;
    }

    // 4b) Capture-phase timeout: liveness passed but a clean frontal capture is
    // not arriving (user drifted off-angle, walked away, camera covered). After
    // 8 s hand control back to the screen, which restarts with a fresh
    // challenge set instead of stalling forever.
    // The flash has gone dark but the front camera is still re-converging its
    // exposure and white balance. Those frames carry a colour cast: embedding
    // one shifts the probe off the template, and MiniFASNet, which scores
    // colour and texture, reads the cast as an attack. Drop them before any of
    // that work is done, and keep holding the capture budget, which is not
    // running yet because no capture is permitted yet.
    if (nowMs < flashSettleUntil.value) {
      stepStart.value = 0;
      contTs.value = nowMs;
      return;
    }

    if (challenges.length > 0) {
      if (stepStart.value === 0) stepStart.value = nowMs;
      else if (nowMs - stepStart.value > 8000) {
        stepStart.value = 0;
        onTimeout('capture');
        return;
      }
    }

    // 4b2) COLOUR-RECOVERY GATE.
    //
    // After the flash the front camera is still walking its white balance back
    // from saturated red, green or blue. A fixed settle window guessed at how
    // long that takes and guessed wrong: captures were still arriving with a
    // heavy cast, measured as channel means of [88,134,101] and [122,99,99]
    // where a neutral skin crop reads about [102,119,137]. A cast crop shifts
    // the recognition embedding AND is read as an attack by MiniFASNet, which
    // scores colour and texture, which is most of why the passive readings
    // swung between 0.03 and 0.88 on the same live face all evening.
    //
    // So stop guessing the duration and measure the recovery: a frame is only
    // eligible once the ROI's colour balance has stopped moving between
    // frames. Self-timing, so it costs nothing when there was no flash and
    // takes as long as it needs on a slow sensor.
    const stU = resize(frame, {
      crop: roiCrop, scale: { width: 16, height: 16 }, rotation: '270deg',
      pixelFormat: 'rgba', dataType: 'uint8',
    });
    let tr = 0, tg = 0, tb = 0;
    for (let p = 0; p < 256; p++) {
      const o = p * 4;
      tr += stU[o]; tg += stU[o + 1]; tb += stU[o + 2];
    }
    const ttot = tr + tg + tb;

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

    // 5) PASSIVE anti-spoof (80x80). Use rgba + strip alpha (the plugin's
    // 3-channel 'rgb' is alpha-first on this device).
    //
    // Input convention, settled by running identical tensors through the
    // original 2.7_80x80_MiniFASNetV2.pth and through the shipped .tflite:
    //
    //   TFLite fed [0,1]    vs PyTorch:  max|diff| = 3e-8   <- same function
    //   TFLite fed 0-255    vs PyTorch:  max|diff| = 0.99   <- different function
    //
    // So the model wants BGR in [0,1], exactly as the reference pipeline's
    // transforms.ToTensor() produces. This code fed raw 0-255, which is not a
    // scaled version of the right answer but a different function altogether:
    // on random noise it returned P(real) = 0.995, confidently calling static a
    // live face. That is the whole explanation for a passive barrier whose
    // output swung between 0.001 and 0.93 on one face and never separated a
    // replay from a person.
    //
    // The earlier note here claimed [0,1] "emits a constant screen-fake for
    // everything". It does do that on LFW-style stills, but those are
    // compressed web photographs, not live sensor captures, so they are not
    // evidence about this model's behaviour on a phone camera. The conversion
    // check is evidence, and it is decisive.
    //
    // Deliberately NOT gain-boosted: it judges capture authenticity and should
    // see the sensor's real response, not a synthetically brightened one.
    // MiniFASNet gets its OWN, WIDER crop.
    //
    // `roiCrop` is 1.25x the detected face box, which is right for the landmark
    // and recognition models: they want the face to fill the frame. MiniFASNet
    // does not. It judges authenticity partly from what surrounds a face, and
    // feeding it a crop tighter than it was trained on puts it out of
    // distribution, where its output is not merely biased but unstable. That is
    // what produced P(real) of 0.93, 0.01 and 0.02 on three consecutive frames
    // of one live face, and it is why the passive barrier could never be armed.
    //
    // 2.7x the face box, which is what these weights are named for: in
    // Silent-Face the leading number IS the factor the detected box is expanded
    // by before cropping. An earlier sweep here preferred 1.6x, but that sweep
    // was run with 0-255 input, which the conversion check has since shown is a
    // different function, so its answer was measuring an artefact.
    //
    // roiCrop is 1.25x the face box, so 2.7x of the box is 2.16x of roiCrop.
    // Clamped to the frame, and centred for the same reason roiCrop is: the
    // guide circle puts the face at frame centre, and mapping the detector's
    // rotated box back to frame pixels is error-prone.
    // The requested crop routinely exceeds the frame, because the guide circle
    // asks the user to FILL it: at that framing 2.16x of roiCrop is larger than
    // the sensor's short side. Clamping it, as this did, silently changes the
    // crop scale as a function of how close the user is standing, so the model
    // sees a different framing every attempt. That is what produced P(real) of
    // 0.710 on one attempt and 0.005 on the next, from the same face.
    //
    // Silent-Face pads in this situation rather than clamping, and so do we:
    // take the largest valid region, resize it to the proportionally smaller
    // size, and centre it in the 80x80 with the border replicated. The face
    // then occupies the same fraction of the input whatever the distance.
    const spDesired = fsize * 2.16;
    const spSize = spDesired <= side ? spDesired : side;
    const spInner = spDesired <= side ? 80 : Math.max(16, Math.round((80 * side) / spDesired));
    const spCrop = { x: (W - spSize) / 2, y: (H - spSize) / 2, width: spSize, height: spSize };
    const spU = resize(frame, {
      crop: spCrop, scale: { width: spInner, height: spInner }, rotation: '270deg',
      pixelFormat: 'rgba', dataType: 'uint8',
    });
    const spOff = Math.floor((80 - spInner) / 2);
    const sp80 = 80 * 80;
    const spIn = new Float32Array(sp80 * 3);
    for (let y = 0; y < 80; y++) {
      // Border replication outside the valid region. With spInner === 80 this
      // reduces to a straight copy, so the common case costs nothing extra.
      const sy = y - spOff < 0 ? 0 : (y - spOff >= spInner ? spInner - 1 : y - spOff);
      for (let x = 0; x < 80; x++) {
        const sx = x - spOff < 0 ? 0 : (x - spOff >= spInner ? spInner - 1 : x - spOff);
        const o = (sy * spInner + sx) * 4;
        const q = (y * 80 + x) * 3;
        // BGR, scaled to [0,1]: the convention the conversion check proved.
        spIn[q] = spU[o + 2] / 255; spIn[q + 1] = spU[o + 1] / 255; spIn[q + 2] = spU[o] / 255;
      }
    }
    const prob = spoof.runSync([spIn])[0] as Float32Array;
    // Silent-Face emits 3 classes: [print-fake, real, screen-fake]. Both the
    // "real" and the "screen" readings travel to the verdict: P(real) alone
    // discards the model's most direct evidence that it is looking at a display.
    const passive = prob.length > 1 ? prob[1] : prob[0]; // P(real)
    const screen = prob.length > 2 ? prob[2] : 0; // P(screen-replay)
    onPassive(passive, screen);

    // Report a presentation attack THE MOMENT it is seen, and stop collecting
    // captures from it. Previously the screen showed a green "liveness passed"
    // as soon as the gestures were done and only contradicted itself seconds
    // later at the verdict, which reads as a bug even when the rejection is
    // correct. Two consecutive frames are required: one high reading is sensor
    // noise, and rejecting a real person on it is the worse error.
    if (screen > screenMax.value) {
      spoofN.value += 1;
      if (spoofN.value >= 2) {
        onSpoof(screen, passive);
        return;
      }
    } else if (spoofN.value > 0) {
      spoofN.value -= 1;
    }

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
    onReady(Array.from(srcBuf as Uint8Array), 256, m5, passive, screen, gain);
  }, [challenges, detector.model, landmarker.model, antispoof.model, paused, flashSlot, flashSince, flashSettleUntil, roiX, roiY, roiS,
      contArmed, contCx, contCy, contSize, contTs, prevChR, prevChG, prevChB]);
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

