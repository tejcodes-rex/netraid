// Camera-free execution of the full recognition pipeline on bundled frames:
//   detect (BlazeFace) -> ROI -> landmarks (FaceLandmarker) -> umeyama align
//   -> flip-TTA embed -> cosine match.
// This is the SAME code and the SAME .tflite models as the live camera path;
// only the frame source differs. It powers the Pipeline Demo screen, which is
// how the pipeline is demonstrated deterministically on Android and on iOS
// simulators (which have no camera).

import type { TensorflowModel } from 'react-native-fast-tflite';
import { decode, nms, pickPrimary } from './blazeface';
import { meshTo5, umeyama, warpAffine, REF5 } from './align';
import { cosine } from './math';
import { embedTTA } from './recognition';
import { base64ToBytes, cropRGB, resizeRGB, toFloat01 } from './imaging';
import { DEMO_ASSETS, DEMO_FRAME_SIZE, type DemoAsset } from './demoAssets';
import { DEFAULT_CONFIG, type Embedding } from './types';
import type { Pt } from './liveness';

export interface DemoFaceRun {
  label: string;
  identity: string;
  /** Displayable PNG (require() id) of the exact processed frame. */
  image: number;
  detectScore: number;
  timings: { detectMs: number; landmarksMs: number; alignMs: number; embedMs: number };
  embedding: Embedding;
}

export interface DemoPair {
  a: string;
  b: string;
  aImage: number;
  bImage: number;
  genuine: boolean; // ground truth: same identity?
  score: number;
  accepted: boolean; // pipeline verdict at the operating threshold
  correct: boolean;
}

export interface DemoReport {
  runs: DemoFaceRun[];
  pairs: DemoPair[];
  allCorrect: boolean;
  avgTotalMs: number;
  threshold: number;
}

export function runDemoFace(
  detector: TensorflowModel,
  landmarker: TensorflowModel,
  asset: DemoAsset,
): DemoFaceRun {
  const S = DEMO_FRAME_SIZE;
  const frame = base64ToBytes(asset.b64);

  // 1) Detect on 128x128.
  let t = Date.now();
  const detOut = detector.runSync([toFloat01(resizeRGB(frame, S, S, 128, 128))]);
  const boxesT = (detOut[0].length > detOut[1].length ? detOut[0] : detOut[1]) as Float32Array;
  const scoresT = (detOut[0].length > detOut[1].length ? detOut[1] : detOut[0]) as Float32Array;
  const box = pickPrimary(nms(decode(boxesT, scoresT, 0.18)));
  const detectMs = Date.now() - t;
  if (!box) throw new Error(`no face detected in ${asset.label}`);

  // 2) Landmarks on a tight square ROI around the detected box (1.25x).
  t = Date.now();
  const cx = ((box.xmin + box.xmax) / 2) * S;
  const cy = ((box.ymin + box.ymax) / 2) * S;
  let roiSize = Math.max(box.xmax - box.xmin, box.ymax - box.ymin) * S * 1.25;
  roiSize = Math.min(S, Math.max(S * 0.3, roiSize));
  const roi = cropRGB(frame, S, S, cx - roiSize / 2, cy - roiSize / 2, roiSize, roiSize);
  const roi256 = resizeRGB(roi.rgb, roi.w, roi.h, 256, 256);
  const lmOut = landmarker.runSync([toFloat01(roi256)]);
  const raw = (lmOut[0].length >= 1404 ? lmOut[0] : lmOut[1]) as Float32Array;
  const lm: Pt[] = [];
  for (let i = 0; i + 2 < raw.length; i += 3) lm.push({ x: raw[i], y: raw[i + 1] });
  const landmarksMs = Date.now() - t;

  // 3) Umeyama alignment to the ArcFace reference, warp to 112x112.
  t = Date.now();
  const M = umeyama(meshTo5(lm), REF5);
  const aligned = warpAffine(roi256, 256, 256, M);
  const alignMs = Date.now() - t;

  // 4) Flip-TTA embedding.
  t = Date.now();
  const embedding = embedTTA(aligned);
  const embedMs = Date.now() - t;

  return {
    label: asset.label,
    identity: asset.identity,
    image: asset.image,
    detectScore: box.score,
    timings: { detectMs, landmarksMs, alignMs, embedMs },
    embedding,
  };
}

/** Score every pair of demo faces and check each verdict against ground truth. */
export function buildReport(runs: DemoFaceRun[]): DemoReport {
  const threshold = DEFAULT_CONFIG.matchThreshold;
  const pairs: DemoPair[] = [];
  for (let i = 0; i < runs.length; i++) {
    for (let j = i + 1; j < runs.length; j++) {
      const genuine = runs[i].identity === runs[j].identity;
      const score = cosine(runs[i].embedding, runs[j].embedding);
      const accepted = score >= threshold;
      pairs.push({
        a: runs[i].label, b: runs[j].label,
        aImage: runs[i].image, bImage: runs[j].image,
        genuine, score, accepted,
        correct: accepted === genuine,
      });
    }
  }
  const avgTotalMs =
    runs.reduce((s, r) =>
      s + r.timings.detectMs + r.timings.landmarksMs + r.timings.alignMs + r.timings.embedMs, 0,
    ) / Math.max(1, runs.length);
  return {
    runs,
    pairs,
    allCorrect: pairs.every((p) => p.correct),
    avgTotalMs,
    threshold,
  };
}

export { DEMO_ASSETS };
