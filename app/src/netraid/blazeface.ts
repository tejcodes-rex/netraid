// BlazeFace (short-range) SSD post-processing: anchor generation, box/keypoint
// decode, sigmoid scoring and non-max suppression. Matches the MediaPipe
// short-range model (128x128 input, 896 anchors, 6 keypoints).
//
// All functions carry the 'worklet' directive so they can run inside the
// vision-camera frame processor thread.

export interface Detection {
  score: number;
  // normalized [0,1] in input space
  xmin: number; ymin: number; xmax: number; ymax: number;
  // 6 keypoints: rightEye, leftEye, nose, mouth, rightEar, leftEar
  keypoints: { x: number; y: number }[];
}

const INPUT = 128;
const NUM_KEYPOINTS = 6;

// SSD anchor options for BlazeFace short-range (MediaPipe defaults).
const STRIDES = [8, 16, 16, 16];
const ANCHOR_OFFSET = 0.5;

// Build the 896 anchors (normalized centers). Pure and worklet-safe; recomputed
// per call (896 iterations, cheap) instead of caching across the JS/worklet edge.
export function buildAnchors(): { cx: number; cy: number }[] {
  'worklet';
  const anchors: { cx: number; cy: number }[] = [];
  let layerId = 0;
  const numLayers = STRIDES.length;
  while (layerId < numLayers) {
    let lastSame = layerId;
    while (lastSame < numLayers && STRIDES[lastSame] === STRIDES[layerId]) lastSame++;
    const repeats = lastSame - layerId;
    // Anchors per location = repeats * (len(aspect_ratios) + interpolated_scale_aspect_ratio)
    // = repeats * (1 + 1) = 2 * repeats for the short-range model. This yields
    // 16*16*2 + 8*8*6 = 896 anchors, matching the model's score/box tensors.
    const perLocation = repeats * 2;
    const stride = STRIDES[layerId];
    const featMap = Math.ceil(INPUT / stride);
    for (let y = 0; y < featMap; y++) {
      for (let x = 0; x < featMap; x++) {
        for (let r = 0; r < perLocation; r++) {
          anchors.push({ cx: (x + ANCHOR_OFFSET) / featMap, cy: (y + ANCHOR_OFFSET) / featMap });
        }
      }
    }
    layerId = lastSame;
  }
  return anchors;
}

function sigmoid(x: number): number {
  'worklet';
  return 1 / (1 + Math.exp(-x));
}

/**
 * Decode raw model outputs.
 * @param boxes  Float32Array length 896*16 (4 box coords + 12 keypoint coords)
 * @param scores Float32Array length 896 (raw logits)
 */
export function decode(boxes: Float32Array, scores: Float32Array, scoreThreshold = 0.6): Detection[] {
  'worklet';
  const anchors = buildAnchors();
  const out: Detection[] = [];
  for (let i = 0; i < anchors.length; i++) {
    const score = sigmoid(scores[i]);
    if (score < scoreThreshold) continue;
    const o = i * 16;
    const a = anchors[i];
    const cx = boxes[o] / INPUT + a.cx;
    const cy = boxes[o + 1] / INPUT + a.cy;
    const w = boxes[o + 2] / INPUT;
    const h = boxes[o + 3] / INPUT;
    const kp: { x: number; y: number }[] = [];
    for (let k = 0; k < NUM_KEYPOINTS; k++) {
      kp.push({
        x: boxes[o + 4 + k * 2] / INPUT + a.cx,
        y: boxes[o + 4 + k * 2 + 1] / INPUT + a.cy,
      });
    }
    out.push({
      score,
      xmin: cx - w / 2, ymin: cy - h / 2, xmax: cx + w / 2, ymax: cy + h / 2,
      keypoints: kp,
    });
  }
  return out;
}

function iou(a: Detection, b: Detection): number {
  'worklet';
  const x1 = Math.max(a.xmin, b.xmin), y1 = Math.max(a.ymin, b.ymin);
  const x2 = Math.min(a.xmax, b.xmax), y2 = Math.min(a.ymax, b.ymax);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
  const areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
  return inter / (areaA + areaB - inter + 1e-6);
}

export function nms(dets: Detection[], iouThreshold = 0.3): Detection[] {
  'worklet';
  const sorted = dets.slice().sort((p, q) => q.score - p.score);
  const keep: Detection[] = [];
  for (const d of sorted) {
    let ok = true;
    for (const k of keep) if (iou(k, d) >= iouThreshold) { ok = false; break; }
    if (ok) keep.push(d);
  }
  return keep;
}

/** Largest-area detection, or null. */
export function pickPrimary(dets: Detection[]): Detection | null {
  'worklet';
  if (!dets.length) return null;
  let best = dets[0];
  let bestArea = (best.xmax - best.xmin) * (best.ymax - best.ymin);
  for (const d of dets) {
    const area = (d.xmax - d.xmin) * (d.ymax - d.ymin);
    if (area > bestArea) { best = d; bestArea = area; }
  }
  return best;
}
