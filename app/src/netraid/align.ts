// Face alignment: estimate a similarity transform (scale + rotation + shift)
// from 5 facial landmarks to the ArcFace reference, then warp the source image to
// a 112x112 RGB crop. This is the single biggest lever on recognition accuracy;
// it replicates insightface's norm_crop. Worklet-safe (pure functions).

import type { Pt } from './liveness';

export const ALIGN_SIZE = 112;

// ArcFace canonical 5-point reference for 112x112 (left eye, right eye, nose,
// left mouth, right mouth).
export const REF5: number[][] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];
const REF = REF5;

// MediaPipe FaceMesh indices for the 5 canonical points (eye centers via corners,
// nose tip, mouth corners).
const MESH5 = {
  leftEye: [33, 133],
  rightEye: [362, 263],
  nose: 1,
  leftMouth: 61,
  rightMouth: 291,
};

function mid(a: Pt, b: Pt): Pt {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/** Pull the 5 alignment points (image-pixel coords) out of a full mesh. */
export function meshTo5(lm: Pt[]): number[][] {
  return [
    [mid(lm[MESH5.leftEye[0]], lm[MESH5.leftEye[1]]).x, mid(lm[MESH5.leftEye[0]], lm[MESH5.leftEye[1]]).y],
    [mid(lm[MESH5.rightEye[0]], lm[MESH5.rightEye[1]]).x, mid(lm[MESH5.rightEye[0]], lm[MESH5.rightEye[1]]).y],
    [lm[MESH5.nose].x, lm[MESH5.nose].y],
    [lm[MESH5.leftMouth].x, lm[MESH5.leftMouth].y],
    [lm[MESH5.rightMouth].x, lm[MESH5.rightMouth].y],
  ];
}

/**
 * Umeyama similarity transform from src (Nx2) to dst (Nx2).
 * Returns the 2x3 affine [a,b,tx, c,d,ty] mapping src -> dst.
 */
export function umeyama(src: number[][], dst: number[][]): number[] {
  const n = src.length;
  const meanS = [0, 0], meanD = [0, 0];
  for (let i = 0; i < n; i++) { meanS[0] += src[i][0]; meanS[1] += src[i][1]; meanD[0] += dst[i][0]; meanD[1] += dst[i][1]; }
  meanS[0] /= n; meanS[1] /= n; meanD[0] /= n; meanD[1] /= n;

  let varS = 0;
  // 2x2 covariance of dst-centered (rows) x src-centered (cols)
  let c00 = 0, c01 = 0, c10 = 0, c11 = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - meanS[0], sy = src[i][1] - meanS[1];
    const dx = dst[i][0] - meanD[0], dy = dst[i][1] - meanD[1];
    varS += sx * sx + sy * sy;
    c00 += dx * sx; c01 += dx * sy; c10 += dy * sx; c11 += dy * sy;
  }
  varS /= n; c00 /= n; c01 /= n; c10 /= n; c11 /= n;

  // SVD of the 2x2 covariance. svd2x2 returns U and the right factor already in
  // transposed (V^T) position, i.e. C = U * diag(S) * V here, so V plays V^T.
  const { U, S, V } = svd2x2(c00, c01, c10, c11);
  // Handle reflection.
  const det = (c00 * c11 - c01 * c10);
  const d = det < 0 ? -1 : 1;
  // R = U * diag(1,d) * V^T, and V already holds V^T, so do not transpose again.
  const R = mul2(U, mulDiag2(V, 1, d, false));
  const scale = (S[0] + d * S[1]) / (varS + 1e-12);

  const a = scale * R[0], b = scale * R[1];
  const c = scale * R[2], dd = scale * R[3];
  const tx = meanD[0] - (a * meanS[0] + b * meanS[1]);
  const ty = meanD[1] - (c * meanS[0] + dd * meanS[1]);
  return [a, b, tx, c, dd, ty];
}

/** Warp source RGB (uint8 HWC, srcW x srcH) to ALIGN_SIZE^2 RGB using affine M (src->dst). */
export function warpAffine(
  src: Uint8Array, srcW: number, srcH: number, M: number[],
): Uint8Array {
  const inv = invertAffine(M);
  const out = new Uint8Array(ALIGN_SIZE * ALIGN_SIZE * 3);
  for (let y = 0; y < ALIGN_SIZE; y++) {
    for (let x = 0; x < ALIGN_SIZE; x++) {
      const sxF = inv[0] * x + inv[1] * y + inv[2];
      const syF = inv[3] * x + inv[4] * y + inv[5];
      const oi = (y * ALIGN_SIZE + x) * 3;
      if (sxF < 0 || syF < 0 || sxF >= srcW - 1 || syF >= srcH - 1) continue;
      // bilinear sample
      const x0 = sxF | 0, y0 = syF | 0;
      const fx = sxF - x0, fy = syF - y0;
      for (let ch = 0; ch < 3; ch++) {
        const i00 = (y0 * srcW + x0) * 3 + ch;
        const i10 = i00 + 3;
        const i01 = i00 + srcW * 3;
        const i11 = i01 + 3;
        out[oi + ch] =
          src[i00] * (1 - fx) * (1 - fy) + src[i10] * fx * (1 - fy) +
          src[i01] * (1 - fx) * fy + src[i11] * fx * fy;
      }
    }
  }
  return out;
}

// ---- tiny 2x2 linear-algebra helpers ----------------------------------------
function mul2(a: number[], b: number[]): number[] {
  return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
          a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3]];
}
function mulDiag2(V: number[], d0: number, d1: number, transpose: boolean): number[] {
  // returns diag(d0,d1) * V^T  when transpose, used as V * diag then transposed
  const Vt = transpose ? [V[0], V[2], V[1], V[3]] : V;
  return [d0 * Vt[0], d0 * Vt[1], d1 * Vt[2], d1 * Vt[3]];
}
function svd2x2(a: number[][] | number, b?: number, c?: number, d?: number):
  { U: number[]; S: number[]; V: number[] } {
  const m00 = a as number, m01 = b!, m10 = c!, m11 = d!;
  const E = (m00 + m11) / 2, F = (m00 - m11) / 2;
  const G = (m10 + m01) / 2, H = (m10 - m01) / 2;
  const Q = Math.hypot(E, H), R = Math.hypot(F, G);
  const s0 = Q + R, s1 = Math.abs(Q - R);
  const a1 = Math.atan2(G, F), a2 = Math.atan2(H, E);
  const theta = (a2 - a1) / 2, phi = (a2 + a1) / 2;
  const U = [Math.cos(phi), -Math.sin(phi), Math.sin(phi), Math.cos(phi)];
  const V = [Math.cos(theta), -Math.sin(theta), Math.sin(theta), Math.cos(theta)];
  return { U, S: [s0, s1], V };
}
function invertAffine(M: number[]): number[] {
  const [a, b, tx, c, d, ty] = M;
  const det = a * d - b * c || 1e-9;
  const ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
  return [ia, ib, -(ia * tx + ib * ty), ic, id, -(ic * tx + id * ty)];
}
