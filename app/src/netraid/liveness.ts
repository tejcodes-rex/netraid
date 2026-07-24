// Offline liveness: active challenge FSM (from Face Mesh landmarks) + passive
// MiniFASNet score. See docs/LIVENESS.md for the math and thresholds.

import type { Challenge, LivenessResult, NetraIDConfig } from './types';

// MediaPipe FaceLandmarker 468-point indices.
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const MOUTH_L = 61;
const MOUTH_R = 291;
const NOSE = 1;
const CHEEK_L = 234;
const CHEEK_R = 454;

const EAR_CLOSED = 0.21;
const EAR_OPEN = 0.25;
const YAW_TURN = 1.15; // yaw ratio for a head turn (~15 degrees); lowered from
// 1.18 after on-device measurement showed natural turns reach ~0.84 / ~1.16.
const SMILE_GAIN = 1.08; // 8% mouth-width increase over the neutral baseline;
// measured natural smiles widen the mouth 6-12%, a fixed 12% was too demanding.
// Blink uses an ADAPTIVE per-eye baseline (see below), not a fixed EAR, because
// eyes behind glasses often never read fully closed.
const BLINK_CLOSE_FRAC = 0.72; // closure = EAR below 72% of this eye's open max
const BLINK_OPEN_FRAC = 0.82;  // reopen  = EAR back above 82% of that max

export type Pt = { x: number; y: number };

function dist(a: Pt, b: Pt): number {
  'worklet';
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function eyeAspectRatio(lm: Pt[], idx: number[]): number {
  'worklet';
  const p1 = lm[idx[0]], p2 = lm[idx[1]], p3 = lm[idx[2]];
  const p4 = lm[idx[3]], p5 = lm[idx[4]], p6 = lm[idx[5]];
  return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4) + 1e-6);
}

export function ear(lm: Pt[]): number {
  'worklet';
  return (eyeAspectRatio(lm, LEFT_EYE) + eyeAspectRatio(lm, RIGHT_EYE)) / 2;
}

export function interOcular(lm: Pt[]): number {
  'worklet';
  return dist(lm[33], lm[263]);
}

export function mouthWidthRatio(lm: Pt[]): number {
  'worklet';
  return dist(lm[MOUTH_L], lm[MOUTH_R]) / (interOcular(lm) + 1e-6);
}

export function yawRatio(lm: Pt[]): number {
  'worklet';
  return dist(lm[NOSE], lm[CHEEK_L]) / (dist(lm[NOSE], lm[CHEEK_R]) + 1e-6);
}

// Thresholds (exported so the worklet can evaluate challenges inline).
export const TH = { earClosed: EAR_CLOSED, earOpen: EAR_OPEN, yawTurn: YAW_TURN, smileGain: SMILE_GAIN };

/**
 * Stateful detector for one active challenge. Feed it landmarks per frame;
 * it returns true once the challenge is satisfied.
 */
export class ChallengeDetector {
  private armed = false;
  private earOpen = 0;
  private baselineMouth = 0;
  private frames = 0;

  constructor(public readonly challenge: Challenge) {}

  update(lm: Pt[]): boolean {
    this.frames++;
    switch (this.challenge) {
      case 'blink': {
        // Adaptive-but-robust: learn the open baseline (running max) but clamp
        // it below 0.42 to reject landmark spikes that would strand the FSM
        // armed. Closure below min(0.24, 72% of baseline); reopen above 82%.
        // A static photo's EAR is constant, so it never dips and never arms.
        const e = ear(lm);
        if (e > this.earOpen && e < 0.42) this.earOpen = e;
        const base = this.earOpen > 0.24 ? this.earOpen : 0.34;
        const closeAt = Math.min(0.24, base * BLINK_CLOSE_FRAC);
        if (e < closeAt) this.armed = true;
        else if (this.armed && e > base * BLINK_OPEN_FRAC) return true;
        return false;
      }
      case 'smile': {
        const w = mouthWidthRatio(lm);
        if (this.frames <= 3) this.baselineMouth = Math.max(this.baselineMouth, w);
        return this.baselineMouth > 0 && w > this.baselineMouth * SMILE_GAIN;
      }
      case 'turn-left':
        return yawRatio(lm) > YAW_TURN;
      case 'turn-right':
        return yawRatio(lm) < 1 / YAW_TURN;
    }
  }
}

/**
 * Cryptographically-seeded random challenge set (anti-replay).
 *
 * Blink is ALWAYS included: a static photograph cannot blink, and blink is the
 * one gesture that cannot be counterfeited by swinging the phone around a
 * printed/displayed photo (which changes perspective geometry enough to mimic
 * head turns). The remaining slots and the overall ORDER stay random.
 */
export function randomChallenges(n: number): Challenge[] {
  const others: Challenge[] = ['smile', 'turn-left', 'turn-right'];
  const bytes = new Uint8Array(others.length + n);
  // react-native-get-random-values polyfills crypto.getRandomValues
  crypto.getRandomValues(bytes);
  const rest = others
    .map((c, i) => ({ c, r: bytes[i] }))
    .sort((a, b) => a.r - b.r)
    .slice(0, Math.max(0, n - 1))
    .map((x) => x.c);
  return ['blink' as Challenge, ...rest]
    .map((c, i) => ({ c, r: bytes[others.length + i] }))
    .sort((a, b) => a.r - b.r)
    .map((x) => x.c);
}

/**
 * Combine active result with the passive anti-spoof score into a final verdict.
 * Both barriers must pass (uncorrelated failure modes -> low joint FAR).
 */
export function decideLiveness(
  activePassed: boolean,
  completed: Challenge[],
  passiveScore: number,
  cfg: NetraIDConfig,
): LivenessResult {
  const passed = activePassed && passiveScore >= cfg.passiveThreshold;
  return { passed, activePassed, passiveScore, completedChallenges: completed };
}
