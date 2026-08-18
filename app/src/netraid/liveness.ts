// Offline liveness: active challenge FSM (from Face Mesh landmarks) + passive
// MiniFASNet score. See docs/LIVENESS.md for the math and thresholds.

import { chromaPassed, type ChromaVerdict } from './chroma';
import type { Challenge, EnrollSpoofMode, LivenessResult, NetraIDConfig } from './types';

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

/**
 * Anti-replay timing floor. A gesture is only accepted as a RESPONSE to the
 * prompt, so it cannot be counted before a human could physically have reacted
 * to seeing it. Simple visual reaction time bottoms out near 200 ms and a
 * volitional facial gesture adds more, so anything completing under 350 ms was
 * already in flight when the prompt appeared: that is a recording playing on
 * its own clock, not a person answering.
 */
export const MIN_REACTION_MS = 350;

/**
 * Frames of a confirmed NEUTRAL face (eyes open / mouth at rest / head frontal)
 * that must be observed AFTER the prompt is issued, before the gesture itself
 * counts. This is what forces the gesture to be a prompt-triggered TRANSITION
 * rather than a state that happened to be on screen: a replayed video caught
 * mid-blink or mid-turn no longer satisfies the step on its first frame.
 */
const NEUTRAL_FRAMES = 2;

// Frontal band for the yaw ratio, used as the neutral state for turn gestures.
const FRONTAL_LO = 0.88;
const FRONTAL_HI = 1.14;

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
export interface GestureBox { value: number }
export interface GestureState {
  /** 1 once closure (blink) evidence has been seen. */
  armed: GestureBox;
  /** Neutral (narrowest) mouth-width baseline for the smile gesture. */
  base: GestureBox;
  /** Learned open-eye EAR baseline for the blink gesture. */
  earOpen: GestureBox;
  /** Count of post-prompt frames confirmed to be in the NEUTRAL state. */
  neutral: GestureBox;
}

export function newGestureState(): GestureState {
  return { armed: { value: 0 }, base: { value: 0 }, earOpen: { value: 0 }, neutral: { value: 0 } };
}

/**
 * Decide whether the current gesture is satisfied by this frame. This is the
 * single implementation used by BOTH the on-device frame-processor worklet and
 * the unit tests, so what ships is what is tested. Every field of `st` is a
 * `{ value }` box: shared values on device, plain objects in tests. All four
 * MUST be cleared whenever a new prompt is issued.
 *
 * Three conditions must all hold, and the last two are what defeat a replayed
 * video of the enrolled person:
 *  1. the gesture geometry itself (EAR dip / mouth widening / yaw),
 *  2. a NEUTRAL face confirmed after the prompt was issued, so the gesture has
 *     to be a transition the prompt caused, not a pose already on screen,
 *  3. `elapsedMs` past the human reaction floor (see MIN_REACTION_MS).
 *
 * `elapsedMs` is measured from the moment the prompt for THIS step appeared.
 */
export function gestureSatisfied(
  c: Challenge,
  lm: Pt[],
  st: GestureState,
  elapsedMs: number,
): boolean {
  'worklet';
  if (c === 'blink') {
    // Adaptive-but-robust blink. We learn this eye's open baseline (running
    // max) so glasses-limited closures still count, but CLAMP the baseline
    // below 0.42: FaceLandmarker throws EAR spikes up to ~0.75 during head
    // motion, and an unclamped max would poison the reopen threshold so the
    // eyes could never read "open" again (permanent-armed stall). A static
    // photo's EAR is constant, so it never dips and never arms.
    const e = ear(lm);
    if (e > st.earOpen.value && e < 0.42) st.earOpen.value = e;
    const b = st.earOpen.value > 0.24 ? st.earOpen.value : 0.34;
    if (st.neutral.value < NEUTRAL_FRAMES) {
      // Eyes must be confirmed OPEN after the prompt before a closure counts.
      if (e > EAR_OPEN - 0.01 && e > b * 0.9) st.neutral.value += 1;
      return false;
    }
    const closeAt = 0.24 < b * BLINK_CLOSE_FRAC ? 0.24 : b * BLINK_CLOSE_FRAC;
    if (e < closeAt) st.armed.value = 1;
    else if (st.armed.value >= 1 && e > b * BLINK_OPEN_FRAC) return elapsedMs >= MIN_REACTION_MS;
    return false;
  }
  if (c === 'smile') {
    // Track the NEUTRAL (narrowest) mouth as the baseline; a smile widens it.
    const w = mouthWidthRatio(lm);
    if (st.base.value === 0 || w < st.base.value) st.base.value = w;
    if (st.neutral.value < NEUTRAL_FRAMES) {
      if (w <= st.base.value * 1.02) st.neutral.value += 1;
      return false;
    }
    return w > st.base.value * SMILE_GAIN && elapsedMs >= MIN_REACTION_MS;
  }
  // Head must be confirmed FRONTAL after the prompt before a turn counts, so a
  // video already showing a turned head cannot satisfy the step on arrival.
  const yr = yawRatio(lm);
  if (st.neutral.value < NEUTRAL_FRAMES) {
    if (yr > FRONTAL_LO && yr < FRONTAL_HI) st.neutral.value += 1;
    return false;
  }
  // The front camera is mirrored: turning the head to the user's left LOWERS
  // the yaw ratio, so the comparisons are flipped relative to a raw feed.
  const turned = c === 'turn-left' ? yr < 1 / YAW_TURN : yr > YAW_TURN;
  return turned && elapsedMs >= MIN_REACTION_MS;
}

/**
 * Stateful detector for one active challenge. Feed it landmarks per frame with
 * the time since that step's prompt appeared; it returns true once the
 * challenge is satisfied. A thin wrapper over `gestureSatisfied`, so the class
 * and the on-device worklet can never drift apart.
 */
export class ChallengeDetector {
  private readonly st = newGestureState();

  constructor(public readonly challenge: Challenge) {}

  update(lm: Pt[], elapsedMs: number = MIN_REACTION_MS): boolean {
    return gestureSatisfied(this.challenge, lm, this.st, elapsedMs);
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
 * Combine the active result with the passive anti-spoof readings into a final
 * verdict. All barriers must pass (uncorrelated failure modes -> low joint FAR).
 *
 * `passiveScore` is the MEDIAN P(real) across the verify captures and
 * `passiveScreen` the MAXIMUM P(screen-replay). The asymmetry is deliberate:
 * each statistic is chosen to be the one an attacker cannot get lucky with.
 * A max over P(real) let a single fortunate frame of a screen replay carry the
 * whole session; a median needs most of the burst to look real. Conversely one
 * frame that clearly reads as a screen is enough to reject.
 */
/**
 * Whether an enrollment burst is allowed to become a template.
 *
 * Separate from the verification gate on purpose. The two see different capture
 * distributions: verification captures arrive after a gesture sequence and a
 * colour flash, with the sensor long settled, while enrollment captures arrive
 * as a burst the moment a face is framed. Reusing `passiveThreshold` here on the
 * assumption that P(real) is P(real) whatever the path rejected genuine
 * enrollments on the target handset, so enrollment carries its own mode and its
 * own threshold, measured against its own path (docs/CALIBRATION.md step 1b).
 *
 * Gates the MEDIAN of the burst for the same reason verification does: one
 * fortunate frame must not carry a photograph through, and one noisy frame must
 * not reject a real person.
 */
export function enrollGatePassed(
  passiveMedian: number, mode: EnrollSpoofMode, threshold: number,
): boolean {
  if (mode !== 'enforce') return true;
  return passiveMedian >= threshold;
}

export function decideLiveness(
  activePassed: boolean,
  completed: Challenge[],
  passiveScore: number,
  passiveScreen: number,
  chroma: ChromaVerdict | null,
  cfg: NetraIDConfig,
): LivenessResult {
  // The screen-replay class is only allowed to reject once its operating point
  // has been measured against live faces on the target hardware; until then it
  // is measured and surfaced, never enforced (see DEFAULT_CONFIG).
  const screenPassed =
    cfg.screenSpoofMode !== 'enforce' || passiveScreen <= cfg.screenSpoofMax;
  // The P(real) floor is under the same discipline as every other gate here: it
  // rejects only once its operating point has been measured on the deployment
  // hardware. It shipped armed on a calibration from a different session and
  // turned away seven of eight genuine attempts on the target handset.
  const passivePassed =
    cfg.passiveMode !== 'enforce' || passiveScore >= cfg.passiveThreshold;
  const passed =
    activePassed &&
    passivePassed &&
    screenPassed &&
    chromaPassed(chroma, cfg.chromaMode, cfg.chromaThreshold);
  return {
    passed,
    activePassed,
    passiveScore,
    passiveScreen,
    chromaScore: chroma && chroma.usable ? chroma.score : null,
    completedChallenges: completed,
  };
}
