import {
  ear,
  mouthWidthRatio,
  yawRatio,
  ChallengeDetector,
  randomChallenges,
  decideLiveness,
  type Pt,
} from '../liveness';
import { DEFAULT_CONFIG } from '../types';

// Build a 468-point mesh with everything at origin, then set the indices the
// liveness math actually reads. Index maps mirror liveness.ts exactly.
function mesh(set: Record<number, [number, number]>): Pt[] {
  const pts: Pt[] = Array.from({ length: 468 }, () => ({ x: 0, y: 0 }));
  for (const k of Object.keys(set)) {
    const i = Number(k);
    pts[i] = { x: set[i][0], y: set[i][1] };
  }
  return pts;
}

// Eye index maps: LEFT_EYE=[362,385,387,263,373,380], RIGHT_EYE=[33,160,158,133,153,144].
// EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|). Open => tall, closed => flat.
function eyeAt(v: number): Record<number, [number, number]> {
  return {
    362: [0, 0], 263: [10, 0], 385: [2, v], 380: [2, -v], 387: [6, v], 373: [6, -v],
    33: [20, 0], 133: [30, 0], 160: [22, v], 144: [22, -v], 158: [26, v], 153: [26, -v],
  };
}
// v=3 -> EAR 0.6 (open); v=0.05 -> EAR 0.01 (deep closure); v=0.9 -> EAR 0.18
// (shallow dip in the 0.16..0.21 band, the landmark-noise regime).
const eye = (open: boolean) => eyeAt(open ? 3 : 0.05);
const eyeShallow = () => eyeAt(0.9);

describe('active liveness math', () => {
  test('EAR is high for open eyes and low for closed eyes', () => {
    expect(ear(mesh(eye(true)))).toBeGreaterThan(0.25);
    expect(ear(mesh(eye(false)))).toBeLessThan(0.21);
  });

  test('adaptive blink fires on a dip-then-recover relative to the eye baseline', () => {
    const d = new ChallengeDetector('blink');
    expect(d.update(mesh(eye(true)))).toBe(false); // learns open baseline
    expect(d.update(mesh(eye(false)))).toBe(false); // closes (relative dip)
    expect(d.update(mesh(eye(true)))).toBe(true); // reopens -> blink detected
  });

  test('a real closure that only partly recovers still fires once past 88% of baseline', () => {
    const d = new ChallengeDetector('blink');
    expect(d.update(mesh(eye(true)))).toBe(false);
    expect(d.update(mesh(eyeShallow()))).toBe(false); // a 70% drop = a real closure
    expect(d.update(mesh(eye(true)))).toBe(true);
  });

  test('a STATIC photo (constant EAR, no dip) can NEVER fire a blink, any # of frames', () => {
    for (const constEar of [eye(true), eye(false), eyeShallow()]) {
      const d = new ChallengeDetector('blink');
      let fired = false;
      for (let i = 0; i < 30; i++) fired = fired || d.update(mesh(constEar));
      expect(fired).toBe(false); // no relative dip on a constant image
    }
  });

  test('small open-eye fluctuation (landmark noise on a live face) does not fire alone', () => {
    const d = new ChallengeDetector('blink');
    // Jitter within ~10% of the open baseline never crosses the 75% close gate.
    expect(d.update(mesh(eye(true)))).toBe(false);
    expect(d.update(mesh(eyeAt(2.7)))).toBe(false); // ~10% lower, not a closure
    expect(d.update(mesh(eye(true)))).toBe(false);
  });

  test('smile challenge fires when mouth widens past the baseline gain', () => {
    const d = new ChallengeDetector('smile');
    const fixedEyes = { 33: [0, 0] as [number, number], 263: [60, 0] as [number, number] };
    const narrow = mesh({ ...fixedEyes, 61: [20, 40], 291: [40, 40] }); // width 20
    const wide = mesh({ ...fixedEyes, 61: [10, 40], 291: [50, 40] }); // width 40 (>12% gain)
    expect(d.update(narrow)).toBe(false); // frames 1-3 establish the neutral baseline
    expect(d.update(narrow)).toBe(false);
    expect(d.update(narrow)).toBe(false);
    expect(d.update(wide)).toBe(true); // frame 4: clear smile past the gain threshold
  });

  test('yaw ratio distinguishes head turn left vs right', () => {
    // yawRatio = |nose-cheekL| / |nose-cheekR|. nose=1, cheekL=234, cheekR=454.
    const left = mesh({ 1: [0, 0], 234: [20, 0], 454: [10, 0] }); // farther from left cheek
    const right = mesh({ 1: [0, 0], 234: [5, 0], 454: [20, 0] });
    expect(yawRatio(left)).toBeGreaterThan(1.3);
    expect(yawRatio(right)).toBeLessThan(1 / 1.3);
    expect(new ChallengeDetector('turn-left').update(left)).toBe(true);
    expect(new ChallengeDetector('turn-right').update(right)).toBe(true);
  });
});

describe('liveness orchestration', () => {
  test('randomChallenges returns n distinct challenges from the pool', () => {
    const c = randomChallenges(3);
    expect(c).toHaveLength(3);
    expect(new Set(c).size).toBe(3);
    const pool = ['blink', 'smile', 'turn-left', 'turn-right'];
    c.forEach((x) => expect(pool).toContain(x));
  });

  test('every challenge set includes blink (static photos cannot blink)', () => {
    for (let i = 0; i < 50; i++) {
      expect(randomChallenges(2)).toContain('blink');
    }
  });

  test('both liveness layers must pass: active challenges AND the passive gate', () => {
    // Active blink/smile/turn FSM is layer 1. Layer 2 is the MiniFASNet P(real)
    // gate, calibrated on-device: live captures measured 0.12-0.74, a laptop
    // screen replay <= 0.04, threshold 0.08 on the max across captures.
    const cfg = DEFAULT_CONFIG;
    expect(decideLiveness(true, ['blink'], 0.9, cfg).passed).toBe(true); // both pass
    expect(decideLiveness(true, ['blink'], 0.03, cfg).passed).toBe(false); // screen replay caught by passive
    expect(decideLiveness(false, [], 0.9, cfg).passed).toBe(false); // active failed
    // Raising the passive threshold re-enables passive gating.
    const strict = { ...cfg, passiveThreshold: 0.6 };
    expect(decideLiveness(true, ['blink'], 0.3, strict).passed).toBe(false); // passive blocks
  });
});
