import {
  ear,
  mouthWidthRatio,
  yawRatio,
  ChallengeDetector,
  randomChallenges,
  decideLiveness,
  enrollGatePassed,
  MIN_REACTION_MS,
  type Pt,
} from '../liveness';
import { consensus } from '../math';
import { DEFAULT_CONFIG } from '../types';

/** A plausible human reaction: past the floor, inside the step budget. */
const REACTED = 900;

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

// Mouth fixtures: inter-ocular 60, mouth corners at 61 / 291.
const EYES_60 = { 33: [0, 0] as [number, number], 263: [60, 0] as [number, number] };
const NEUTRAL_MOUTH = mesh({ ...EYES_60, 61: [20, 40], 291: [40, 40] }); // width 20
const WIDE_MOUTH = mesh({ ...EYES_60, 61: [10, 40], 291: [50, 40] }); // width 40

// Head-pose fixtures: yawRatio = |nose-cheekL| / |nose-cheekR|.
const FRONTAL = mesh({ 1: [0, 0], 234: [10, 0], 454: [10, 0] }); // ratio 1.0
const TURNED_RIGHT = mesh({ 1: [0, 0], 234: [20, 0], 454: [10, 0] }); // ratio 2.0
const TURNED_LEFT = mesh({ 1: [0, 0], 234: [5, 0], 454: [20, 0] }); // ratio 0.25

describe('active liveness math', () => {
  test('EAR is high for open eyes and low for closed eyes', () => {
    expect(ear(mesh(eye(true)))).toBeGreaterThan(0.25);
    expect(ear(mesh(eye(false)))).toBeLessThan(0.21);
  });

  test('adaptive blink fires on a dip-then-recover relative to the eye baseline', () => {
    const d = new ChallengeDetector('blink');
    expect(d.update(mesh(eye(true)), REACTED)).toBe(false); // neutral frame 1
    expect(d.update(mesh(eye(true)), REACTED)).toBe(false); // neutral frame 2
    expect(d.update(mesh(eye(false)), REACTED)).toBe(false); // closes (relative dip)
    expect(d.update(mesh(eye(true)), REACTED)).toBe(true); // reopens -> blink detected
  });

  test('a real closure that only partly recovers still fires once past 88% of baseline', () => {
    const d = new ChallengeDetector('blink');
    expect(d.update(mesh(eye(true)), REACTED)).toBe(false);
    expect(d.update(mesh(eye(true)), REACTED)).toBe(false);
    expect(d.update(mesh(eyeShallow()), REACTED)).toBe(false); // a 70% drop = a real closure
    expect(d.update(mesh(eye(true)), REACTED)).toBe(true);
  });

  test('a STATIC photo (constant EAR, no dip) can NEVER fire a blink, any # of frames', () => {
    for (const constEar of [eye(true), eye(false), eyeShallow()]) {
      const d = new ChallengeDetector('blink');
      let fired = false;
      for (let i = 0; i < 30; i++) fired = fired || d.update(mesh(constEar), REACTED);
      expect(fired).toBe(false); // no relative dip on a constant image
    }
  });

  test('small open-eye fluctuation (landmark noise on a live face) does not fire alone', () => {
    const d = new ChallengeDetector('blink');
    // Jitter within ~10% of the open baseline never crosses the 75% close gate.
    expect(d.update(mesh(eye(true)), REACTED)).toBe(false);
    expect(d.update(mesh(eye(true)), REACTED)).toBe(false);
    expect(d.update(mesh(eyeAt(2.7)), REACTED)).toBe(false); // ~10% lower, not a closure
    expect(d.update(mesh(eye(true)), REACTED)).toBe(false);
  });

  test('smile challenge fires when mouth widens past the baseline gain', () => {
    const d = new ChallengeDetector('smile');
    expect(d.update(NEUTRAL_MOUTH, REACTED)).toBe(false); // frames 1-2 confirm neutral
    expect(d.update(NEUTRAL_MOUTH, REACTED)).toBe(false);
    expect(d.update(NEUTRAL_MOUTH, REACTED)).toBe(false); // still at the baseline
    expect(d.update(WIDE_MOUTH, REACTED)).toBe(true); // clear smile past the gain
  });

  test('yaw ratio distinguishes head turn left vs right (front camera is mirrored)', () => {
    // yawRatio = |nose-cheekL| / |nose-cheekR|. nose=1, cheekL=234, cheekR=454.
    // The preview is mirrored, so a HIGH ratio is the user turning right.
    expect(yawRatio(TURNED_RIGHT)).toBeGreaterThan(1.3);
    expect(yawRatio(TURNED_LEFT)).toBeLessThan(1 / 1.3);
    expect(yawRatio(FRONTAL)).toBeCloseTo(1, 5);

    for (const [c, turned] of [
      ['turn-right', TURNED_RIGHT],
      ['turn-left', TURNED_LEFT],
    ] as const) {
      const d = new ChallengeDetector(c);
      expect(d.update(FRONTAL, REACTED)).toBe(false); // frames 1-2 confirm frontal
      expect(d.update(FRONTAL, REACTED)).toBe(false);
      expect(d.update(turned, REACTED)).toBe(true);
    }
  });
});

// Replay-attack regression suite. Each case below is a way a recording of the
// enrolled person can present a technically correct gesture; none of them is a
// response to the prompt, and none may pass.
describe('anti-replay: a gesture only counts as a reply to the prompt', () => {
  test('a gesture completing faster than a human can react is rejected', () => {
    const d = new ChallengeDetector('blink');
    d.update(mesh(eye(true)), 0);
    d.update(mesh(eye(true)), 40);
    d.update(mesh(eye(false)), 80);
    // Geometrically a perfect blink, but 120 ms after the prompt appeared it
    // cannot be an answer to it: the recording was already mid-blink.
    expect(d.update(mesh(eye(true)), 120)).toBe(false);
    expect(MIN_REACTION_MS).toBeGreaterThan(120);
  });

  test('the same blink DOES count once it lands inside the reaction window', () => {
    const d = new ChallengeDetector('blink');
    d.update(mesh(eye(true)), 200);
    d.update(mesh(eye(true)), 300);
    d.update(mesh(eye(false)), 500);
    expect(d.update(mesh(eye(true)), 620)).toBe(true);
  });

  test('a video already showing a turned head cannot satisfy a turn on arrival', () => {
    const d = new ChallengeDetector('turn-right');
    // Every frame is past the turn threshold, but the head is never observed
    // frontal after the prompt, so no prompt-caused transition ever happens.
    let fired = false;
    for (let i = 0; i < 30; i++) {
      fired = fired || d.update(TURNED_RIGHT, REACTED + i * 60);
    }
    expect(fired).toBe(false);
  });

  test('a video already showing a smile cannot satisfy the smile on arrival', () => {
    const d = new ChallengeDetector('smile');
    let fired = false;
    for (let i = 0; i < 30; i++) {
      fired = fired || d.update(WIDE_MOUTH, REACTED + i * 60);
    }
    expect(fired).toBe(false); // the widest mouth becomes its own baseline
  });

  test('a video caught mid-blink cannot satisfy the blink on arrival', () => {
    const d = new ChallengeDetector('blink');
    // Closed eyes at prompt time, then reopening: the old detector armed on the
    // very first frame and fired on the second. Now the eyes must be confirmed
    // OPEN after the prompt before any closure counts.
    expect(d.update(mesh(eye(false)), REACTED)).toBe(false);
    expect(d.update(mesh(eye(true)), REACTED + 60)).toBe(false);
    // ...and a genuine blink after that neutral observation still works.
    expect(d.update(mesh(eye(true)), REACTED + 120)).toBe(false);
    expect(d.update(mesh(eye(false)), REACTED + 180)).toBe(false);
    expect(d.update(mesh(eye(true)), REACTED + 240)).toBe(true);
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

  test('the reaction window is bounded at both ends, which is what a recording cannot satisfy', () => {
    // A recording can contain every gesture NetraID knows. What it cannot do is
    // produce the demanded one, starting from a confirmed neutral face, inside a
    // window that opens when the prompt appears and closes 2.8 s later, three
    // times running, in an order drawn after the recording was made.
    //
    // Both ends matter. Without the floor, a video already mid-blink on arrival
    // satisfies the step instantly. Without a tight ceiling, an attacker just
    // lets the recording run until the demanded gesture comes around, and the
    // reaction test decays into "does this video contain a blink somewhere".
    expect(MIN_REACTION_MS).toBeGreaterThanOrEqual(300);
    // Well under the 6 s that let an attacker simply run a recording until the
    // demanded gesture came around.
    expect(DEFAULT_CONFIG.challengeTimeoutMs).toBeLessThanOrEqual(4000);
    // And comfortably above a human's 350 ms to 1.2 s reaction, plus the time
    // each step spends confirming a neutral face first. At 2.8 s real users
    // timed out repeatedly and never reached a verdict.
    expect(DEFAULT_CONFIG.challengeTimeoutMs).toBeGreaterThanOrEqual(3500);
  });

  test('steps are drawn WITHOUT replacement, so a recording gets one chance each', () => {
    // Repeating a gesture within a session would hand a recording a second shot
    // at the same moment. Two steps rather than three because the gesture layer
    // no longer carries the anti-replay burden alone: the passive screen class
    // is armed. Three steps at a tight window rejected real users outright.
    for (let i = 0; i < 50; i++) {
      const set = randomChallenges(DEFAULT_CONFIG.numChallenges);
      expect(set).toHaveLength(DEFAULT_CONFIG.numChallenges);
      expect(new Set(set).size).toBe(DEFAULT_CONFIG.numChallenges);
    }
  });

  test('the passive barriers REPORT until a distribution has been measured', () => {
    // They were armed on three samples from one session and immediately turned
    // the real user away: the model read median P(real) 0.710 on one attempt
    // and 0.005 on the next, from the same face, because the crop scale was
    // silently varying with how close the subject stood.
    //
    // Three captures is not a distribution. The app now records EVERY processed
    // frame's reading and logs per-attempt percentiles, so an operating point
    // can be chosen from the tails across many live and attack attempts.
    const cfg = DEFAULT_CONFIG;
    expect(cfg.passiveMode).toBe('report');
    expect(cfg.screenSpoofMode).toBe('report');
    // A reading that would fail an armed gate still passes, and still travels
    // to the verdict so it can be collected.
    expect(decideLiveness(true, ['blink'], 0.005, 0.995, null, cfg).passed).toBe(true);
    expect(decideLiveness(true, ['blink'], 0.005, 0.995, null, cfg).passiveScore).toBe(0.005);
    // The ACTIVE barrier still rejects on its own, so the app is not open.
    expect(decideLiveness(false, [], 0.9, 0.02, null, cfg).passed).toBe(false);
  });

  test('the passive floor rejects once armed against measured hardware', () => {
    // Arming is the deployment's decision, taken after reading the live
    // distribution off its own handsets (docs/CALIBRATION.md).
    const armed = { ...DEFAULT_CONFIG, passiveMode: 'enforce' as const };
    expect(decideLiveness(true, ['blink'], 0.9, 0.02, null, armed).passed).toBe(true);
    expect(decideLiveness(true, ['blink'], 0.03, 0.02, null, armed).passed).toBe(false);
    const strict = { ...armed, passiveThreshold: 0.6 };
    expect(decideLiveness(true, ['blink'], 0.3, 0.02, null, strict).passed).toBe(false);
  });

  test('the enrollment gate rejects nobody until it is armed', () => {
    // Enrollment is the root of trust, so it carries its own gate. It ships in
    // `report` because its capture distribution differs from verification's:
    // six shots fired as a burst the moment a face is framed, with no gesture
    // phase and no chroma flash ahead of them. Borrowing the verification
    // threshold rejected genuine enrollments on the target handset, which is
    // exactly the failure `report` exists to prevent.
    const { enrollSpoofMode, enrollPassiveThreshold } = DEFAULT_CONFIG;
    expect(enrollSpoofMode).toBe('report');
    // A reading far below the threshold still enrolls while unarmed.
    expect(enrollGatePassed(0.001, enrollSpoofMode, enrollPassiveThreshold)).toBe(true);
    expect(enrollGatePassed(0.001, 'off', enrollPassiveThreshold)).toBe(true);
  });

  test('an armed enrollment gate refuses a template built from a photograph', () => {
    // Once an operating point has been measured on the enrollment path, arming
    // it must actually stop a print or a screen from becoming an identity: no
    // amount of liveness at verification can repair a template built from one.
    expect(enrollGatePassed(0.02, 'enforce', 0.08)).toBe(false);
    expect(enrollGatePassed(0.31, 'enforce', 0.08)).toBe(true);
    // The boundary belongs to the genuine user: a false reject at a toll plaza
    // is a worker who cannot mark attendance.
    expect(enrollGatePassed(0.08, 'enforce', 0.08)).toBe(true);
  });

  test('an armed chroma challenge can reject on its own, and reports its score', () => {
    const armed = { ...DEFAULT_CONFIG, chromaMode: 'enforce' as const };
    // `score` is the face-to-room response RATIO. A replayed display moves no
    // more than the room does (measured: 0.25 and 0.71); a live face moves
    // several times more (measured: 4.3 and 8.0).
    const flat = { score: 0.42, faceLift: -0.0013, bgLift: -0.0051, slotsMeasured: 5, samples: 15, usable: true };
    const lit = { score: 4.3, faceLift: -0.0128, bgLift: -0.003, slotsMeasured: 5, samples: 15, usable: true };
    // Everything else passes; only the face's response to our own light differs.
    expect(decideLiveness(true, ['blink'], 0.5, 0.02, flat, armed).passed).toBe(false);
    expect(decideLiveness(true, ['blink'], 0.5, 0.02, lit, armed).passed).toBe(true);
    // The measured value is always surfaced, so an operator can see the evidence.
    expect(decideLiveness(true, ['blink'], 0.5, 0.02, lit, armed).chromaScore).toBe(4.3);
    expect(decideLiveness(true, ['blink'], 0.5, 0.02, null, armed).chromaScore).toBeNull();
  });

  test('a single lucky frame no longer carries a replay through an armed passive gate', () => {
    // Aggregation, not arming, is what this covers: with the gate armed, the
    // median must be what it sees, so one fortunate frame cannot carry a replay.
    const cfg = { ...DEFAULT_CONFIG, passiveMode: 'enforce' as const };
    // Three captures of a screen replay: one frame reads unusually "real".
    const perFrameReal = [0.02, 0.31, 0.03];
    const perFrameScreen = [0.95, 0.44, 0.97];
    const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
    // The old aggregate (max P(real), screen class ignored) accepted this.
    expect(Math.max(...perFrameReal)).toBeGreaterThanOrEqual(cfg.passiveThreshold);
    // The shipped aggregate rejects it on both counts.
    expect(
      decideLiveness(true, ['blink'], med(perFrameReal), consensus(perFrameScreen, 2), null, cfg).passed,
    ).toBe(false);
  });

  test('one noisy frame cannot reject a real person', () => {
    // The complaint that sent us here: a live user occasionally saw "spoof".
    // Two of three captures read clean, one spiked. A max would have rejected;
    // the consensus does not.
    const cfg = DEFAULT_CONFIG;
    const screen = [0.91, 0.06, 0.09]; // one spike, two clean
    expect(Math.max(...screen)).toBeGreaterThan(cfg.screenSpoofMax); // old rule: rejected
    expect(consensus(screen, 2)).toBeLessThan(cfg.screenSpoofMax); // shipped rule: accepted
    expect(decideLiveness(true, ['blink'], 0.4, consensus(screen, 2), null, cfg).passed).toBe(true);
  });
});
