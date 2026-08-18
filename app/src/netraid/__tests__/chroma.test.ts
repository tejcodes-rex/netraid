import {
  randomFlashSequence,
  chromaResponse,
  chromaPassed,
  FLASH_RGB,
  type ChromaSample,
  type FlashColor,
} from '../chroma';
import { DEFAULT_CONFIG } from '../types';

// randomFlashSequence uses crypto.getRandomValues, which react-native-get-random-values
// polyfills on device. Node 18+ exposes it on globalThis.crypto; this shim keeps the
// test independent of the runtime's crypto surface.
beforeAll(() => {
  if (typeof globalThis.crypto?.getRandomValues !== 'function') {
    let n = 0;
    (globalThis as any).crypto = {
      getRandomValues: (a: Uint8Array) => {
        for (let i = 0; i < a.length; i++) a[i] = (n = (n * 1103515245 + 12345) & 0xff);
        return a;
      },
    };
  }
});

/**
 * Synthesise what the camera sees.
 *
 * `lift` is how far the FACE's chromaticity moves toward the emitted colour: a
 * live face reflects the screen and lifts, a display replaying a recording
 * emits its own light and does not.
 *
 * `awb` is a global shift applied to the face AND the background alike, which
 * is what the sensor's auto-white-balance does when the screen changes colour.
 * It is the reason face-only measurement failed on hardware, so every test here
 * has to survive it.
 *
 * `bgLift` is a response on the BACKGROUND, which our flash should barely
 * reach: it falls off with the square of distance.
 */
function samplesFor(
  seq: FlashColor[], lift: number, noise = 0, awb = 0, bgLift = 0,
  /** How much harder the face CENTRE responds than its RIM. A head curves away
   * from the phone, so its centre returns more of our light; a photograph or a
   * display is flat and responds uniformly, giving 1. */
  shape = 1,
): ChromaSample[] {
  const out: ChromaSample[] = [];
  const toward = (c: FlashColor, base: number[], amount: number) => {
    const v = [...base];
    if (c === 'red') { v[0] += amount; v[1] -= amount / 2; v[2] -= amount / 2; }
    if (c === 'green') { v[1] += amount; v[0] -= amount / 2; v[2] -= amount / 2; }
    if (c === 'blue') { v[2] += amount; v[0] -= amount / 2; v[1] -= amount / 2; }
    return v;
  };
  for (let slot = 0; slot < seq.length; slot++) {
    for (let k = 0; k < 3; k++) {
      // Neutral skin sits near equal-energy; the sign of the jitter alternates
      // so it averages out and only the lift survives.
      const j = noise * (k % 2 ? 1 : -1);
      const face = toward(seq[slot], [1 / 3 + j, 1 / 3 - j, 1 / 3], lift + awb);
      const bg = toward(seq[slot], [1 / 3, 1 / 3, 1 / 3], bgLift + awb);
      // The rim carries the base response; the centre carries `shape` times it.
      const rim = toward(seq[slot], [1 / 3 + j, 1 / 3 - j, 1 / 3], lift + awb);
      const centre = toward(seq[slot], [1 / 3 + j, 1 / 3 - j, 1 / 3], lift * shape + awb);
      // Luminance: our light raises the whole subject, and on a curved one it
      // raises the centre by `shape` times as much as the rim.
      const isLit = seq[slot] !== 'dark';
      const ambient = 60;
      const ourLight = isLit ? 90 : 0;
      out.push({
        slot,
        r: face[0], g: face[1], b: face[2],
        bgR: bg[0], bgG: bg[1], bgB: bg[2],
        cR: centre[0], cG: centre[1], cB: centre[2],
        pR: rim[0], pG: rim[1], pB: rim[2],
        cY: ambient + ourLight * shape,
        pY: ambient + ourLight,
      });
    }
  }
  return out;
}

describe('chroma challenge sequence', () => {
  test('every colour has a screen value and DARK is the neutral reference', () => {
    for (const c of ['red', 'green', 'blue', 'dark'] as FlashColor[]) {
      expect(FLASH_RGB[c]).toMatch(/^#[0-9A-F]{6}$/i);
    }
    // Not white: white is our light too, so a white slot is not an unlit
    // reference, and the shape measurement has nothing to compare against.
    expect(FLASH_RGB.dark).toBe('#000000');
  });

  test('sequences alternate lit and neutral slots and use >= 2 distinct colours', () => {
    for (let i = 0; i < 25; i++) {
      const seq = randomFlashSequence(5);
      expect(seq).toHaveLength(5);
      expect(seq[1]).toBe('dark');
      expect(seq[3]).toBe('dark');
      const lit = seq.filter((c) => c !== 'dark');
      expect(lit.length).toBeGreaterThanOrEqual(2);
      // With one colour only, the score would measure ambient drift instead of
      // a response to our light.
      expect(new Set(lit).size).toBeGreaterThanOrEqual(2);
    }
  });

  test('no two adjacent slots carry the same colour', () => {
    for (let i = 0; i < 25; i++) {
      const seq = randomFlashSequence(5);
      for (let k = 1; k < seq.length; k++) expect(seq[k]).not.toBe(seq[k - 1]);
    }
  });
});

describe('chroma response separates a lit face from a replayed display', () => {
  const seq: FlashColor[] = ['red', 'dark', 'green', 'dark', 'blue'];

  test('a head curves away from the phone, so its centre responds harder than its rim', () => {
    // Skin is close to Lambertian: a patch returns our light in proportion to
    // the cosine of the angle between its surface normal and the phone. The
    // nose and forehead face the screen head-on, the cheeks and jaw curve away.
    const v = chromaResponse(seq, samplesFor(seq, 0.03, 0, 0, 0.004, 1.9));
    expect(v.usable).toBe(true);
    expect(v.score).toBeGreaterThan(DEFAULT_CONFIG.chromaThreshold);
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
  });

  test('a FLAT surface responds uniformly and is rejected, at any brightness', () => {
    // A photograph and a display share one surface normal across their whole
    // area, so centre and rim respond identically and the ratio is 1. This holds
    // whatever the panel shows, however bright it is, and however far away it is
    // held, which is what makes it something no recording can defeat.
    for (const strength of [0.01, 0.05, 0.2]) {
      const v = chromaResponse(seq, samplesFor(seq, strength, 0, 0, 0.004, 1.0));
      expect(v.usable).toBe(true);
      expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(false);
    }
  });

  test('a replayed display is flat, so our light adds no contrast to it', () => {
    const v = chromaResponse(seq, samplesFor(seq, 0.03, 0, 0, 0.012, 1.0));
    expect(v.usable).toBe(true);
    expect(v.score).toBeCloseTo(1, 2);
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(false);
  });

  test('a worker standing against a wall still passes', () => {
    // The reason the background reference had to go: in a toll booth or a
    // doorway the wall is lit almost as much as the face, and a face-vs-room
    // measurement then reads like a spoof. Nothing about the subject changed,
    // so the shape measurement must be indifferent to it, and it is: it never
    // looks outside the face.
    const v = chromaResponse(seq, samplesFor(seq, 0.03, 0, 0, 0.028, 1.9));
    expect(v.usable).toBe(true);
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
  });

  test('a sequence with no dark slot is unusable, and never a rejection', () => {
    // Without an unlit phase there is no baseline contrast to compare against.
    // Absence of a measurement must never be reported as an attack: in direct
    // sunlight, where a phone screen cannot compete, this is the case that
    // keeps a real worker from being turned away.
    const lit: FlashColor[] = ['red', 'green', 'blue'];
    const v = chromaResponse(lit, samplesFor(lit, 0.03, 0, 0, 0.004, 1.9));
    expect(v.usable).toBe(false);
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
  });

  test('tracking only ONE of the emitted colours is not enough', () => {
    // A replay whose own content happens to be reddish moves the red channel in
    // every slot. The score takes the MINIMUM ratio across emitted colours, so
    // an accidental correlation with one colour cannot carry the verdict.
    const s = samplesFor(seq, 0.004, 0, 0, 0.012).map((x) => ({ ...x, r: x.r + 0.06 }));
    const v = chromaResponse(seq, s);
    expect(v.score).toBeLessThan(DEFAULT_CONFIG.chromaThreshold);
  });

  test('sensor noise uncorrelated with the sequence does not manufacture a pass', () => {
    const v = chromaResponse(seq, samplesFor(seq, 0.004, 0.04, 0, 0.012));
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(false);
  });
});

describe('auto-white-balance cannot fake or erase a chroma response', () => {
  test('a global AWB shift is cancelled, so it cannot manufacture a score', () => {
    // This is the failure that made the face-only measurement useless on the
    // target handset: turn the screen red and the sensor pulls the WHOLE image
    // back toward neutral, which moves the face's chromaticity in lockstep with
    // the flash for reasons that have nothing to do with a face being present.
    // A display replaying a recording gets the same AWB treatment, so face-only
    // measurement scores it exactly like a live face.
    const seq = randomFlashSequence(5);
    const replayUnderAwb = samplesFor(seq, 0, 0, 0.05, 0);
    const v = chromaResponse(seq, replayUnderAwb);
    // The face tracked the sequence strongly on its own...
    expect(Math.abs(v.faceLift)).toBeGreaterThan(0.02);
    // ...but the room tracked it just as hard, so the ratio is ~1 and nothing
    // about a face has been demonstrated.
    expect(v.score).toBeLessThan(DEFAULT_CONFIG.chromaThreshold);
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(false);
  });

  test('a real face still scores through the same AWB shift', () => {
    // The differential keeps only the part our flash produced by actually
    // landing on a near subject, which is what AWB cannot remove.
    const seq = randomFlashSequence(5);
    // A genuine face response on top of a shared AWB shift. The AWB component
    // is common to both regions, so it survives in the ratio only as a
    // compression toward 1: it can weaken a true positive but never invent one.
    const live = samplesFor(seq, 0.03, 0, 0.02, 0, 1.9);
    const v = chromaResponse(seq, live);
    expect(v.score).toBeGreaterThan(DEFAULT_CONFIG.chromaThreshold);
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
  });

  test('a response shared with the background does not count as liveness', () => {
    // Our flash falls off with distance, so a genuine near face responds far
    // more than the room behind it. A response the background shares equally
    // came from something else and is not evidence of a face in front of us.
    const seq = randomFlashSequence(5);
    const v = chromaResponse(seq, samplesFor(seq, 0.05, 0, 0, 0.05));
    expect(Math.abs(v.faceLift)).toBeGreaterThan(0.02);
    // Face and room moved together, so the ratio is ~1: whatever caused it was
    // not our flash landing on a near subject.
    expect(v.score).toBeLessThan(DEFAULT_CONFIG.chromaThreshold);
  });
});

describe('chroma gate policy', () => {
  const seq: FlashColor[] = ['red', 'dark', 'green', 'dark', 'blue'];

  test('an UNUSABLE reading never rejects: no measurement is not evidence', () => {
    // Face lost after the first slot, so only one slot was ever sampled.
    const v = chromaResponse(seq, samplesFor(seq, 0).filter((s) => s.slot === 0));
    expect(v.usable).toBe(false);
    expect(chromaPassed(v, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
  });

  test('report mode measures without failing anyone', () => {
    const v = chromaResponse(seq, samplesFor(seq, 0));
    expect(v.score).toBeLessThan(DEFAULT_CONFIG.chromaThreshold);
    expect(chromaPassed(v, 'report', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
    expect(chromaPassed(v, 'off', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
  });

  test('a session that never ran the challenge is not penalised', () => {
    expect(chromaPassed(null, 'enforce', DEFAULT_CONFIG.chromaThreshold)).toBe(true);
  });

  test('the shipped default is OFF: it could not be armed, so it does not run', () => {
    // The one gate in this module that ships enforcing. It earns that because
    // its operating point was measured on the target handset rather than
    // assumed: face-to-room response ratio 4.3 and 8.0 live, 0.25 and 0.71 for
    // a video replayed on a second display, with the threshold at 2.0 between
    // them. It is also the only barrier a recording cannot satisfy, since the
    // colour sequence does not exist until the session starts.
    // The ratio separates cleanly in principle and in these tests, where the
    // background is genuinely background. On hardware, at the framing the guide
    // circle asks for, the frame border is mostly more subject, so a live user
    // measured 0.43 and 1.24 and arming rejected them. Sampling a region proven
    // to be outside the subject is the fix; a threshold is not.
    // Sound physics that never separated on hardware, at the cost of 1.8 s of
    // flashing on every verification and no function at all in daylight. A
    // barrier that cannot be armed is a delay, not a barrier. The code stays for
    // indoor deployments that calibrate it; the default does not pay for it.
    expect(DEFAULT_CONFIG.chromaMode).toBe('off');
    expect(DEFAULT_CONFIG.chromaThreshold).toBe(1.25);
  });
});
