// Chroma challenge: active screen-illumination liveness, fully offline.
//
// WHY THIS EXISTS
// The gesture challenge set is finite. Blink, smile, turn-left, turn-right is
// four items, so a determined attacker can record all four and replay whichever
// one is asked for. Randomising the order raises the cost of that attack; it
// does not remove it. Adding more gestures does not remove it either, it only
// makes the recording longer.
//
// The fix is to stop asking for something the subject can pre-perform, and
// start asking for something only a physical face in front of THIS phone can
// produce: a reflection of light this phone chooses at verification time.
//
// The device turns its own screen into a coloured lamp and flashes a sequence
// picked from a CSPRNG the instant verification starts. A real face reflects
// that light, so the skin's colour balance tracks the sequence. A face on
// another display is emitting its own light; our flashes barely perturb it, so
// its colour balance does not track ours. Nothing about the sequence exists
// before the session begins, so there is nothing to record.
//
// This is the barrier that scales: the challenge space is not four gestures, it
// is every sequence of colours and durations the phone can emit.

import type { ChromaMode } from './types';

export type FlashColor = 'red' | 'green' | 'blue' | 'dark';

/** Screen colours for each slot.
 *
 * The neutral reference is DARK, not white. White was our own light too, so
 * every slot in the sequence was illuminated by the phone and there was no
 * unlit reference to compare against. That mattered once the verdict moved to
 * measuring the SHAPE of the subject: shape shows up as the difference between
 * how the scene looks under our light and how it looks without it, and with no
 * dark slot that difference was never observable. */
export const FLASH_RGB: Record<FlashColor, string> = {
  red: '#FF2D1A',
  green: '#12FF6A',
  blue: '#2C6BFF',
  dark: '#000000',
};

/** The chromatic (non-neutral) colours, and the channel each one drives. */
const CHROMATIC: FlashColor[] = ['red', 'green', 'blue'];

/** One frame's measurement of the face's colour balance during a flash slot. */
export interface ChromaSample {
  /** Index into the emitted sequence this frame was captured under. */
  slot: number;
  /** Face-region chromaticity: each channel over the channel sum, so overall
   * exposure and auto-gain cancel out and only the colour balance remains. */
  r: number;
  g: number;
  b: number;
  /** The same measurement taken on the frame BORDER, away from the face.
   *
   * Face chromaticity on its own is not a usable signal, because the front
   * camera's auto-white-balance corrects exactly what it measures: turn the
   * screen red and AWB pulls the whole image back toward neutral within a few
   * hundred milliseconds, erasing the lift. Measured on the target handset, a
   * live face produced a lift of -0.008 to +0.002, which is noise.
   *
   * AWB is a GLOBAL gain, so it moves the face and the background together.
   * The difference between them survives it. And the difference is exactly what
   * the physics predicts: our flash falls off with the square of distance, so a
   * face at arm's length is lit strongly by it while the room behind is not.
   * A face on another display has no such gradient, because the display emits
   * its own light and our flash barely perturbs it. */
  bgR: number;
  bgG: number;
  bgB: number;
  /** Chromaticity of the CENTRE of the face region, and of its outer ring.
   *
   * This is the measurement the verdict is taken on, and it needs no background
   * at all. The frame border turned out to be a poor reference: at the 40 cm the
   * guide circle asks for, it is mostly the subject's own neck and shoulders,
   * lit by the same flash at the same distance, so the ratio collapsed toward 1
   * on genuine users (measured: 0.43 and 1.24).
   *
   * A face carries its own reference, in its shape. Skin is close to Lambertian,
   * so how much of our light a patch returns falls off with the cosine of the
   * angle between its surface normal and the phone. The nose and forehead face
   * the screen almost head-on; the cheeks and jawline curve away at a grazing
   * angle and return far less. So on a real head, the centre tracks our flash
   * substantially harder than the rim does.
   *
   * A photograph or a display is FLAT. Every part of it shares one surface
   * normal, so it responds uniformly and the ratio is 1 whatever its content,
   * brightness or distance. That is a property of the geometry, not of the
   * image, which is why no recording can produce it.
   *
   * Both regions are the same subject at nearly the same distance, so global
   * auto-exposure and auto-white-balance divide out exactly as before. */
  cR: number;
  cG: number;
  cB: number;
  pR: number;
  pG: number;
  pB: number;
  /** Mean LUMINANCE of the face centre and of its rim, in this frame.
   *
   * Chromaticity cannot carry the shape measurement, and putting it there was
   * the mistake: chromaticity is each channel over the channel sum, so it
   * divides intensity out by construction, while Lambertian falloff IS an
   * intensity effect. Once the flash dominates the ambient light, every lit part
   * of the face converges on the same colour balance no matter how much light it
   * received. Measured on a real head: centre 0.0336 against rim 0.0331, a ratio
   * of 0.97, indistinguishable from a flat panel.
   *
   * Luminance keeps exactly what chromaticity throws away. Auto-exposure is a
   * global gain, so it cancels in the centre-to-rim ratio taken within one
   * frame, and what survives is how much more of our light the parts facing the
   * phone returned than the parts curving away from it. */
  cY: number;
  pY: number;
}

export interface ChromaVerdict {
  /** How much MORE contrast our light creates between the centre of the face
   * and its rim, compared with the same scene unlit.
   *
   * On a head, switching on a near light raises the centre far more than the
   * rim, because the rim curves away and meets that light at a grazing angle,
   * so this rises above 1. A photograph or a display is flat: one surface
   * normal everywhere, so our light raises all of it by the same factor and the
   * number stays at 1, whatever the panel shows, however bright it is, and
   * however far away it is held. Geometry rather than image content, which is
   * why no recording can carry it.
   *
   * A RATIO OF RATIOS, so everything global divides out: auto-exposure,
   * auto-white-balance, screen brightness, skin tone and distance all scale
   * both regions and both phases together.
   */
  score: number;
  /** Whole-face lift, kept for diagnosis and for calibration records. */
  faceLift: number;
  /** Frame-border lift. Diagnostic only: see the ChromaSample docs for why it
   * is not a usable reference at the framing this app asks for. */
  bgLift: number;
  /** Face-centre and face-rim CHROMATICITY lifts. Diagnostic: these are what
   * the shape verdict used to be taken on, before it moved to luminance. */
  centreLift: number;
  rimLift: number;
  /** Mean centre-to-rim luminance ratio while our light was on, and while it
   * was off. `score` is the first divided by the second. */
  litContrast: number;
  darkContrast: number;
  /** Slots that produced at least one usable sample. */
  slotsMeasured: number;
  samples: number;
  /** False only when the sequence is too short or too few slots were sampled. */
  usable: boolean;
}

/**
 * A CSPRNG flash sequence: `n` slots, never two identical in a row, always
 * containing at least two different chromatic colours so the response can be
 * differenced. DARK slots are interleaved as the unlit reference.
 */
export function randomFlashSequence(n: number): FlashColor[] {
  const bytes = new Uint8Array(n * 2);
  // react-native-get-random-values polyfills crypto.getRandomValues
  crypto.getRandomValues(bytes);
  const seq: FlashColor[] = [];
  for (let i = 0; i < n; i++) {
    // Alternate lit and neutral slots: a lift is only meaningful against a
    // reference measured moments earlier under the same ambient light.
    if (i % 2 === 1) {
      seq.push('dark');
      continue;
    }
    let pick = CHROMATIC[bytes[i] % CHROMATIC.length];
    if (pick === seq[i - 2]) pick = CHROMATIC[(bytes[i + n] + 1) % CHROMATIC.length];
    seq.push(pick);
  }
  // Guarantee at least two distinct chromatic colours; with one colour the
  // score would measure ambient drift rather than a response to our light.
  const lit = seq.filter((c) => c !== 'dark');
  if (new Set(lit).size < 2 && seq.length >= 3) {
    const other = CHROMATIC.filter((c) => c !== lit[0]);
    seq[seq.length - 1] = other[bytes[0] % other.length];
  }
  return seq;
}

/**
 * Score how strongly the face's colour balance tracked the emitted sequence.
 *
 * For each chromatic colour actually emitted, compare that channel's mean
 * chromaticity while the screen was showing the colour against its mean while
 * the screen was showing anything else. A face lit by our screen lifts the
 * matching channel every time; a display replaying a recording does not,
 * because its own emission dominates whatever our screen adds.
 *
 * The result is in chromaticity units (each channel is a fraction of the
 * channel sum, so all three sum to 1). Positive means the face responded.
 */
export function chromaResponse(seq: FlashColor[], samples: ChromaSample[]): ChromaVerdict {
  const chan = (s: ChromaSample, c: FlashColor) => (c === 'red' ? s.r : c === 'green' ? s.g : s.b);
  const bgChan = (s: ChromaSample, c: FlashColor) =>
    (c === 'red' ? s.bgR : c === 'green' ? s.bgG : s.bgB);
  const cChan = (s: ChromaSample, c: FlashColor) =>
    (c === 'red' ? s.cR : c === 'green' ? s.cG : s.cB);
  const pChan = (s: ChromaSample, c: FlashColor) =>
    (c === 'red' ? s.pR : c === 'green' ? s.pG : s.pB);
  const slotsSeen = new Set(samples.map((s) => s.slot).filter((i) => i < seq.length));
  const emitted = CHROMATIC.filter((c) =>
    seq.some((x, i) => x === c && slotsSeen.has(i)),
  );

  const ratios: number[] = [];
  const faceLifts: number[] = [];
  const bgLifts: number[] = [];
  const centreLifts: number[] = [];
  const rimLifts: number[] = [];
  for (const c of emitted) {
    const on: number[] = [];
    const off: number[] = [];
    const bgOn: number[] = [];
    const bgOff: number[] = [];
    const cOn: number[] = [];
    const cOff: number[] = [];
    const pOn: number[] = [];
    const pOff: number[] = [];
    for (const s of samples) {
      if (s.slot >= seq.length) continue;
      if (seq[s.slot] === c) {
        on.push(chan(s, c)); bgOn.push(bgChan(s, c));
        cOn.push(cChan(s, c)); pOn.push(pChan(s, c));
      } else {
        off.push(chan(s, c)); bgOff.push(bgChan(s, c));
        cOff.push(cChan(s, c)); pOff.push(pChan(s, c));
      }
    }
    if (!on.length || !off.length) continue;
    const faceLift = mean(on) - mean(off);
    // The background's response to the same flash is the reference. Whatever
    // the sensor did globally (auto-white-balance, auto-exposure) applied to
    // both regions equally, so dividing removes it. What survives is the part
    // of the response that came from our light actually reaching a near
    // subject, because illumination falls off with the square of distance.
    //
    // MAGNITUDES, not signed values: on this sensor the response is negative,
    // because auto-white-balance pulls the lit channel down harder than the
    // reflected light pushes it up. The direction is the sensor's convention.
    // What identifies a face is that the near region moved far MORE than the
    // far one, whichever way the sensor chose to express it.
    const bgLift = bgOn.length && bgOff.length ? mean(bgOn) - mean(bgOff) : 0;
    // The verdict: face CENTRE against face RIM. Both are the same subject, so
    // any global sensor correction divides out, and what remains is whether the
    // surface curves away from the phone the way a head does.
    const centreLift = cOn.length && cOff.length ? mean(cOn) - mean(cOff) : 0;
    const rimLift = pOn.length && pOff.length ? mean(pOn) - mean(pOff) : 0;
    faceLifts.push(faceLift);
    bgLifts.push(bgLift);
    centreLifts.push(centreLift);
    rimLifts.push(rimLift);
    ratios.push(Math.abs(centreLift) / (Math.abs(rimLift) + RATIO_EPS));
  }

  // Every emitted colour must respond more on the face than on the background.
  // Taking the MINIMUM rather than the average means one accidental correlation
  // cannot carry the verdict: a replay would have to happen to track every
  // colour we chose, in the right region, at the right moment.
  // THE VERDICT: how much more centre-to-rim contrast our light creates than
  // the scene has without it. Computed per frame as a ratio inside that frame,
  // so auto-exposure cannot touch it, then averaged over the lit slots and over
  // the dark ones.
  const litC: number[] = [];
  const darkC: number[] = [];
  for (const s of samples) {
    if (s.slot >= seq.length) continue;
    if (s.pY <= 0) continue;
    (seq[s.slot] === 'dark' ? darkC : litC).push(s.cY / s.pY);
  }
  const litContrast = litC.length ? mean(litC) : 0;
  const darkContrast = darkC.length ? mean(darkC) : 0;
  const shape = darkContrast > 0 ? litContrast / darkContrast : 0;
  const score = shape;
  return {
    score,
    faceLift: faceLifts.length ? faceLifts[argMaxAbs(faceLifts)] : 0,
    bgLift: bgLifts.length ? bgLifts[argMaxAbs(bgLifts)] : 0,
    centreLift: centreLifts.length ? centreLifts[argMaxAbs(centreLifts)] : 0,
    rimLift: rimLifts.length ? rimLifts[argMaxAbs(rimLifts)] : 0,
    litContrast,
    darkContrast,
    slotsMeasured: slotsSeen.size,
    samples: samples.length,
    // A ratio is only meaningful once the flash moved SOMETHING. In bright
    // sunlight a phone screen cannot compete with the sun, both regions sit in
    // noise, and dividing noise by noise is not evidence either way. That case
    // is unusable, which is never a rejection.
    // Usable once both phases were actually observed. Without a dark slot and a
    // lit slot there is no contrast to compare, and a missing measurement is
    // never a rejection.
    usable: slotsSeen.size >= 3 && litC.length >= 2 && darkC.length >= 1
      && litContrast > 0 && darkContrast > 0,
  };
}

/** Guards the ratio when the background response is essentially zero. */
const RATIO_EPS = 0.0015;

/** Index of the entry with the largest magnitude, for reporting the worst case. */
function argMaxAbs(xs: number[]): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i]) > Math.abs(xs[best])) best = i;
  return best;
}

function mean(xs: number[]): number {
  let t = 0;
  for (const x of xs) t += x;
  return t / xs.length;
}

/**
 * Apply the configured mode. `report` measures and surfaces the score without
 * failing anyone, which is how a deployment collects its own operating point
 * before arming the gate; `enforce` rejects below the threshold.
 *
 * An UNUSABLE reading (face lost during the flash, too few slots) is never a
 * rejection: absence of a measurement is not evidence of an attack.
 */
export function chromaPassed(v: ChromaVerdict | null, mode: ChromaMode, threshold: number): boolean {
  if (mode !== 'enforce' || v === null || !v.usable) return true;
  return v.score >= threshold;
}
