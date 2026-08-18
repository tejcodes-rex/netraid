// NetraID public API, the drop-in surface the Datalake 3.0 app calls.
//
//   import { NetraID } from '@netraid/face-auth';
//   await NetraID.init();
//   await NetraID.enroll({ personId });
//   const r = await NetraID.verify({ requireLiveness: true });
//
// The heavy real-time work (detection, landmarks, liveness FSM) runs in the
// vision-camera frame processor (see frameProcessor.ts); this module orchestrates
// enrollment, matching, persistence and sync.

import { loadRecognitionModel, embedTTA, isLoaded } from './recognition';
import { initStore, saveTemplate, loadTemplates, insertAttendance } from './store';
import { robustAverage, bestMatch, aggregateMatches, cosine, median, consensus } from './math';
import { decideLiveness, enrollGatePassed } from './liveness';
import type { ChromaVerdict } from './chroma';
import { startSync } from './sync';
import { getDeviceId } from './keys';
import uuid from 'react-native-uuid';
import {
  DEFAULT_CONFIG,
  type NetraIDConfig,
  type VerifyResult,
  type LivenessResult,
  type Challenge,
} from './types';
import { logCalibration, r4 } from './calibration';

let config: NetraIDConfig = DEFAULT_CONFIG;
let initPromise: Promise<void> | null = null;

/** Thrown when an enrolling face already matches a different enrolled person.
 * One face = one identity: blocks duplicate identities (and the attendance
 * fraud of one worker enrolling under two IDs). */
export class DuplicateFaceError extends Error {
  constructor(public readonly existingPersonId: string, score: number) {
    super(`Face already enrolled as ${existingPersonId} (similarity ${score.toFixed(2)})`);
    this.name = 'DuplicateFaceError';
  }
}

/** Thrown when the frames offered for enrollment do not read as a live face.
 * Enrollment is the root of trust: a template built from a photograph or a
 * screen makes every later verification against it meaningless, and no amount
 * of liveness at verification time can repair it. The floor is the same
 * device-calibrated P(real) statistic the verification path uses. */
export class SpoofedEnrollmentError extends Error {
  constructor(public readonly passiveScore: number) {
    super(
      `Enrollment frames did not read as a live face (anti-spoof ${passiveScore.toFixed(2)})`,
    );
    this.name = 'SpoofedEnrollmentError';
  }
}

/** A new template this similar to another person's template is the same face.
 * Genuinely different people measure <= 0.33 even in the LFW worst tail
 * (typically < 0.1); the same face re-enrolled sloppily (angle/lighting/
 * glasses variation) was measured slipping under 0.6 on-device, so the guard
 * sits between the two regimes. */
const DUPLICATE_FACE_THRESHOLD = 0.45;

export interface CapturedFace {
  /** Aligned 112x112 RGB crop (uint8, HWC) produced by the frame processor. */
  rgb: Uint8Array;
  livenessActivePassed: boolean;
  completedChallenges: Challenge[];
  /** MiniFASNet P(real) for this capture. */
  passiveScore: number;
  /** MiniFASNet P(screen-replay) for this capture. */
  passiveScreen: number;
  /** Laplacian-variance sharpness of the crop (see quality.ts). */
  sharpness: number;
}

async function init(cfg?: Partial<NetraIDConfig>): Promise<void> {
  config = { ...DEFAULT_CONFIG, ...cfg };
  // Concurrency-safe: several screens call init() on mount; the store must
  // open exactly once. The recognition model is NOT awaited here: it loads
  // through the useRecognitionModel hook on the capture screens, and
  // enroll()/verify() await it on demand. Blocking init on it would stall
  // screens that never mount a camera (e.g. the dashboard reading counts).
  if (!initPromise) {
    initPromise = initStore().then(() => startSync()).catch((e) => {
      initPromise = null; // allow retry on next call
      throw e;
    });
  }
  await initPromise;
}

/**
 * Enroll a person from multi-shot captures. The sharpest crops are kept, each
 * is embedded with flip-TTA, embedding outliers (a bad crop that slipped past
 * the quality gates) are dropped, and only the averaged template is stored.
 */
async function enroll(args: {
  personId: string;
  captures: CapturedFace[];
}): Promise<void> {
  if (!isLoaded()) await loadRecognitionModel(); // load on demand if init raced
  const bySharpness = [...args.captures].sort((a, b) => b.sharpness - a.sharpness);
  const keep = bySharpness.slice(0, Math.max(3, Math.ceil(bySharpness.length * 0.8)));
  // Anti-spoof at the ROOT OF TRUST. Enrollment previously ran with no liveness
  // barrier at all, so a photograph held to the lens could become an identity.
  // The median is used for the same reason as at verification: one fortunate
  // frame must not carry a print through, and one noisy frame must not reject a
  // real person.
  //
  // Whether that median REJECTS is governed by `enrollSpoofMode`, which ships as
  // `report`: the enrollment burst is a different capture distribution from the
  // verification one, so the verify threshold is not evidence about it. The
  // reading is recorded on every save either way, which is what lets the gate be
  // armed later against numbers measured on this path. See docs/CALIBRATION.md.
  const passive = median(keep.map((c) => c.passiveScore));
  const gateOk = enrollGatePassed(
    passive, config.enrollSpoofMode, config.enrollPassiveThreshold,
  );
  if (config.enrollSpoofMode !== 'off') {
    logCalibration({
      stage: 'enroll',
      outcome: gateOk ? 'accept' : 'reject',
      mode: config.enrollSpoofMode,
      passiveMedian: r4(passive),
      real: keep.map((c) => r4(c.passiveScore)),
      screen: keep.map((c) => r4(c.passiveScreen)),
      sharp: keep.map((c) => Math.round(c.sharpness)),
      shots: keep.length,
    });
  }
  if (!gateOk) throw new SpoofedEnrollmentError(passive);

  const embs = keep.map((c) => embedTTA(c.rgb));
  const template = robustAverage(embs);
  if (__DEV__) {
    console.log(
      '[enroll] ' + args.personId + ' shots: ' +
        keep.map((c, i) =>
          `sharp ${c.sharpness.toFixed(0)} cos ${cosine(embs[i], template).toFixed(3)}`,
        ).join(' | '),
    );
  }

  // One face, one identity: re-enrolling the same personId updates the
  // template, but the same face under a NEW id is rejected.
  const others = (await loadTemplates()).filter((t) => t.personId !== args.personId);
  const clash = bestMatch(template, others);
  if (clash.personId && clash.score >= DUPLICATE_FACE_THRESHOLD) {
    throw new DuplicateFaceError(clash.personId, clash.score);
  }

  await saveTemplate(args.personId, template);
}

/**
 * Verify from several captured frames: enforce liveness, embed each with
 * flip-TTA, and aggregate (majority person, median score, margin over the
 * second-best person). On success, record an attendance event into the
 * encrypted local queue.
 */
async function verify(args: {
  captures: CapturedFace[];
  requireLiveness?: boolean;
  /** Result of the screen-illumination challenge for this session, when the
   * host screen ran one. Omitted means the challenge did not run, which is
   * never treated as evidence of an attack. */
  chroma?: ChromaVerdict | null;
}): Promise<VerifyResult> {
  const t0 = Date.now();
  const { captures } = args;
  if (!isLoaded()) await loadRecognitionModel(); // load on demand if init raced

  const first = captures[0];
  // Median P(real), and the screen-replay score at least two captures agreed
  // on. Neither statistic can be swung by a single frame, in either direction:
  // one lucky frame cannot carry a replay through, and one noisy frame cannot
  // reject a real person. See decideLiveness.
  const liveness: LivenessResult = decideLiveness(
    first.livenessActivePassed,
    first.completedChallenges,
    median(captures.map((c) => c.passiveScore)),
    consensus(captures.map((c) => c.passiveScreen), 2),
    args.chroma ?? null,
    config,
  );
  if ((args.requireLiveness ?? true) && !liveness.passed) {
    return { ok: false, score: 0, liveness, reason: 'spoof',
      elapsedMs: Date.now() - t0 };
  }

  const templates = await loadTemplates();
  const perFrame = captures.map((c) => bestMatch(embedTTA(c.rgb), templates));
  const match = aggregateMatches(perFrame);
  if (__DEV__) {
    console.log(
      '[verify] frames=' +
        perFrame.map((m, i) =>
          `${m.personId || '-'}:${m.score.toFixed(3)}(2nd ${m.secondScore.toFixed(2)}, sharp ${captures[i].sharpness.toFixed(0)}, live ${captures[i].passiveScore.toFixed(3)}, scr ${captures[i].passiveScreen.toFixed(3)})`,
        ).join(' ') +
        ` -> ${match.personId || 'NONE'}:${match.score.toFixed(3)} in ${Date.now() - t0}ms`,
    );
  }
  const marginOk =
    match.secondScore < 0 ||
    match.score - match.secondScore >= config.matchMargin;

  if (!match.personId || match.score < config.matchThreshold || !marginOk) {
    return { ok: false, score: Math.max(0, match.score), liveness,
      reason: 'no-match', elapsedMs: Date.now() - t0 };
  }

  await insertAttendance({
    id: uuid.v4() as string,
    personId: match.personId,
    ts: Date.now(),
    siteId: config.siteId,
    deviceId: await getDeviceId(),
    livenessPassed: liveness.passed,
    matchScore: match.score,
    syncState: 'pending',
    attempts: 0,
    createdAt: Date.now(),
  });

  return { ok: true, personId: match.personId, score: match.score, liveness,
    elapsedMs: Date.now() - t0 };
}

export const NetraID = { init, enroll, verify };
export * from './types';
