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
import { robustAverage, bestMatch, aggregateMatches, cosine } from './math';
import { decideLiveness } from './liveness';
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
  passiveScore: number;
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
}): Promise<VerifyResult> {
  const t0 = Date.now();
  const { captures } = args;
  if (!isLoaded()) await loadRecognitionModel(); // load on demand if init raced

  const first = captures[0];
  const liveness: LivenessResult = decideLiveness(
    first.livenessActivePassed,
    first.completedChallenges,
    Math.max(...captures.map((c) => c.passiveScore)),
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
          `${m.personId || '-'}:${m.score.toFixed(3)}(2nd ${m.secondScore.toFixed(2)}, sharp ${captures[i].sharpness.toFixed(0)}, live ${captures[i].passiveScore.toFixed(3)})`,
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
