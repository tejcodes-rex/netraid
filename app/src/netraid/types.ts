// NetraID public + internal types.

export type PersonId = string;

/** A face embedding: 512-d L2-normalized Float32 vector from MobileFaceNet. */
export type Embedding = Float32Array;

export interface EnrolledTemplate {
  personId: PersonId;
  embedding: Embedding;
  createdAt: number;
}

export interface AttendanceRecord {
  id: string; // client-generated UUID, idempotency key for sync
  personId: PersonId;
  ts: number; // epoch ms
  siteId?: string;
  lat?: number;
  lng?: number;
  deviceId: string;
  livenessPassed: boolean;
  matchScore: number;
  syncState: 'pending' | 'synced' | 'failed';
  attempts: number;
  createdAt: number;
}

export type Challenge = 'blink' | 'smile' | 'turn-left' | 'turn-right';

export interface LivenessResult {
  passed: boolean;
  activePassed: boolean;
  passiveScore: number; // 0..1, higher = more "live"
  completedChallenges: Challenge[];
}

export interface VerifyResult {
  ok: boolean;
  personId?: PersonId;
  score: number; // cosine similarity to best match
  liveness: LivenessResult;
  reason?: 'no-face' | 'spoof' | 'no-match' | 'low-quality';
  elapsedMs: number;
}

export interface NetraIDConfig {
  /** Cosine acceptance threshold (operating point). Calibrated per deployment. */
  matchThreshold: number;
  /** Required gap between the best and second-best person. Only enforced when a
   * second person is enrolled; blocks lookalike confusion near the threshold. */
  matchMargin: number;
  /** Frames aggregated per verification (majority person, median score). */
  verifyShots: number;
  /** Candidate frames captured per enrollment; the sharpest are kept and
   * embedding outliers dropped before averaging into the template. */
  enrollShots: number;
  /** Passive anti-spoof acceptance threshold (0..1). */
  passiveThreshold: number;
  /** Number of active challenges required, issued in random order. */
  numChallenges: number;
  /** Per-challenge timeout (ms). */
  challengeTimeoutMs: number;
  siteId?: string;
}

export const DEFAULT_CONFIG: NetraIDConfig = {
  matchThreshold: 0.38, // operating point for w600k_mbf on-device: same-person
  // frontal captures sit ~0.5+, different people <0.3 (see ml/scripts/check_separation.py).
  matchMargin: 0.08,
  verifyShots: 3,
  enrollShots: 6,
  passiveThreshold: 0.08, // MiniFASNet P(real), gated on the MAX across the verify
  // captures. Calibrated on-device (Vivo V2246, 24 Jul 2026): live-face captures
  // measured 0.12-0.74, a laptop-screen replay of the same face measured <= 0.04.
  // 0.08 sits 2x above the worst spoof reading and 2.8x below the lowest live one.
  numChallenges: 2,
  challengeTimeoutMs: 5000,
};
