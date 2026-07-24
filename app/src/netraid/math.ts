// Small, dependency-free vector math used for face matching.
// Kept allocation-light because match() runs on the hot path.

import type { Embedding } from './types';

export function l2normalize(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const inv = 1 / (Math.sqrt(s) + 1e-9);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}

/** Cosine similarity of two L2-normalized vectors == dot product. */
export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export interface MatchResult {
  personId: string;
  score: number;
  /** Cosine to the best template of a DIFFERENT person (-1 if none). Used for
   * the acceptance margin: a genuine match is both high and well-separated. */
  secondScore: number;
}

/**
 * Brute-force nearest template by cosine. For field deployments (hundreds of
 * enrolled staff per device) this is sub-millisecond. For thousands, swap in an
 * ANN index (e.g. sqlite-vec) behind this same signature.
 */
export function bestMatch(
  probe: Embedding,
  templates: { personId: string; embedding: Embedding }[],
): MatchResult {
  let best: MatchResult = { personId: '', score: -1, secondScore: -1 };
  for (const t of templates) {
    const s = cosine(probe, t.embedding);
    if (s > best.score) {
      if (t.personId !== best.personId) best.secondScore = best.score;
      best = { personId: t.personId, score: s, secondScore: best.secondScore };
    } else if (t.personId !== best.personId && s > best.secondScore) {
      best.secondScore = s;
    }
  }
  return best;
}

/**
 * Combine per-frame match results into one verdict (multi-frame verification).
 * The winner needs a strict majority of frames; its score is the MEDIAN of the
 * winning frames, so one blurred or badly aligned probe cannot decide alone.
 */
export function aggregateMatches(perFrame: MatchResult[]): MatchResult {
  const wins = new Map<string, MatchResult[]>();
  for (const m of perFrame) {
    if (!m.personId) continue;
    const arr = wins.get(m.personId) ?? [];
    arr.push(m);
    wins.set(m.personId, arr);
  }
  let top: MatchResult[] = [];
  let topId = '';
  for (const [id, arr] of wins) {
    if (arr.length > top.length) { top = arr; topId = id; }
  }
  if (!topId || top.length * 2 <= perFrame.length) {
    return { personId: '', score: -1, secondScore: -1 };
  }
  return {
    personId: topId,
    score: median(top.map((m) => m.score)),
    secondScore: Math.max(...top.map((m) => m.secondScore)),
  };
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Horizontal flip of a square uint8 RGB (HWC) image, for flip-TTA embedding. */
export function mirrorRGB(rgb: Uint8Array, size: number): Uint8Array {
  const out = new Uint8Array(rgb.length);
  for (let y = 0; y < size; y++) {
    const row = y * size * 3;
    for (let x = 0; x < size; x++) {
      const src = row + x * 3;
      const dst = row + (size - 1 - x) * 3;
      out[dst] = rgb[src]; out[dst + 1] = rgb[src + 1]; out[dst + 2] = rgb[src + 2];
    }
  }
  return out;
}

/** Average several L2-normalized embeddings (multi-shot enrollment) and renormalize. */
export function averageEmbeddings(embs: Float32Array[]): Float32Array {
  const dim = embs[0].length;
  const acc = new Float32Array(dim);
  for (const e of embs) for (let i = 0; i < dim; i++) acc[i] += e[i];
  for (let i = 0; i < dim; i++) acc[i] /= embs.length;
  return l2normalize(acc);
}

/**
 * Average embeddings, dropping shots that disagree with the consensus (a bad
 * crop that slipped past the quality gates). Falls back to the plain average
 * when too few shots survive to be meaningful.
 */
export function robustAverage(embs: Float32Array[]): Float32Array {
  if (embs.length <= 3) return averageEmbeddings(embs);
  const centroid = averageEmbeddings(embs);
  const sims = embs.map((e) => cosine(e, centroid));
  const mean = sims.reduce((a, b) => a + b, 0) / sims.length;
  const kept = embs.filter((_, i) => sims[i] >= mean - 0.15);
  return averageEmbeddings(kept.length >= 3 ? kept : embs);
}
