import {
  l2normalize, cosine, bestMatch, averageEmbeddings,
  aggregateMatches, median, mirrorRGB, robustAverage,
} from '../math';
import { assessCrop, MIN_SHARPNESS } from '../quality';

const f = (...xs: number[]) => Float32Array.from(xs);
const norm = (v: Float32Array) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

describe('vector math (face matching core)', () => {
  test('l2normalize returns a unit vector', () => {
    const out = l2normalize(f(3, 4)); // |(3,4)| = 5
    expect(norm(out)).toBeCloseTo(1, 6);
    expect(out[0]).toBeCloseTo(0.6, 6);
    expect(out[1]).toBeCloseTo(0.8, 6);
  });

  test('cosine of identical normalized vectors is 1, orthogonal is 0', () => {
    const a = l2normalize(f(1, 2, 3, 4));
    expect(cosine(a, a)).toBeCloseTo(1, 6);
    const x = l2normalize(f(1, 0));
    const y = l2normalize(f(0, 1));
    expect(cosine(x, y)).toBeCloseTo(0, 6);
  });

  test('bestMatch picks the closest enrolled template by cosine', () => {
    const probe = l2normalize(f(1, 0.1, 0));
    const templates = [
      { personId: 'A', embedding: l2normalize(f(0, 1, 0)) },
      { personId: 'B', embedding: l2normalize(f(1, 0, 0)) }, // closest to probe
      { personId: 'C', embedding: l2normalize(f(0, 0, 1)) },
    ];
    const r = bestMatch(probe, templates);
    expect(r.personId).toBe('B');
    expect(r.score).toBeGreaterThan(0.9);
  });

  test('an impostor scores far below a genuine match (separability)', () => {
    const enrolled = l2normalize(f(0.9, 0.2, 0.1, 0.05));
    const genuine = l2normalize(f(0.88, 0.22, 0.12, 0.04)); // same person, jittered
    const impostor = l2normalize(f(-0.3, 0.7, -0.5, 0.4)); // different
    const tpl = [{ personId: 'P', embedding: enrolled }];
    expect(bestMatch(genuine, tpl).score).toBeGreaterThan(0.27); // above DEFAULT_CONFIG threshold
    expect(bestMatch(impostor, tpl).score).toBeLessThan(0.27);
  });

  test('averageEmbeddings produces a renormalized unit template (multi-shot enroll)', () => {
    const out = averageEmbeddings([
      l2normalize(f(1, 0, 0)),
      l2normalize(f(0.9, 0.1, 0)),
      l2normalize(f(0.95, 0.05, 0.05)),
    ]);
    expect(norm(out)).toBeCloseTo(1, 6);
  });

  test('bestMatch tracks the second-best person for the margin rule', () => {
    const probe = l2normalize(f(1, 0.1, 0));
    const templates = [
      { personId: 'A', embedding: l2normalize(f(1, 0, 0)) },
      { personId: 'B', embedding: l2normalize(f(0.7, 0.7, 0)) },
      { personId: 'C', embedding: l2normalize(f(0, 0, 1)) },
    ];
    const r = bestMatch(probe, templates);
    expect(r.personId).toBe('A');
    expect(r.secondScore).toBeCloseTo(
      cosine(probe, templates[1].embedding), 6,
    );
    // With a single enrolled person there is no second-best.
    expect(bestMatch(probe, [templates[0]]).secondScore).toBe(-1);
  });
});

describe('multi-frame aggregation (verify)', () => {
  const m = (personId: string, score: number, secondScore = -1) =>
    ({ personId, score, secondScore });

  test('majority person wins with the median of its scores', () => {
    const r = aggregateMatches([m('A', 0.55), m('A', 0.61), m('B', 0.4)]);
    expect(r.personId).toBe('A');
    expect(r.score).toBeCloseTo(0.58, 6); // median of [0.55, 0.61]
  });

  test('one blurred outlier frame cannot decide the verdict', () => {
    const r = aggregateMatches([m('A', 0.6), m('A', 0.58), m('A', 0.1)]);
    expect(r.personId).toBe('A');
    expect(r.score).toBeCloseTo(0.58, 6); // median, not dragged down by 0.1
  });

  test('no strict majority means no match', () => {
    const r = aggregateMatches([m('A', 0.6), m('B', 0.62), m('C', 0.61)]);
    expect(r.personId).toBe('');
    const r2 = aggregateMatches([m('A', 0.6), m('B', 0.62)]);
    expect(r2.personId).toBe('');
  });

  test('keeps the worst-case confusion score across frames', () => {
    const r = aggregateMatches([m('A', 0.6, 0.2), m('A', 0.62, 0.35)]);
    expect(r.secondScore).toBeCloseTo(0.35, 6);
  });

  test('median handles odd and even counts', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });
});

describe('robustAverage (enrollment outlier rejection)', () => {
  test('drops a shot that disagrees with the consensus', () => {
    const good = [
      l2normalize(f(1, 0.02, 0)),
      l2normalize(f(0.98, 0, 0.03)),
      l2normalize(f(1, 0.01, 0.01)),
    ];
    const bad = l2normalize(f(-0.2, 1, 0.4)); // wildly different crop
    const tpl = robustAverage([...good, bad]);
    // The template should stay close to the good cluster, unmoved by the outlier.
    expect(cosine(tpl, averageEmbeddings(good))).toBeGreaterThan(0.999);
  });

  test('small sets are averaged as-is', () => {
    const a = l2normalize(f(1, 0));
    const b = l2normalize(f(0.9, 0.1));
    expect(cosine(robustAverage([a, b]), averageEmbeddings([a, b]))).toBeCloseTo(1, 6);
  });
});

describe('flip-TTA and crop quality', () => {
  test('mirrorRGB flips horizontally and is an involution', () => {
    // 2x2 RGB image with distinct pixels.
    const img = Uint8Array.from([
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12,
    ]);
    const flipped = mirrorRGB(img, 2);
    expect([...flipped.slice(0, 3)]).toEqual([4, 5, 6]);
    expect([...flipped.slice(3, 6)]).toEqual([1, 2, 3]);
    expect([...mirrorRGB(flipped, 2)]).toEqual([...img]);
  });

  test('sharpness separates textured crops from flat/blurred ones', () => {
    const n = 112 * 112 * 3;
    const flat = new Uint8Array(n).fill(128);
    const textured = new Uint8Array(n);
    for (let i = 0; i < n; i++) textured[i] = (i / 3 | 0) % 2 ? 200 : 60;
    expect(assessCrop(flat).sharpness).toBeLessThan(MIN_SHARPNESS);
    expect(assessCrop(textured).sharpness).toBeGreaterThan(MIN_SHARPNESS);
    expect(assessCrop(flat).brightness).toBeCloseTo(128, 0);
  });
});
