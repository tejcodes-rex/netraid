import { umeyama, warpAffine, meshTo5, REF5, ALIGN_SIZE } from '../align';
import type { Pt } from '../liveness';

function apply(M: number[], p: number[]): number[] {
  return [M[0] * p[0] + M[1] * p[1] + M[2], M[3] * p[0] + M[4] * p[1] + M[5]];
}

describe('face alignment (umeyama similarity transform)', () => {
  test('recovers an exact similarity transform from corresponding points', () => {
    // Construct dst by applying a known scale + rotation + translation to REF5.
    const s = 1.5, th = (20 * Math.PI) / 180, tx = 5, ty = -3;
    const known = [s * Math.cos(th), -s * Math.sin(th), tx, s * Math.sin(th), s * Math.cos(th), ty];
    const dst = REF5.map((p) => apply(known, p));

    const M = umeyama(REF5, dst);
    // The solved transform must map every source point onto its target.
    REF5.forEach((p, i) => {
      const out = apply(M, p);
      expect(out[0]).toBeCloseTo(dst[i][0], 4);
      expect(out[1]).toBeCloseTo(dst[i][1], 4);
    });
  });

  test('meshTo5 extracts eye centers, nose and mouth corners from a full mesh', () => {
    const lm: Pt[] = Array.from({ length: 468 }, () => ({ x: 0, y: 0 }));
    lm[33] = { x: 10, y: 10 }; lm[133] = { x: 20, y: 10 }; // left eye corners -> center (15,10)
    lm[362] = { x: 40, y: 10 }; lm[263] = { x: 50, y: 10 }; // right eye -> (45,10)
    lm[1] = { x: 30, y: 25 }; // nose
    lm[61] = { x: 18, y: 40 }; lm[291] = { x: 42, y: 40 }; // mouth corners
    const five = meshTo5(lm);
    expect(five[0]).toEqual([15, 10]);
    expect(five[1]).toEqual([45, 10]);
    expect(five[2]).toEqual([30, 25]);
    expect(five[3]).toEqual([18, 40]);
    expect(five[4]).toEqual([42, 40]);
  });
});

describe('warpAffine (bilinear crop sampler)', () => {
  test('samples in-bounds source pixels and leaves out-of-bounds pixels blank', () => {
    // 8x8 solid red source. Identity transform: output (x,y) samples source (x,y).
    const W = 8, H = 8;
    const src = new Uint8Array(W * H * 3);
    for (let i = 0; i < W * H; i++) { src[i * 3] = 255; src[i * 3 + 1] = 0; src[i * 3 + 2] = 0; }
    const identity = [1, 0, 0, 0, 1, 0];
    const out = warpAffine(src, W, H, identity);

    expect(out.length).toBe(ALIGN_SIZE * ALIGN_SIZE * 3);
    // Top-left output pixel maps into the red source region.
    expect(out[0]).toBe(255); // R
    expect(out[1]).toBe(0); // G
    // A pixel well outside the 8x8 source stays zero (skipped).
    const far = (50 * ALIGN_SIZE + 50) * 3;
    expect(out[far]).toBe(0);
  });
});
