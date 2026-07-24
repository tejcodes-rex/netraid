import { base64ToBytes, cropRGB, resizeRGB } from '../imaging';

describe('base64ToBytes (demo asset decoding)', () => {
  const cases: [string, number[]][] = [
    ['AQID', [1, 2, 3]],
    ['AQIDBA==', [1, 2, 3, 4]],
    ['AQIDBAU=', [1, 2, 3, 4, 5]],
    ['', []],
  ];
  test.each(cases)('decodes %s', (b64, bytes) => {
    expect([...base64ToBytes(b64)]).toEqual(bytes);
  });

  test('roundtrips a full byte range', () => {
    const buf = Uint8Array.from({ length: 256 }, (_, i) => i);
    const b64 = Buffer.from(buf).toString('base64');
    expect([...base64ToBytes(b64)]).toEqual([...buf]);
  });
});

describe('resizeRGB / cropRGB', () => {
  test('identity resize preserves pixels', () => {
    const img = Uint8Array.from({ length: 4 * 4 * 3 }, (_, i) => (i * 7) % 256);
    expect([...resizeRGB(img, 4, 4, 4, 4)]).toEqual([...img]);
  });

  test('downscale of a flat image stays flat', () => {
    const img = new Uint8Array(8 * 8 * 3).fill(90);
    const out = resizeRGB(img, 8, 8, 4, 4);
    expect(out.every((v) => v === 90)).toBe(true);
  });

  test('crop clamps to bounds and extracts the right region', () => {
    // 2x2 image, distinct pixels per channel-0.
    const img = Uint8Array.from([
      10, 0, 0, 20, 0, 0,
      30, 0, 0, 40, 0, 0,
    ]);
    const c = cropRGB(img, 2, 2, 1, 1, 5, 5); // over-extends, clamps to 1x1
    expect(c.w).toBe(1);
    expect(c.h).toBe(1);
    expect(c.rgb[0]).toBe(40);
  });
});
