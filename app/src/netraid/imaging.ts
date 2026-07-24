// Pure raster helpers for the camera-free pipeline path (Pipeline Demo screen,
// iOS simulator). The live camera path does its resizing on the GPU via
// vision-camera-resize-plugin; these mirror that behavior in plain TypeScript.

const B64 =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_LOOKUP = new Uint8Array(128);
for (let i = 0; i < B64.length; i++) B64_LOOKUP[B64.charCodeAt(i)] = i;

/** Decode base64 to bytes. (Hermes lacks a global atob; this is dependency-free.) */
export function base64ToBytes(b64: string): Uint8Array {
  let end = b64.length;
  while (end > 0 && b64[end - 1] === '=') end--;
  const n = Math.floor((end * 3) / 4);
  const out = new Uint8Array(n);
  let o = 0;
  for (let i = 0; i + 3 < end; i += 4) {
    const x =
      (B64_LOOKUP[b64.charCodeAt(i)] << 18) |
      (B64_LOOKUP[b64.charCodeAt(i + 1)] << 12) |
      (B64_LOOKUP[b64.charCodeAt(i + 2)] << 6) |
      B64_LOOKUP[b64.charCodeAt(i + 3)];
    out[o++] = x >> 16;
    out[o++] = (x >> 8) & 0xff;
    out[o++] = x & 0xff;
  }
  const rem = end % 4;
  if (rem >= 2) {
    const i = end - rem;
    let x = (B64_LOOKUP[b64.charCodeAt(i)] << 18) | (B64_LOOKUP[b64.charCodeAt(i + 1)] << 12);
    if (rem === 3) x |= B64_LOOKUP[b64.charCodeAt(i + 2)] << 6;
    out[o++] = x >> 16;
    if (rem === 3) out[o++] = (x >> 8) & 0xff;
  }
  return out;
}

/** Bilinear resize of uint8 RGB (HWC). */
export function resizeRGB(
  src: Uint8Array, srcW: number, srcH: number, dstW: number, dstH: number,
): Uint8Array {
  const out = new Uint8Array(dstW * dstH * 3);
  const sx = srcW / dstW, sy = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0; else if (fy > srcH - 1) fy = srcH - 1;
    const y0 = Math.max(0, Math.min(Math.floor(fy), srcH - 2));
    const wy = fy - y0;
    for (let x = 0; x < dstW; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0; else if (fx > srcW - 1) fx = srcW - 1;
      const x0 = Math.max(0, Math.min(Math.floor(fx), srcW - 2));
      const wx = fx - x0;
      const i00 = (y0 * srcW + x0) * 3;
      const i10 = i00 + 3;
      const i01 = i00 + srcW * 3;
      const i11 = i01 + 3;
      const oi = (y * dstW + x) * 3;
      for (let c = 0; c < 3; c++) {
        out[oi + c] = Math.round(
          src[i00 + c] * (1 - wx) * (1 - wy) + src[i10 + c] * wx * (1 - wy) +
          src[i01 + c] * (1 - wx) * wy + src[i11 + c] * wx * wy,
        );
      }
    }
  }
  return out;
}

/** Crop a region (clamped to bounds) out of uint8 RGB (HWC). */
export function cropRGB(
  src: Uint8Array, srcW: number, srcH: number,
  x: number, y: number, w: number, h: number,
): { rgb: Uint8Array; w: number; h: number } {
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(srcW, Math.round(x + w));
  const y1 = Math.min(srcH, Math.round(y + h));
  const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);
  const out = new Uint8Array(cw * ch * 3);
  for (let row = 0; row < ch; row++) {
    const s = ((y0 + row) * srcW + x0) * 3;
    out.set(src.subarray(s, s + cw * 3), row * cw * 3);
  }
  return { rgb: out, w: cw, h: ch };
}

/** uint8 RGB -> Float32 [0,1] NHWC, the input format of the detection/landmark models. */
export function toFloat01(rgb: Uint8Array): Float32Array {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) out[i] = rgb[i] / 255;
  return out;
}
