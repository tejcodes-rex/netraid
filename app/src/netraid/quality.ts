// Capture-quality scoring for aligned 112x112 face crops. Recognition accuracy
// is dominated by input quality, so both enrollment and verification rank and
// filter their crops instead of trusting a single frame.

const SIZE = 112;

export interface CropQuality {
  /** Variance of the 3x3 Laplacian over the gray crop. Low = motion blur / defocus. */
  sharpness: number;
  /** Mean gray level 0..255. Flags under/over-exposed captures. */
  brightness: number;
}

/** Absolute floor below which a crop is discarded outright (severe blur). Kept
 * deliberately lenient; ranking (keep the sharpest N) does the real work. */
export const MIN_SHARPNESS = 8;

export function assessCrop(rgb: Uint8Array): CropQuality {
  const n = SIZE * SIZE;
  const gray = new Float32Array(n);
  let sum = 0;
  for (let p = 0, q = 0; p < n; p++, q += 3) {
    const g = 0.299 * rgb[q] + 0.587 * rgb[q + 1] + 0.114 * rgb[q + 2];
    gray[p] = g;
    sum += g;
  }
  const brightness = sum / n;

  // Laplacian variance on the interior (borders skipped).
  let lapSum = 0, lapSq = 0;
  const m = (SIZE - 2) * (SIZE - 2);
  for (let y = 1; y < SIZE - 1; y++) {
    for (let x = 1; x < SIZE - 1; x++) {
      const i = y * SIZE + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - SIZE] - gray[i + SIZE];
      lapSum += lap;
      lapSq += lap * lap;
    }
  }
  const mean = lapSum / m;
  return { sharpness: lapSq / m - mean * mean, brightness };
}
