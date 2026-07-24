// On-device face embedding via react-native-fast-tflite (MobileFaceNet fp16).
//
// The model expects RGB 112x112 normalized to (x-127.5)/127.5 in NHWC layout,
// and outputs a 512-d embedding.
//
// IMPORTANT (device note): the imperative loadTensorflowModel(require(...), delegate)
// call HANGS forever on some mid-range Android devices (seen on Vivo V2246): the
// returned promise never resolves or rejects, so any enroll/verify that awaits it
// stalls at "loading model". The useTensorflowModel() hook loads the exact same
// asset reliably (it is what the detection/landmark/anti-spoof models already use).
// So we load through the hook and stash the handle for the imperative embed() path.

import { useEffect } from 'react';
import {
  useTensorflowModel,
  type TensorflowModel,
} from 'react-native-fast-tflite';
import { l2normalize, mirrorRGB, averageEmbeddings } from './math';
import type { Embedding } from './types';

export const FACE_SIZE = 112;
export const EMBED_DIM = 512;

let model: TensorflowModel | null = null;

/** Store the model handle loaded by the hook so embed() can run it. */
export function setRecognitionModel(m: TensorflowModel | null): void {
  if (m) model = m;
}

/**
 * React hook that loads the recognition model the proven-reliable way and
 * publishes it to this module. Mount it in any screen that enrolls or verifies.
 * Returns the load state ('loading' | 'loaded' | 'error') for UI/diagnostics.
 */
export function useRecognitionModel(): string {
  // float32 weights: the fp16 variant hangs the TFLite interpreter on some
  // mid-range devices (Vivo V2246), and int8 emits NaN on this CPU. float32 is
  // the most compatible format and loads the same way the detection models do.
  const m = useTensorflowModel(
    require('../../assets/models/mobilefacenet_f32.tflite'),
  );
  useEffect(() => {
    if (m.model) setRecognitionModel(m.model);
  }, [m.model]);
  return m.state;
}

/**
 * Wait until the recognition model published by useRecognitionModel() is ready.
 * Kept so existing call sites (init/enroll/verify) work unchanged; it no longer
 * does the imperative load that hangs.
 */
export async function loadRecognitionModel(): Promise<void> {
  if (model) return;
  const t0 = Date.now();
  while (!model && Date.now() - t0 < 20000) {
    await new Promise((r) => setTimeout(r, 100));
  }
  if (!model) {
    throw new Error('Recognition model not ready (hook not mounted or load failed)');
  }
}

/**
 * Run the embedder on an already-aligned RGB crop.
 * @param rgb  Uint8 RGB pixels, length FACE_SIZE*FACE_SIZE*3 (HWC).
 * @returns    L2-normalized 512-d embedding.
 */
export function embed(rgb: Uint8Array): Embedding {
  if (!model) throw new Error('Recognition model not loaded');
  const input = preprocess(rgb);
  const outputs = model.runSync([input]);
  const raw = outputs[0] as Float32Array;
  return l2normalize(raw);
}

/**
 * Flip test-time augmentation: average the embeddings of the crop and its
 * mirror image. Standard ArcFace-family trick; costs one extra inference and
 * measurably tightens same-person similarity under pose/lighting variation.
 */
export function embedTTA(rgb: Uint8Array): Embedding {
  return averageEmbeddings([embed(rgb), embed(mirrorRGB(rgb, FACE_SIZE))]);
}

/** uint8 RGB HWC -> Float32 NHWC normalized to [-1,1]. */
function preprocess(rgb: Uint8Array): Float32Array {
  const n = FACE_SIZE * FACE_SIZE * 3;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rgb[i] - 127.5) / 127.5;
  return out;
}

export function isLoaded(): boolean {
  return model != null;
}
