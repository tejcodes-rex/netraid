"""
NetraID, Step 2: build a REAL calibration set from aligned faces and convert
MobileFaceNet (ONNX) -> TFLite int8 with proper per-tensor quantization.

Why real calibration matters: int8 needs representative activations. Random data
collapses accuracy (saturated tensors). We align real LFW faces with the same
ArcFace 5-point pipeline used at inference, so the int8 ranges are correct.

Outputs:
  ml/data/aligned_faces.npy        (N,112,112,3) uint8 RGB   - reused by benchmark
  ml/data/aligned_labels.npy       (N,) identity indices
  ml/data/calib_nhwc.npy           (K,112,112,3) float32 normalized
  ml/models/tf_mbf/*.tflite        float32 / float16 / int8 variants
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, ONNX_MBF, TF_DIR, preprocess_rgb, align_face  # noqa: E402

DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
N_CALIB = 256


def load_real_faces(max_imgs: int = 1200):
    """Fetch LFW (real, diverse faces) and align each to 112x112 RGB."""
    from sklearn.datasets import fetch_lfw_people
    print("Fetching LFW (first run downloads ~200MB)...")
    lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0,
                           slice_=(slice(0, 250), slice(0, 250)), funneled=True)
    imgs, labels = lfw.images, lfw.target
    if imgs.max() <= 1.0 + 1e-6:
        imgs = imgs * 255.0
    imgs = imgs.astype(np.uint8)                       # (N,250,250,3) RGB

    aligned, alabels, fails = [], [], 0
    for i in range(min(max_imgs, len(imgs))):
        bgr = imgs[i][:, :, ::-1].copy()              # RGB -> BGR for insightface
        crop_rgb = align_face(bgr)
        if crop_rgb is None:
            fails += 1
            continue
        aligned.append(crop_rgb)
        alabels.append(int(labels[i]))
        if (i + 1) % 200 == 0:
            print(f"  aligned {len(aligned)}/{i+1} (det fails: {fails})")
    print(f"Aligned {len(aligned)} faces, {fails} detection failures.")
    return np.asarray(aligned, np.uint8), np.asarray(alabels, np.int64)


def main() -> int:
    faces, labels = load_real_faces()
    if len(faces) < 50:
        print("Too few aligned faces; aborting.")
        return 1
    np.save(DATA / "aligned_faces.npy", faces)
    np.save(DATA / "aligned_labels.npy", labels)

    # Representative dataset: normalized NHWC float32
    idx = np.random.RandomState(0).permutation(len(faces))[:N_CALIB]
    calib = np.stack([preprocess_rgb(faces[i])[0] for i in idx])     # (K,3,112,112)
    calib_nhwc = np.transpose(calib, (0, 2, 3, 1)).astype(np.float32)  # (K,112,112,3)
    np.save(DATA / "calib_nhwc.npy", calib_nhwc)
    print(f"Calibration set: {calib_nhwc.shape} -> {DATA/'calib_nhwc.npy'}")

    # Convert with onnx2tf using real calibration (already normalized -> mean 0, std 1)
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(ONNX_MBF),
        "-o", str(TF_DIR),
        "-oiqt", "-qt", "per-channel",  # per-channel >> per-tensor for conv accuracy
        "-cind", "input.1", str(DATA / "calib_nhwc.npy"),
        "[[[[0.0,0.0,0.0]]]]", "[[[[1.0,1.0,1.0]]]]",
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print("onnx2tf failed.")
        return r.returncode

    print("\nConverted TFLite models:")
    for f in sorted(TF_DIR.glob("*.tflite")):
        print(f"  {f.name:55s} {f.stat().st_size/1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
