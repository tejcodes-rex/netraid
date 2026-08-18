"""Measure same-person vs different-person cosine separation for the float32
MobileFaceNet tflite that ships in the app. Confirms the model discriminates and
picks a sane match threshold."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import preprocess_rgb, tflite_embed, l2norm

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "tf_mbf" / "w600k_mbf_float32.tflite"

faces = np.load(ROOT / "data" / "aligned_faces.npy")      # (N,112,112,3) uint8 RGB
labels = np.load(ROOT / "data" / "aligned_labels.npy")    # (N,)
print(f"faces {faces.shape}, identities {len(np.unique(labels))}")

# Embed a subset for speed.
rng = np.random.RandomState(0)
idx = rng.permutation(len(faces))[:400]
embs, labs = [], []
for i in idx:
    x = preprocess_rgb(faces[i])                 # (1,3,112,112)
    y = tflite_embed(str(MODEL), x)
    embs.append(l2norm(y))
    labs.append(int(labels[i]))
embs = np.stack(embs)
labs = np.array(labs)
print(f"embedded {len(embs)}; embedding norm mean {np.linalg.norm(embs,axis=1).mean():.3f}, "
      f"any NaN: {np.isnan(embs).any()}")

# Pairwise cosine.
S = embs @ embs.T
same, diff = [], []
for i in range(len(embs)):
    for j in range(i + 1, len(embs)):
        (same if labs[i] == labs[j] else diff).append(S[i, j])
same, diff = np.array(same), np.array(diff)
print(f"same-person pairs {len(same)}: mean {same.mean():.3f} p5 {np.percentile(same,5):.3f} "
      f"p50 {np.percentile(same,50):.3f}")
print(f"diff-person pairs {len(diff)}: mean {diff.mean():.3f} p95 {np.percentile(diff,95):.3f} "
      f"p99 {np.percentile(diff,99):.3f} max {diff.max():.3f}")

# Best threshold by accuracy.
best_t, best_acc = 0, 0
for t in np.linspace(0.1, 0.7, 61):
    acc = ((same >= t).sum() + (diff < t).sum()) / (len(same) + len(diff))
    if acc > best_acc:
        best_acc, best_t = acc, t
tar = (same >= best_t).mean()
far = (diff >= best_t).mean()
print(f"best threshold {best_t:.3f} -> acc {best_acc:.3f}, TAR {tar:.3f}, FAR {far:.3f}")
# Threshold at ~0.1% FAR (security-oriented operating point).
for t in np.linspace(0.2, 0.7, 51):
    if (diff >= t).mean() <= 0.001:
        print(f"FAR<=0.1% at threshold {t:.3f}; TAR there {(same>=t).mean():.3f}")
        break
