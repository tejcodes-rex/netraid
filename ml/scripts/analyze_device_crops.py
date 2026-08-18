"""Read raw 112x112x3 RGB crops dumped from the device, save them as PNG for
visual inspection, and score them with the float32 tflite that ships in the app.
Tells us if on-device alignment is degrading embeddings vs the lab pipeline."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import preprocess_rgb, tflite_embed, l2norm, cosine

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "tf_mbf" / "w600k_mbf_float32.tflite"
DUMP = ROOT / "data" / "device_crops"

def load_rgb(p: Path) -> np.ndarray:
    raw = np.frombuffer(p.read_bytes(), dtype=np.uint8)
    assert raw.size == 112 * 112 * 3, f"{p.name}: {raw.size} bytes (expected {112*112*3})"
    return raw.reshape(112, 112, 3)

embs = {}
for p in sorted(DUMP.glob("*.rgb")):
    img = load_rgb(p)
    Image.fromarray(img, "RGB").save(p.with_suffix(".png"))
    y = tflite_embed(str(MODEL), preprocess_rgb(img))
    embs[p.stem] = l2norm(y)
    print(f"{p.stem:12s} mean={img.mean():.1f} std={img.std():.1f} "
          f"embNaN={np.isnan(y).any()} -> saved {p.with_suffix('.png').name}")

names = list(embs)
print("\ncosine between device crops:")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"  {names[i]} vs {names[j]}: {cosine(embs[names[i]], embs[names[j]]):.3f}")
