"""
NetraID, Step 4: stage the on-device models into the React Native app bundle.

Copies/renames the real converted artifacts into app/assets/models/ with the
names the app's require() calls expect, and extracts the FaceLandmarker .tflite
from the MediaPipe .task bundle (a zip).
"""
from __future__ import annotations
import shutil
import zipfile
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
RAW = ML / "models" / "raw"
TF = ML / "models" / "tf_mbf"
DEST = ML.parent / "app" / "assets" / "models"
DEST.mkdir(parents=True, exist_ok=True)

COPIES = [
    (TF / "w600k_mbf_integer_quant.tflite", "mobilefacenet_int8.tflite"),
    (TF / "w600k_mbf_float16.tflite", "mobilefacenet_fp16.tflite"), # delegate-friendly fallback
    (RAW / "blaze_face_short_range.tflite", "blazeface_short_range.tflite"),
]


def main() -> None:
    staged = []
    for src, name in COPIES:
        if src.exists():
            shutil.copy2(src, DEST / name)
            staged.append((name, (DEST / name).stat().st_size / 1e6))

    # FaceLandmarker: the .task is a zip; pull out the landmark .tflite.
    task = RAW / "face_landmarker.task"
    if task.exists():
        with zipfile.ZipFile(task) as z:
            tfl = [n for n in z.namelist() if n.endswith(".tflite") and "landmark" in n.lower()]
            pick = tfl[0] if tfl else next((n for n in z.namelist() if n.endswith(".tflite")), None)
            if pick:
                with z.open(pick) as f, open(DEST / "face_landmarker.tflite", "wb") as out:
                    out.write(f.read())
                staged.append(("face_landmarker.tflite", (DEST / "face_landmarker.tflite").stat().st_size / 1e6))

    total = 0.0
    print("Staged into app/assets/models/:")
    for name, mb in staged:
        total += mb
        print(f"  {name:34s} {mb:6.2f} MB")
    print(f"  {'TOTAL':34s} {total:6.2f} MB")
    print("\nMiniFASNet passive anti-spoof is staged separately by "
          "scripts/05_convert_minifasnet.py -> minifasnet_int8.tflite.")


if __name__ == "__main__":
    main()
