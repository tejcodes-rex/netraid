"""
NetraID, Step 1: download all open-source pretrained models.

Every model here is permissively licensed (MIT / Apache-2.0), satisfying the
hackathon's "open-source only, no extra licenses" constraint.

  - MobileFaceNet recognition  : insightface buffalo_s  -> w600k_mbf.onnx   (MIT)
    (the pack also ships det_500m.onnx, a tiny SCRFD face detector)
  - Face detection (mobile)     : MediaPipe BlazeFace short-range            (Apache-2.0)
  - Face landmarks (468/478)    : MediaPipe FaceLandmarker (.task bundle)    (Apache-2.0)
  - Passive anti-spoofing       : MiniFASNet x2 (Silent-Face-Anti-Spoofing)  (Apache-2.0)

Run:  python scripts/01_download_models.py
Output: ml/models/raw/<files>  +  ml/models/raw/manifest.json
"""
from __future__ import annotations
import json
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]          # ml/
RAW = ROOT / "models" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# (name, url, license, kind). kind drives post-processing (zip extraction etc.)
DOWNLOADS = [
    (
        "buffalo_s.zip",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "MIT", "zip",
    ),
    (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "Apache-2.0", "file",
    ),
    (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        "Apache-2.0", "file",
    ),
    (
        "2.7_80x80_MiniFASNetV2.pth",
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/"
        "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        "Apache-2.0", "file",
    ),
    (
        "4_0_0_80x80_MiniFASNetV1SE.pth",
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/"
        "resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
        "Apache-2.0", "file",
    ),
]


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size/1e6:.2f} MB)")
        return True
    try:
        with requests.get(url, stream=True, timeout=60,
                          headers={"User-Agent": "NetraID/1.0"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"  {dest.name}", leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"  [ok]   {dest.name} ({dest.stat().st_size/1e6:.2f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    print(f"Downloading models into {RAW}\n")
    manifest = []
    ok_count = 0
    for name, url, lic, kind in DOWNLOADS:
        dest = RAW / name
        ok = download(url, dest)
        entry = {"name": name, "license": lic, "url": url, "ok": ok}
        if ok:
            ok_count += 1
            entry["size_bytes"] = dest.stat().st_size
            if kind == "zip":
                with zipfile.ZipFile(dest) as z:
                    z.extractall(RAW)
                    entry["extracted"] = z.namelist()
                print(f"  [unzip] {name} -> {entry['extracted']}")
        manifest.append(entry)

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {ok_count}/{len(DOWNLOADS)} downloaded. Manifest -> {RAW/'manifest.json'}")
    # Report key file: the MobileFaceNet recognition model
    mbf = RAW / "buffalo_s" / "w600k_mbf.onnx"
    if mbf.exists():
        print(f"\nMobileFaceNet (FP32 ONNX): {mbf} = {mbf.stat().st_size/1e6:.2f} MB")
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(main())
