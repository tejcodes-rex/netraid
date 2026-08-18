"""
NetraID, step 5: convert the MiniFASNet passive anti-spoof model (PyTorch .pth)
to ONNX, then to TFLite int8, and stage it for the app.

Uses the open-source MiniFASNet architecture (Apache-2.0) fetched into ml/minivision/.
Output adds a softmax so index 1 is P(live).
"""
from __future__ import annotations
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))


def _ensure_model_lib():
    """Fetch the open-source MiniFASNet architecture (Apache-2.0) if absent."""
    import urllib.request as u
    base = ("https://raw.githubusercontent.com/minivision-ai/"
            "Silent-Face-Anti-Spoofing/master/src/")
    pkg = ML / "minivision"
    (pkg / "model_lib").mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").touch()
    (pkg / "model_lib" / "__init__.py").touch()
    for rel in ("model_lib/MiniFASNet.py", "utility.py"):
        dst = pkg / rel
        if not dst.exists():
            dst.write_bytes(u.urlopen(base + rel, timeout=30).read())


_ensure_model_lib()
from minivision.model_lib.MiniFASNet import MiniFASNetV2  # noqa: E402

RAW = ML / "models" / "raw"
OUT = ML / "models" / "minifas"
OUT.mkdir(parents=True, exist_ok=True)
DEST = ML.parent / "app" / "assets" / "models"

PTH = RAW / "2.7_80x80_MiniFASNetV2.pth"   # 80x80 input, kernel (5,5)
KERNEL = (5, 5)


class SoftmaxWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        return self.sm(self.net(x))


def load_model() -> nn.Module:
    net = MiniFASNetV2(conv6_kernel=KERNEL, num_classes=3)
    state = torch.load(PTH, map_location="cpu")
    # trained with DataParallel -> strip the 'module.' prefix
    clean = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
    net.load_state_dict(clean)
    net.eval()
    return SoftmaxWrap(net).eval()


def main() -> int:
    model = load_model()
    dummy = torch.randn(1, 3, 80, 80)
    onnx_path = OUT / "minifasnet.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["prob"], opset_version=13,
        dynamo=False,
    )
    print(f"ONNX written: {onnx_path} ({onnx_path.stat().st_size/1e6:.2f} MB)")

    # representative calibration: real aligned faces resized to 80x80, normalized to [0,1]
    faces = np.load(ML / "data" / "aligned_faces.npy")[:256]
    import PIL.Image as Image
    calib = np.stack([
        np.asarray(Image.fromarray(f).resize((80, 80)), np.float32) / 255.0
        for f in faces
    ]).astype(np.float32)                                   # (K,80,80,3) NHWC [0,1]
    np.save(ML / "data" / "calib_minifas.npy", calib)

    cmd = [
        sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(OUT),
        "-oiqt", "-qt", "per-channel",
        "-cind", "input", str(ML / "data" / "calib_minifas.npy"),
        "[[[[0.0,0.0,0.0]]]]", "[[[[1.0,1.0,1.0]]]]",
    ]
    if subprocess.run(cmd, cwd=str(ML)).returncode != 0:
        return 1

    int8 = OUT / "minifasnet_integer_quant.tflite"
    if int8.exists():
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / "minifasnet_int8.tflite").write_bytes(int8.read_bytes())
        print(f"Staged minifasnet_int8.tflite ({int8.stat().st_size/1e6:.2f} MB)")
    print("\nTFLite variants:")
    for f in sorted(OUT.glob("*.tflite")):
        print(f"  {f.name:45s} {f.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
