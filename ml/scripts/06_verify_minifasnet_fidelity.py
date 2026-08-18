"""NetraID, step 6: prove the shipped MiniFASNet still computes MiniFASNet.

GROUND TRUTH FOR THE ANTI-SPOOF INPUT CONVENTION. Run this before changing
anything about how frames are fed to the passive model, and after any
re-conversion of the weights.

The app once fed this model raw 0-255 pixels while the converted graph expects
[0,1]. That is not a scaling error, it is a different function: fed 0-255 the
model returns P(real) = 0.995 on RANDOM NOISE, confidently calling static a live
face. Every threshold derived from it was therefore meaningless, a video replay
passed every barrier built on top of it, and the symptom looked like "the
liveness logic is wrong" for as long as nobody checked the model itself.

Expected result:

    TFLite fed [0,1]   vs PyTorch:  max|diff| ~ 3e-8   PASS
    TFLite fed 0-255   vs PyTorch:  max|diff| ~ 0.99   (shown for contrast)

If the [0,1] figure is not near zero, the conversion is broken and nothing
downstream can be trusted. Fix that before touching a single threshold.

Usage:  python ml/scripts/06_verify_minifasnet_fidelity.py
"""
import sys, numpy as np, torch
from collections import OrderedDict
from pathlib import Path
ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from minivision.model_lib.MiniFASNet import MiniFASNetV2
from ai_edge_litert.interpreter import Interpreter

PTH = str(ML / "models" / "raw" / "2.7_80x80_MiniFASNetV2.pth")
TFL = str(ML.parent / "app" / "assets" / "models" / "minifasnet_fp32.tflite")

net = MiniFASNetV2(conv6_kernel=(5, 5), num_classes=3)
state = torch.load(PTH, map_location='cpu')
net.load_state_dict(OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state.items()))
net.eval()

it = Interpreter(model_path=TFL); it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]

rng = np.random.default_rng(0)
imgs = rng.integers(0, 256, size=(6, 80, 80, 3)).astype(np.float32)

def torch_probs(x_nhwc):
    # Reference pipeline: cv2 BGR image -> ToTensor() -> [0,1], CHW
    t = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2).copy()) / 255.0
    with torch.no_grad():
        return torch.softmax(net(t), dim=1).numpy()

def tfl_probs(x_nhwc):
    o = []
    for i in range(len(x_nhwc)):
        it.set_tensor(inp['index'], np.ascontiguousarray(x_nhwc[i:i+1], np.float32)); it.invoke()
        o.append(it.get_tensor(out['index'])[0].copy())
    return np.stack(o)

ref = torch_probs(imgs)                 # PyTorch, [0,1] as the reference code does
t255 = tfl_probs(imgs)                  # TFLite fed 0-255  (what the app does)
t01 = tfl_probs(imgs / 255.0)           # TFLite fed [0,1]

np.set_printoptions(precision=4, suppress=True)
print("PyTorch  ([0,1] input, reference pipeline):")
print(ref)
print("\nTFLite fed 0-255  (what the app feeds):")
print(t255, "\n  max|diff| vs PyTorch =", np.abs(t255 - ref).max())
print("\nTFLite fed [0,1]:")
print(t01, "\n  max|diff| vs PyTorch =", np.abs(t01 - ref).max())
