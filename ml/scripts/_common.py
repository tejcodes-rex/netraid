"""Shared helpers for the NetraID ML pipeline: alignment, preprocessing, TFLite IO."""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]            # ml/
RAW = ROOT / "models" / "raw"
TF_DIR = ROOT / "models" / "tf_mbf"
ONNX_MBF = RAW / "w600k_mbf.onnx"

# ArcFace recognition input contract (insightface w600k_mbf):
#   RGB, 112x112, normalized (x - 127.5) / 127.5  -> range [-1, 1], NCHW for ONNX.
IMG_SIZE = 112


def preprocess_rgb(face_rgb_uint8: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB aligned crop -> (1,3,112,112) float32 normalized (NCHW)."""
    x = face_rgb_uint8.astype(np.float32)
    x = (x - 127.5) / 127.5
    x = np.transpose(x, (2, 0, 1))[None, ...]          # HWC -> 1,C,H,W
    return x


def nchw_to_nhwc(x_nchw: np.ndarray) -> np.ndarray:
    return np.transpose(x_nchw, (0, 2, 3, 1))


def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(l2norm(a) @ l2norm(b))


# ---- ONNX reference embedder -------------------------------------------------
_onnx_sess = None


def onnx_embed(x_nchw: np.ndarray) -> np.ndarray:
    global _onnx_sess
    if _onnx_sess is None:
        import onnxruntime as ort
        _onnx_sess = ort.InferenceSession(str(ONNX_MBF),
                                          providers=["CPUExecutionProvider"])
    name = _onnx_sess.get_inputs()[0].name
    return _onnx_sess.run(None, {name: x_nchw.astype(np.float32)})[0]


# ---- TFLite (LiteRT) embedder ------------------------------------------------
_tflite_cache: dict = {}


def _get_interpreter(model_path: str):
    """Cache one interpreter per model (creating one per call exhausts native
    resources and is slow)."""
    key = str(model_path)
    if key not in _tflite_cache:
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=key)
        it.allocate_tensors()
        _tflite_cache[key] = it
    return _tflite_cache[key]


def tflite_embed(model_path: str, x_nchw: np.ndarray):
    """Run a converted (NHWC) tflite model on an NCHW input; returns float embedding."""
    it = _get_interpreter(model_path)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    x = nchw_to_nhwc(x_nchw).astype(np.float32)
    if inp["dtype"] in (np.int8, np.uint8):
        scale, zp = inp["quantization"]
        lo, hi = (-128, 127) if inp["dtype"] == np.int8 else (0, 255)
        x = np.clip(np.round(x / scale + zp), lo, hi).astype(inp["dtype"])
    it.set_tensor(inp["index"], x)
    it.invoke()
    y = it.get_tensor(out["index"]).astype(np.float32)
    if out["dtype"] in (np.int8, np.uint8):
        scale, zp = out["quantization"]
        y = (y - zp) * scale
    return y


# ---- Face alignment (insightface SCRFD detector + 5-pt norm crop) ------------
_aligner = None


def get_aligner():
    """FaceAnalysis with detection only; downloads buffalo_s det model on first use."""
    global _aligner
    if _aligner is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"])
        app.prepare(ctx_id=-1, det_size=(320, 320))     # ctx_id=-1 -> CPU
        _aligner = app
    return _aligner


def align_face(img_bgr: np.ndarray):
    """Detect the largest face and return an aligned 112x112 RGB uint8 crop, or None."""
    from insightface.utils import face_align
    faces = get_aligner().get(img_bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    crop_bgr = face_align.norm_crop(img_bgr, f.kps, image_size=IMG_SIZE)
    return crop_bgr[:, :, ::-1].copy()                  # BGR -> RGB
