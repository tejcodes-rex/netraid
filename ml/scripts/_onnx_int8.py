import sys; from itertools import combinations; from pathlib import Path; import numpy as np, os
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import preprocess_rgb, onnx_embed, l2norm, cosine, ONNX_MBF
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
R=Path(__file__).resolve().parents[1]
q8=str(R/"models/raw/w600k_mbf_int8.onnx")
if not os.path.exists(q8):
    quantize_dynamic(str(ONNX_MBF), q8, weight_type=QuantType.QInt8)
print("int8 ONNX size: %.2f MB"%(os.path.getsize(q8)/1e6),flush=True)
s8=ort.InferenceSession(q8,providers=['CPUExecutionProvider']); n8=s8.get_inputs()[0].name
def e8(x): return s8.run(None,{n8:x.astype(np.float32)})[0]
faces=np.load(R/"data/aligned_faces.npy"); labels=np.load(R/"data/aligned_labels.npy")
idx=np.random.RandomState(1).permutation(len(faces))[:300]; faces,labels=faces[idx],labels[idx]
def es(fn): return np.asarray([l2norm(fn(preprocess_rgb(f))) for f in faces])
def acc(E):
    by={}
    for i,l in enumerate(labels): by.setdefault(int(l),[]).append(i)
    g=np.array([E[a]@E[b] for ids in by.values() for a,b in combinations(ids,2)])
    rng=np.random.RandomState(0); im=[]
    while len(im)<20000:
        a,b=rng.randint(0,len(labels),2)
        if labels[a]!=labels[b]: im.append(float(E[a]@E[b]))
    im=np.array(im); 
    best=0;bt=0
    for t in np.linspace(0,1,201):
        a=(np.sum(g>=t)+np.sum(im<t))/(len(g)+len(im))
        if a>best:best,bt=a,t
    far=np.quantile(im,0.999); tar=float(np.mean(g>=far))
    return best*100,bt,tar*100
ref=es(onnx_embed); E8=es(e8)
fid=float(np.mean([cosine(ref[i],E8[i]) for i in range(len(E8))]))
a0,t0,tar0=acc(ref); a8,t8,tar8=acc(E8)
print(f"FP32 ONNX  acc={a0:.2f}% thr={t0:.3f} TAR@FAR1e-3={tar0:.2f}%",flush=True)
print(f"INT8 ONNX  acc={a8:.2f}% thr={t8:.3f} TAR@FAR1e-3={tar8:.2f}% fidelity={fid:.4f}",flush=True)
