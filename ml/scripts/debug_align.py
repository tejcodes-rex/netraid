"""Reproduce the device alignment off-device: load the dumped 256 ROI + its 5
landmarks, run BOTH OpenCV's correct similarity warp and a port of the app's
umeyama+warpAffine, and compare. Saves PNGs and scores against the enrolled crop."""
import json, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import preprocess_rgb, tflite_embed, l2norm, cosine

D = Path(__file__).resolve().parents[1] / "data" / "device_crops"
MODEL = Path(__file__).resolve().parents[1] / "models" / "tf_mbf" / "w600k_mbf_float32.tflite"

REF5 = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                 [41.5493,92.3655],[70.7299,92.2041]], np.float32)

roi = np.frombuffer((D/"roi.rgb").read_bytes(), np.uint8).reshape(256,256,3)
j = json.loads((D/"roi_m5.json").read_text())
m5 = np.array(j["m5"], np.float32)
print("m5:\n", m5)
cv2.imwrite(str(D/"roi.png"), roi[:,:,::-1])

# --- 1) OpenCV correct similarity (the reference) ---
M_cv, _ = cv2.estimateAffinePartial2D(m5, REF5, method=cv2.LMEDS)
aligned_cv = cv2.warpAffine(roi, M_cv, (112,112))
cv2.imwrite(str(D/"aligned_cv.png"), aligned_cv[:,:,::-1])
print("cv2 M:\n", M_cv)

# --- 2) Port of the app's umeyama (align.ts) ---
def svd2x2(c00,c01,c10,c11):
    E=(c00+c11)/2; F=(c00-c11)/2; G=(c10+c01)/2; H=(c10-c01)/2
    Q=np.hypot(E,H); R=np.hypot(F,G)
    s0=Q+R; s1=abs(Q-R)
    a1=np.arctan2(G,F); a2=np.arctan2(H,E)
    th=(a2-a1)/2; ph=(a2+a1)/2
    U=[np.cos(ph),-np.sin(ph),np.sin(ph),np.cos(ph)]
    V=[np.cos(th),-np.sin(th),np.sin(th),np.cos(th)]
    return U,[s0,s1],V
def mul2(a,b): return [a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3]]
def mulDiag2(V,d0,d1): return [d0*V[0],d0*V[1],d1*V[2],d1*V[3]]
def app_umeyama(src,dst):
    n=len(src); meanS=src.mean(0); meanD=dst.mean(0)
    varS=0.0; c00=c01=c10=c11=0.0
    for i in range(n):
        sx,sy=src[i]-meanS; dx,dy=dst[i]-meanD
        varS+=sx*sx+sy*sy; c00+=dx*sx; c01+=dx*sy; c10+=dy*sx; c11+=dy*sy
    varS/=n; c00/=n; c01/=n; c10/=n; c11/=n
    U,S,V=svd2x2(c00,c01,c10,c11)
    det=c00*c11-c01*c10; d=-1 if det<0 else 1
    R=mul2(U,mulDiag2(V,1,d))
    scale=(S[0]+d*S[1])/(varS+1e-12)
    a=scale*R[0]; b=scale*R[1]; c=scale*R[2]; dd=scale*R[3]
    tx=meanD[0]-(a*meanS[0]+b*meanS[1]); ty=meanD[1]-(c*meanS[0]+dd*meanS[1])
    return np.array([[a,b,tx],[c,dd,ty]],np.float32)
M_app=app_umeyama(m5,REF5)
aligned_app=cv2.warpAffine(roi,M_app,(112,112))
cv2.imwrite(str(D/"aligned_app.png"),aligned_app[:,:,::-1])
print("app M:\n",M_app)

# --- score each against the enrolled crop ---
enroll=np.frombuffer((D/"enroll.rgb").read_bytes(),np.uint8).reshape(112,112,3)
e_enroll=l2norm(tflite_embed(str(MODEL),preprocess_rgb(enroll)))
for name,img in [("cv",aligned_cv),("app",aligned_app)]:
    e=l2norm(tflite_embed(str(MODEL),preprocess_rgb(img)))
    print(f"{name:4s} aligned vs enroll cosine = {cosine(e,e_enroll):.3f}")
