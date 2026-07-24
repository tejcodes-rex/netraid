# NetraID, Offline Liveness / Anti-Spoofing Design

The brief explicitly requires *"basic offline anti-spoofing measures (e.g., requiring the
user to blink, smile, or turn their head) to prevent attendance fraud via photographs or
screens."* NetraID goes beyond the minimum with **two independent, fully-offline barriers**.
Both must pass before a match is even attempted.

```
 frame ─► [Barrier 1: ACTIVE challenge FSM]  AND  [Barrier 2: PASSIVE CNN]  ─► allow match
            blink / turn / smile (random)         MiniFASNet print/replay
```

This defeats the three common attacks:
| Attack | Beaten by |
|---|---|
| Printed photo | Active (no blink/3D motion) **+** passive (no skin micro-texture/Moiré) |
| Phone/laptop screen replay | Passive (screen Moiré/reflection) **+** random challenge order |
| Pre-recorded video of the person | Randomized challenge sequence + per-step timeout |

---

## Barrier 1, Active challenge-response (model-free, uses Face Mesh)

The 468-point MediaPipe FaceLandmarker mesh gives us everything; no extra model needed
(0 MB added). A small **finite-state machine** issues a **randomized** sequence of 2-3
challenges with per-step timeouts, e.g. `blink → turn-left → smile`. Randomization +
timing is what defeats replayed video.

### Blink, Eye Aspect Ratio (EAR)
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```
- Open eye ≈ 0.30; a blink drops EAR below **~0.21** then recovers.
- Detect a blink = EAR < 0.21 for ≥ 2 consecutive frames, then return above 0.25.
- Mesh indices (fit the 6-point formula):
  - Left eye:  `[362, 385, 387, 263, 373, 380]`
  - Right eye: `[33, 160, 158, 133, 153, 144]`

### Smile, normalized mouth width
- Use mouth-corner width `||61 - 291||` divided by inter-ocular distance (scale-invariant).
- A smile widens this ratio ~10-20% over the neutral baseline captured at session start.
- (Pure Mouth-Aspect-Ratio detects *open mouth*, not smile, we track corner width.)

### Head turn, geometric yaw
```
yaw_ratio = ||nose(1) - leftCheek(234)|| / ||nose(1) - rightCheek(454)||
```
- ≈ 1.0 frontal; **> 1.3 or < 0.75** ⇒ head turned (~15-20°).
- Pitch (nod) analogously from nose `1`, forehead `10`, chin `152`.
- For higher precision we optionally run `solvePnP` on 6 mesh points → Euler angles.

### Anti-replay hardening
- Challenge **order and count are randomized per session** (CSPRNG).
- Each step has a short timeout (e.g. 4 s); the full sequence must complete in one session.
- Optional: bind the challenge nonce into the attendance record for server-side audit.

---

## Barrier 2, Passive anti-spoof (MiniFASNet)

`minivision Silent-Face-Anti-Spoofing` (Apache-2.0). A MobileFaceNet-derived classifier
trained with **Fourier-spectrum auxiliary supervision**, it learns the frequency artefacts
that prints and screens introduce (Moiré, missing high-frequency skin texture).

- Two scale models exist (2.7× and 4.0× crop expansion, 80×80 input). We ship **one
  float32 model (1.68 MB)**: the int8 variant emits saturated one-hots/NaN (measured on
  desktop LiteRT and the target phone), the same failure family as the int8 recognition
  model. Input convention matters: **BGR channel order, raw 0-255 floats** (Silent-Face
  skips ToTensor scaling); fed [0,1] RGB the model degenerates to a constant verdict.
- Runs on every capture (and a dev-only probe); outputs [print-fake, real, screen-fake].
- **Gate armed and calibrated on the target device**: accept at max P(real) ≥ 0.08 across
  the verify captures. Measured live faces 0.12-0.74; a laptop-screen replay of the
  enrolled user ≤ 0.04 (P(screen-fake) up to 0.99). See BENCHMARKS.md §3c.
- Converted to TFLite via the same ONNX → onnx2tf path as the recognition model.

> Note: passive RGB anti-spoofing alone does not generalize perfectly across devices and
> novel attacks. That is exactly why NetraID **combines** it with the active challenge. The
> two are uncorrelated failure modes, so the joint false-accept rate is far lower than either
> alone.

---

## Decision logic

```
live = active_challenge_passed AND (passive_score >= PASSIVE_THRESHOLD)
if not live:        -> reject, log spoof attempt (no match attempted)
else:               -> proceed to alignment + embedding + cosine match
```

Thresholds (`PASSIVE_THRESHOLD`, EAR, yaw) are configurable per deployment and calibrated on
a small on-site validation set during pilot, to adapt to camera hardware and lighting.

## Tuning for NHAI field conditions (lighting / demographics)
- **Harsh sunlight / low light**: liveness uses geometry (ratios), which is largely
  illumination-invariant; recognition robustness comes from MobileFaceNet's training +
  optional auto-exposure lock and a low-light capture hint in the UI.
- **Diverse Indian demographics**: the recognition backbone is trained on WebFace600K
  (highly diverse); thresholds are validated on an India-representative pilot set.
