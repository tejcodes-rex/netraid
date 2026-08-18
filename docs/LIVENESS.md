# NetraID, Offline Liveness / Anti-Spoofing Design

The brief explicitly requires *"basic offline anti-spoofing measures (e.g., requiring the
user to blink, smile, or turn their head) to prevent attendance fraud via photographs or
screens."* NetraID implements that, and binds it to the identity capture.

**What is enforced today**, and this is stated plainly because a liveness claim that
overstates itself is worse than a modest one:

| Barrier | State | Why |
|---|---|---|
| Active challenge (reaction test) | **enforced** | Model-free geometry, so it is unaffected by sensor response or illumination |
| Continuity binding | **enforced** | Ties the liveness proof to the frames the identity is read from |
| Passive CNN, MiniFASNet | **measured, not enforced** | Recorded on every attempt; its operating point was not stable enough on our test handset to arm (see "Calibrating" below) |
| Chroma challenge | **off** | Sound physics, but it cannot work in daylight and did not separate on the target handset |

A gate is armed only where a measurement supports it. `docs/CALIBRATION.md` is the
procedure for arming the passive gate on deployment hardware, and it is roughly a
thirty-minute job.
Both must pass before a match is even attempted.

```
 frame ─► [1: ACTIVE gestures]  AND  [2: PASSIVE CNN]  AND  [3: CHROMA challenge] ─► match
           reaction-timed             MiniFASNet             screen-as-lamp,
           blink / turn / smile       print / replay         reflection response
```

This defeats the three common attacks:
| Attack | Beaten by |
|---|---|
| Printed photo | Active (no blink/3D motion) **+** passive (no skin micro-texture/Moiré) |
| Phone/laptop screen replay | Passive screen-class gate (Moiré/reflection) **+** the reaction window |
| Pre-recorded video of the person | Reaction window + post-prompt neutral requirement + one-shot session |
| A recording of **every** gesture, played on demand | **Chroma challenge** (Barrier 3): the response required did not exist when the recording was made |

> **The honest limit of a gesture challenge.** Blink, smile, turn-left, turn-right
> is four items. An attacker can record all four and play whichever one is asked
> for, and adding a fifth or a tenth gesture does not change that, it only makes
> the recording longer. Randomising the order raises the cost of the attack; it
> does not remove it.
>
> Two things actually remove it, and NetraID does both. **Timing**: a gesture is
> only counted as a reply to the prompt, so it has to arrive in a window the
> attacker cannot predict, from a neutral start (see "Anti-replay hardening").
> **A challenge that cannot be pre-performed at all**: the chroma challenge in
> Barrier 3 asks for a reflection of light this phone picks at verification time.
> There is nothing to record, because the question does not exist until the
> session starts.

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

The active barrier is run as a **reaction test**, not a gesture search. Each step
opens a window the moment its prompt is drawn, and the gesture is only counted as an
answer to that prompt. Four rules, all in `gestureSatisfied()` (`app/src/netraid/liveness.ts`)
and all unit-tested in `__tests__/liveness.test.ts`:

1. **Reaction floor, `MIN_REACTION_MS = 350`.** Simple visual reaction time bottoms
   out near 200 ms and a volitional facial gesture adds more. Anything completing
   sooner was already in flight when the prompt appeared, which is a recording
   running on its own clock. Rejected.
2. **Post-prompt neutral requirement.** The face must be observed in the gesture's
   *neutral* state (eyes open / mouth at rest / head frontal) for 2 processed frames
   **after** the prompt, before the gesture itself can count. This is what forces a
   prompt-caused **transition** instead of a pose that happened to be on screen. A
   replay caught mid-blink or already showing a turned head no longer satisfies the
   step on arrival, which is precisely how a video used to walk through it.
3. **Window ceiling = `challengeTimeoutMs` (6 s).** Late is also not an answer.
4. **One-shot session.** A timeout discards *all* progress and issues a brand-new
   random set. The earlier build kept completed steps and re-rolled only the stuck
   gesture, which let an attacker hold one session open until the gestures their
   recording contained came up. That re-roll is gone.

On top of that the challenge **order and count are randomised per session** (CSPRNG),
and blink is always present because a static photo cannot produce one.

Net effect on a looping replay: every step must land inside a ~5.6 s window, from a
neutral start, at a phase the attacker does not control, and any miss ends the whole
attempt rather than costing one step.

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
- Both ends of that output travel to the verdict, and neither aggregate can be swung by
  a single frame in either direction:
  - **median** P(real) across the verify captures ≥ `passiveThreshold` (0.08). This is
    the statistic the gate reads when armed.
  - **consensus** P(screen-fake), the value at least 2 of the 3 captures reached,
    compared against `screenSpoofMax` (0.85). **Measured and reported, not enforced**, for
    the reason below.
- Gating the *max* of P(real) (the earlier behaviour) meant one fortunate frame out of
  three carried the whole session. The median closes that: a replay has to look real for
  most of the burst, not once.

> **Why the screen-fake class ships unarmed.** MiniFASNet emits a 3-class softmax, so
> P(print) + P(real) + P(screen) = 1. The device calibration measured P(real) on live
> faces (0.12-0.74) and both classes on spoofs, but never P(screen) on a live face. A
> genuine face at the low end of that range leaves up to 0.88 to distribute across the
> two fake classes, so a threshold derived from spoof samples alone can reject a real
> person by construction, and in field testing it did. The reading is surfaced on every
> verification so the operating point can be measured on the target handset and the gate
> then armed, exactly as `chromaMode` is. An uncalibrated gate is not a strong gate; it
> is an arbitrary one, and it can fail open as easily as it fails closed.

- **No barrier reports a pass before every barrier has ruled.** The card shown between
  the gestures and the verdict is deliberately not green and reads "Gestures verified,
  checking": claiming a pass and then rejecting, or restarting, reads as a malfunction to
  an observer even when the rejection is correct.
- **Calibrated on the target device**: measured live faces P(real) 0.12-0.74; a
  laptop-screen replay of the enrolled user ≤ 0.04 with P(screen-fake) up to 0.99.
  See BENCHMARKS.md §3c.
- Converted to TFLite via the same ONNX → onnx2tf path as the recognition model.

> Note: passive RGB anti-spoofing alone does not generalize perfectly across devices and
> novel attacks. That is exactly why NetraID **combines** it with the active challenge. The
> two are uncorrelated failure modes, so the joint false-accept rate is far lower than either
> alone.

---

## Barrier 3, Chroma challenge (screen-as-lamp), `app/src/netraid/chroma.ts`

This is the barrier that answers "what if they record every gesture you have".

The phone turns **its own screen into a coloured lamp** and flashes a sequence drawn
from a CSPRNG the instant verification begins: five ~360 ms slots, lit colours
alternating with white references, no two adjacent alike, always at least two
different colours. A real face reflects that light, so the skin's colour balance
tracks the sequence. A face on another display is emitting its own light, and our
flashes barely perturb it, so its colour balance does not track ours.

**Nothing about the sequence exists before the session starts, so there is nothing to
record.** The challenge space stops being four gestures and becomes every sequence of
colours and durations the phone can emit.

### Capture is held until the sensor recovers
No recognition frame is captured while the flash is up, and none for `chromaSettleMs`
(700 ms) after it ends. The front camera spends the sequence adapting its exposure and
white balance to saturated colour, and the frames immediately after it goes dark still
carry that cast. Embedding one shifts the probe away from the template, and MiniFASNet,
which scores colour and texture, reads the cast as an attack. This is the same failure
the 1.2 s cold-start warm-up exists to prevent, re-created mid-session. The capture-phase
budget is likewise held while the flash owns the screen, so the allowance is not spent
before the first capture is even permitted.

### The measurement
- Per frame, the mean of the face ROI in each channel, converted to **chromaticity**
  (`R/(R+G+B)` and so on). Using the ratio rather than raw brightness makes the
  measurement immune to front-camera auto-exposure and auto-gain, which would
  otherwise swamp the signal.
- The first 150 ms of every slot is discarded, so a frame that arrived while the
  display was still switching is never credited to the previous colour.
- Score = for each emitted colour, the lift of *its* channel during *its* slots over
  the same channel during every other slot, then the **minimum** across colours. The
  minimum, not the mean: a replay whose content happens to be reddish would correlate
  with the red slots by accident, and must not be able to buy the verdict with one
  lucky colour. It has to track every colour we chose.
- Nothing else runs during the flash. The gesture FSM would read distorted landmarks,
  MiniFASNet would judge a colour-cast frame, and a capture emitted here would carry
  the cast into the embedding.

### Status: off by default
`chromaMode` ships as **`off`**. It costs the user roughly two seconds of flashing on every
verification, it cannot work in daylight because a phone screen is not a meaningful share
of the light outdoors, and on the target handset it did not separate a live face from a
replay. A barrier that cannot be armed is a delay rather than a barrier, so the default
does not pay for it. The implementation and its tests are kept for indoor deployments that
calibrate it. The physics is device-independent but the operating
point is not: screen brightness, front-camera auto-white-balance and ambient light all
scale the lift. Until the threshold is read off the target hardware, the score is
measured and displayed on every verification and **rejects no one**. An unusable
reading (face lost mid-flash, too few slots sampled) is never a rejection either:
absence of a measurement is not evidence of an attack.

### Calibrating the chroma gate
1. Build with `chromaMode: 'report'` (the default) and verify **10 live users** on the
   target handset, indoors and in daylight. Record the `chroma` value on each result card.
2. Repeat with the attack you care about: the same faces replayed on a phone and on a
   laptop, held to fill the oval.
3. Set `chromaThreshold` midway between the two clusters on a log scale, then switch
   `chromaMode` to `'enforce'`. The shipped `1.25` is a ratio, not an absolute lift: it asks that the face respond more to our light than its surroundings do. It has not been validated on hardware.
4. Re-run step 1 to confirm the false-reject rate at the chosen point.

---

## Decision logic

```
live =      active_challenges_passed_as_reactions
     AND    median(P_real)        >= PASSIVE_THRESHOLD
     AND    consensus(P_screen,2) <= SCREEN_SPOOF_MAX   (when armed and calibrated)
     AND    chroma_response       >= CHROMA_THRESHOLD   (when armed and usable)
if not live:        -> reject, log spoof attempt (no match attempted)
else:               -> proceed to alignment + embedding + cosine match
```

**Enrollment is gated too, on its own measurement.** The passive floor is evaluated in
`enroll()` as well as `verify()`. Enrollment is the root of trust: a template built from a
photograph or a screen makes every later verification against it meaningless, and no
amount of liveness at verification time can repair it. `SpoofedEnrollmentError` is thrown
before any template is written.

The gate is governed by `enrollSpoofMode`, which ships as **`report`** for the same reason
`chromaMode` does, and for a reason worth stating plainly because it cost us a build.
Enrollment and verification see **different capture distributions**: verification captures
arrive after a gesture sequence and a colour flash, with the sensor long settled, while
enrollment captures arrive as a six-shot burst the moment a face is framed. The
verification threshold was borrowed for enrollment on the assumption that P(real) is
P(real) regardless of path. It is not, and the borrowed number rejected genuine
enrollments on the target handset. `enrollPassiveThreshold` is therefore a separate field,
measured against the enrollment path, and armed only after that measurement. See
`CALIBRATION.md` step 1b.

Thresholds (`PASSIVE_THRESHOLD`, EAR, yaw) are configurable per deployment and calibrated on
a small on-site validation set during pilot, to adapt to camera hardware and lighting.

## Tuning for NHAI field conditions (lighting / demographics)
- **Harsh sunlight / low light**: liveness uses geometry (ratios), which is largely
  illumination-invariant; recognition robustness comes from MobileFaceNet's training +
  optional auto-exposure lock and a low-light capture hint in the UI.
- **Diverse Indian demographics**: the recognition backbone is trained on WebFace600K
  (highly diverse); thresholds are validated on an India-representative pilot set.
