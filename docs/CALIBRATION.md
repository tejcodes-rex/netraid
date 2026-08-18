# NetraID, Calibrating the Liveness Gates on Your Own Hardware

Three gates ship **measured but not armed**:

| Gate | Config field | What it judges |
|---|---|---|
| Passive screen-replay | `screenSpoofMode` | MiniFASNet's P(screen) at verification |
| Chroma challenge | `chromaMode` | Whether the face reflected our screen light |
| Enrollment anti-spoof | `enrollSpoofMode` | MiniFASNet's P(real) across the enrollment burst |

Each is computed and recorded on every attempt, and none of them rejects anyone until an
operator has read its operating point off the handset the deployment actually uses.

This is deliberate. A threshold carried over from another device is not a security
control, it is a guess, and a guess fails open as readily as it fails closed. The gate
that rejects a genuine worker at a toll plaza and the gate that admits a photograph are
the same miscalibration seen from two sides.

The enrollment gate is listed separately from the verification one for a reason that cost
us a build: the two see **different capture distributions**. Verification captures arrive
after a gesture sequence and a colour flash, with the sensor long settled. Enrollment
captures arrive as a six-shot burst the moment a face is framed. Borrowing the
verification threshold for enrollment rejected genuine enrollments on the target device.
Measure each gate against its own path.

The procedure below takes about half an hour.

---

## 0. How to read the numbers off the handset

Two ways, and for a real calibration run you want the second.

**On screen, development builds only.** The verification screen prints each layer's verdict
for the attempt just decided, which is enough to sanity-check a single attempt while
developing. It is compiled out of release builds: a field user has no use for model
probabilities, and an attacker who can read the exact margin by which they failed has been
handed a tuning signal.

**Over USB, which is what the procedure below assumes.** Every attempt, at both
enrollment and verification, emits one machine-readable line in release builds:

```
adb logcat -s ReactNativeJS:V | grep NETRAID_CALIB
```

```
NETRAID_CALIB {"stage":"verify","outcome":"accept","reason":null,"score":0.5123,
  "elapsedMs":402,"real":[0.412,0.377,0.455],"screen":[0.081,0.104,0.066],
  "sharp":[46,51,44],"chroma":0.0193,"challenges":["blink","turn-left"]}
NETRAID_CALIB {"stage":"enroll","outcome":"accept","mode":"report",
  "passiveMedian":0.3140,"real":[0.29,0.31,0.34,0.28,0.36,0.31],
  "screen":[0.11,0.09,0.13,0.10,0.08,0.12],"sharp":[52,49,55,47,51,50],"shots":6}
```

A calibration run is dozens of numbers per subject. Transcribing them by hand off a phone
screen is how a transcription error ends up baked into a security threshold, so capture
the log and compute from the file. The payload carries model scores and timings only:
never an embedding, an image, or a person id.

## 0b. What you are measuring

Each verification records, per capture:

| Column | Meaning |
|---|---|
| `real` | MiniFASNet P(real) for that capture. Higher is more live. |
| `screen` | MiniFASNet P(screen-replay) for that capture. Higher means "this is a display". |
| `sharp` | Laplacian-variance sharpness of the crop, for context only. |
| `chroma` | Chromaticity lift measured during the flash. Higher means the face reflected our light. `n/a` means the measurement was not usable and, correctly, is not evidence of anything. |

MiniFASNet emits a **three-class softmax**: P(print) + P(real) + P(screen) = 1. This is why
`screen` cannot be thresholded from spoof samples alone. A genuine face reading `real 0.12`
has 0.88 left to distribute across the two fake classes and can legitimately read high on
`screen`. You need the live distribution before you can draw a line.

---

## 1. Collect the live distribution

Verify **at least 10 different people**, once each, on the target handset. Vary what the
field actually varies:

- indoors under artificial light
- outdoors in daylight, and with the sun behind the subject
- a dim or shaded location
- at least two subjects wearing spectacles
- at least one subject wearing a hard hat or cap

Record `real` (take the median of the three frames), `screen` (take the value at least two
of the three frames reached), and `chroma` for each person.

You are looking for the **worst** live reading in each column: the lowest `real`, the
highest `screen`, the lowest `chroma`. Those three numbers define the edge of the genuine
population on your hardware.

## 1b. Collect the enrollment distribution

The same 10 people, enrolled once each on the target handset, under the same spread of
lighting. Take `passiveMedian` from each `"stage":"enroll"` line.

This is a **separate** distribution from step 1 and it is normally lower, because the
enrollment burst starts as soon as a face is framed rather than after a gesture sequence.
Do not reuse the verification numbers here.

Then enroll the same faces from a printed photograph and from a phone screen, and take
`passiveMedian` for those. Enrollment is the root of trust: a template built from a
photograph makes every later verification against it meaningless, and no amount of
liveness at verification can repair it. This gate is the only thing standing there.

## 2. Collect the attack distribution

Repeat with the attacks that matter, using the same faces:

| Attack | How |
|---|---|
| Printed photograph | A colour print, held to fill the oval |
| Phone-screen photo | The subject's photo displayed on a second phone |
| Laptop-screen photo | The same, on a laptop display |
| Video replay | A recording of the subject performing blink, smile and both head turns, played back on demand |

Record the same three columns. You are looking for the **best** attack reading: the highest
`real`, the lowest `screen`, the highest `chroma`.

> The video replay is the important one. If it is being stopped only by the gesture
> barrier, the chroma gate is what closes the remaining gap, because the colour sequence
> is drawn from a CSPRNG when the session starts and cannot have been recorded.

## 3. Set the thresholds

You now have two clusters per column. Set each threshold **between** them, and prefer the
side that protects the genuine population, because a false reject at a toll plaza is a
worker who cannot mark attendance.

| Config field | Set to |
|---|---|
| `passiveThreshold` | Below the lowest live median `real`, above the highest attack `real`. Shipped default 0.08 sits between 0.12 and 0.04 as measured on a Vivo V2246. |
| `screenSpoofMax` | Above the highest live consensus `screen`, below the lowest attack `screen`. |
| `chromaThreshold` | Between the clusters on a **log scale**, since the lift scales with screen brightness and ambient light. The shipped 0.012 is a placeholder, not a measurement. |
| `enrollPassiveThreshold` | Below the lowest live enrollment `passiveMedian` from step 1b, above the highest photo/screen enrollment reading. The shipped 0.08 is inherited from the verification gate and is **not** a measurement of this path. |

If a column's two clusters **overlap**, that gate cannot be armed on this hardware. Leave
it in `report` and rely on the other two barriers. Say so in the deployment record rather
than picking a threshold inside the overlap, which would reject genuine users and admit
attacks at the same time.

## 4. Arm the gates

In `app/src/netraid/types.ts → DEFAULT_CONFIG`:

```ts
screenSpoofMode: 'enforce',
screenSpoofMax: <your number>,
chromaMode: 'enforce',
chromaThreshold: <your number>,
enrollSpoofMode: 'enforce',
enrollPassiveThreshold: <your number>,
```

Rebuild and reinstall.

## 5. Confirm the false-reject rate

Re-run steps 1 and 1b with the gates armed. Every one of the 10 live subjects must enroll
successfully, and must then pass verification on the first or second attempt. If any subject fails repeatedly, a threshold is inside the live
distribution: widen it and repeat. Do not ship a configuration you have not re-measured
after arming.

Record the final numbers, the handset model and the date in the deployment record. A
threshold without a measurement behind it cannot be audited, and at a later inquiry the
question will be why a specific worker was rejected on a specific day.

---

## Recalibrate when

- the handset model changes (different sensor, different auto-white-balance behaviour)
- screen brightness policy changes, which scales the chroma lift directly
- a new attack class appears in the field

The recognition threshold (`matchThreshold`, `matchMargin`) is calibrated separately
against an India-representative validation set; see `BENCHMARKS.md`.
