# NetraID, Trying the App

A five minute walkthrough for evaluating the APK on a real handset. It also explains what the
app is checking at each step, because several of the gates are deliberately strict and it is
useful to know why a rejection happened.

**Requires** an Android 8.0 or newer device. No network connection is needed at any point.

## 1. Install

Download and install `NetraID-universal.apk`. It carries every processor architecture, so it
runs on any phone or emulator without picking a variant.

Android will warn that the app is from an unknown source, because it is not distributed
through the Play Store. Allow the installation.

On first launch, **grant the camera permission**. The app cannot function without it and asks
for nothing else: no storage, no location, no network permission.

## 2. Enrol a person

Open **Enroll Personnel**.

**Hold the phone at about 40 cm, with your face inside the ring.** The ring and the fit
percentage above it tell you how you are doing:

| Ring colour | Meaning |
|---|---|
| White | No face found, or too far away |
| Amber | Face found but not well framed. Move closer or centre yourself |
| Green | Framing is good. Hold still |

**Watch the six dots below the ring.** Each fills in as a shot is accepted, and the counter in
the top right reads `0/6 CAPTURED` through to `6/6`. Hold steady and let all six fill.

**A shot is only accepted when the frame passes every quality gate**, so the dots fill in at
the pace the camera gives usable frames rather than at a fixed rate. A frame is rejected, and
no dot appears, if the face is off-angle, motion-blurred, too dark or too bright, or if the
detector is not confident enough. This is deliberate: an enrolment template is built once and
every future verification is compared against it, so a soft or badly lit shot would degrade
recognition for that person permanently.

**Only after all six dots are filled does the Save button become usable.** Until then it reads
`Capture 6 angles`. Type the personnel ID, for example `NHAI-04821`, and press **Save
Template**.

### What is happening underneath

The six shots are ranked by sharpness, the best are kept, each is converted to a 512-dimension
embedding, embeddings that disagree with the rest are discarded as bad crops, and the survivors
are averaged into one template. **Only that template is stored. No photograph is written to
disk at any point**, and the store is encrypted with SQLCipher using a key held in the Android
Keystore.

### Two rejections you may see, and what they mean

**"This face is already enrolled as ..."** One face maps to one identity. Enrolling the same
face under a second ID is refused, which prevents a worker holding two attendance records.
To re-enrol, use the same ID; that updates the existing template.

**"These frames did not read as a live face."** The passive anti-spoof gate at enrolment. It
ships in reporting mode by default, so it will not normally fire.

## 3. Verify

Open **Verify Attendance**.

**Step one, follow the prompts.** The app asks for two gestures drawn at random from blink,
smile, turn left and turn right. Do each one as soon as it appears on screen.

The prompts are a **reaction test**, not a checklist. Each step requires the app to see a
settled, neutral face after the prompt appears, and then the gesture no sooner than 350 ms and
no later than 4 seconds. That window matters: it is why a recording of somebody performing all
four gestures cannot simply be played back, since the demanded gesture has to arrive inside a
window that opens at a moment chosen after the recording was made.

If a step is missed, the whole attempt restarts with a **new random pair**. Completed steps are
not carried over, which is what stops an attempt being stretched out until a convenient gesture
comes around.

**Step two, hold still.** After the gestures the card reads *Gestures verified, checking*, and
a counter shows `capturing 1/3`, `2/3`, `3/3`. Keep your face square to the camera and hold
steady until all three are taken.

The same quality gates apply here as at enrolment, plus a **frontality gate**: since the last
gesture may have left your head turned, the app waits until you are facing the camera again
before it captures. If it seems to pause, look straight at the lens.

**Step three, the verdict.** The card shows the matched personnel ID and a similarity score.
Anything at 0.38 or above with a clear margin over the next-best person is an accept; on the
test handset a genuine match scores between 0.65 and 0.86.

### What is happening underneath

Three captures are aligned, embedded, and matched against every enrolled template. The verdict
uses the aggregate across all three plus a margin rule, so a single lucky frame cannot carry a
match, and a near-tie between two people is refused rather than guessed.

The whole verdict computes in **371 to 604 ms**. The interaction takes longer than that only
because of the gestures, which are there to establish a live subject.

## 4. Things worth trying

| Test | Expected |
|---|---|
| Verify as the enrolled person | Accepted, score typically 0.65 to 0.86 |
| Verify as somebody who is not enrolled | Rejected, face not enrolled on this device |
| Hold up a printed photograph or a photo on a screen | Rejected: a photograph cannot blink, smile or turn on demand |
| Complete the gestures, then swap in a photo before the captures | Attempt void: the face that proved liveness must be the same continuously tracked face the identity is read from |
| **Turn off mobile data and Wi-Fi entirely, then enrol and verify** | Works identically. Nothing about authentication touches the network |
| Verify in a dim room, then in bright light | Accepted in both. The gates that decide a verdict are geometric, not appearance-based |

### One limitation stated plainly

A **video replay of an enrolled person, held steady**, can pass. The passive CNN that would
catch it is computed and recorded on every attempt but is **not armed**, because its operating
point was not stable enough on our test handset to justify a threshold, and a gate calibrated
on the wrong hardware rejects genuine workers. `docs/CALIBRATION.md` is the procedure for
measuring and arming it on deployment hardware, and it takes about thirty minutes.

We would rather state this than claim a detection rate we have not measured.

## 5. If something does not work

| Symptom | Cause |
|---|---|
| Camera is black | The permission was declined. Grant it in Android settings, under Apps, NetraID, Permissions |
| Dots do not fill during enrolment | Framing or lighting. Get the ring to green, hold still, and avoid strong backlight |
| Prompts keep restarting | A gesture arrived outside its window. Do each one promptly, as soon as the prompt appears |
| It pauses after the gestures | The frontality gate. Look straight at the camera |
| "Face not enrolled on this device" | Templates are per device by design and are never uploaded. Enrol on the handset you are testing |
