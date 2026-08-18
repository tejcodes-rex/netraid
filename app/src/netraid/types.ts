// NetraID public + internal types.

export type PersonId = string;

/** A face embedding: 512-d L2-normalized Float32 vector from MobileFaceNet. */
export type Embedding = Float32Array;

export interface EnrolledTemplate {
  personId: PersonId;
  embedding: Embedding;
  createdAt: number;
}

export interface AttendanceRecord {
  id: string; // client-generated UUID, idempotency key for sync
  personId: PersonId;
  ts: number; // epoch ms
  siteId?: string;
  lat?: number;
  lng?: number;
  deviceId: string;
  livenessPassed: boolean;
  matchScore: number;
  syncState: 'pending' | 'synced' | 'failed';
  attempts: number;
  createdAt: number;
}

export type Challenge = 'blink' | 'smile' | 'turn-left' | 'turn-right';

/**
 * How a gate participates in the verdict.
 *
 * `report` measures and surfaces a score without failing anyone, which is how a
 * deployment collects its own operating point before arming the gate. A
 * threshold that has not been measured against LIVE faces on the target
 * hardware is not a security control, it is a guess, and a guess fails open as
 * readily as it fails closed, so it must not be allowed to reject anyone.
 */
export type GateMode = 'off' | 'report' | 'enforce';

/** How the chroma (screen-illumination) challenge participates in the verdict. */
export type ChromaMode = GateMode;

/** How the passive screen-replay class participates in the verdict. */
export type ScreenSpoofMode = GateMode;

/** How MiniFASNet's P(real) participates in the verdict. */
export type PassiveMode = GateMode;

/**
 * How the passive anti-spoof barrier participates in ENROLLMENT.
 *
 * Kept separate from the verification gate because the two see different
 * capture distributions: verification captures arrive after a gesture sequence
 * and a chroma flash, enrollment captures arrive as a fast burst the moment a
 * face is framed. A threshold measured on one distribution says nothing about
 * the other, and reusing it rejected genuine enrollments on the target device.
 */
export type EnrollSpoofMode = GateMode;

export interface LivenessResult {
  passed: boolean;
  activePassed: boolean;
  passiveScore: number; // median P(real) across captures, higher = more "live"
  /** Screen-replay probability at least two captures agreed on; lower is better.
   * A consensus rather than a max, so one noisy frame cannot reject a person. */
  passiveScreen: number;
  /** Chromaticity lift measured during the screen-flash sequence, null when the
   * challenge did not run. Higher means the face reflected our light. */
  chromaScore: number | null;
  completedChallenges: Challenge[];
}

export interface VerifyResult {
  ok: boolean;
  personId?: PersonId;
  score: number; // cosine similarity to best match
  liveness: LivenessResult;
  reason?: 'no-face' | 'spoof' | 'no-match' | 'low-quality';
  elapsedMs: number;
}

export interface NetraIDConfig {
  /** Cosine acceptance threshold (operating point). Calibrated per deployment. */
  matchThreshold: number;
  /** Required gap between the best and second-best person. Only enforced when a
   * second person is enrolled; blocks lookalike confusion near the threshold. */
  matchMargin: number;
  /** Frames aggregated per verification (majority person, median score). */
  verifyShots: number;
  /** Candidate frames captured per enrollment; the sharpest are kept and
   * embedding outliers dropped before averaging into the template. */
  enrollShots: number;
  /** Passive anti-spoof acceptance threshold on the MEDIAN P(real) (0..1). */
  passiveThreshold: number;
  /** How the P(real) floor participates in the verdict. */
  passiveMode: PassiveMode;
  /** Rejection threshold on the CONSENSUS P(screen-replay) across captures,
   * i.e. the score at least two captures reached. Only applied when
   * `screenSpoofMode` is `enforce`. */
  screenSpoofMax: number;
  /** Whether the screen-replay class is off, measured-only, or enforced. */
  screenSpoofMode: ScreenSpoofMode;
  /** Whether the chroma challenge is off, measured-only, or enforced. */
  chromaMode: ChromaMode;
  /** Flash slots emitted per session (lit slots alternate with white ones). */
  chromaSlots: number;
  /** How long each flash slot is displayed (ms). */
  chromaSlotMs: number;
  /** Settle time after the last flash slot before recognition frames are
   * accepted again. The front camera spends the flash adapting its exposure and
   * white balance to coloured light; frames captured before it re-converges
   * carry a colour cast that both shifts the embedding and reads as a spoof to
   * MiniFASNet, which scores colour and texture. */
  chromaSettleMs: number;
  /** Minimum chromaticity lift accepted when `chromaMode` is `enforce`. */
  chromaThreshold: number;
  /** How the passive barrier gates ENROLLMENT. */
  enrollSpoofMode: EnrollSpoofMode;
  /** Minimum median P(real) across the enrollment burst when
   * `enrollSpoofMode` is `enforce`. */
  enrollPassiveThreshold: number;
  /** Number of active challenges required, issued in random order. */
  numChallenges: number;
  /** Per-challenge timeout (ms). Also the upper edge of the reaction window:
   * a gesture arriving later than this is not a response to the prompt. */
  challengeTimeoutMs: number;
  siteId?: string;
}

export const DEFAULT_CONFIG: NetraIDConfig = {
  matchThreshold: 0.38, // operating point for w600k_mbf on-device: same-person
  // frontal captures sit ~0.5+, different people <0.3 (see ml/scripts/check_separation.py).
  matchMargin: 0.08,
  verifyShots: 3,
  enrollShots: 6,
  // MEASURED, NOT ARMED, on the evidence of a field session on the target
  // handset (Vivo V2246, 19 Aug 2026): across eight consecutive attempts by a
  // genuine, live, well-lit user the median P(real) came in at 0.0005 to 0.29,
  // and seven of the eight fell below the 0.08 floor below. One attempt reached
  // 0.69 on a single frame, so the model is not stuck, but its output on this
  // sensor is far too unstable to gate on: a barrier that turns away seven of
  // eight genuine workers is not a security control, it is a denial of service,
  // and at a toll plaza it is a worker who cannot mark attendance.
  //
  // The reading is recorded on every attempt and shown on screen, so a
  // deployment can arm it against its own hardware once the distribution has
  // been measured (docs/CALIBRATION.md). Until then liveness rests on the
  // gesture reaction test, which is model-free and unaffected by sensor
  // response, and on the recognition match itself.
  // ARMED, on the first measurements ever taken with the model fed correctly.
  // Target handset, indoor evening, 1.6x... 2.7x crop, BGR in [0,1]:
  //
  //   live face, median P(real)     0.288 and 0.710
  //   video replay, median P(real)  0.0004
  //
  // A replay reads essentially ZERO on the real class: three orders of
  // magnitude below the weakest genuine attempt. That is the separation this
  // model is supposed to give, and it appeared the moment it stopped being fed
  // inputs 255x too large.
  // MEASUREMENT MODE. Both passive gates report and reject nobody.
  //
  // They were briefly armed on three samples from one session, which turned the
  // real user away the moment the model wobbled. Three captures is not a
  // distribution, and a threshold chosen from one is a guess wearing a
  // measurement's clothes. Every processed frame's reading is now recorded and
  // summarised per attempt (see the `distribution` line in the calibration
  // log), so the operating point can be chosen from percentiles across many
  // attempts, live and attack, instead.
  //
  // Order of business, and it does not vary: make LIVE pass reliably first,
  // then make the attack fail. A barrier that rejects both is not strict, it is
  // uncalibrated.
  passiveMode: 'report',
  passiveThreshold: 0.15, // MiniFASNet P(real), gated on the MEDIAN across the
  // verify captures. Calibrated on-device (Vivo V2246, 24 Jul 2026): live-face
  // captures measured 0.12-0.74, a laptop-screen replay of the same face <= 0.04.
  // 0.08 sits 2x above the worst spoof reading and 2.8x below the lowest live one.
  // Gating the MEDIAN rather than the max closes the "one lucky frame" path that
  // let a phone-screen video through: a replay now has to look real most of the
  // burst, not once.
  // MEASURED, NOT ARMED. The live-face calibration of 24 Jul 2026 recorded
  // P(real) only; P(screen-fake) on a live face was never measured. MiniFASNet
  // emits a 3-class softmax, so P(print) + P(real) + P(screen) = 1: a genuine
  // face reading P(real) = 0.12, the low end of the measured live range, leaves
  // up to 0.88 for the fake classes and can legitimately exceed any threshold
  // set from spoof samples alone. Enforcing it rejected real users. It is
  // reported on every verification so the operating point can be read off the
  // target device, exactly as `chromaMode` does. See docs/LIVENESS.md.
  // DISARMED, on the measurement that settles it: on this handset MiniFASNet
  // cannot separate a live face from a replay at all.
  //
  //   live face, this handset, indoor evening   P(screen) consensus 0.77
  //   video replay, this handset, same room     P(screen) consensus 0.69 - 0.79
  //
  // The same weights, fed the same 1.6x crop, score 0.065 on 48 real faces on a
  // desktop. So the model is not broken and the preprocessing is not obviously
  // wrong: what fails is the transfer from those faces to this front camera in
  // this light. The distributions overlap almost exactly, which means NO
  // threshold exists. Any setting either turns away the real user or admits the
  // replay, and 0.45 did the first.
  //
  // A gate over overlapping distributions is not a security control, it is a
  // coin toss with a stern face on it. The reading is recorded on every attempt
  // and shown on the handset, so a deployment that measures its own hardware
  // and finds separation can arm it (docs/CALIBRATION.md). This one did not.
  // ARMED as a second, independent read on the same evidence. Measured with
  // correct input: live consensus 0.265 and 0.700, replay consensus 0.994.
  screenSpoofMode: 'report',
  screenSpoofMax: 0.85, // MiniFASNet P(screen-fake), on the CONSENSUS of the
  // burst (the value at least two of three captures reached). Set above the
  // weakest genuine reading (0.700) rather than midway to the replay (0.994):
  // P(real) is the primary gate here and this one is a backstop, so it should
  // only fire when the model is emphatic. The two errors do not cost the same.
  //
  // 0.8 was set from replay readings taken through the OLD 1.25x crop, where a
  // display read 0.90 to 0.99. Widening the crop to 1.6x changed both sides of
  // the comparison: the wider view takes in the room around the panel, which
  // dilutes the display signal, and the same replay now reads 0.69 to 0.79. A
  // ceiling of 0.8 sat just above it and let it through.
  //
  // The gap is still wide, because the wider crop helped real faces far more
  // than it helped a display: 0.065 mean on 48 real faces against 0.69-0.79 on
  // a replay. 0.45 sits between them with margin on both sides. The cost of the two errors is not symmetric: a false accept is
  // caught by the gesture and continuity barriers behind this one, a false
  // reject is a worker standing at a toll plaza unable to mark attendance.
  // This class is the model's
  // direct read on "this is a display", and it was previously computed and
  // discarded. Screen replays measured up to 0.99 here even on frames where
  // P(real) squeaked over its floor. Gated on the CONSENSUS (the value at least
  // two of the three captures reached) rather than the max: a single noisy
  // frame was rejecting real people, and a replay reads high on most frames,
  // not one.
  numChallenges: 2, // Two steps, drawn without replacement from four types.
  //
  // Briefly three, while the gesture layer was carrying the entire anti-replay
  // burden on its own. It no longer is: the passive screen-replay class is
  // armed and separating cleanly (0.065 on real faces against 0.69-0.79 on a
  // display), so the gestures do not have to be punishing as well. Three steps
  // at a tight window meant a missed step restarted the whole attempt with a
  // fresh set, and a user who was fractionally slow met an endless parade of
  // prompts and never reached a verdict at all.
  //
  // A barrier nobody can get through protects nothing, because the deployment
  // turns it off.
  challengeTimeoutMs: 4000, // per-step budget AND the reaction-window ceiling.
  //
  // This is the number that decides whether a recording can be walked through a
  // challenge, and 6000 was far too generous. Every step already requires a
  // confirmed NEUTRAL face after its prompt and then the gesture no sooner than
  // MIN_REACTION_MS, so a video cannot arrive mid-gesture. But with six seconds
  // to play with, an attacker holding a recording of all four gestures could
  // simply let it run until the demanded one came around, and the reaction test
  // degraded into "does this video contain a blink somewhere".
  //
  // A human answering a prompt they are watching for reacts in roughly 350 ms
  // to 1.2 s, and each step spends part of its budget confirming a NEUTRAL face
  // before the gesture can even begin to count. 2.8 s left too little room for
  // that on a real user and produced constant restarts; 4 s is comfortable,
  // while still being well under the 6 s that let an attacker simply run a
  // recording until the demanded gesture came around.
  // The chroma challenge ships MEASURED but not ARMED. Its physics is device
  // independent but its operating point is not: screen brightness, front-camera
  // auto-white-balance and ambient light all scale the lift, so the threshold
  // has to be read off the target hardware (see docs/LIVENESS.md, "Calibrating
  // the chroma gate") before it can reject anyone. `report` surfaces the score
  // on every verification so that calibration is a five-minute job.
  // DISARMED AGAIN, on evidence. The ratio is sound physics and it does separate
  // when the reference is clean, but on this handset the reference is not:
  // at the 40 cm the guide circle asks for, the subject's head, neck and
  // shoulders fill the frame, so the "background" ring is largely more of the
  // same subject at the same distance, lit by the same flash. Measured on a
  // genuine user the ratio then came out at 0.43 and 1.24, indistinguishable
  // from a replay, and armed at 1.25 it rejected the real user repeatedly.
  //
  // A background reference has to be background. Fixing this means sampling a
  // region proven to be outside the subject rather than assuming the frame
  // border is, which is a real change and not a threshold tweak. Until then the
  // score is recorded and not enforced, which is the same rule every other gate
  // here follows.
  // OFF by default.
  //
  // The physics is sound and the code is kept, but it did not earn its place in
  // a field flow. It needs the phone's screen to be a meaningful share of the
  // light on the subject, which rules out the daylight half of a highway
  // worker's day outright. It costs every user 1.8 s of flashing plus the
  // settle time behind it, on every single verification. And across an evening
  // of measurement on the target handset it never produced a stable separation
  // between a live face and a replayed one: the face-to-room ratio was
  // contaminated by the subject's own shoulders at the framing this app asks
  // for, and the face-shape contrast came out at 0.99 on a real head.
  //
  // A barrier that cannot be armed is not a barrier, it is a delay. Deployments
  // that run indoors under controlled light can enable it after calibrating on
  // their own hardware (docs/CALIBRATION.md); the default should not pay for it.
  chromaMode: 'off',
  chromaSlots: 5, // lit / white / lit / white / lit -> ~1.8 s. Briefly cut to 3
  // to save a worker's time while the challenge was only advisory. Restored:
  // this is now the barrier that actually stops a recorded replay, `usable`
  // needs three sampled slots, and two lit slots leave no margin for a frame
  // lost to a blink or a head turn.
  chromaSlotMs: 360,
  chromaSettleMs: 700, // front-camera AWB/AE re-convergence after the flash
  chromaThreshold: 1.25, // The face must respond MORE than the room, not
  // several times more.
  //
  // 2.0 sat between the measured clusters (replay <= 0.71, live 4.3 and 8.0)
  // and looked like the obvious midpoint. It is not, because the live cluster
  // is not a fixed property of a face: the ratio depends on how much nearer the
  // face is than whatever is behind it. A worker standing against a wall, in a
  // toll booth, or in a doorway lights the background almost as much as their
  // own face, and the ratio collapses toward 1 without anything being wrong.
  // Choosing a midpoint between two clusters assumes both are tight; only the
  // attack cluster is.
  //
  // What geometry actually guarantees is the ORDERING: a face cannot be further
  // from the phone than the scene behind it, so its response can approach the
  // background's but not fall below it. A replayed display inverts that, and
  // measurably did: 0.25 and 0.71, the room moving more than the "face". The
  // threshold therefore sits just above the degenerate case of a face flat
  // against its own background, not halfway to a best case that only holds in
  // an open room.
  //
  // Dimensionless, so unlike an absolute lift it does not have to be re-derived
  // for every room, screen brightness and skin tone. Re-measure per handset
  // family all the same (docs/CALIBRATION.md).
  // Enrollment is the root of trust: a template built from a photograph makes
  // every later verification against it meaningless, and no amount of liveness
  // at verification can repair it. So the barrier belongs here. But it ships
  // MEASURED, NOT ARMED, because `passiveThreshold` above was calibrated on the
  // VERIFY capture distribution, and enrollment's is different: six shots fired
  // as a burst the moment a face is framed, with no gesture phase and no chroma
  // flash ahead of them. Borrowing the verify number rejected genuine
  // enrollments on the target device. `report` records the median on every
  // save so the enrollment operating point can be read off the handset and this
  // gate armed against its own distribution. See docs/CALIBRATION.md.
  enrollSpoofMode: 'report',
  enrollPassiveThreshold: 0.08, // inherited from the verify gate, NOT yet
  // measured on the enrollment distribution. Do not arm before measuring.
};
