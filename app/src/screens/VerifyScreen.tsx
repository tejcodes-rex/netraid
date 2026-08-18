import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, Pressable, StyleSheet, Text, View } from 'react-native';
import { useIsFocused } from '@react-navigation/native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { useSharedValue } from 'react-native-worklets-core';
import { NetraID, DEFAULT_CONFIG, type CapturedFace, type Challenge, type VerifyResult } from '../netraid';
import { useRecognitionModel } from '../netraid/recognition';
import { useNetraFrameProcessor } from '../netraid/frameProcessor';
import { randomChallenges } from '../netraid/liveness';
import { logCalibration, r4 } from '../netraid/calibration';
import { consensus, median } from '../netraid/math';
import { lampOn, lampOff } from '../netraid/screenLamp';
import {
  chromaResponse, randomFlashSequence, FLASH_RGB,
  type ChromaSample, type ChromaVerdict, type FlashColor,
} from '../netraid/chroma';
import { countTemplates } from '../netraid/store';
import { ChallengePrompt } from '../components/ChallengePrompt';

const REASON_LABEL: Record<NonNullable<VerifyResult['reason']>, string> = {
  spoof: 'presentation attack (photo, screen or replayed video)',
  'no-match': 'face not enrolled on this device',
  'no-face': 'no face detected',
  'low-quality': 'capture too blurred or poorly lit',
};

const CHALLENGE_LABEL: Record<Challenge, string> = {
  blink: 'Please blink',
  smile: 'Please smile',
  'turn-left': 'Turn your head left',
  'turn-right': 'Turn your head right',
};

export function VerifyScreen({ navigation }: { navigation: any }) {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();
  const [done, setDone] = useState<Challenge[]>([]);
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [dbg, setDbg] = useState('starting...');
  const [faceOk, setFaceOk] = useState(false);
  const [fit, setFit] = useState(0);
  const [shots, setShots] = useState(0);
  const [timedOut, setTimedOut] = useState('');
  const [holdHint, setHoldHint] = useState('');
  const [resetKey, setResetKey] = useState(0);
  const [enrolledCount, setEnrolledCount] = useState<number | null>(null);
  // Chroma challenge. `flashPhase` gates rendering and capture; the shared
  // values are what the frame-processor worklet reads to tag each measurement
  // with the colour that was actually on screen when the frame arrived.
  const [flashPhase, setFlashPhase] = useState<'idle' | 'running' | 'settling' | 'done'>('idle');
  const [flashColor, setFlashColor] = useState<FlashColor>('dark');
  const flashPhaseRef = useRef<'idle' | 'running' | 'settling' | 'done'>('idle');
  const flashSlot = useSharedValue(-1);
  const flashSince = useSharedValue(0);
  // Wall-clock instant the post-flash colour cast has cleared. Read by the
  // frame processor, which drops frames before it and holds the capture budget.
  const flashSettleUntil = useSharedValue(0);
  // Every processed frame's [P(real), P(screen)] for this attempt.
  const passiveSamples = useRef<[number, number][]>([]);
  const chromaSamples = useRef<ChromaSample[]>([]);
  const chromaVerdict = useRef<ChromaVerdict | null>(null);
  // Per-capture anti-spoof readings for the attempt that produced the verdict
  // on screen. Surfaced in release builds: the thresholds these gates use can
  // only be set from live readings on the target handset, and a release APK
  // has no Metro console to read them from.
  const [calib, setCalib] = useState<{
    real: number[]; screen: number[]; sharp: number[]; chroma: ChromaVerdict | null;
  } | null>(null);
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [attack, setAttack] = useState<
    { screen: number; real: number; continuity?: 'lost' | 'jump' } | null
  >(null);
  // Camera runs only while this screen is focused and the app is foregrounded;
  // fast app-switching or stacked navigation must never leave it running.
  const isFocused = useIsFocused();
  const [appActive, setAppActive] = useState(AppState.currentState === 'active');
  // Refs, not state: the frame-processor callbacks are captured once by the
  // worklet, so they must read/write stable objects.
  const capturesRef = useRef<CapturedFace[]>([]);
  const verifying = useRef(false);
  const doneRef = useRef<Challenge[]>([]);
  useRecognitionModel(); // loads the embedder via the reliable hook path
  // Count comes from config, never a literal: hard-coding 2 here meant raising
  // numChallenges to 3 changed nothing, because every later reset re-rolls
  // challengesRef.current.length, which was seeded from this line.
  const [challenges, setChallenges] = useState(
    () => randomChallenges(DEFAULT_CONFIG.numChallenges),
  );
  const challengesRef = useRef(challenges);
  useEffect(() => { challengesRef.current = challenges; }, [challenges]);

  useEffect(() => {
    if (!hasPermission) requestPermission();
    NetraID.init().then(async () => {
      setEnrolledCount(await countTemplates());
    }).catch(() => setEnrolledCount(null));
    const sub = AppState.addEventListener('change', (s) => setAppActive(s === 'active'));
    return () => sub.remove();
  }, [hasPermission, requestPermission]);

  /** Abandon any flash in progress and clear everything it measured. */
  const resetChroma = useCallback(() => {
    lampOff();
    if (flashTimer.current) clearTimeout(flashTimer.current);
    flashTimer.current = null;
    flashSlot.value = -1;
    // A stale settle deadline would suppress the NEXT attempt's frames for up
    // to chromaSettleMs, so it dies with the attempt that set it.
    flashSettleUntil.value = 0;
    chromaSamples.current = [];
    chromaVerdict.current = null;
    flashPhaseRef.current = 'idle';
    setFlashPhase('idle');
  }, [flashSlot, flashSettleUntil]);

  /** Full reset for another attempt: fresh challenges, empty capture buffer. */
  const verifyAgain = useCallback(() => {
    capturesRef.current = [];
    verifying.current = false;
    doneRef.current = [];
    setShots(0);
    setDone([]);
    setResult(null);
    setAttack(null);
    setCalib(null);
    passiveSamples.current = [];
    setFit(0);
    setHoldHint('');
    setTimedOut('');
    resetChroma();
    setChallenges(randomChallenges(DEFAULT_CONFIG.numChallenges));
    setResetKey((k) => k + 1); // restarts the liveness FSM
  }, [resetChroma]);

  /**
   * Run the screen-illumination challenge once the gestures are done, before
   * any recognition frame is captured. The sequence is drawn now, from a
   * CSPRNG, so it cannot have been recorded; each slot's colour is published to
   * the worklet with the timestamp it went up, and the worklet reports how the
   * face's colour balance responded. Recognition captures are blocked until it
   * finishes, because a colour-cast crop would poison the embedding.
   */
  const runChromaChallenge = useCallback(() => {
    if (DEFAULT_CONFIG.chromaMode === 'off') {
      flashPhaseRef.current = 'done';
      setFlashPhase('done');
      return;
    }
    const seq = randomFlashSequence(DEFAULT_CONFIG.chromaSlots);
    chromaSamples.current = [];
    // Drive the screen at full brightness for the flash: at the brightness a
    // handset actually sits at in the field, the emitted light is too weak for
    // the face's response to rise above sensor noise.
    lampOn();
    flashPhaseRef.current = 'running';
    setFlashPhase('running');

    const show = (i: number) => {
      if (i >= seq.length) {
        flashSlot.value = -1;
        chromaVerdict.current = chromaResponse(seq, chromaSamples.current);
        // Do NOT open the capture gate on the same tick the flash ends. The
        // front camera has just spent the whole sequence adapting its exposure
        // and white balance to saturated colour, and the first frames after it
        // goes dark still carry that cast. Embedding one shifts the probe away
        // from the template, and MiniFASNet, which scores colour and texture,
        // reads the cast as a presentation attack. This is the same failure the
        // 1.2 s cold-start warm-up exists to prevent, re-created mid-session.
        lampOff();
        flashSettleUntil.value = Date.now() + DEFAULT_CONFIG.chromaSettleMs;
        flashPhaseRef.current = 'settling';
        setFlashPhase('settling');
        flashTimer.current = setTimeout(() => {
          flashPhaseRef.current = 'done';
          setFlashPhase('done');
        }, DEFAULT_CONFIG.chromaSettleMs);
        return;
      }
      setFlashColor(seq[i]);
      // Publish the slot AND the instant it went up: the worklet discards
      // frames that arrived while the display was still switching, which would
      // otherwise be credited to the previous colour.
      flashSlot.value = i;
      flashSince.value = Date.now();
      flashTimer.current = setTimeout(() => show(i + 1), DEFAULT_CONFIG.chromaSlotMs);
    };
    show(0);
  }, [flashSlot, flashSince, flashSettleUntil]);

  // Gestures complete -> run the flash challenge exactly once per attempt.
  useEffect(() => {
    if (done.length >= challenges.length && flashPhase === 'idle' && !result && !attack) {
      runChromaChallenge();
    }
  }, [done.length, challenges.length, flashPhase, result, attack, runChromaChallenge]);

  // Leaving the screen mid-flash must not strand the handset at full brightness.
  useEffect(() => () => {
    lampOff();
    if (flashTimer.current) clearTimeout(flashTimer.current);
  }, []);

  const frameProcessor = useNetraFrameProcessor(challenges, {
    onChallengeProgress: (d) => { setFaceOk(true); setDone(d); doneRef.current = d; },
    onNoFace: () => { setFaceOk(false); setFit(0); },
    onChallengeTimeout: (reason) => {
      // A step ran out of its reaction window, or the capture phase stalled.
      // Either way the whole attempt restarts with a brand-new random set and
      // zero progress. The earlier behaviour kept completed steps and re-rolled
      // only the stuck gesture, which let anyone holding up a recording keep
      // one session alive until the gestures their video contained came up.
      // A session is now a single, short, all-or-nothing reaction test.
      // NOTE: all state updates are siblings here, never nested inside an
      // updater; nesting desynced the on-screen prompt from the FSM.
      if (verifying.current) return;
      const n = challengesRef.current.length;
      const restarted = doneRef.current.length > 0;
      // A timeout ends an attempt without ever producing a verdict, so without
      // this line a run that never got through the gestures leaves NO trace in
      // the calibration log at all, and looks from the outside like the app
      // doing nothing.
      logCalibration({ stage: 'verify', outcome: 'timeout', reason, steps: n, done: doneRef.current.length });
      capturesRef.current = [];
      doneRef.current = [];
      setShots(0);
      setDone([]);
      setChallenges(randomChallenges(n));
      setResetKey((k) => k + 1);
      // Say which phase actually stalled. Restarting after the gestures went
      // green, with the generic "took too long", is what read as a glitch: the
      // user had done everything asked and the app appeared to forget it.
      setTimedOut(
        reason === 'capture'
          ? 'Could not get a clear, front-facing frame. Look straight at the camera and hold still.'
          : restarted
            ? 'Took too long, starting over with new gestures'
            : 'Follow the prompt as soon as it appears',
      );
      setTimeout(() => setTimedOut(''), 3000);
    },
    onDebug: (m) => {
      if (__DEV__) {
        setDbg(m); // dev overlay only; in release this state churn would just tax the JS thread
        // Calibration probe + gesture telemetry -> Metro, for threshold tuning.
        if (m.indexOf('PASV') === 0 || m.indexOf('ear=') > 0) console.log('[liveness]', m);
      }
      if (m.indexOf('FIT ') === 0) {
        const pct = parseInt(m.slice(4), 10) || 0;
        setFit(pct);
        setFaceOk(pct >= 60);
      } else if (m.indexOf('FRONT') === 0) {
        // Liveness done but the capture gate wants a straight, well-framed face.
        setHoldHint('Look straight at the camera and hold still');
      } else if (m.indexOf('ear=') >= 0) {
        setFit(100);
        setFaceOk(true);
      }
    },
    onChromaSample: (slot, r, g, b, bgR, bgG, bgB, cR, cG, cB, pR, pG, pB, cY, pY) => {
      if (flashPhaseRef.current === 'running') {
        chromaSamples.current.push({ slot, r, g, b, bgR, bgG, bgB, cR, cG, cB, pR, pG, pB, cY, pY });
      }
    },
    onPassiveSample: (real, screen) => {
      // Accumulate the whole attempt's distribution. Three captures is not a
      // sample you can set a security threshold from.
      if (passiveSamples.current.length < 400) passiveSamples.current.push([real, screen]);
    },
    onContinuityBreak: (reason) => {
      // The face that proved liveness is not the face still in front of the
      // camera, so nothing captured from here can be attributed to it. Void the
      // whole attempt rather than the capture: a fresh gesture set is the only
      // way to re-establish who is there.
      if (verifying.current) return;
      verifying.current = true;
      capturesRef.current = [];
      setShots(0);
      logCalibration({ stage: 'verify', outcome: 'void', reason: 'continuity-' + reason });
      setAttack({ screen: 0, real: 0, continuity: reason });
    },
    onSpoofEvidence: (screen, real) => {
      // Two consecutive frames read as a display. End the attempt here and say
      // so, instead of collecting captures and only revealing it at the verdict.
      // Frames from the flash AND from the settle window that follows it are
      // colour-cast, so their anti-spoof reading is not evidence of anything.
      // Only those two phases are excluded: evidence seen during the GESTURE
      // phase is on an uncast frame and is exactly when a replay should be
      // caught, so suppressing it there (as `!== 'done'` did) would throw away
      // the earliest honest reading of the attempt.
      const phase = flashPhaseRef.current;
      if (verifying.current || phase === 'running' || phase === 'settling') return;
      verifying.current = true;
      capturesRef.current = [];
      setShots(0);
      setAttack({ screen, real });
      logCalibration({ stage: 'verify', outcome: 'abort', reason: 'spoof-early', screen: r4(screen), real: r4(real) });
    },
    onCapture: async (rgb, passiveScore, passiveScreen, sharpness) => {
      // The frame processor only emits captures AFTER every active challenge is
      // satisfied, so the gesture barrier is passed by construction here.
      // (Reading the `done` state instead would be stale inside this
      // worklet-created callback.) verifying.current gates re-entry AND stops
      // capture while a result card is shown (the camera stays live but we
      // ignore its frames until the user acts), so a finished verdict never
      // spawns a new one.
      if (verifying.current) return;
      // Never embed a frame lit by a coloured flash: the cast would shift the
      // template. Recognition waits for the chroma challenge to finish.
      if (flashPhaseRef.current !== 'done') return;
      setHoldHint('');
      capturesRef.current.push({
        rgb,
        livenessActivePassed: true,
        completedChallenges: challenges,
        passiveScore,
        passiveScreen,
        sharpness,
      });
      setShots(capturesRef.current.length);
      if (capturesRef.current.length < DEFAULT_CONFIG.verifyShots) return;
      verifying.current = true;
      // Distribution over the WHOLE attempt, which is what a threshold has to be
      // chosen from. Reported as percentiles rather than a mean: the tail is
      // what decides whether a real user is turned away.
      const pct = (xs: number[], q: number) => {
        const a = [...xs].sort((x, y) => x - y);
        return a.length ? a[Math.min(a.length - 1, Math.floor(q * (a.length - 1)))] : 0;
      };
      const allReal = passiveSamples.current.map((v) => v[0]);
      const allScreen = passiveSamples.current.map((v) => v[1]);
      logCalibration({
        stage: 'distribution',
        frames: allReal.length,
        real: {
          min: r4(pct(allReal, 0)), p10: r4(pct(allReal, 0.1)), p50: r4(pct(allReal, 0.5)),
          p90: r4(pct(allReal, 0.9)), max: r4(pct(allReal, 1)),
        },
        screen: {
          min: r4(pct(allScreen, 0)), p10: r4(pct(allScreen, 0.1)), p50: r4(pct(allScreen, 0.5)),
          p90: r4(pct(allScreen, 0.9)), max: r4(pct(allScreen, 1)),
        },
      });
      const readings = {
        real: capturesRef.current.map((c) => c.passiveScore),
        screen: capturesRef.current.map((c) => c.passiveScreen),
        sharp: capturesRef.current.map((c) => c.sharpness),
        chroma: chromaVerdict.current,
      };
      setCalib(readings);
      try {
        const r = await NetraID.verify({
          captures: capturesRef.current,
          requireLiveness: true,
          chroma: chromaVerdict.current,
        });
        setResult(r);
        logCalibration({
          stage: 'verify',
          outcome: r.ok ? 'accept' : 'reject',
          reason: r.reason ?? null,
          score: r4(r.score),
          elapsedMs: Math.round(r.elapsedMs),
          real: readings.real.map(r4),
          screen: readings.screen.map(r4),
          sharp: readings.sharp.map((v) => Math.round(v)),
          chroma: readings.chroma && readings.chroma.usable ? r4(readings.chroma.score) : null,
          chromaFace: readings.chroma ? r4(readings.chroma.faceLift) : null,
          chromaBg: readings.chroma ? r4(readings.chroma.bgLift) : null,
          chromaLit: readings.chroma ? r4(readings.chroma.litContrast) : null,
          chromaDark: readings.chroma ? r4(readings.chroma.darkContrast) : null,
          chromaUsable: readings.chroma ? readings.chroma.usable : null,
          challenges,
        });
      } catch (e) {
        // A thrown verify (e.g. the embedder briefly reloading) must NEVER
        // leave verifying.current stuck true, that would silently lock the
        // screen until a remount. Surface it and let the user retry.
        console.log('[verify] error', String(e));
        logCalibration({ stage: 'verify', outcome: 'error', error: String(e) });
        setResult({
          ok: false, score: 0, reason: 'low-quality', elapsedMs: 0,
          liveness: {
            passed: false, activePassed: false, passiveScore: 0, passiveScreen: 1,
            chromaScore: null, completedChallenges: [],
          },
        });
      }
    },
  }, resetKey, result !== null || attack !== null, {
    slot: flashSlot, since: flashSince, settleUntil: flashSettleUntil,
  });

  if (!device || !hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.info}>Grant camera permission to continue...</Text>
      </View>
    );
  }

  const current = challenges[done.length];

  if (enrolledCount === 0) {
    return (
      <View style={[styles.center, styles.fill]}>
        <Text style={styles.emptyTitle}>No personnel enrolled</Text>
        <Text style={styles.emptySub}>
          Enroll at least one person on this device before verifying attendance.
        </Text>
        <View style={{ marginTop: 24, alignSelf: 'stretch', marginHorizontal: 48 }}>
          <Pressable style={styles.actionBtn} onPress={() => navigation.replace('Enroll')}>
            <Text style={styles.actionText}>Enroll Personnel</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.fill}>
      <Camera
        // Keep the camera CONTINUOUSLY active while the screen is visible.
        // Toggling isActive off (after a result) then on does not reliably
        // resume frame delivery on this device, the session gets stranded and
        // only a full screen remount recovers it. Staying active avoids the
        // deactivate/reactivate cycle entirely; capture is gated in software
        // (verifying.current + result), not by stopping the camera.
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isFocused && appActive}
        frameProcessor={frameProcessor}
        pixelFormat="rgb"
      />
      {/* The chroma challenge: the screen becomes the lamp. It covers the
          preview deliberately, both so the emitted colour is not diluted by
          dark preview pixels and so the subject holds still through it. */}
      {flashPhase === 'running' && (
        <View pointerEvents="none" style={[StyleSheet.absoluteFill, { backgroundColor: FLASH_RGB[flashColor] }]}>
          <View style={styles.flashCaption}>
            <Text style={styles.flashText}>Hold still</Text>
            <Text style={styles.flashSub}>Checking that this is a live face</Text>
          </View>
        </View>
      )}
      {!result && !attack && flashPhase !== 'running' && (
        <View pointerEvents="none" style={styles.guideWrap}>
          <View style={[styles.faceOval, { borderColor: fit >= 60 ? '#1FD27A' : fit >= 30 ? '#F2B705' : 'rgba(255,255,255,0.6)' }]} />
          <Text style={[styles.fitPct, { color: fit >= 60 ? '#1FD27A' : fit >= 30 ? '#F2B705' : '#fff' }]}>
            {fit > 0 ? `Face fit ${fit}%` : 'No face'}
          </Text>
          <Text style={styles.guideHint}>
            {timedOut ? timedOut
              : fit >= 60 ? 'Good, hold still and follow the prompt'
              : fit > 0 ? 'Move closer and center your face in the circle'
              : 'Fit your face in the circle, hold about 40 cm away'}
          </Text>
        </View>
      )}
      {__DEV__ && <Text style={styles.debug}>{dbg}</Text>}
      {__DEV__ && (result || attack) && calib && (
        // LIVENESS DIAGNOSTIC, development builds only.
        //
        // Every layer's own verdict and the number it was taken on. Invaluable
        // while calibrating, and wrong to ship: a field user has no use for
        // model probabilities, and an attacker reading the exact margin by
        // which they failed is being handed a tuning signal. Metro inlines
        // __DEV__, so this compiles out of release entirely.
        //
        // Release builds carry the same numbers in one machine-readable log
        // line per attempt instead, which is what docs/CALIBRATION.md uses.
        <View style={styles.calib} pointerEvents="none">
          <Text style={styles.calibTitle}>Liveness diagnostic</Text>
          {calib.real.map((v, i) => (
            <Text key={i} style={styles.calibLine}>
              {`frame ${i + 1}   real ${v.toFixed(3)}   screen ${calib.screen[i].toFixed(3)}   sharp ${calib.sharp[i].toFixed(0)}`}
            </Text>
          ))}
          <Text style={styles.calibLine}>
            {`real   median ${median(calib.real).toFixed(3)}  min ${DEFAULT_CONFIG.passiveThreshold}` +
             `  ${DEFAULT_CONFIG.passiveMode}`}
          </Text>
          <Text style={styles.calibLine}>
            {`screen consensus ${consensus(calib.screen, 2).toFixed(3)}  max ${DEFAULT_CONFIG.screenSpoofMax}` +
             `  ${DEFAULT_CONFIG.screenSpoofMode}`}
          </Text>
          <Text style={styles.calibLine}>
            {`chroma ${DEFAULT_CONFIG.chromaMode === 'off' ? 'off'
              : calib.chroma && calib.chroma.usable ? calib.chroma.score.toFixed(3) : 'unusable'}`}
          </Text>
        </View>
      )}
      <View style={styles.overlay}>
        {!result && !attack && flashPhase !== 'running' && current && (
          <ChallengePrompt
            label={CHALLENGE_LABEL[current]}
            step={done.length + 1}
            total={challenges.length}
          />
        )}
        {!result && !attack && (flashPhase === 'done' || flashPhase === 'settling') && !current && (
          // Deliberately NOT "Liveness passed". The gestures are only the first
          // of three barriers; the anti-spoof verdict is not in yet. Claiming a
          // pass here and then rejecting seconds later reads as a broken app
          // even when the rejection is correct.
          <View style={styles.holdCard}>
            <Text style={styles.holdText}>Gestures verified · checking</Text>
            <Text style={styles.holdSub}>
              {holdHint || `Checking authenticity, capturing ${Math.min(shots + 1, DEFAULT_CONFIG.verifyShots)}/${DEFAULT_CONFIG.verifyShots}...`}
            </Text>
          </View>
        )}
        {attack && (
          <View style={[styles.card, { backgroundColor: '#b00020' }]}>
            <Text style={styles.cardTitle}>
              {attack.continuity ? '✗ Attempt void' : '✗ Presentation attack'}
            </Text>
            <Text style={styles.cardLine}>
              {attack.continuity === 'lost'
                ? 'The face that completed the challenge left the frame before the '
                  + 'photo was taken, so the two cannot be attributed to the same person.'
                : attack.continuity === 'jump'
                  ? 'The face in frame was replaced after the challenge was completed. '
                    + 'Liveness and identity must be proved by the same subject.'
                  : `This is a display, not a face. Anti-spoof: screen ${attack.screen.toFixed(2)}, `
                    + `real ${attack.real.toFixed(2)}.`}
            </Text>
            <Text style={styles.cardLine}>No attendance was recorded.</Text>
            <View style={styles.cardActions}>
              <Pressable style={[styles.actionBtn, styles.actionGhost]} onPress={verifyAgain}>
                <Text style={styles.actionText}>Try again</Text>
              </Pressable>
              <Pressable style={styles.actionBtn} onPress={() => navigation.goBack()}>
                <Text style={styles.actionText}>Done</Text>
              </Pressable>
            </View>
          </View>
        )}
        {result && (
          <ResultCard
            result={result}
            onAgain={verifyAgain}
            onDone={() => navigation.goBack()}
          />
        )}
      </View>
    </View>
  );
}

function ResultCard({ result, onAgain, onDone }: {
  result: VerifyResult; onAgain: () => void; onDone: () => void;
}) {
  const ok = result.ok;
  return (
    <View style={[styles.card, { backgroundColor: ok ? '#0a7d33' : '#b00020' }]}>
      <Text style={styles.cardTitle}>
        {ok ? '✓ Verified' : '✗ Not verified'}
      </Text>
      <Text style={styles.cardLine}>
        {ok ? `Person: ${result.personId}` : `Reason: ${REASON_LABEL[result.reason ?? 'low-quality']}`}
      </Text>
      <Text style={styles.cardLine}>Score: {result.score.toFixed(3)}</Text>
      <Text style={styles.cardLine}>
        Liveness: {result.liveness.passed ? 'passed' : 'failed'} · {result.elapsedMs}ms
      </Text>
      {(result.reason === 'spoof' || result.liveness.chromaScore !== null) && (
        // Show WHICH barrier rejected: it is the difference between "the app
        // said no" and being able to point at the evidence during an audit.
        <Text style={styles.cardLine}>
          Anti-spoof: real {result.liveness.passiveScore.toFixed(2)} · screen{' '}
          {result.liveness.passiveScreen.toFixed(2)}
          {result.liveness.chromaScore !== null &&
            ` · chroma ${result.liveness.chromaScore.toFixed(3)}`}
        </Text>
      )}
      <View style={styles.cardActions}>
        <Pressable style={[styles.actionBtn, styles.actionGhost]} onPress={onAgain}>
          <Text style={styles.actionText}>Verify again</Text>
        </Pressable>
        <Pressable style={styles.actionBtn} onPress={onDone}>
          <Text style={styles.actionText}>Done</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  info: { color: '#333', fontSize: 16 },
  overlay: { position: 'absolute', bottom: 60, left: 0, right: 0, alignItems: 'center' },
  guideWrap: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
  faceOval: { width: 250, height: 330, borderRadius: 165, borderWidth: 3, backgroundColor: 'transparent' },
  fitPct: { marginTop: 14, fontSize: 18, fontWeight: '700', textShadowColor: '#000', textShadowRadius: 4 },
  // Deliberately NOT green. Green is the verdict colour, and this card is shown
  // while two of the three barriers have not ruled yet. Painting it green here
  // and then rejecting (or restarting) is what reads as a glitch to an observer,
  // who quite reasonably reports that the app "said it passed".
  holdCard: {
    backgroundColor: 'rgba(20,22,26,0.92)', borderWidth: 1, borderColor: '#F2B705',
    paddingVertical: 16, paddingHorizontal: 24, borderRadius: 16, width: '85%',
    alignItems: 'center',
  },
  flashCaption: { position: 'absolute', bottom: 90, left: 0, right: 0, alignItems: 'center' },
  flashText: { color: 'rgba(0,0,0,0.72)', fontSize: 26, fontWeight: '800' },
  flashSub: { color: 'rgba(0,0,0,0.55)', fontSize: 14, marginTop: 4 },
  holdText: { color: '#F2B705', fontSize: 20, fontWeight: '700' },
  holdSub: { color: '#fff', fontSize: 14, marginTop: 4 },
  guideHint: { marginTop: 22, color: '#fff', fontSize: 14, textAlign: 'center', paddingHorizontal: 16,
    paddingVertical: 8, backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: 10, overflow: 'hidden' },
  debug: { position: 'absolute', top: 50, left: 8, right: 8, color: '#0f0', fontSize: 12,
    fontFamily: 'monospace', backgroundColor: 'rgba(0,0,0,0.55)', padding: 6 },
  calib: {
    position: 'absolute', top: 44, left: 10, right: 10, padding: 8, borderRadius: 8,
    backgroundColor: 'rgba(0,0,0,0.66)',
  },
  calibTitle: { color: '#8DE8B4', fontSize: 11, fontWeight: '700', marginBottom: 3 },
  calibLine: { color: '#DCEFE4', fontSize: 11, fontFamily: 'monospace' },
  card: { padding: 20, borderRadius: 16, width: '85%' },
  cardTitle: { color: '#fff', fontSize: 22, fontWeight: '700', marginBottom: 8 },
  cardLine: { color: '#fff', fontSize: 15, marginTop: 2 },
  cardActions: { flexDirection: 'row', gap: 12, marginTop: 16 },
  actionBtn: {
    flex: 1, backgroundColor: 'rgba(255,255,255,0.22)', borderRadius: 10,
    paddingVertical: 12, alignItems: 'center',
  },
  actionGhost: { backgroundColor: 'rgba(0,0,0,0.25)' },
  actionText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  emptyTitle: { color: '#fff', fontSize: 22, fontWeight: '700' },
  emptySub: {
    color: '#8DA298', fontSize: 14, textAlign: 'center', marginTop: 8,
    marginHorizontal: 32, lineHeight: 20,
  },
});
