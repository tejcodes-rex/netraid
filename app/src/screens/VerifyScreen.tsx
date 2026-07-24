import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, Pressable, StyleSheet, Text, View } from 'react-native';
import { useIsFocused } from '@react-navigation/native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { NetraID, DEFAULT_CONFIG, type CapturedFace, type Challenge, type VerifyResult } from '../netraid';
import { useRecognitionModel } from '../netraid/recognition';
import { useNetraFrameProcessor } from '../netraid/frameProcessor';
import { randomChallenges } from '../netraid/liveness';
import { countTemplates } from '../netraid/store';
import { ChallengePrompt } from '../components/ChallengePrompt';

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
  const [challenges, setChallenges] = useState(() => randomChallenges(2));
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

  /** Full reset for another attempt: fresh challenges, empty capture buffer. */
  const verifyAgain = useCallback(() => {
    capturesRef.current = [];
    verifying.current = false;
    doneRef.current = [];
    setShots(0);
    setDone([]);
    setResult(null);
    setFit(0);
    setHoldHint('');
    setTimedOut('');
    setChallenges(randomChallenges(2));
    setResetKey((k) => k + 1); // restarts the liveness FSM
  }, []);

  const frameProcessor = useNetraFrameProcessor(challenges, {
    onChallengeProgress: (d) => { setFaceOk(true); setDone(d); doneRef.current = d; },
    onNoFace: () => { setFaceOk(false); setFit(0); },
    onChallengeTimeout: () => {
      // A stage ran out of its 10 s budget. Never punish completed progress:
      // swap ONLY the stuck gesture for a fresh random one (anti-replay stays,
      // the set order and content remain unpredictable). Blink is the one
      // gesture that cannot be swapped away (photos cannot blink), so a stuck
      // blink keeps its slot and the user gets a steadiness hint.
      // NOTE: all state updates are siblings here, never nested inside an
      // updater; nesting desynced the on-screen prompt from the FSM.
      if (verifying.current) return;
      const prev = challengesRef.current;
      const doneCount = doneRef.current.length;
      const flash = (msg: string) => {
        setTimedOut(msg);
        setTimeout(() => setTimedOut(''), 2500);
      };
      if (doneCount >= prev.length) {
        // Capture phase timed out (face drifted post-liveness): full restart.
        capturesRef.current = [];
        doneRef.current = [];
        setShots(0);
        setDone([]);
        setChallenges(randomChallenges(prev.length));
        setResetKey((k) => k + 1);
        flash('Could not get a clear look, starting over');
        return;
      }
      const stuck = prev[doneCount];
      if (stuck === 'blink') {
        flash('Hold the phone steady and blink');
        return; // blink keeps its slot; the FSM already restarted the timer
      }
      const pool: Challenge[] = ['smile', 'turn-left', 'turn-right'];
      const options = pool.filter((c) => !prev.includes(c));
      if (options.length === 0) return;
      const next = [...prev];
      next[doneCount] = options[Math.floor(Math.random() * options.length)];
      setChallenges(next);
      flash('Switched to a different gesture');
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
    onCapture: async (rgb, passiveScore, sharpness) => {
      // The frame processor only emits captures AFTER every active challenge is
      // satisfied, so liveness is passed by construction here. (Reading the `done`
      // state instead would be stale inside this worklet-created callback.)
      // verifying.current gates re-entry AND stops capture while a result card
      // is shown (the camera stays live but we ignore its frames until the user
      // acts), so a finished verdict never spawns a new one.
      if (verifying.current) return;
      setHoldHint('');
      capturesRef.current.push({
        rgb,
        livenessActivePassed: true,
        completedChallenges: challenges,
        passiveScore,
        sharpness,
      });
      setShots(capturesRef.current.length);
      if (capturesRef.current.length < DEFAULT_CONFIG.verifyShots) return;
      verifying.current = true;
      try {
        const r = await NetraID.verify({
          captures: capturesRef.current,
          requireLiveness: true,
        });
        setResult(r);
      } catch (e) {
        // A thrown verify (e.g. the embedder briefly reloading) must NEVER
        // leave verifying.current stuck true, that would silently lock the
        // screen until a remount. Surface it and let the user retry.
        console.log('[verify] error', String(e));
        setResult({
          ok: false, score: 0, reason: 'low-quality', elapsedMs: 0,
          liveness: { passed: false, activePassed: false, passiveScore: 0, completedChallenges: [] },
        });
      }
    },
  }, resetKey, result !== null);

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
      {!result && (
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
      <View style={styles.overlay}>
        {!result && current && (
          <ChallengePrompt
            label={CHALLENGE_LABEL[current]}
            step={done.length + 1}
            total={challenges.length}
          />
        )}
        {!result && !current && (
          <View style={styles.holdCard}>
            <Text style={styles.holdText}>Liveness passed</Text>
            <Text style={styles.holdSub}>
              {holdHint || `Hold still, capturing ${Math.min(shots + 1, DEFAULT_CONFIG.verifyShots)}/${DEFAULT_CONFIG.verifyShots}...`}
            </Text>
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
        {ok ? `Person: ${result.personId}` : `Reason: ${result.reason}`}
      </Text>
      <Text style={styles.cardLine}>Score: {result.score.toFixed(3)}</Text>
      <Text style={styles.cardLine}>
        Liveness: {result.liveness.passed ? 'passed' : 'failed'} · {result.elapsedMs}ms
      </Text>
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
  holdCard: { backgroundColor: 'rgba(10,125,51,0.92)', paddingVertical: 16, paddingHorizontal: 24, borderRadius: 16, width: '85%', alignItems: 'center' },
  holdText: { color: '#fff', fontSize: 20, fontWeight: '700' },
  holdSub: { color: '#fff', fontSize: 14, marginTop: 4 },
  guideHint: { marginTop: 22, color: '#fff', fontSize: 14, textAlign: 'center', paddingHorizontal: 16,
    paddingVertical: 8, backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: 10, overflow: 'hidden' },
  debug: { position: 'absolute', top: 50, left: 8, right: 8, color: '#0f0', fontSize: 12,
    fontFamily: 'monospace', backgroundColor: 'rgba(0,0,0,0.55)', padding: 6 },
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
