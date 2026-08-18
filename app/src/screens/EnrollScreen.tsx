import React, { useEffect, useMemo, useState } from 'react';
import { AppState, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useIsFocused } from '@react-navigation/native';
import {
  Camera, useCameraDevice, useCameraPermission,
} from 'react-native-vision-camera';
import {
  NetraID, DuplicateFaceError, SpoofedEnrollmentError,
  type Challenge, type CapturedFace,
} from '../netraid';
import { useRecognitionModel } from '../netraid/recognition';
import { useNetraFrameProcessor } from '../netraid/frameProcessor';
import { randomChallenges } from '../netraid/liveness';
import { color, font, radius, space, type } from '../theme';
import { PrimaryButton } from '../components/ui';

// Multi-shot enrollment: capture several candidate frames; enroll() keeps the
// sharpest, drops embedding outliers, and averages the rest into the template.
const SHOTS = 6;

export function EnrollScreen({ navigation }: { navigation: any }) {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();
  const [personId, setPersonId] = useState('');
  const [captures, setCaptures] = useState<CapturedFace[]>([]);
  const [done, setDone] = useState(false);
  const [fit, setFit] = useState(0);
  const [status, setStatus] = useState('');
  // Bumped on each retake to remount the camera and reset the frame processor;
  // reactivating the camera in place after the 6-shot session does not reliably
  // resume frames on this device, the session must be recreated.
  const [attempt, setAttempt] = useState(0);
  // Camera runs only while focused + foregrounded (parity with VerifyScreen).
  const isFocused = useIsFocused();
  const [appActive, setAppActive] = useState(AppState.currentState === 'active');
  useRecognitionModel(); // loads the embedder via the reliable hook path
  // Enrollment needs a clean, frontal face shot rather than a gesture sequence,
  // so no active challenge runs here. The PASSIVE anti-spoof barrier is still
  // enforced, inside enroll(): a template built from a photograph would make
  // every later verification against it meaningless. With no challenges, the
  // frame processor emits a
  // capture as soon as a well-framed face is present.
  const challenges = useMemo<Challenge[]>(() => [], []);

  useEffect(() => {
    if (!hasPermission) requestPermission();
    NetraID.init();
    const sub = AppState.addEventListener('change', (s) => setAppActive(s === 'active'));
    return () => sub.remove();
  }, [hasPermission, requestPermission]);

  const frameProcessor = useNetraFrameProcessor(challenges, {
    onChallengeProgress: () => {},
    onNoFace: () => setFit(0),
    onDebug: (m) => {
      if (m.indexOf('FIT ') === 0) setFit(parseInt(m.slice(4), 10) || 0);
      else if (m.indexOf('ear=') >= 0) setFit(100);
    },
    onCapture: (rgb, passiveScore, passiveScreen, sharpness) => {
      setCaptures((prev) =>
        prev.length >= SHOTS
          ? prev
          : [...prev, { rgb, livenessActivePassed: true, completedChallenges: challenges, passiveScore, passiveScreen, sharpness }],
      );
    },
  }, attempt, done || captures.length >= SHOTS);

  const canSave = personId.trim().length > 0 && captures.length >= SHOTS;

  const [saving, setSaving] = useState(false);

  async function save() {
    if (saving || done) return; // ignore repeat taps while a save is in flight
    setSaving(true);
    setStatus('Saving template...');
    try {
      await NetraID.enroll({ personId: personId.trim(), captures });
      setStatus('Enrolled');
      setDone(true);
      setTimeout(() => navigation.goBack(), 1200);
    } catch (e) {
      if (e instanceof DuplicateFaceError) {
        setStatus(`Save failed: this face is already enrolled as ${e.existingPersonId}`);
      } else if (e instanceof SpoofedEnrollmentError) {
        setStatus(
          'Save failed: these frames did not read as a live face. Enroll the person ' +
          'directly, not from a photo or a screen.',
        );
      } else {
        setStatus('Save failed: ' + String(e));
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <SafeAreaView style={s.safe} edges={['top']}>
      <View style={s.headerRow}>
        <Text style={type.h2}>Enroll Personnel</Text>
        <Text style={[type.label]}>{captures.length}/{SHOTS} CAPTURED</Text>
      </View>

      <View style={s.cameraBox}>
        {device && hasPermission ? (
          <Camera
            // Continuously active while visible (until the template is saved).
            // Reactivating a stopped camera strands the session on this device;
            // the SHOTS cap is enforced in onCapture, not by stopping frames.
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={isFocused && appActive && !done}
            frameProcessor={frameProcessor}
            pixelFormat="rgb"
          />
        ) : (
          <View style={s.center}><Text style={type.dim}>Grant camera permission...</Text></View>
        )}
        <View style={[s.ring, { borderColor: fit >= 60 ? '#1FD27A' : fit >= 30 ? '#F2B705' : 'rgba(255,255,255,0.5)' }]} />
        <Text style={s.fitText}>{fit > 0 ? `Fit ${fit}%${fit >= 60 ? ', hold still + follow prompt' : ', center your face'}` : 'Fit your face in the ring, ~40 cm'}</Text>
        <View style={s.shots}>
          {Array.from({ length: SHOTS }).map((_, i) => (
            <View key={i} style={[s.shotDot, i < captures.length && s.shotDotOn]} />
          ))}
        </View>
      </View>

      <View style={s.form}>
        <Text style={type.label}>PERSONNEL ID</Text>
        <TextInput
          value={personId}
          onChangeText={setPersonId}
          placeholder="e.g. NHAI-04821"
          placeholderTextColor={color.inkFaint}
          autoCapitalize="characters"
          style={s.input}
        />
        <View style={{ marginTop: space(4) }}>
          <PrimaryButton
            title={done ? '✓ Enrolled' : saving ? 'Saving...' : canSave ? 'Save Template' : `Capture ${SHOTS} angles`}
            onPress={save}
            disabled={!canSave || done || saving}
          />
        </View>
        {status ? (
          <Text style={[s.status, status.indexOf('failed') >= 0 && { color: '#FF6B6B' }]}>{status}</Text>
        ) : null}
        {captures.length > 0 && captures.length < SHOTS && !done && !saving ? (
          <Text style={s.enrollHint}>Hold your face centered and still, {SHOTS - captures.length} more...</Text>
        ) : null}
        {captures.length > 0 && !done && !saving ? (
          <Pressable onPress={() => { setCaptures([]); setStatus(''); setAttempt((a) => a + 1); }}>
            <Text style={s.retake}>{captures.length >= SHOTS ? 'Retake shots' : 'Restart capture'}</Text>
          </Pressable>
        ) : null}
        <Text style={s.hint}>
          Only an encrypted embedding is stored, never your photo. Move slightly between
          captures for better accuracy in varied light.
        </Text>
      </View>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: color.bg },
  headerRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: space(4), paddingVertical: space(3),
  },
  cameraBox: {
    margin: space(4), height: 360, borderRadius: radius.lg, overflow: 'hidden',
    backgroundColor: '#000', borderColor: color.line2, borderWidth: 1,
  },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  ring: {
    position: 'absolute', alignSelf: 'center', top: 40, width: 220, height: 280,
    borderRadius: 140, borderWidth: 3,
  },
  fitText: {
    position: 'absolute', alignSelf: 'center', top: 330, color: '#fff', fontSize: 13,
    backgroundColor: 'rgba(0,0,0,0.5)', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8, overflow: 'hidden',
  },
  shots: { position: 'absolute', bottom: 16, alignSelf: 'center', flexDirection: 'row', gap: space(2) },
  shotDot: { width: 30, height: 5, borderRadius: 3, backgroundColor: 'rgba(255,255,255,0.25)' },
  shotDotOn: { backgroundColor: color.green },
  form: { paddingHorizontal: space(4) },
  input: {
    marginTop: space(2), backgroundColor: color.panel, borderColor: color.line2, borderWidth: 1,
    borderRadius: radius.md, paddingHorizontal: space(4), paddingVertical: space(3.5),
    color: color.ink, fontFamily: font.mono, fontSize: 15, letterSpacing: 1,
  },
  hint: { ...type.dim, marginTop: space(4), lineHeight: 18, fontSize: 12 },
  status: {
    marginTop: space(3), color: color.green, fontFamily: font.mono, fontSize: 13,
    textAlign: 'center',
  },
  retake: {
    marginTop: space(3), color: color.inkDim, fontFamily: font.mono, fontSize: 12,
    textAlign: 'center', textDecorationLine: 'underline',
  },
  enrollHint: {
    marginTop: space(3), color: color.amber, fontFamily: font.mono, fontSize: 12,
    textAlign: 'center',
  },
});
