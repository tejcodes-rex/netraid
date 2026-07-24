import React, { useEffect, useRef, useState } from 'react';
import { Image, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useRecognitionModel } from '../netraid/recognition';
import {
  DEMO_ASSETS, buildReport, runDemoFace,
  type DemoReport, type DemoFaceRun,
} from '../netraid/demoPipeline';
import { color, font, radius, space, type } from '../theme';
import { Card } from '../components/ui';

// Runs the full recognition pipeline on bundled reference frames, no camera
// needed. This is the deterministic committee demo on Android and the
// cross-platform proof on the iOS simulator: identical TypeScript, identical
// .tflite models, only the frame source differs from the live path.
export function PipelineDemoScreen() {
  const detector = useTensorflowModel(require('../../assets/models/blazeface_short_range.tflite'));
  const landmarker = useTensorflowModel(require('../../assets/models/face_landmarker.tflite'));
  const recogState = useRecognitionModel();
  const [runs, setRuns] = useState<DemoFaceRun[]>([]);
  const [report, setReport] = useState<DemoReport | null>(null);
  const [error, setError] = useState('');
  const started = useRef(false);

  const ready = !!detector.model && !!landmarker.model && recogState === 'loaded';

  useEffect(() => {
    if (!ready || started.current) return;
    started.current = true;
    const det = detector.model!, lmk = landmarker.model!;
    const done: DemoFaceRun[] = [];
    // Process one frame per tick so stage results paint progressively.
    const step = (i: number) => {
      if (i >= DEMO_ASSETS.length) {
        setReport(buildReport(done));
        return;
      }
      try {
        done.push(runDemoFace(det, lmk, DEMO_ASSETS[i]));
        setRuns([...done]);
        setTimeout(() => step(i + 1), 30);
      } catch (e) {
        setError(String(e));
      }
    };
    step(0);
  }, [ready, detector.model, landmarker.model]);

  return (
    <SafeAreaView style={s.safe} edges={['bottom']}>
      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <Text style={[type.dim, { lineHeight: 19 }]}>
          Full offline pipeline on bundled LFW reference frames: BlazeFace detect,
          FaceLandmarker, umeyama alignment, flip-TTA MobileFaceNet embedding, cosine
          match. Same code and models as the live camera, on any platform.
        </Text>

        {!ready && !error ? (
          <Card style={{ marginTop: space(4) }}>
            <Text style={[type.body, { padding: space(4) }]}>Loading models...</Text>
          </Card>
        ) : null}
        {error ? (
          <Card style={{ marginTop: space(4) }}>
            <Text style={[type.body, { padding: space(4), color: '#FF6B6B' }]}>{error}</Text>
          </Card>
        ) : null}

        {runs.map((r) => (
          <Card key={r.label} style={{ marginTop: space(3) }}>
            <View style={[s.rowBetween, { padding: space(4) }]}>
              <Image source={r.image} style={s.face} />
              <View style={{ flex: 1, marginLeft: space(3.5) }}>
                <View style={s.rowBetween}>
                  <Text style={type.body}>{r.label}</Text>
                  <Text style={s.mono}>face {Math.round(r.detectScore * 100)}%</Text>
                </View>
                <Text style={[s.mono, { marginTop: space(1.5), color: color.inkDim }]}>
                  detect {r.timings.detectMs}ms · landmarks {r.timings.landmarksMs}ms{'\n'}
                  align {r.timings.alignMs}ms · embed {r.timings.embedMs}ms
                </Text>
              </View>
            </View>
          </Card>
        ))}

        {report ? (
          <>
            <Text style={[type.label, { marginTop: space(5) }]}>MATCH MATRIX · threshold {report.threshold}</Text>
            {report.pairs.map((p) => (
              <Card key={p.a + p.b} style={{ marginTop: space(2.5) }}>
                <View style={[s.rowBetween, { padding: space(3.5) }]}>
                  <View style={s.pairFaces}>
                    <Image source={p.aImage} style={s.faceSmall} />
                    <Image source={p.bImage} style={[s.faceSmall, { marginLeft: -10 }]} />
                  </View>
                  <View style={{ flex: 1, marginLeft: space(3) }}>
                    <Text style={[type.body, { fontSize: 13 }]}>{p.a}  vs  {p.b}</Text>
                    <Text style={[s.mono, { marginTop: 2, color: color.inkDim }]}>
                      {p.genuine ? 'same person' : 'different people'} · cosine {p.score.toFixed(3)}
                    </Text>
                  </View>
                  <Text style={[s.verdict, { color: p.correct ? color.green : '#FF6B6B' }]}>
                    {p.correct ? (p.genuine ? 'MATCH ✓' : 'REJECT ✓') : 'WRONG ✗'}
                  </Text>
                </View>
              </Card>
            ))}
            <Card style={{ marginTop: space(4), borderColor: report.allCorrect ? color.green : '#FF6B6B', borderWidth: 1 }}>
              <View style={{ padding: space(4) }}>
                <Text style={[type.h2, { color: report.allCorrect ? color.green : '#FF6B6B' }]}>
                  {report.allCorrect ? 'All verdicts correct' : 'Verdict errors, inspect above'}
                </Text>
                <Text style={[type.dim, { marginTop: space(1.5) }]}>
                  Avg full-pipeline time {Math.round(report.avgTotalMs)} ms per face, entirely offline.
                </Text>
              </View>
            </Card>
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: color.bg },
  scroll: { padding: space(4), paddingBottom: space(10) },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  mono: { fontFamily: font.mono, fontSize: 11, color: color.ink },
  verdict: { fontFamily: font.mono, fontSize: 13, marginLeft: space(3) },
  face: { width: 64, height: 64, borderRadius: radius.md, backgroundColor: '#222' },
  faceSmall: {
    width: 42, height: 42, borderRadius: 21, backgroundColor: '#222',
    borderWidth: 1.5, borderColor: color.line2,
  },
  pairFaces: { flexDirection: 'row', alignItems: 'center' },
});
