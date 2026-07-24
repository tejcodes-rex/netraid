import React, { useCallback, useEffect, useState } from 'react';
import {
  ScrollView, StyleSheet, Text, View, useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import NetInfo from '@react-native-community/netinfo';
import { NetraID } from '../netraid';
import { countTemplates, countPendingAttendance } from '../netraid/store';
import { color, font, space, type } from '../theme';
import { Card, Metric, PrimaryButton, StatusPill } from '../components/ui';

// Measured, reproducible numbers (docs/BENCHMARKS.md): LFW verify accuracy of
// the shipped FP32 recognition model, and the on-disk size of all four models.
const LFW_ACCURACY = '99.76';
const MODEL_BUNDLE_MB = '17.3';

export function HomeScreen({ navigation }: { navigation: any }) {
  const { width } = useWindowDimensions();
  const wide = width > 520;
  const [enrolled, setEnrolled] = useState<number | null>(null);
  const [pending, setPending] = useState<number | null>(null);
  const [online, setOnline] = useState(false);

  const refresh = useCallback(async () => {
    try {
      await NetraID.init();
      setEnrolled(await countTemplates());
      setPending(await countPendingAttendance());
    } catch {
      // DB not ready yet; the next focus refresh will pick it up.
    }
  }, []);

  useEffect(() => {
    refresh();
    const unsubFocus = navigation.addListener('focus', refresh);
    const unsubNet = NetInfo.addEventListener((s) => {
      setOnline(!!(s.isConnected && s.isInternetReachable));
    });
    return () => { unsubFocus(); unsubNet(); };
  }, [navigation, refresh]);

  return (
    <SafeAreaView style={s.safe} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* brand */}
        <View style={s.header}>
          <View style={s.logo}>
            <Text style={s.logoMark}>◉</Text>
          </View>
          <View style={{ flex: 1 }}>
            <Text style={type.h2}>NetraID</Text>
            <Text style={[type.label, { marginTop: 2 }]}>NHAI · Datalake 3.0</Text>
          </View>
          <StatusPill label="EDGE" value="OFFLINE" tone="green" />
        </View>

        {/* hero */}
        <Card style={{ marginTop: space(4) }}>
          <View style={s.hero}>
            <Text style={s.heroKicker}>FIELD AUTHENTICATION</Text>
            <Text style={s.heroTitle}>Verify identity{'\n'}without a network.</Text>
            <Text style={[type.dim, { marginTop: space(2.5), lineHeight: 20 }]}>
              On-device face recognition + liveness in under a second. Works in zero-network
              highway zones. Nothing leaves the device until you reconnect.
            </Text>
            <View style={[s.actions, wide && { flexDirection: 'row' }]}>
              <View style={{ flex: 1 }}>
                <PrimaryButton title="Verify Attendance" onPress={() => navigation.navigate('Verify')} />
              </View>
              <View style={{ flex: 1 }}>
                <PrimaryButton title="Enroll Personnel" tone="ghost" onPress={() => navigation.navigate('Enroll')} />
              </View>
            </View>
          </View>
        </Card>

        {/* status metrics (live device data + measured benchmarks) */}
        <View style={s.metrics}>
          <Metric label="Enrolled" value={enrolled === null ? '...' : String(enrolled)} />
          <Metric label="Pending Sync" value={pending === null ? '...' : String(pending)} tone="amber" />
        </View>
        <View style={[s.metrics, { marginTop: space(3) }]}>
          <Metric label="Accuracy · LFW" value={LFW_ACCURACY} suffix="%" tone="cyan" />
          <Metric label="Models" value={MODEL_BUNDLE_MB} suffix="MB" />
        </View>

        {/* device line */}
        <Card style={{ marginTop: space(4) }}>
          <View style={{ padding: space(4), gap: space(3) }}>
            <Text style={type.label}>Device Status</Text>
            <Row
              k="Connectivity"
              v={online ? 'Online · sync active' : 'Offline · queued sync ready'}
              tone={online ? 'green' : 'amber'}
            />
            <Row k="Secure store" v="SQLCipher · key in Keystore" />
            <Row
              k="Sync queue"
              v={pending ? `${pending} pending · purged on server ACK` : 'Empty · no PII retained'}
            />
          </View>
        </Card>

        <Card style={{ marginTop: space(3) }}>
          <View style={{ padding: space(4) }}>
            <Text style={type.label}>Cross-Platform Proof</Text>
            <Text style={[type.dim, { marginTop: space(1.5), lineHeight: 18 }]}>
              Run the full recognition pipeline on bundled reference frames, no camera
              needed. Identical code path on Android and iOS.
            </Text>
            <View style={{ marginTop: space(3) }}>
              <PrimaryButton title="Run Pipeline Demo" tone="ghost" onPress={() => navigation.navigate('PipelineDemo')} />
            </View>
          </View>
        </Card>

        <Text style={s.foot}>100% offline inference · open-source · ap-south-1</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function Row({ k, v, tone = 'green' }: { k: string; v: string; tone?: 'green' | 'amber' }) {
  return (
    <View style={s.row}>
      <View style={[s.rowDot, { backgroundColor: tone === 'amber' ? color.amber : color.green }]} />
      <Text style={[type.dim, { width: 110 }]}>{k}</Text>
      <Text style={[type.body, { flex: 1, fontFamily: font.mono, fontSize: 12 }]}>{v}</Text>
    </View>
  );
}

const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: color.bg },
  scroll: { padding: space(4), paddingBottom: space(10) },
  header: { flexDirection: 'row', alignItems: 'center', gap: space(3) },
  logo: {
    width: 44, height: 44, borderRadius: 12, backgroundColor: color.greenDeep,
    alignItems: 'center', justifyContent: 'center',
  },
  logoMark: { color: color.green, fontSize: 22 },
  hero: { padding: space(5) },
  heroKicker: { fontFamily: font.mono, fontSize: 10, color: color.green, letterSpacing: 2 },
  heroTitle: { fontFamily: font.displayHeavy, fontSize: 30, color: color.ink, marginTop: space(2), lineHeight: 34, letterSpacing: -0.6 },
  actions: { gap: space(3), marginTop: space(5) },
  metrics: { flexDirection: 'row', gap: space(3), marginTop: space(4) },
  row: { flexDirection: 'row', alignItems: 'center', gap: space(2.5) },
  rowDot: { width: 7, height: 7, borderRadius: 4 },
  foot: { fontFamily: font.mono, fontSize: 10, color: color.inkFaint, textAlign: 'center', marginTop: space(6), letterSpacing: 0.8 },
});
