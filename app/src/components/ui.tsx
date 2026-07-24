// Reusable NetraID UI primitives, consistent with the web Command Center.
import React from 'react';
import {
  Pressable, StyleSheet, Text, View, type ViewStyle, type StyleProp,
} from 'react-native';
import { color, radius, space, type, shadow, font } from '../theme';

export function Card({ children, style }: { children: React.ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[s.card, style]}>{children}</View>;
}

export function CardHead({ title, tag }: { title: string; tag?: string }) {
  return (
    <View style={s.head}>
      <View style={s.accent} />
      <Text style={type.title}>{title}</Text>
      {tag ? <Text style={[type.label, { marginLeft: 'auto' }]}>{tag}</Text> : null}
    </View>
  );
}

export function StatusPill({ label, value, tone = 'green' }:
  { label: string; value: string; tone?: 'green' | 'amber' | 'red' }) {
  const c = tone === 'amber' ? color.amber : tone === 'red' ? color.red : color.green;
  return (
    <View style={s.pill}>
      <View style={[s.dot, { backgroundColor: c }]} />
      <Text style={s.pillLabel}>{label}</Text>
      <Text style={s.pillValue}>{value}</Text>
    </View>
  );
}

export function Metric({ label, value, suffix, tone = 'green' }:
  { label: string; value: string; suffix?: string; tone?: 'green' | 'amber' | 'cyan' }) {
  const c = tone === 'amber' ? color.amber : tone === 'cyan' ? color.cyan : color.green;
  return (
    <Card style={{ flex: 1, minWidth: 150 }}>
      <View style={[s.kpiAccent, { backgroundColor: c }]} />
      <View style={{ padding: space(4) }}>
        <Text style={type.label}>{label}</Text>
        <Text style={[type.metric, { marginTop: space(2) }]}>
          {value}
          {suffix ? <Text style={{ fontSize: 14, color: color.inkDim }}>{suffix}</Text> : null}
        </Text>
      </View>
    </Card>
  );
}

export function PrimaryButton({ title, onPress, tone = 'green', disabled }:
  { title: string; onPress: () => void; tone?: 'green' | 'ghost'; disabled?: boolean }) {
  const ghost = tone === 'ghost';
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        s.btn,
        ghost ? s.btnGhost : s.btnGreen,
        pressed && { opacity: 0.85, transform: [{ scale: 0.99 }] },
        disabled && { opacity: 0.4 },
      ]}>
      <Text style={[s.btnText, { color: ghost ? color.green : color.onGreen }]}>{title}</Text>
    </Pressable>
  );
}

const s = StyleSheet.create({
  card: {
    backgroundColor: color.panel, borderColor: color.line, borderWidth: 1,
    borderRadius: radius.md, overflow: 'hidden', ...shadow.card,
  },
  head: {
    flexDirection: 'row', alignItems: 'center', gap: space(2.5),
    paddingVertical: space(3.5), paddingHorizontal: space(4),
    borderBottomColor: color.line, borderBottomWidth: 1,
  },
  accent: { width: 3, height: 16, borderRadius: 2, backgroundColor: color.green },
  kpiAccent: { position: 'absolute', left: 0, top: 0, bottom: 0, width: 3 },
  pill: {
    flexDirection: 'row', alignItems: 'center', gap: space(2),
    paddingVertical: space(2), paddingHorizontal: space(3),
    borderRadius: radius.pill, borderColor: color.line2, borderWidth: 1,
    backgroundColor: color.bg2,
  },
  dot: { width: 8, height: 8, borderRadius: 4 },
  pillLabel: { fontFamily: font.mono, fontSize: 10, color: color.inkFaint, letterSpacing: 1.2 },
  pillValue: { fontFamily: font.monoMed, fontSize: 11, color: color.ink },
  btn: {
    borderRadius: radius.md, paddingVertical: space(4), alignItems: 'center', justifyContent: 'center',
  },
  btnGreen: { backgroundColor: color.green },
  btnGhost: { backgroundColor: 'transparent', borderColor: color.line2, borderWidth: 1 },
  btnText: { fontFamily: font.display, fontSize: 15, letterSpacing: 0.2 },
});
