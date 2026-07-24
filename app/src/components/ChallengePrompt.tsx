// Liveness challenge prompt overlay, calm, legible, with a lane-dash progress
// track that matches the highway design language.
import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { color, font, radius, space } from '../theme';

export function ChallengePrompt({ label, step, total }:
  { label: string; step: number; total: number }) {
  const pulse = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 900, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0, duration: 900, useNativeDriver: true }),
      ]),
    ).start();
  }, [pulse]);

  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.06] });

  return (
    <View style={s.wrap}>
      <Text style={s.kicker}>LIVENESS CHECK · STEP {step}/{total}</Text>
      <Animated.Text style={[s.prompt, { transform: [{ scale }] }]}>{label}</Animated.Text>
      <View style={s.track}>
        {Array.from({ length: total }).map((_, i) => (
          <View key={i} style={[s.seg, i < step - 1 && s.segDone, i === step - 1 && s.segActive]} />
        ))}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    alignItems: 'center', paddingVertical: space(5), paddingHorizontal: space(6),
    borderRadius: radius.lg, backgroundColor: 'rgba(8,13,11,0.82)',
    borderColor: color.line2, borderWidth: 1, width: '88%',
  },
  kicker: { fontFamily: font.mono, fontSize: 10, color: color.green, letterSpacing: 2 },
  prompt: { fontFamily: font.displayHeavy, fontSize: 26, color: color.ink, marginTop: space(2.5) },
  track: { flexDirection: 'row', gap: space(2), marginTop: space(4) },
  seg: { width: 38, height: 5, borderRadius: 3, backgroundColor: color.line2 },
  segActive: { backgroundColor: color.green },
  segDone: { backgroundColor: color.greenDeep },
});
