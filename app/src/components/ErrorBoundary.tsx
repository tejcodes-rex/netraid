import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { color, font, space, type } from '../theme';
import { PrimaryButton } from './ui';

interface Props { children: React.ReactNode }
interface State { error: Error | null }

/**
 * Last-resort crash barrier: a render error anywhere in the tree shows a
 * recoverable screen instead of a dead black window. Field devices have no
 * dev to shake-reload; "Try again" remounts the tree.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <View style={s.wrap}>
        <Text style={type.h2}>Something went wrong</Text>
        <Text style={s.detail} numberOfLines={4}>{String(this.state.error)}</Text>
        <View style={{ marginTop: space(5), alignSelf: 'stretch' }}>
          <PrimaryButton title="Try again" onPress={() => this.setState({ error: null })} />
        </View>
      </View>
    );
  }
}

const s = StyleSheet.create({
  wrap: {
    flex: 1, backgroundColor: color.bg, alignItems: 'center',
    justifyContent: 'center', padding: space(6),
  },
  detail: {
    marginTop: space(3), color: color.inkDim, fontFamily: font.mono,
    fontSize: 12, textAlign: 'center',
  },
});
