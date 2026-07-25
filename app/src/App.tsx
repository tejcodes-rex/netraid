import React from 'react';
import { StatusBar } from 'react-native';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { color, font } from './theme';
import { HomeScreen } from './screens/HomeScreen';
import { VerifyScreen } from './screens/VerifyScreen';
import { EnrollScreen } from './screens/EnrollScreen';
import { PipelineDemoScreen } from './screens/PipelineDemoScreen';
import { ErrorBoundary } from './components/ErrorBoundary';

const Stack = createNativeStackNavigator();

const navTheme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    background: color.bg,
    card: color.panel,
    text: color.ink,
    border: color.line,
    primary: color.green,
  },
};

// Deep links: lets tooling (and the iOS CI demo recording) open a screen
// directly, e.g. `xcrun simctl openurl booted netraid://demo`.
const linking = {
  prefixes: ['netraid://'],
  config: {
    // Deep-linked screens stack ON TOP of Home, so the back button returns to
    // the dashboard instead of exiting the app.
    initialRouteName: 'Home' as const,
    screens: {
      Home: '',
      Verify: 'verify',
      Enroll: 'enroll',
      PipelineDemo: 'demo',
    },
  },
};

export default function App(props: { demo?: boolean }) {
  // `demo` is set by the iOS AppDelegate when the app is launched with "--demo"
  // (the CI simulator run): open the camera-free Pipeline Demo directly.
  const initialRoute = props?.demo ? 'PipelineDemo' : 'Home';
  return (
    <SafeAreaProvider>
      <ErrorBoundary>
      <StatusBar barStyle="light-content" backgroundColor={color.bg} />
      <NavigationContainer theme={navTheme} linking={linking}>
        <Stack.Navigator
          initialRouteName={initialRoute}
          screenOptions={{
            headerStyle: { backgroundColor: color.bg },
            headerTitleStyle: { fontFamily: font.display, color: color.ink },
            headerTintColor: color.green,
            contentStyle: { backgroundColor: color.bg },
          }}>
          <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
          <Stack.Screen name="Verify" component={VerifyScreen} options={{ title: 'Verify' }} />
          <Stack.Screen name="Enroll" component={EnrollScreen} options={{ headerShown: false }} />
          <Stack.Screen name="PipelineDemo" component={PipelineDemoScreen} options={{ title: 'Pipeline Demo' }} />
        </Stack.Navigator>
      </NavigationContainer>
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}
