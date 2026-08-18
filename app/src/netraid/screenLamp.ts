import { NativeModules, Platform } from 'react-native';

/**
 * The screen, used as a lamp for the chroma liveness challenge.
 *
 * The challenge measures how a face reflects light this phone chooses at
 * verification time. That requires the screen to actually emit some. Measured
 * on the target handset with system brightness at 12/255 (~5%, which is where a
 * phone sits at dusk), the face's response to a full-screen flash was 0.002 in
 * chromaticity units: noise, and not enough to arm the barrier on.
 *
 * Brightness is raised for the duration of the flash only and handed straight
 * back. It is a per-window override, so no system setting changes and Android
 * releases it if the app loses focus.
 */
const Lamp = NativeModules.ScreenLamp as
  | { setBrightness: (level: number) => void }
  | undefined;

/** Full brightness while the flash is up. */
export function lampOn() {
  if (Platform.OS === 'android') Lamp?.setBrightness(1.0);
}

/** Hand brightness back to the system. Safe to call when it was never raised. */
export function lampOff() {
  if (Platform.OS === 'android') Lamp?.setBrightness(-1);
}
