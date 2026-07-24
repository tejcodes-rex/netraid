// NetraID design system, shared with the web Command Center.
// "Highway Control Room": near-black green-slate, highway-sign signal green,
// amber for spoof warnings, IBM Plex / Archivo type. Single source of truth.

export const color = {
  bg: '#070C0A',
  bg2: '#0A120E',
  panel: '#0E1714',
  panel2: '#101B16',
  line: '#1C2A24',
  line2: '#26382F',
  ink: '#E9F1ED',
  inkDim: '#8DA298',
  inkFaint: '#5E726A',
  green: '#1FD27A',
  greenDeep: '#0C7C49',
  amber: '#F4B740',
  red: '#FF5D5D',
  cyan: '#3BD6C6',
  onGreen: '#04140C',
} as const;

// 4-pt spacing scale.
export const space = (n: number) => n * 4;

export const radius = { sm: 10, md: 14, lg: 20, pill: 999 } as const;

// Fonts must be linked via react-native.config.js + `npx react-native-asset`.
// Display: Archivo (700/800/900) · Body: IBM Plex Sans · Mono: IBM Plex Mono.
export const font = {
  display: 'Archivo-Bold',
  displayHeavy: 'Archivo-Black',
  body: 'IBMPlexSans-Regular',
  bodyMed: 'IBMPlexSans-Medium',
  bodySemi: 'IBMPlexSans-SemiBold',
  mono: 'IBMPlexMono-Regular',
  monoMed: 'IBMPlexMono-Medium',
} as const;

export const type = {
  h1: { fontFamily: font.displayHeavy, fontSize: 30, letterSpacing: -0.6, color: color.ink },
  h2: { fontFamily: font.display, fontSize: 20, letterSpacing: -0.3, color: color.ink },
  title: { fontFamily: font.bodySemi, fontSize: 16, color: color.ink },
  body: { fontFamily: font.body, fontSize: 14, color: color.ink },
  dim: { fontFamily: font.body, fontSize: 13, color: color.inkDim },
  mono: { fontFamily: font.mono, fontSize: 12, color: color.inkDim, letterSpacing: 0.4 },
  label: {
    fontFamily: font.mono, fontSize: 10, color: color.inkFaint,
    letterSpacing: 1.6, textTransform: 'uppercase' as const,
  },
  metric: { fontFamily: font.displayHeavy, fontSize: 34, letterSpacing: -0.8, color: color.ink },
} as const;

export const shadow = {
  card: {
    shadowColor: '#000', shadowOpacity: 0.5, shadowRadius: 24,
    shadowOffset: { width: 0, height: 14 }, elevation: 10,
  },
} as const;
