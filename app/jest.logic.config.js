// Unit tests for the pure on-device algorithms (matching, liveness, alignment).
// These run in plain Node, no native modules, no React Native runtime. We bypass
// the project babel.config.js (which pulls the worklets plugin used only on device)
// and transform TypeScript with the React Native preset alone.
module.exports = {
  testEnvironment: 'node',
  roots: ['<rootDir>/src/netraid/__tests__'],
  testMatch: ['**/*.test.ts'],
  transform: {
    '^.+\\.tsx?$': [
      'babel-jest',
      { configFile: false, presets: ['@react-native/babel-preset'] },
    ],
  },
};
