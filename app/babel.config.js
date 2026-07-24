module.exports = {
  presets: ['@react-native/babel-preset'],
  plugins: [
    // Required by react-native-vision-camera frame processors + worklets.
    // Must be listed LAST.
    ['react-native-worklets-core/plugin'],
  ],
};
