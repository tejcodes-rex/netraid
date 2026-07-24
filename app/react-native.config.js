// Links the Archivo + IBM Plex font files (drop the .ttf files in assets/fonts/
// then run `npx react-native-asset`). Also bundles the .tflite model assets.
module.exports = {
  project: { ios: {}, android: {} },
  assets: ['./assets/fonts/', './assets/models/'],
};
