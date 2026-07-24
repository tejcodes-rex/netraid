const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * The on-device models are bundled as binary assets, so .tflite must be a
 * recognized asset extension for require('../assets/models/*.tflite') to resolve
 * and for the files to be copied into release builds.
 *
 * @type {import('metro-config').MetroConfig}
 */
const defaultConfig = getDefaultConfig(__dirname);

const config = {
  resolver: {
    assetExts: [...defaultConfig.resolver.assetExts, 'tflite', 'bin'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
