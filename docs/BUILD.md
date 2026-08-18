# NetraID, Build and Deploy (Android)

This documents exactly how the shipped APK was produced and how to reproduce it, including
the verified artifacts and one Windows-specific gotcha.

## Result (verified)

- **`dist/NetraID-android.apk`**, a signed, standalone Android build (50.8 MB, arm64-v8a).
- It is fully offline: the JavaScript bundle (`assets/index.android.bundle`) and all four
  models are packaged inside the APK, so it runs with no Metro server and no network.
- Models embedded (seen with `unzip -l` on the APK, stored under `res/`):
  BlazeFace 0.22 MB, FaceLandmarker 2.44 MB, MobileFaceNet **float32 13.0 MB**, MiniFASNet
  float32 1.68 MB. That is the ≈ 17.3 MB on-device stack quoted in `BENCHMARKS.md`, now physically
  inside the package. (float32 recognition is shipped because int8/fp16 failed on the target
  mid-range device, see `BENCHMARKS.md` §4.)
- 28 native libraries compiled from source, including `libVisionCameraTflite`,
  `libtensorflowlite_jni`, `libtensorflowlite_gpu_jni`, `libop-sqlite`, `librnworklets`,
  `libhermes`.
- Manifest: `package com.netraid`, `minSdkVersion 26` (Android 8.0), `targetSdkVersion 34`.
- Signed with the standard Android debug keystore (`android/app/debug.keystore`), which is the
  normal signing path for a prototype. For Play distribution, generate a release keystore and
  point `signingConfigs.release` at it (`android/app/build.gradle`).

## Toolchain

- JDK 17 (Temurin 17.0.19), portable, no admin needed.
- Android SDK: platform `android-35`, build-tools `35.0.0`, platform-tools, NDK
  `26.1.10909125`, CMake `3.22.1`.
- Gradle `8.10.2` (via the wrapper), React Native `0.76.5`.

## One-time setup

```bash
# 1) JS dependencies
cd app && npm install --legacy-peer-deps

# 2) Stage the on-device models into app/assets/models (real .tflite files)
cd ../ml && python scripts/01_download_models.py && python scripts/02_convert_to_tflite.py
python scripts/04_export_app_models.py

# 3) Point Gradle at your SDK
echo "sdk.dir=/absolute/path/to/android-sdk" > app/android/local.properties
```

## Build

```bash
cd app/android
# Debug (loads JS from Metro over USB; good for development):
./gradlew assembleDebug -PreactNativeArchitectures=arm64-v8a
# Standalone release (self-contained, offline, models + JS bundled):
./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
# Release for every supported ABI (this is what ships):
./gradlew assembleRelease
```

Outputs land in `app/android/app/build/outputs/apk/{debug,release}/`.

### Building on Windows

Nothing extra to do: the native build stages itself at a short path
automatically. What follows is the reason, because the failures it prevents are
badly misreported and cost real time to diagnose.

Windows caps path APIs at **260 characters** for files and **248** for creating
a directory. A stock React Native 0.76 project passes both on its own from a
normally nested checkout, and produces two different errors, neither of which
mentions path length:

| Symptom | Actual cause |
|---|---|
| `ninja: error: mkdir(RNVisionCameraResizePluginSpec_autolinked_build/...): No such file or directory` | The codegen object path for an autolinked module reaches **306** characters. Reads like missing state; is not. |
| `ninja: error: manifest 'build.ninja' still dirty after 100 tries`, only for `armeabi-v7a` | ninja checks a prefab config through the literal relative path `<abi>/../prefab/<abi>/...`. For `armeabi-v7a` that string is **263** characters so the check fails, and ninja treats the file as an input to `build.ninja` it can never satisfy. The same path for `arm64-v8a` is **259** and builds cleanly, which is why the failure looks ABI-specific and intermittent. |

Cleaning fixes neither, and repeated cleaning makes the second one look like
corrupted state. `app/android/settings.gradle` stages every module's native
build under `<drive>:\nxb\<checkout-key>` on Windows, which puts every
generated path back inside the limit and, usefully, makes the build independent
of how deeply the repository was cloned. macOS, Linux and CI keep the stock
layout. Override the location with `-PnetraidCxxRoot=<path>`.

## Deploy to a phone

```bash
adb install -r dist/NetraID-android.apk      # or the release apk path above
# For the debug variant during development:
cd app && npx react-native run-android        # builds, installs, starts Metro
```

Then on the device: grant the camera permission, enroll a person (multi-shot, the app stores
only the embedding), and verify (perform the random blink / smile / head-turn challenge). All
inference is on-device.

## Dependency notes (RN 0.76 specifics that bit us, now pinned)

- `@react-native-community/cli` and `@react-native/metro-config` are required dev dependencies
  on RN 0.76 (the CLI was split out of core). Without the CLI, autolinking cannot find the
  Android package name; without metro-config, release JS bundling fails.
- `react-native-screens` is pinned to `4.4.0`. Newer 4.2x releases use prop types that RN
  0.76's codegen cannot parse (`Unknown prop type ...`).
- `op-sqlite` is configured with `{ "sqlcipher": true }` in `package.json` so the local
  database is AES encrypted at rest.
- `metro.config.js` adds `tflite` to `assetExts` so the models resolve and get bundled.

## Windows long-path note

On Windows, one autolinked codegen target for `vision-camera-resize-plugin` produces a very
deep build path. A release build (the `RelWithDebInfo` folder adds characters) can exceed the
260-character `MAX_PATH` limit when the repository sits at a deep location, and `ninja` fails
with `mkdir(...): No such file or directory`. Debug builds stay just under the limit.

Fix: build from a short real path. Copying the project to, for example, `C:\nx` and building
from `C:\nx\app\android` resolves it. A `subst` virtual drive or a junction does **not** work,
because the React Native codegen calls `realpath`, which resolves the alias back to the long
path and then fails with "this and base files have different roots". `LongPathsEnabled=1`
alone is not sufficient for the NDK `ninja` shipped with r26.
