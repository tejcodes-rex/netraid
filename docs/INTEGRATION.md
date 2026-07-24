# NetraID, Integration Guide (Datalake 3.0)

NetraID is built to drop into the existing **Datalake 3.0 React Native** app with no
backend changes for the offline path. This guide covers install → models → permissions →
the 3-line API.

## 1. Install
```bash
npm install react-native-vision-camera react-native-worklets-core \
  vision-camera-resize-plugin react-native-fast-tflite \
  @op-engineering/op-sqlite react-native-mmkv react-native-keychain \
  @react-native-community/netinfo react-native-get-random-values react-native-uuid
cd ios && pod install && cd ..   # iOS only
```
Add the worklets plugin to `babel.config.js` (see `app/babel.config.js`) and enable
SQLCipher for op-sqlite in `package.json`:
```json
"op-sqlite": { "sqlcipher": true }
```

## 2. Stage the models
The real `.tflite` files are produced by the ML pipeline and copied into the bundle:
```bash
cd ml && python scripts/01_download_models.py && python scripts/02_convert_to_tflite.py
python scripts/04_export_app_models.py     # -> app/assets/models/*.tflite
```
`react-native.config.js` already lists `assets/models/` and `assets/fonts/`; run:
```bash
npx react-native-asset
```

## 3. Permissions and minimum OS
- **Android** `AndroidManifest.xml`: `<uses-permission android:name="android.permission.CAMERA"/>`
- **iOS** `Info.plist`: `NSCameraUsageDescription` → "Used for offline attendance verification."
- **Android**: `minSdkVersion 26` (Android 8.0), exactly the brief floor.
- **iOS**: the recognition + liveness logic and the int8 models carry no iOS version
  dependency and run on iOS 12. The practical floor is set by the host app's React Native
  toolchain: this reference app pins RN 0.76 (Xcode floor iOS 15.1), and the camera frame
  processor (vision-camera v4) needs iOS 13+. To ship against an iOS 12 device target, embed
  the NetraID module in a Datalake host built on an RN version whose floor is iOS 12 (for
  example RN 0.71, which supported iOS 12.4); the module sources compile unchanged.

## 4. Use it, 3 lines
```tsx
import { NetraID, DuplicateFaceError } from './netraid';

await NetraID.init();                                    // open encrypted DB + start sync
await NetraID.enroll({ personId: 'NHAI-04821', captures });   // 6-shot, outlier-rejected
const r = await NetraID.verify({ captures, requireLiveness: true }); // 3-frame aggregate
// r -> { ok, personId, score, liveness, elapsedMs }
```
`captures` come from the frame processor (`useNetraFrameProcessor`), which only emits
quality-gated, liveness-passed, aligned 112×112 crops with a sharpness score. `enroll`
throws `DuplicateFaceError` if the face is already enrolled under a different ID (one face,
one identity). Or mount the ready-made screens (`HomeScreen`, `EnrollScreen`,
`VerifyScreen`, `PipelineDemoScreen`) directly into your navigator, they already use the
shared design system in `src/theme.ts`.

## 5. Backend (optional, additive)
Deploy the sync API (see `backend/README.md`) and point the app at it:
```bash
NETRAID_API_BASE=https://<api-id>.execute-api.ap-south-1.amazonaws.com
```
The device works fully offline without this; sync just drains the queue when online.

## 6. Tuning knobs (`src/netraid/types.ts → DEFAULT_CONFIG`)
| Field | Meaning | Default |
|---|---|---|
| `matchThreshold` | cosine acceptance (median of winning frames) | 0.38 (calibrate on pilot) |
| `matchMargin` | required gap over the best different person | 0.08 |
| `verifyShots` | frames aggregated per verification | 3 |
| `enrollShots` | candidate shots per enrollment | 6 |
| `passiveThreshold` | passive anti-spoof acceptance | 0.0 (active challenge is the primary gate) |
| `numChallenges` | active liveness steps | 2 |
| `challengeTimeoutMs` | per-step budget; timeout re-issues a fresh random set | 5000 |

## 7. Delegate selection
All four models load on the **CPU delegate** via `react-native-fast-tflite`: on the target
mid-range hardware the GPU/NNAPI delegates failed to load some graphs (and int8 emitted NaN,
fp16 stalled, see `BENCHMARKS.md` §4), while the CPU path is robust and already meets the
< 1 s budget with margin. No high-end GPU is required anywhere.
