# NetraID, Integration Guide (Datalake 3.0)

NetraID is built to drop into the existing **Datalake 3.0 React Native** app with no
backend changes for the offline path. This guide covers install → models → permissions →
the 3-line API.

## 0. Integration model: a module, not a second app

Datalake 3.0 is a large app with a great many screens and modules already in it. NetraID
is deliberately shaped so that adding it does not require understanding, touching or
re-testing any of them:

- **One directory, no globals.** Everything lives under `src/netraid/`. No interceptors,
  no navigation side effects, no monkey-patching, no shared singletons with the host app.
  Its only long-lived hook is a single NetInfo subscription owned by `sync.ts` and
  released by `stopSync()`. Deleting the directory and the two call sites removes it
  completely.
- **A 3-function surface** (`init` / `enroll` / `verify`) plus optional drop-in screens.
  The host app decides where attendance sits in its own navigation; NetraID never
  reaches back into it.
- **Its own storage.** A separate SQLCipher database and keychain entry. It does not
  read, migrate or share the host app's tables, so no existing feature's data model
  changes and the two cannot corrupt each other.
- **Additive backend.** The offline path needs no server at all. The sync API is a new,
  independent endpoint; no existing API contract is modified.
- **Camera lifecycle is scoped to the screen.** The frame processor runs only while the
  attendance screen is focused and the app is foregrounded, so no other module's camera,
  CPU or memory behaviour changes when NetraID is not on screen.
- **Flaggable by construction.** Because the module is additive and has exactly two call
  sites, gating it behind the host app's existing feature-flag mechanism is a one-line
  change; it can be enabled for a pilot circle and switched off without rolling back the
  host app.

The one genuinely shared resource is **app size**: four TFLite models, **≈ 17.3 MB**
bundled (see ARCHITECTURE.md §5), and that is the whole cost the rest of the app pays.

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
| `passiveThreshold` | floor on the **median** P(real) across captures | 0.08 (calibrate on pilot) |
| `screenSpoofMax` | ceiling on the P(screen-replay) that **2 of 3** captures reached | 0.5 (**calibrate first**) |
| `screenSpoofMode` | screen-replay class: `off` / `report` / `enforce` | `report` |
| `numChallenges` | active liveness steps | 2 |
| `challengeTimeoutMs` | per-step reaction window; a miss restarts the whole attempt | 6000 |
| `chromaMode` | screen-illumination challenge: `off` / `report` / `enforce` | `report` |
| `chromaSlots` / `chromaSlotMs` | flash slots and their duration | 5 / 360 |
| `chromaSettleMs` | capture hold after the flash, for AWB/AE re-convergence | 700 |
| `chromaThreshold` | minimum reflection response when armed | 0.012 (**calibrate first**) |
| `enrollSpoofMode` | passive gate at **enrollment**: `off` / `report` / `enforce` | `report` |
| `enrollPassiveThreshold` | floor on the median P(real) across the enrollment burst | 0.08 (**calibrate first**) |

> `chromaMode`, `screenSpoofMode` and `enrollSpoofMode` all ship as `report`: each one
> runs and records its score on every attempt, but rejects no one until you have read its
> operating point off your own hardware. Release builds print the per-attempt readings on
> screen and emit one machine-readable line per attempt, so a whole run comes off the
> handset with `adb logcat -s ReactNativeJS:V | grep NETRAID_CALIB`. Follow
> `CALIBRATION.md` and arm all three. A threshold copied from another device is not a
> security control, and `enrollSpoofMode` in particular must be measured against the
> enrollment path rather than borrowed from the verification one: the two see different
> capture distributions and the borrowed number rejects genuine enrollments.

## 7. Delegate selection
All four models load on the **CPU delegate** via `react-native-fast-tflite`: on the target
mid-range hardware the GPU/NNAPI delegates failed to load some graphs (and int8 emitted NaN,
fp16 stalled, see `BENCHMARKS.md` §4), while the CPU path is robust and already meets the
< 1 s budget with margin. No high-end GPU is required anywhere.
