/**
 * Calibration telemetry.
 *
 * Every gate in this module ships with a three-state `GateMode`, and a gate can
 * only move from `report` to `enforce` once its operating point has been read
 * off the hardware the deployment will actually run on. That reading has to come
 * from somewhere, and a release APK has no Metro console attached.
 *
 * So each attempt emits one machine-readable line, in release builds as well as
 * debug. `adb logcat -s ReactNativeJS:V | grep NETRAID_CALIB` pulls a whole
 * calibration run off a handset over nothing but a USB cable, which is what
 * makes the procedure in docs/CALIBRATION.md something an operator can follow
 * instead of transcribing dozens of numbers off a phone screen by hand.
 *
 * Nothing here is PII: the payload carries model scores and timings, never an
 * embedding, an image, or a person id.
 */
export const CALIB_TAG = 'NETRAID_CALIB';

/** Round to 4 dp so a log line stays readable without losing gate resolution. */
export const r4 = (n: number) => Number(n.toFixed(4));

export function logCalibration(payload: Record<string, unknown>) {
  console.log(CALIB_TAG, JSON.stringify(payload));
}
