package com.netraid

import android.view.WindowManager
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

/**
 * Drives the screen as a lamp for the chroma liveness challenge.
 *
 * The challenge asks the face to reflect light the phone chooses at
 * verification time, which only works if the screen actually emits enough of
 * it. Measured on the target handset: with the system brightness at 12 of 255,
 * roughly 5 percent, the face's colour response to a full-screen flash was
 * 0.002 in chromaticity units, indistinguishable from noise, and the barrier
 * meant to stop a recorded replay could not be armed at all. A field device at
 * dusk sits exactly there.
 *
 * This raises the brightness of THIS WINDOW only, for the ~1.8 s the flash is
 * up, and hands control straight back. It writes no system setting and needs no
 * permission: WindowManager.LayoutParams.screenBrightness is a per-window
 * override that Android drops when the window loses focus, so a crash or a
 * backgrounded app cannot leave a handset stuck at full brightness draining its
 * battery in the field.
 */
class ScreenLampModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

  override fun getName() = "ScreenLamp"

  /**
   * @param level 0.0..1.0, or a negative value to hand brightness back to the
   *   system (WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE).
   */
  @ReactMethod
  fun setBrightness(level: Double) {
    val activity = currentActivity ?: return
    activity.runOnUiThread {
      val window = activity.window ?: return@runOnUiThread
      val params: WindowManager.LayoutParams = window.attributes
      params.screenBrightness =
          if (level < 0) WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
          else level.coerceIn(0.0, 1.0).toFloat()
      window.attributes = params
    }
  }
}
