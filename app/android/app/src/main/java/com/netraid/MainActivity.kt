package com.netraid

import android.os.Bundle
import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.fabricEnabled
import com.facebook.react.defaults.DefaultReactActivityDelegate

class MainActivity : ReactActivity() {

  /**
   * Returns the name of the main component registered from JavaScript. This is used to schedule
   * rendering of the component.
   */
  override fun getMainComponentName(): String = "netraid"

  /**
   * Returns the instance of the [ReactActivityDelegate]. We use [DefaultReactActivityDelegate]
   * which allows you to enable New Architecture with a single boolean flags [fabricEnabled]
   */
  override fun createReactActivityDelegate(): ReactActivityDelegate =
      DefaultReactActivityDelegate(this, mainComponentName, fabricEnabled)

  /**
   * Discard any saved fragment state on restore.
   *
   * react-native-screens holds the navigation stack as fragments and refuses to
   * be restored from a Bundle: its ScreenFragment constructor throws
   * IllegalStateException("Screen fragments should never be restored"). Android
   * hands a saved state back whenever it recreates this activity, which happens
   * routinely after the process is killed in the background, and this handset
   * class kills background apps aggressively. The result was a crash to a blank
   * screen on reopening the app, with no user action able to recover it.
   *
   * Passing null makes the activity rebuild its navigation from scratch, which
   * is correct here: every screen's state is either transient (a live camera
   * session) or already persisted in the encrypted store.
   */
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(null)
  }
}
