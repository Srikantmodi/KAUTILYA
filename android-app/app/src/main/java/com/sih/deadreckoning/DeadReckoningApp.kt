/**
 * DeadReckoningApp.kt — Application subclass for one-time SDK initialization
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 1 (App Shell, UI & Navigation Frontend)
 *
 * MapLibre requires MapLibre.getInstance(context) before any MapView is created.
 * This Application class ensures it's called exactly once at app startup.
 *
 * Referenced in AndroidManifest.xml as android:name=".DeadReckoningApp"
 */
package com.sih.deadreckoning

import android.app.Application
import android.util.Log
import org.maplibre.android.MapLibre

class DeadReckoningApp : Application() {
    override fun onCreate() {
        super.onCreate()
        MapLibre.getInstance(this)
        Log.i("DeadReckoningApp", "MapLibre SDK initialized")
    }
}
