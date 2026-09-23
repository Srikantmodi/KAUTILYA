/**
 * MainActivity.kt — Single-activity host for the dead reckoning app
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 1 (App Shell, UI & Navigation Frontend)
 *
 * This is the app entry point. It is purely presentational per §2.2:
 *   - Handles runtime permissions (location)
 *   - Manages fragment navigation: CalibrationScreen → MapFragment
 *   - Instantiates bridge objects (FusionBridge, ModeManager, etc.)
 *   - Does NOT orchestrate the sensor → calibration → inference → fusion
 *     data pipeline — that wiring is the responsibility of the pipeline
 *     coordinator (out of Member 1's scope per the task card).
 *
 * References:
 *   PRD §2.2  — "Frontend... thin, purely presentational"
 *   PRD §3.1  — MapLibre, not Google Maps
 *   PRD §7 guardrail #10 — nothing on the live nav path depends on network
 */
package com.sih.deadreckoning

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.sih.deadreckoning.calibration.AlignmentManager
import com.sih.deadreckoning.fusion.FusionBridge
import com.sih.deadreckoning.fusion.NavState
import com.sih.deadreckoning.mode.ModeManager
import com.sih.deadreckoning.mode.NavigationMode
import com.sih.deadreckoning.ui.CalibrationScreen
import com.sih.deadreckoning.ui.MapFragment
import com.sih.deadreckoning.ui.TelemetryOverlay

/**
 * Single-activity host managing permission flow and fragment navigation.
 *
 * Flow:
 *   1. Check/request location permissions
 *   2. Show [CalibrationScreen] for device alignment
 *   3. On calibration complete, transition to [MapFragment] + [TelemetryOverlay]
 *
 * Bridge objects (FusionBridge, ModeManager, AlignmentManager) are instantiated
 * here and injected into fragments, but this activity does NOT wire the data
 * pipeline itself. That's the coordinator's job (not Member 1's scope).
 */
class MainActivity : AppCompatActivity(), CalibrationScreen.CalibrationListener {

    companion object {
        private const val TAG = "MainActivity"

        /**
         * Fragment tag constants for fragment manager lookups.
         */
        private const val TAG_CALIBRATION = "calibration_screen"
        private const val TAG_MAP = "map_fragment"
    }

    /** Stable container ID — generated once at class load. */
    private val fragmentContainerId = View.generateViewId()

    /* ── Bridge instances (owned by Activity, injected into fragments) ──── */

    private lateinit var fusionBridge: FusionBridge
    private lateinit var modeManager: ModeManager
    private lateinit var alignmentManager: AlignmentManager

    /* ── UI ───────────────────────────────────────────────────────────────── */

    private var telemetryOverlay: TelemetryOverlay? = null
    private lateinit var rootContainer: FrameLayout

    /* ── Permission handling ──────────────────────────────────────────────── */

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fineGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        val coarseGranted = permissions[Manifest.permission.ACCESS_COARSE_LOCATION] ?: false

        if (fineGranted) {
            Log.i(TAG, "Fine location permission granted")
            onPermissionsReady()
        } else if (coarseGranted) {
            Log.w(TAG, "Only coarse location granted — fine is required for GNSS")
            Toast.makeText(
                this,
                "Fine location required for accurate navigation",
                Toast.LENGTH_LONG
            ).show()
            // Proceed anyway — GnssQualityMonitor will handle degraded accuracy
            onPermissionsReady()
        } else {
            Log.e(TAG, "Location permission denied")
            Toast.makeText(
                this,
                "Location permission is required for navigation",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /* ── Activity lifecycle ───────────────────────────────────────────────── */

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Create the root FrameLayout container programmatically
        rootContainer = FrameLayout(this).apply {
            id = fragmentContainerId
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        setContentView(rootContainer)

        // Instantiate bridges
        fusionBridge = FusionBridge()
        modeManager = ModeManager()
        alignmentManager = AlignmentManager()

        Log.i(TAG, "Bridges instantiated: FusionBridge, ModeManager, AlignmentManager")

        // Check permissions and proceed
        if (savedInstanceState == null) {
            checkAndRequestPermissions()
        }
    }

    private fun checkAndRequestPermissions() {
        val fineLocation = ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_FINE_LOCATION
        )

        if (fineLocation == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Location permission already granted")
            onPermissionsReady()
        } else {
            Log.i(TAG, "Requesting location permissions")
            locationPermissionLauncher.launch(
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                )
            )
        }
    }

    /**
     * Called once location permissions are granted. Starts the calibration flow.
     */
    private fun onPermissionsReady() {
        showCalibrationScreen()
    }

    /* ── Fragment navigation ──────────────────────────────────────────────── */

    private fun showCalibrationScreen() {
        val fragment = CalibrationScreen.newInstance().apply {
            setAlignmentManager(alignmentManager)
            setCalibrationListener(this@MainActivity)
        }

        supportFragmentManager.beginTransaction()
            .replace(fragmentContainerId, fragment, TAG_CALIBRATION)
            .commit()

        Log.i(TAG, "Showing CalibrationScreen")
    }

    /**
     * Called by CalibrationScreen when yaw calibration is complete.
     * Transitions to the MapFragment with TelemetryOverlay.
     */
    override fun onCalibrationComplete() {
        Log.i(TAG, "Calibration complete — transitioning to map")
        showMapScreen()
    }

    private fun showMapScreen() {
        // Create the telemetry overlay
        telemetryOverlay = TelemetryOverlay(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.TOP or Gravity.START
                topMargin = 48
                leftMargin = 16
            }
            elevation = 8f
        }

        // Create nav state provider backed by real bridges
        val navStateProvider = object : MapFragment.NavStateProvider {
            override fun getNavState(): NavState? {
                return try {
                    fusionBridge.getNavState()
                } catch (e: UnsatisfiedLinkError) {
                    // C++ not linked yet — fall back to null (MapFragment uses mock)
                    null
                }
            }

            override fun getCurrentMode(): NavigationMode {
                return modeManager.getCurrentMode()
            }

            // Session origin — defaults until first real GNSS fix sets it
            private var originLat = 28.6139
            private var originLon = 77.2090

            override fun getSessionOriginLat(): Double = originLat
            override fun getSessionOriginLon(): Double = originLon
        }

        val mapFragment = MapFragment.newInstance().apply {
            setNavStateProvider(navStateProvider)
            setTelemetryOverlay(telemetryOverlay!!)
        }

        supportFragmentManager.beginTransaction()
            .replace(fragmentContainerId, mapFragment, TAG_MAP)
            .commit()

        // Add the telemetry overlay on top of the fragment container
        rootContainer.addView(telemetryOverlay)

        Log.i(TAG, "Showing MapFragment + TelemetryOverlay")
    }

    /**
     * Allow skipping calibration for development/testing.
     * Called if the user long-presses during calibration (or via debug menu).
     */
    fun skipCalibration() {
        Log.w(TAG, "Calibration skipped — using mock/default values")
        showMapScreen()
    }

    override fun onDestroy() {
        telemetryOverlay = null
        super.onDestroy()
    }
}
