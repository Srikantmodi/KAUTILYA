/**
 * CalibrationScreen.kt — Calibration progress UI
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 1 (App Shell, UI & Navigation Frontend)
 *
 * Purely presentational per §2.2 — displays calibration state from
 * Member 3's AlignmentManager, never computes calibration math itself.
 *
 * Shows:
 *   - Pitch / Roll / Yaw angle gauges
 *   - Yaw calibration progress indicator
 *   - Instructional text ("Place phone flat and drive straight")
 *   - Auto-transitions to MapFragment when calibration is complete
 *
 * References:
 *   PRD §6.2   — calibration pipeline
 *   api-contracts.md §3  — CalibrationState / RotationEstimate
 */
package com.sih.deadreckoning.ui

import android.animation.ValueAnimator
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.sih.deadreckoning.calibration.AlignmentManager

/**
 * Fragment displaying the device calibration progress.
 *
 * Shown on app startup before navigation begins. Reads calibration
 * state from [AlignmentManager] and displays pitch/roll/yaw values
 * with a progress indicator for yaw calibration.
 *
 * Once yaw is calibrated ([RotationEstimate.yawCalibrated] == true),
 * signals the host activity to transition to the map view.
 *
 * This fragment contains ZERO calibration math. It only reads the
 * [RotationEstimate] data class and formats it for display.
 */
class CalibrationScreen : Fragment() {

    companion object {
        private const val TAG = "CalibrationScreen"

        /** Polling interval for calibration state (ms). */
        private const val POLL_INTERVAL_MS = 200L

        fun newInstance(): CalibrationScreen = CalibrationScreen()
    }

    /**
     * Callback interface for signaling calibration completion to the host.
     * Implemented by MainActivity to trigger fragment transition.
     */
    interface CalibrationListener {
        fun onCalibrationComplete()
    }

    /* ── External dependencies ───────────────────────────────────────────── */

    private var alignmentManager: AlignmentManager? = null
    private var calibrationListener: CalibrationListener? = null

    fun setAlignmentManager(manager: AlignmentManager) {
        alignmentManager = manager
    }

    fun setCalibrationListener(listener: CalibrationListener) {
        calibrationListener = listener
    }

    /* ── UI elements ─────────────────────────────────────────────────────── */

    private lateinit var titleText: TextView
    private lateinit var instructionText: TextView
    private lateinit var pitchLabel: TextView
    private lateinit var rollLabel: TextView
    private lateinit var yawLabel: TextView
    private lateinit var pitchValue: TextView
    private lateinit var rollValue: TextView
    private lateinit var yawValue: TextView
    private lateinit var yawStatusText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var statusText: TextView

    /* ── Polling state ───────────────────────────────────────────────────── */

    private val handler = Handler(Looper.getMainLooper())
    private var isPolling = false

    private val pollRunnable = object : Runnable {
        override fun run() {
            if (!isPolling) return
            updateCalibrationDisplay()
            handler.postDelayed(this, POLL_INTERVAL_MS)
        }
    }

    /* ── Fragment lifecycle ───────────────────────────────────────────────── */

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val root = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(48, 96, 48, 48)
            setBackgroundColor(Color.parseColor("#1A1A2E"))
        }

        // Title
        titleText = createText("Device Calibration", 28f, true, Color.WHITE).also {
            root.addView(it)
        }
        addSpacer(root, 16)

        // Instruction
        instructionText = createText(
            "Place your phone in the vehicle mount.\nDrive straight for a few seconds.",
            16f, false, Color.parseColor("#B0BEC5")
        ).also {
            it.gravity = Gravity.CENTER
            root.addView(it)
        }
        addSpacer(root, 48)

        // Pitch row
        val pitchRow = createAngleRow("PITCH").also { root.addView(it) }
        pitchLabel = pitchRow.getChildAt(0) as TextView
        pitchValue = pitchRow.getChildAt(1) as TextView
        addSpacer(root, 12)

        // Roll row
        val rollRow = createAngleRow("ROLL").also { root.addView(it) }
        rollLabel = rollRow.getChildAt(0) as TextView
        rollValue = rollRow.getChildAt(1) as TextView
        addSpacer(root, 12)

        // Yaw row
        val yawRow = createAngleRow("YAW").also { root.addView(it) }
        yawLabel = yawRow.getChildAt(0) as TextView
        yawValue = yawRow.getChildAt(1) as TextView
        addSpacer(root, 24)

        // Yaw calibration status
        yawStatusText = createText(
            "⏳ Waiting for yaw calibration...",
            14f, false, Color.parseColor("#FF9800")
        ).also {
            it.gravity = Gravity.CENTER
            root.addView(it)
        }
        addSpacer(root, 24)

        // Progress bar
        progressBar = ProgressBar(
            requireContext(), null,
            android.R.attr.progressBarStyleHorizontal
        ).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            max = 100
            progress = 0
            isIndeterminate = false
        }
        root.addView(progressBar)
        addSpacer(root, 16)

        // Status text
        statusText = createText(
            "Calibrating...", 14f, false, Color.parseColor("#90CAF9")
        ).also {
            it.gravity = Gravity.CENTER
            root.addView(it)
        }

        return root
    }

    override fun onResume() {
        super.onResume()
        startPolling()
    }

    override fun onPause() {
        stopPolling()
        super.onPause()
    }

    /* ── Calibration state display ───────────────────────────────────────── */

    /**
     * Read calibration state from AlignmentManager and update the UI.
     * Called at POLL_INTERVAL_MS by the polling handler.
     *
     * This method ONLY reads and displays — no calibration math.
     */
    private fun updateCalibrationDisplay() {
        val manager = alignmentManager
        if (manager == null) {
            // AlignmentManager not yet injected — show waiting state
            statusText.text = "Waiting for sensor initialization..."
            return
        }

        val rotation = manager.getCurrentRotation()

        // Display angles in degrees for human readability
        val pitchDeg = Math.toDegrees(rotation.pitchRad.toDouble())
        val rollDeg = Math.toDegrees(rotation.rollRad.toDouble())
        val yawDeg = Math.toDegrees(rotation.yawRad.toDouble())

        pitchValue.text = String.format("%.1f°", pitchDeg)
        rollValue.text = String.format("%.1f°", rollDeg)
        yawValue.text = String.format("%.1f°", yawDeg)

        // Color the values based on magnitude (purely visual feedback)
        pitchValue.setTextColor(angleColor(pitchDeg))
        rollValue.setTextColor(angleColor(rollDeg))

        if (rotation.yawCalibrated) {
            // Yaw is calibrated — show success and transition
            yawValue.setTextColor(Color.parseColor("#4CAF50"))
            yawStatusText.text = "✅ Yaw calibrated!"
            yawStatusText.setTextColor(Color.parseColor("#4CAF50"))
            progressBar.progress = 100
            statusText.text = "Calibration complete — starting navigation..."

            // Delay transition slightly so the user sees the "complete" state
            stopPolling()
            handler.postDelayed({
                calibrationListener?.onCalibrationComplete()
            }, 1500)
        } else {
            // Still calibrating — show progress
            yawValue.setTextColor(Color.parseColor("#FF9800"))
            yawStatusText.text = "⏳ Drive straight to calibrate yaw..."

            // Animate progress to show activity (indeterminate-ish).
            // Since we don't know % complete, pulse between 20-80%.
            val elapsed = System.currentTimeMillis() % 4000
            val phase = (elapsed / 4000.0 * 100).toInt().coerceIn(20, 80)
            progressBar.progress = phase

            statusText.text = "Collecting calibration data..."
        }
    }

    /**
     * Map angle magnitude to a color: small = green, medium = amber, large = red.
     * Purely visual — not a threshold check.
     */
    private fun angleColor(degrees: Double): Int {
        val abs = kotlin.math.abs(degrees)
        return when {
            abs < 10.0 -> Color.parseColor("#4CAF50")   // green — small tilt
            abs < 30.0 -> Color.parseColor("#FF9800")   // amber — moderate
            else       -> Color.parseColor("#F44336")   // red — extreme
        }
    }

    /* ── UI helpers ───────────────────────────────────────────────────────── */

    private fun createAngleRow(label: String): LinearLayout {
        return LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )

            val labelTv = createText(label, 16f, true, Color.parseColor("#78909C"))
            labelTv.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            addView(labelTv)

            val valueTv = createText("—°", 20f, true, Color.WHITE)
            valueTv.gravity = Gravity.END
            addView(valueTv)
        }
    }

    private fun createText(
        text: String, sizeSp: Float, bold: Boolean, color: Int
    ): TextView {
        return TextView(requireContext()).apply {
            this.text = text
            textSize = sizeSp
            setTextColor(color)
            if (bold) setTypeface(typeface, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
    }

    private fun addSpacer(parent: LinearLayout, heightDp: Int) {
        val scale = resources.displayMetrics.density
        parent.addView(View(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                (heightDp * scale).toInt()
            )
        })
    }

    /* ── Polling control ─────────────────────────────────────────────────── */

    private fun startPolling() {
        if (isPolling) return
        isPolling = true
        handler.post(pollRunnable)
    }

    private fun stopPolling() {
        isPolling = false
        handler.removeCallbacks(pollRunnable)
    }
}
