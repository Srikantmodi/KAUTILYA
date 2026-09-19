package com.sih.deadreckoning.calibration

import com.sih.deadreckoning.sensor.ImuSample
import android.util.Log
import kotlin.math.sqrt
import kotlin.math.acos

/**
 * DeviceOrientationDetector — detects phone remount or dislodgement mid-drive.
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 3 (Calibration & Device Alignment)
 *
 * Monitors raw ImuSample streams for:
 *  1. Angular jerk spikes (sudden rotation → phone was picked up or mount slipped)
 *  2. Sustained gravity vector shifts (phone remounted at a new angle)
 *
 * When either condition is detected, calls [onRemountDetected] so the
 * caller (typically AlignmentManager) can reset calibration state.
 *
 * Usage:
 *   val detector = DeviceOrientationDetector {
 *       alignmentManager.reset()
 *   }
 *   sensorBridge.start { sample ->
 *       alignmentManager.feedSample(sample)
 *       detector.onSample(sample)
 *   }
 */
class DeviceOrientationDetector(
    private val onRemountDetected: () -> Unit
) {
    companion object {
        private const val TAG = "DeviceOrientationDetector"

        /** Angular velocity spike threshold for jerk detection (rad/s) */
        const val JERK_THRESHOLD_RADS = 1.5f

        /** Number of jerk spikes within JERK_WINDOW_MS to trigger detection */
        const val JERK_COUNT_THRESHOLD = 5

        /** Rolling window for counting jerk spikes (ms) */
        const val JERK_WINDOW_MS = 500L

        /** Angle change threshold for gravity-direction-shift detection (radians) */
        const val GRAVITY_SHIFT_THRESHOLD_RAD = 0.26f  // ~15°

        /** Sustained duration for gravity shift (ms) before detection triggers */
        const val GRAVITY_SHIFT_SUSTAINED_MS = 2000L

        /** Low-pass filter coefficient for gravity EMA */
        const val EMA_ALPHA = 0.05f

        /** Minimum samples before shift detection activates (avoid false positives at start) */
        const val MIN_SAMPLES_BEFORE_DETECTION = 100
    }

    // ─── Jerk detection state ─────────────────────────────────────────────
    private val jerkTimestamps = ArrayDeque<Long>()

    // ─── Gravity shift detection state ────────────────────────────────────
    private var gravityEma = floatArrayOf(0f, 0f, -9.80665f)
    private var referenceGravity = floatArrayOf(0f, 0f, -9.80665f)
    private var shiftStartMs: Long = -1L
    private var sampleCount: Int = 0
    private var gravityInitialized = false

    /**
     * Process one raw ImuSample. Call this for every sample in the IMU stream.
     * Thread-safe via caller ensuring single-threaded access (same HandlerThread
     * as SensorBridge). Does NOT require locking.
     */
    fun onSample(sample: ImuSample) {
        sampleCount++

        // ── 1. Jerk detection ──────────────────────────────────────────────
        val gyroMag = sqrt(
            sample.gx * sample.gx +
            sample.gy * sample.gy +
            sample.gz * sample.gz
        )

        if (gyroMag > JERK_THRESHOLD_RADS) {
            jerkTimestamps.addLast(sample.timestampMs)
        }

        // Remove timestamps older than JERK_WINDOW_MS
        val cutoff = sample.timestampMs - JERK_WINDOW_MS
        while (jerkTimestamps.isNotEmpty() && jerkTimestamps.first() < cutoff) {
            jerkTimestamps.removeFirst()
        }

        if (jerkTimestamps.size >= JERK_COUNT_THRESHOLD) {
            jerkTimestamps.clear()
            triggerRemount("Jerk spike count exceeded threshold (${JERK_COUNT_THRESHOLD} in ${JERK_WINDOW_MS}ms)")
            return
        }

        // ── 2. Gravity direction shift detection ───────────────────────────
        // Update EMA of raw acceleration
        gravityEma[0] = EMA_ALPHA * sample.ax + (1f - EMA_ALPHA) * gravityEma[0]
        gravityEma[1] = EMA_ALPHA * sample.ay + (1f - EMA_ALPHA) * gravityEma[1]
        gravityEma[2] = EMA_ALPHA * sample.az + (1f - EMA_ALPHA) * gravityEma[2]

        // Only activate after enough samples for EMA to stabilize
        if (sampleCount < MIN_SAMPLES_BEFORE_DETECTION) {
            if (sampleCount == MIN_SAMPLES_BEFORE_DETECTION) {
                // Snapshot stable reference
                referenceGravity = gravityEma.copyOf()
                gravityInitialized = true
            }
            return
        }

        if (!gravityInitialized) return

        // Compute angle between current EMA and reference gravity direction
        val angle = angleBetween(gravityEma, referenceGravity)

        if (angle > GRAVITY_SHIFT_THRESHOLD_RAD) {
            if (shiftStartMs < 0L) {
                shiftStartMs = sample.timestampMs
            } else if (sample.timestampMs - shiftStartMs > GRAVITY_SHIFT_SUSTAINED_MS) {
                shiftStartMs = -1L
                // Update reference to new position after remount settles
                referenceGravity = gravityEma.copyOf()
                val deg = Math.toDegrees(angle.toDouble())
                triggerRemount("Sustained gravity direction shift: ${"%.1f".format(deg)} deg")
            }
        } else {
            shiftStartMs = -1L  // condition no longer met — reset timer
        }
    }

    /** Reset detector state (call when calibration is manually reset). */
    fun reset() {
        jerkTimestamps.clear()
        gravityEma = floatArrayOf(0f, 0f, -9.80665f)
        referenceGravity = floatArrayOf(0f, 0f, -9.80665f)
        shiftStartMs = -1L
        sampleCount = 0
        gravityInitialized = false
    }

    // ─── Private helpers ──────────────────────────────────────────────────

    private fun triggerRemount(reason: String) {
        Log.w(TAG, "Remount detected: $reason")
        onRemountDetected()
    }

    /**
     * Compute the angle (radians) between two 3D vectors.
     * Returns value in [0, π].
     */
    private fun angleBetween(a: FloatArray, b: FloatArray): Float {
        val magA = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        val magB = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
        if (magA < 1e-6f || magB < 1e-6f) return 0f
        val dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (magA * magB)
        val dotClamped = dot.coerceIn(-1f, 1f)
        return acos(dotClamped)
    }
}
