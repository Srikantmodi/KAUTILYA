/**
 * SensorBridge.kt — Raw IMU data acquisition
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 2 (Sensor & GNSS Data Acquisition)
 *
 * Registers SensorManager listeners for TYPE_ACCELEROMETER / TYPE_GYROSCOPE,
 * merges latest-of-each into one [ImuSample] per gyro event (gyro is typically
 * the faster/primary sensor clock; accel is merged at its latest value).
 *
 * IMPORTANT — this class emits RAW values only. Do NOT level, filter, bias-
 * correct, or otherwise "clean" the data here.  Calibration (Member 3) and
 * inference (Member 6) apply their own transforms downstream. §2.3 explicitly
 * warns that raw gyro cannot be trusted uncorrected — that's AlignmentManager's
 * responsibility, not ours.
 *
 * Coordinate frame: Android sensor coordinate system (phone body frame).
 * Units: accel → m/s², gyro → rad/s  (Android default).
 * Timestamps: epoch milliseconds (matches imu_types.h / api-contracts.md §0).
 *
 * Threading: sensor callbacks fire on a dedicated [HandlerThread], never the
 * UI thread.  The user-provided callback is invoked on that same thread; if
 * the consumer needs to post to main, that's the consumer's responsibility.
 *
 * References:
 *   PRD §2.1  — canonical data flow (SensorBridge is the first stage)
 *   PRD §6.2  — do NOT pre-process here
 *   api-contracts.md §5  — exact interface surface
 */
package com.sih.deadreckoning.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log

/* ─────────────────────────────────────────────────────────────────────────────
 * ImuSample — Kotlin-side mirror of core-engine ImuSample (imu_types.h)
 *
 * Defined here per the PRD folder structure (no separate data-class file) and
 * api-contracts.md §1 Kotlin-side mirror specification.
 *
 * timestampMs  — epoch milliseconds, NOT boot-relative nanoseconds.
 * ax,ay,az     — raw accelerometer, phone frame, m/s² (includes gravity).
 * gx,gy,gz     — raw gyroscope, phone frame, rad/s.
 * ───────────────────────────────────────────────────────────────────────────── */
data class ImuSample(
    val timestampMs: Long,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float
)

/**
 * Raw IMU sensor bridge.
 *
 * Usage:
 * ```
 * val bridge = SensorBridge(context)
 * bridge.start { sample -> buffer.push(sample) }
 * // ... later
 * bridge.stop()
 * ```
 *
 * @param context  Android [Context] used to obtain the [SensorManager].
 */
class SensorBridge(context: Context) {

    companion object {
        private const val TAG = "SensorBridge"

        /**
         * Sensor delay hint — SENSOR_DELAY_GAME (~50 Hz on most devices) is a
         * reasonable middle-ground between power consumption and meeting the
         * 10 Hz pipeline update rate with plenty of headroom.  The actual
         * delivery rate is device-dependent; we never rely on exact timing.
         */
        private const val SENSOR_DELAY = SensorManager.SENSOR_DELAY_GAME
    }

    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accelSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val gyroSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    /* Latest accelerometer reading — written by accel listener, read by gyro
     * listener.  Access is guarded by @Volatile + the fact that both listeners
     * dispatch on the same single-threaded Handler, so there are no torn reads. */
    @Volatile private var latestAccelValues: FloatArray? = null
    @Volatile private var latestAccelTimestampNs: Long = 0L

    private var sensorThread: HandlerThread? = null
    private var sensorHandler: Handler? = null
    private var callback: ((ImuSample) -> Unit)? = null
    @Volatile private var running = false

    /**
     * Offset to convert [SensorEvent.timestamp] (nanoseconds since boot, may
     * or may not include deep-sleep time depending on OEM) to epoch ms.
     * Computed once at [start] time:
     *   offset = System.currentTimeMillis() - elapsedRealtimeMs
     * and then applied to every event.
     *
     * This is not perfectly accurate (event delivery latency introduces a few
     * ms of jitter) but matches imu_types.h's epoch-ms convention and is the
     * standard pragmatic approach for Android sensor timestamps.
     */
    private var bootToEpochOffsetMs: Long = 0L

    /* ── Listener implementations ──────────────────────────────────────────── */

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            // Store latest accel; a merged ImuSample will be emitted on the
            // next gyro event.
            latestAccelValues = event.values.copyOf()
            latestAccelTimestampNs = event.timestamp
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
            Log.d(TAG, "Accelerometer accuracy changed: $accuracy")
        }
    }

    private val gyroListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val accel = latestAccelValues ?: return  // no accel yet — skip

            val timestampMs = nanosToEpochMs(event.timestamp)

            val sample = ImuSample(
                timestampMs = timestampMs,
                ax = accel[0], ay = accel[1], az = accel[2],
                gx = event.values[0], gy = event.values[1], gz = event.values[2]
            )

            callback?.invoke(sample)
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
            Log.d(TAG, "Gyroscope accuracy changed: $accuracy")
        }
    }

    /* ── Public API (matches api-contracts.md §5) ──────────────────────────── */

    /**
     * Start streaming raw IMU samples.
     *
     * @param callback  invoked on the sensor thread for each merged
     *                  accel+gyro sample.  Never invoked on the UI thread.
     * @throws IllegalStateException  if called when already started, or if the
     *         device lacks accelerometer/gyroscope hardware.
     */
    fun start(callback: (ImuSample) -> Unit) {
        check(!running) { "SensorBridge is already running. Call stop() first." }

        requireNotNull(accelSensor) {
            "Device has no TYPE_ACCELEROMETER sensor — cannot proceed."
        }
        requireNotNull(gyroSensor) {
            "Device has no TYPE_GYROSCOPE sensor — cannot proceed."
        }

        this.callback = callback

        // Compute boot-to-epoch offset once, used for all timestamp conversions.
        bootToEpochOffsetMs =
            System.currentTimeMillis() - SystemClock.elapsedRealtime()

        // Dedicated background thread for sensor callbacks.
        val thread = HandlerThread("SensorBridge-Thread").also { it.start() }
        sensorThread = thread
        sensorHandler = Handler(thread.looper)

        sensorManager.registerListener(
            accelListener, accelSensor, SENSOR_DELAY, sensorHandler
        )
        sensorManager.registerListener(
            gyroListener, gyroSensor, SENSOR_DELAY, sensorHandler
        )

        running = true
        Log.i(TAG, "Started IMU acquisition (accel + gyro).")
    }

    /**
     * Stop streaming and release sensor resources.
     *
     * Safe to call multiple times or when not started (no-op in that case).
     */
    fun stop() {
        if (!running) return

        sensorManager.unregisterListener(accelListener)
        sensorManager.unregisterListener(gyroListener)

        sensorThread?.quitSafely()
        sensorThread = null
        sensorHandler = null

        latestAccelValues = null
        latestAccelTimestampNs = 0L
        callback = null
        running = false

        Log.i(TAG, "Stopped IMU acquisition.")
    }

    /**
     * @return `true` if the bridge is actively streaming.
     */
    fun isRunning(): Boolean = running

    /* ── Internals ─────────────────────────────────────────────────────────── */

    /**
     * Convert a [SensorEvent.timestamp] (nanoseconds since boot) to epoch ms.
     *
     * Uses the offset computed once at [start] time — fast, no syscall per event.
     */
    private fun nanosToEpochMs(eventTimestampNs: Long): Long {
        // SensorEvent.timestamp is typically elapsedRealtimeNanos on most
        // devices post-KitKat, but some OEMs use uptimeNanos.  The single-
        // offset approach is the standard pragmatic conversion; precision is
        // ±single-digit ms, which is well within the pipeline's 100ms
        // inference interval.
        //
        // NOTE (Edge Case): Because elapsedRealtime includes deep sleep and 
        // uptimeNanos does not, if the device goes into deep sleep between 
        // start() and a sensor event, the timestamp could drift by the duration 
        // of the deep sleep. For continuous navigation (screen on), this is ~0.
        val elapsedMs = eventTimestampNs / 1_000_000L
        return elapsedMs + bootToEpochOffsetMs
    }
}
