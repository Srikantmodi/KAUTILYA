package com.sih.deadreckoning.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import androidx.annotation.GuardedBy

/**
 * Kotlin-side mirror of the C++ ImuSample struct (core-engine/include/imu_types.h).
 *
 * Units — per api-contracts.md §0 (Global Conventions):
 *  - timestampMs : epoch milliseconds (Long)
 *  - ax, ay, az  : raw accelerometer, phone frame, m/s² (includes gravity)
 *  - gx, gy, gz  : raw gyroscope, phone frame, rad/s
 *
 * NOTE: No magnetometer field here — mag is zero-filled at the JNI boundary
 * when marshaling to the C++ ImuSample struct that has float mag[3].
 * This matches the Kotlin-side mirror in api-contracts.md §1.
 */
data class ImuSample(
    val timestampMs: Long,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float
)

/**
 * SensorBridge — raw IMU data acquisition.
 *
 * Sits at the very top of the canonical data flow (PRD §2.1):
 *   Phone IMU (accel, gyro) → [SensorBridge] → 10Hz ring buffer → downstream
 *
 * Design decisions:
 *  - Registers listeners at SENSOR_DELAY_GAME (~50Hz on most devices) so the
 *    downstream SensorDataBuffer has enough raw samples to decimate to 10Hz.
 *  - Merges latest accelerometer + latest gyroscope into one ImuSample on every
 *    accelerometer event (accel typically fires at equal or higher rate than gyro).
 *  - Runs listeners on a dedicated HandlerThread to keep the main/UI thread free
 *    (PRD §7 guardrail #8 — never block the UI thread from the sensor path).
 *  - Emits RAW, UNLEVELED, UNFILTERED data — leveling/gravity-removal is
 *    Member 3's (Calibration) responsibility. Do NOT add any filtering here
 *    (PRD §2.3 context).
 *  - Thread-safe: all mutable state guarded by [lock].
 */
class SensorBridge(context: Context) {

    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accelerometer: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val gyroscope: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    // ---- mutable state, guarded by [lock] ----

    private val lock = Any()

    @GuardedBy("lock")
    private var callback: ((ImuSample) -> Unit)? = null

    @GuardedBy("lock")
    private var running = false

    // Latest gyro reading — kept so we can merge with the next accel event.
    @GuardedBy("lock")
    private var latestGyro = floatArrayOf(0f, 0f, 0f)

    @GuardedBy("lock")
    private var gyroReceived = false

    // Dedicated background thread for sensor callbacks.
    private var sensorThread: HandlerThread? = null
    private var sensorHandler: Handler? = null

    // ---- Sensor listeners ----

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val sample: ImuSample
            val cb: ((ImuSample) -> Unit)?

            synchronized(lock) {
                if (!running) return
                cb = callback ?: return

                // Convert Android's boot-time nanos to epoch milliseconds.
                // SensorEvent.timestamp is nanoseconds since boot (elapsedRealtimeNanos).
                // We convert via: epochMs = System.currentTimeMillis() - (elapsedRealtimeNanos - event.timestamp) / 1_000_000
                val nowMs = System.currentTimeMillis()
                val elapsedNs = android.os.SystemClock.elapsedRealtimeNanos()
                val eventAgeMs = (elapsedNs - event.timestamp) / 1_000_000L
                val epochMs = nowMs - eventAgeMs

                sample = ImuSample(
                    timestampMs = epochMs,
                    ax = event.values[0],
                    ay = event.values[1],
                    az = event.values[2],
                    gx = latestGyro[0],
                    gy = latestGyro[1],
                    gz = latestGyro[2]
                )
            }

            // Invoke callback outside the lock to avoid holding it during
            // potentially slow downstream processing.
            cb?.invoke(sample)
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* no-op */ }
    }

    private val gyroListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            synchronized(lock) {
                if (!running) return
                latestGyro[0] = event.values[0]
                latestGyro[1] = event.values[1]
                latestGyro[2] = event.values[2]
                gyroReceived = true
            }
            // No sample emitted here — merged into the next accel event.
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* no-op */ }
    }

    // ---- Public API (matches api-contracts.md §5) ----

    /**
     * Begin streaming raw IMU samples via [callback].
     *
     * Each invocation of [callback] delivers one [ImuSample] containing the
     * latest accelerometer reading merged with the most recent gyroscope reading.
     * Samples arrive at roughly the accelerometer's hardware rate (~50–100 Hz
     * depending on device); downstream [SensorDataBuffer] decimates to 10 Hz.
     *
     * @throws IllegalStateException if a sensor is unavailable on this device.
     */
    fun start(callback: (ImuSample) -> Unit) {
        synchronized(lock) {
            if (running) return // idempotent

            requireNotNull(accelerometer) {
                "TYPE_ACCELEROMETER sensor not available on this device"
            }
            requireNotNull(gyroscope) {
                "TYPE_GYROSCOPE sensor not available on this device"
            }

            this.callback = callback
            this.gyroReceived = false
            this.latestGyro = floatArrayOf(0f, 0f, 0f)
            this.running = true
        }

        // Create a dedicated thread for sensor callbacks.
        val thread = HandlerThread("SensorBridge-IMU").also { it.start() }
        val handler = Handler(thread.looper)
        sensorThread = thread
        sensorHandler = handler

        // Register gyro first so latestGyro is populated before the first accel event.
        sensorManager.registerListener(
            gyroListener, gyroscope, SensorManager.SENSOR_DELAY_GAME, handler
        )
        sensorManager.registerListener(
            accelListener, accelerometer, SensorManager.SENSOR_DELAY_GAME, handler
        )
    }

    /**
     * Stop streaming. Safe to call multiple times or if never started.
     * After this returns, no further callbacks will be invoked.
     */
    fun stop() {
        synchronized(lock) {
            if (!running) return
            running = false
            callback = null
        }

        sensorManager.unregisterListener(accelListener)
        sensorManager.unregisterListener(gyroListener)

        sensorThread?.quitSafely()
        sensorThread = null
        sensorHandler = null
    }
}
