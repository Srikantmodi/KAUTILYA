package com.sih.deadreckoning.calibration

import com.sih.deadreckoning.sensor.ImuSample
import android.util.Log

/**
 * AlignmentManager — Android-side calibration bridge
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 3 (Calibration & Device Alignment)
 *
 * This class is the Kotlin entry point for the C++ calibration engine
 * (core-engine/src/calibration.cpp). It bridges the raw ImuSample stream
 * from Member 2's SensorBridge to the native EKF and yaw optimizer,
 * and exposes leveled acceleration to downstream consumers.
 *
 * Thread safety:
 *  - [feedSample] may be called from SensorBridge's HandlerThread.
 *  - [levelAccelerometer] and [getCurrentRotation] may be called from
 *    inference/fusion threads.
 *  - The C++ layer handles its own spinlock; Kotlin is a thin wrapper.
 *
 * API (from api-contracts.md §3):
 *  - feedSample(ImuSample)             → triggers JNI calibrationUpdate
 *  - levelAccelerometer(FloatArray)    → JNI levelAccelerometer
 *  - getCurrentRotation(): RotationEstimate
 *  - resetCalibration()
 */
class AlignmentManager {

    companion object {
        private const val TAG = "AlignmentManager"

        init {
            // Load the native library containing calibration.cpp
            try {
                System.loadLibrary("core_engine")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library core_engine: ${e.message}", e)
            }
        }
    }

    // ─── JNI native declarations (must match JniBridge.h exactly) ────────

    /**
     * Feed a raw IMU sample into the C++ calibration EKF.
     * Maps to: Java_com_sih_deadreckoning_calibration_AlignmentManager_calibrationUpdate
     *
     * @param timestampMs  epoch milliseconds
     * @param accel        FloatArray[3]: [ax, ay, az] raw, phone-frame, m/s²
     * @param gyro         FloatArray[3]: [gx, gy, gz] raw, phone-frame, rad/s
     */
    private external fun calibrationUpdate(
        timestampMs: Long,
        accel: FloatArray,
        gyro: FloatArray
    )

    /**
     * Remove gravity and project into vehicle frame.
     * Maps to: Java_com_sih_deadreckoning_calibration_AlignmentManager_levelAccelerometer
     *
     * NOTE: This is public because Member 6's TFLiteInferenceEngine calls it
     * directly to level raw samples before ML windowing.
     *
     * @param rawAccel  FloatArray[3]: phone-frame m/s² (includes gravity)
     * @return          FloatArray[3]: vehicle-frame m/s² (gravity removed)
     *                  [0]=forward, [1]=lateral, [2]=vertical
     */
    external fun levelAccelerometer(rawAccel: FloatArray): FloatArray

    /**
     * Get current calibration state from the C++ engine.
     * Maps to: Java_com_sih_deadreckoning_calibration_AlignmentManager_getCurrentRotation
     *
     * @return FloatArray[4]: [pitch_rad, roll_rad, yaw_rad, yaw_calibrated_float]
     *         yaw_calibrated_float is 1.0f when yaw has been solved, 0.0f otherwise
     */
    private external fun getCurrentRotationNative(): FloatArray

    /**
     * Reset calibration state in C++ engine.
     */
    private external fun resetCalibration(): Unit


    // ─── Public Kotlin API ────────────────────────────────────────────────

    /**
     * Data class representing the current device orientation estimate.
     * All angles in radians per api-contracts.md §0.
     */
    data class RotationEstimate(
        val pitchRad: Float,
        val rollRad: Float,
        val yawRad: Float,
        val yawCalibrated: Boolean  ///< true once pooled yaw solve has run
    )

    /**
     * Feed a raw ImuSample from SensorBridge into the calibration engine.
     *
     * Call this for EVERY raw sample. The C++ layer handles decimation
     * of the gate check internally — you do not need to decimate before
     * calling this function.
     *
     * @param sample  Raw ImuSample from SensorBridge (unleveled, includes gravity)
     */
    fun feedSample(sample: ImuSample) {
        val accel = floatArrayOf(sample.ax, sample.ay, sample.az)
        val gyro  = floatArrayOf(sample.gx, sample.gy, sample.gz)
        try {
            calibrationUpdate(sample.timestampMs, accel, gyro)
        } catch (e: Exception) {
            Log.e(TAG, "JNI calibrationUpdate threw: ${e.message}", e)
        }
    }

    private var lastCalibratedState = false

    /**
     * Get the current calibration rotation estimate.
     */
    fun getCurrentRotation(): RotationEstimate {
        return try {
            val arr = getCurrentRotationNative()
            val isCalibrated = arr[3] > 0.5f
            if (isCalibrated && !lastCalibratedState) {
                lastCalibratedState = true
                Log.i(TAG, "Calibration state updated: Full 3D locked! Pitch=${arr[0]} rad, Roll=${arr[1]} rad, Yaw=${arr[2]} rad")
            }
            // arr[4]: [pitch, roll, yaw, yaw_calibrated_as_float]
            RotationEstimate(
                pitchRad      = arr[0],
                rollRad       = arr[1],
                yawRad        = arr[2],
                yawCalibrated = isCalibrated
            )
        } catch (e: Exception) {
            Log.e(TAG, "JNI getCurrentRotation threw: ${e.message}", e)
            RotationEstimate(0.0f, 0.0f, 0.0f, false)
        }
    }

    /**
     * Alias for feedSample per PRD §6.2 contract convention.
     */
    fun update(sample: ImuSample) = feedSample(sample)

    /**
     * Reset all calibration state. Call on session start or when
     * DeviceOrientationDetector signals a phone remount.
     */
    fun reset() {
        try {
            lastCalibratedState = false
            resetCalibration()
            Log.i(TAG, "Native calibration state reset successfully")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "Native resetCalibration not linked: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "resetCalibration threw: ${e.message}", e)
        }
    }

    /**
     * Alias for reset() per PRD §6.2 contract convention.
     */
    fun resetCalibrationState() = reset()
}
