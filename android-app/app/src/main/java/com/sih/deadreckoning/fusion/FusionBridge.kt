/**
 * FusionBridge.kt — Thin JNI wrapper for the C++ UKF Fusion Engine
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * This is a PURE PASS-THROUGH — one Kotlin method per exposed C++ function
 * in JniBridge.h (api-contracts.md §9). No business logic. If a "quick fix"
 * appears here, it's a sign the C++ core is being bypassed (§7 guardrail #1).
 *
 * References:
 *   api-contracts.md §9, PRD §6.7, JniBridge.h fusion section
 */
package com.sih.deadreckoning.fusion

import android.util.Log

/**
 * Kotlin-side mirror of NavState (C++ struct from imu_types.h).
 * Populated by unpacking the 26-element float array from fusionGetNavState().
 */
data class NavState(
    /** Position [east, north, up] in meters, ENU relative to session start. */
    val positionE: Float,
    val positionN: Float,
    val positionU: Float,

    /** Velocity [ve, vn, vu] in m/s, ENU. */
    val velocityE: Float,
    val velocityN: Float,
    val velocityU: Float,

    /** Heading in radians, ENU convention (0=East, CCW+). */
    val headingRad: Float,

    /** Gyro bias estimates [bx, by, bz] in rad/s. */
    val gyroBiasX: Float,
    val gyroBiasY: Float,
    val gyroBiasZ: Float,

    /** 4×4 position+heading covariance block, row-major (16 floats). */
    val covariance: FloatArray
) {
    /** Convenience: forward speed magnitude (m/s). */
    val speed: Float get() = kotlin.math.sqrt(velocityE * velocityE + velocityN * velocityN)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is NavState) return false
        return positionE == other.positionE && positionN == other.positionN &&
                positionU == other.positionU && velocityE == other.velocityE &&
                velocityN == other.velocityN && velocityU == other.velocityU &&
                headingRad == other.headingRad && covariance.contentEquals(other.covariance)
    }

    override fun hashCode(): Int {
        var result = positionE.hashCode()
        result = 31 * result + positionN.hashCode()
        result = 31 * result + headingRad.hashCode()
        result = 31 * result + covariance.contentHashCode()
        return result
    }
}

/**
 * Thin JNI bridge to the C++ UKF fusion engine.
 *
 * All methods are direct pass-throughs to their C++ counterparts declared
 * in JniBridge.h. No business logic, no data transformation, no caching.
 *
 * Usage:
 * ```
 * val fusion = FusionBridge()
 * fusion.predict(deltaV, dtSeconds)
 * fusion.updateGnss(gnssFix)
 * val state = fusion.getNavState()
 * ```
 */
class FusionBridge {

    companion object {
        private const val TAG = "FusionBridge"

        init {
            try {
                System.loadLibrary("core_engine")
                Log.i(TAG, "core_engine native library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load core_engine native library: ${e.message}")
            }
        }
    }

    /* ── Native (JNI) method declarations ────────────────────────────────── */
    /* These match the declarations in JniBridge.h exactly.                  */

    /**
     * UKF predict step — strapdown integration using ML-derived delta-v.
     * @param deltaV     velocity change (m/s) from ML model
     * @param dtSeconds  time step duration (seconds)
     */
    external fun fusionPredict(deltaV: Float, dtSeconds: Float)

    /**
     * UKF GNSS measurement update — loosely-coupled correction.
     * @param ts         GnssFix timestamp_ms
     * @param lat        latitude, degrees (WGS-84)
     * @param lon        longitude, degrees (WGS-84)
     * @param speed      ground speed, m/s
     * @param bearing    course over ground, radians (ENU: 0=East, CCW+)
     * @param accuracy   horizontal accuracy, meters
     * @param satellites satellite count
     * @param quality    quality score [0.0, 1.0]
     */
    external fun fusionUpdateGnss(
        ts: Long,
        lat: Double, lon: Double,
        speed: Float,
        bearing: Float,
        accuracy: Float,
        satellites: Int,
        quality: Float
    )

    /**
     * UKF map-match measurement update — closed-loop HMM correction.
     * Skipped internally when confidence < threshold (0.7).
     * @param pseudoMeasurement  FloatArray[2]: [lateral_offset_m, heading_offset_rad]
     * @param confidence         match confidence [0.0, 1.0]
     */
    external fun fusionUpdateMapMatch(
        pseudoMeasurement: FloatArray,
        confidence: Float
    )

    /**
     * Zero Angular Rate Update — correct gyro bias during straight-line.
     */
    external fun fusionApplyZaru()

    /**
     * Zero velocity Update — hard-clamp velocity when stationary.
     * Call when stationary_probability > 0.95 from ML model output.
     */
    external fun fusionApplyZupt()

    /**
     * Apply error-state neural network residual correction (§2.5).
     * @param deltaP    FloatArray[3]: position correction [de, dn, du] meters
     * @param deltaV    FloatArray[3]: velocity correction [dve, dvn, dvu] m/s
     * @param deltaPsi  heading correction, radians
     */
    external fun fusionApplyEsNnResidual(
        deltaP: FloatArray,
        deltaV: FloatArray,
        deltaPsi: Float
    )

    /**
     * Retrieve the current fused navigation state as a raw float array.
     * @return FloatArray[26]:
     *           [0..2]   position — east, north, up (meters, ENU)
     *           [3..5]   velocity — ve, vn, vu (m/s, ENU)
     *           [6]      heading — radians (ENU: 0=East, CCW+)
     *           [7..9]   gyro_bias — bx, by, bz (rad/s)
     *           [10..25] covariance — 4×4, row-major (position+heading block)
     */
    external fun fusionGetNavState(): FloatArray

    /* ── Convenience methods ──────────────────────────────────────────────── */

    /**
     * Get the current navigation state as a structured NavState object.
     * Unpacks the 26-element float array from the native call.
     */
    fun getNavState(): NavState {
        val raw = fusionGetNavState()
        return NavState(
            positionE  = raw[0],
            positionN  = raw[1],
            positionU  = raw[2],
            velocityE  = raw[3],
            velocityN  = raw[4],
            velocityU  = raw[5],
            headingRad = raw[6],
            gyroBiasX  = raw[7],
            gyroBiasY  = raw[8],
            gyroBiasZ  = raw[9],
            covariance = raw.sliceArray(10..25)
        )
    }

    /**
     * Convenience: predict using the InferenceResult from Member 6's engine.
     */
    fun predict(deltaV: Float, dtSeconds: Float) {
        fusionPredict(deltaV, dtSeconds)
    }

    /**
     * Convenience: update GNSS from a GnssFix data class (Member 2).
     * Expects the GnssFix from GnssQualityMonitor with bearing already
     * converted to radians ENU convention.
     */
    fun updateGnss(
        timestampMs: Long,
        latDeg: Double,
        lonDeg: Double,
        speedMps: Float,
        bearingRad: Float,
        accuracyM: Float,
        satelliteCount: Int,
        qualityScore: Float
    ) {
        fusionUpdateGnss(
            timestampMs, latDeg, lonDeg,
            speedMps, bearingRad, accuracyM,
            satelliteCount, qualityScore
        )
    }

    /**
     * Convenience: update map match from MapMatchResult (Member 5).
     */
    fun updateMapMatch(pseudoMeasurement: FloatArray, confidence: Float) {
        fusionUpdateMapMatch(pseudoMeasurement, confidence)
    }
}
