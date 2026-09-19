package com.sih.deadreckoning.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssStatus
import android.location.LocationManager
import android.os.Build
import android.os.Looper
import androidx.annotation.GuardedBy
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

/**
 * Kotlin-side mirror of the C++ GnssFix struct (core-engine/include/imu_types.h).
 *
 * Units — per api-contracts.md §0 (Global Conventions):
 *  - timestampMs    : epoch milliseconds (Long)
 *  - latDeg/lonDeg  : degrees (WGS-84)
 *  - speedMps       : meters/second
 *  - bearingDeg     : degrees (0 = North, clockwise) — matches Android's native GNSS
 *                     bearing field; converted to radians by consumers (api-contracts §1)
 *  - accuracyM      : estimated horizontal accuracy in meters (68th percentile radius)
 *  - satelliteCount  : number of satellites used in the fix
 *  - qualityScore    : [0.0, 1.0] composite quality, computed internally from
 *                     satelliteCount + accuracyM + fix-age (api-contracts §5)
 */
data class GnssFix(
    val timestampMs: Long,
    val latDeg: Double, val lonDeg: Double,
    val speedMps: Float, val bearingDeg: Float,
    val accuracyM: Float, val satelliteCount: Int,
    val qualityScore: Float
)

/**
 * GnssQualityMonitor — GNSS data acquisition with composite quality scoring.
 *
 * Sits in the canonical data flow (PRD §2.1) as the GNSS branch, feeding:
 *  - [GnssFix.qualityScore] → Member 4's ModeManager (§6.5) for millisecond-
 *    scale GNSS_AIDED_INS ↔ PURE_DR transitions.
 *  - Full [GnssFix] → Member 4's FusionBridge (ukf_fusion::update_gnss).
 *
 * Design decisions:
 *  - Uses [FusedLocationProviderClient] at ~1 Hz (1000 ms interval), matching
 *    the IO-VNBD dataset's real GPS rate (PRD §2.1 implementation notes).
 *  - Satellite count obtained via [GnssStatus.Callback] on the system
 *    [LocationManager] — FusedLocationProvider doesn't expose sat count directly.
 *  - Quality score is a normalized composite of three sub-scores:
 *      1. Satellite score    : clamp(satCount / 12, 0, 1) — 12+ sats ≈ excellent
 *      2. Accuracy score     : clamp(1 - accuracyM / 50, 0, 1) — <50 m is usable
 *      3. Freshness score    : clamp(1 - fixAgeMs / 3000, 0, 1) — >3 s stale ≈ lost
 *    Final: weighted average (0.3 × sat + 0.4 × accuracy + 0.3 × freshness).
 *    Thresholds chosen to roughly align with api-contracts §7 starting defaults
 *    (satCount ≥ 4, accuracy ≤ 15 m, fixAge ≤ 2000 ms → high quality).
 *  - Does NOT filter, smooth, or modify GNSS positions — raw pass-through
 *    plus a computed quality score, nothing else.
 *  - Thread-safe: all mutable state guarded by [lock].
 */
class GnssQualityMonitor(context: Context) {

    companion object {
        // ---- Quality score weights ----
        private const val W_SATELLITE: Float = 0.3f
        private const val W_ACCURACY: Float = 0.4f
        private const val W_FRESHNESS: Float = 0.3f

        // ---- Sub-score parameters ----
        /** Satellite count at which sat-score saturates to 1.0. */
        private const val SAT_COUNT_EXCELLENT: Float = 12f

        /** Accuracy (m) at which accuracy-score drops to 0.0. */
        private const val ACCURACY_FLOOR_M: Float = 50f

        /** Fix age (ms) at which freshness-score drops to 0.0. */
        private const val FRESHNESS_FLOOR_MS: Float = 3000f

        /** Location update interval target, milliseconds (~1 Hz). */
        private const val LOCATION_INTERVAL_MS: Long = 1000L

        /** Fastest location update we'll accept, milliseconds. */
        private const val LOCATION_FASTEST_MS: Long = 500L
    }

    private val fusedClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val locationManager: LocationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    // ---- Mutable state ----

    private val lock = Any()

    @GuardedBy("lock")
    private var onFixCallback: ((GnssFix) -> Unit)? = null

    @GuardedBy("lock")
    private var running = false

    /** Latest satellite count from the GnssStatus callback. */
    @GuardedBy("lock")
    private var currentSatelliteCount: Int = 0

    /** Timestamp (epoch ms) of the most recent fix delivered to the callback. */
    @GuardedBy("lock")
    private var lastFixEpochMs: Long = 0L

    // ---- Callbacks ----

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val location = result.lastLocation ?: return

            val fix: GnssFix
            val cb: ((GnssFix) -> Unit)?

            synchronized(lock) {
                if (!running) return
                cb = onFixCallback ?: return

                val nowMs = System.currentTimeMillis()
                val fixTimeMs = location.time  // epoch ms
                val fixAgeMs = (nowMs - fixTimeMs).coerceAtLeast(0L)

                val satCount = currentSatelliteCount

                val qualityScore = computeQualityScore(
                    satelliteCount = satCount,
                    accuracyM = if (location.hasAccuracy()) location.accuracy else ACCURACY_FLOOR_M,
                    fixAgeMs = fixAgeMs
                )

                fix = GnssFix(
                    timestampMs = fixTimeMs,
                    latDeg = location.latitude,
                    lonDeg = location.longitude,
                    speedMps = if (location.hasSpeed()) location.speed else 0f,
                    bearingDeg = if (location.hasBearing()) location.bearing else 0f,
                    accuracyM = if (location.hasAccuracy()) location.accuracy else Float.MAX_VALUE,
                    satelliteCount = satCount,
                    qualityScore = qualityScore
                )

                lastFixEpochMs = fixTimeMs
            }

            // Invoke outside the lock.
            cb?.invoke(fix)
        }
    }

    private val gnssStatusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            // Count only satellites actually used in the fix (not just visible).
            var usedCount = 0
            for (i in 0 until status.satelliteCount) {
                if (status.usedInFix(i)) usedCount++
            }
            synchronized(lock) {
                currentSatelliteCount = usedCount
            }
        }
    }

    // ---- Public API (matches api-contracts.md §5) ----

    /**
     * Begin streaming GNSS fixes at ~1 Hz via [onFix].
     *
     * Each [GnssFix] includes a [GnssFix.qualityScore] computed from satellite
     * count, horizontal accuracy, and fix freshness.
     *
     * Requires ACCESS_FINE_LOCATION permission to be granted BEFORE calling.
     * Throws [SecurityException] if permission is missing at runtime.
     */
    @SuppressLint("MissingPermission") // Caller is responsible for runtime permission checks.
    fun start(onFix: (GnssFix) -> Unit) {
        synchronized(lock) {
            if (running) return // idempotent
            onFixCallback = onFix
            currentSatelliteCount = 0
            lastFixEpochMs = 0L
            running = true
        }

        // Register satellite-count listener.
        try {
            locationManager.registerGnssStatusCallback(
                gnssStatusCallback, android.os.Handler(Looper.getMainLooper())
            )
        } catch (_: SecurityException) {
            // On some devices/emulators without GNSS hardware this can throw.
            // We still proceed with satCount=0 rather than crashing.
        }

        // Request fused location updates at ~1 Hz.
        val request = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            LOCATION_INTERVAL_MS
        )
            .setMinUpdateIntervalMillis(LOCATION_FASTEST_MS)
            .build()

        fusedClient.requestLocationUpdates(
            request,
            locationCallback,
            Looper.getMainLooper()
        )
    }

    /**
     * Stop streaming. Safe to call multiple times or if never started.
     */
    fun stop() {
        synchronized(lock) {
            if (!running) return
            running = false
            onFixCallback = null
        }

        fusedClient.removeLocationUpdates(locationCallback)
        try {
            locationManager.unregisterGnssStatusCallback(gnssStatusCallback)
        } catch (_: Exception) {
            // Tolerate if not registered.
        }
    }

    // ---- Quality scoring (internal) ----

    /**
     * Composite GNSS quality score in [0.0, 1.0].
     *
     * Sub-scores:
     *  - Satellite  : linear ramp 0→1 over [0, SAT_COUNT_EXCELLENT] sats
     *  - Accuracy   : linear ramp 1→0 over [0, ACCURACY_FLOOR_M] meters
     *  - Freshness  : linear ramp 1→0 over [0, FRESHNESS_FLOOR_MS] ms
     *
     * Final = weighted sum (W_SATELLITE + W_ACCURACY + W_FRESHNESS = 1.0).
     */
    private fun computeQualityScore(
        satelliteCount: Int,
        accuracyM: Float,
        fixAgeMs: Long
    ): Float {
        val satScore = (satelliteCount.toFloat() / SAT_COUNT_EXCELLENT).coerceIn(0f, 1f)
        val accScore = (1f - accuracyM / ACCURACY_FLOOR_M).coerceIn(0f, 1f)
        val freshScore = (1f - fixAgeMs.toFloat() / FRESHNESS_FLOOR_MS).coerceIn(0f, 1f)

        return (W_SATELLITE * satScore + W_ACCURACY * accScore + W_FRESHNESS * freshScore)
            .coerceIn(0f, 1f)
    }
}
