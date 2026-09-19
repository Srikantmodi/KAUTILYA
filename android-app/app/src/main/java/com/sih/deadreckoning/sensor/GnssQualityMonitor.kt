/**
 * GnssQualityMonitor.kt — GNSS acquisition with quality scoring
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 2 (Sensor & GNSS Data Acquisition)
 *
 * Uses [FusedLocationProviderClient] at ~1 Hz (matching IO-VNBD's real GPS
 * rate) to produce [GnssFix] objects that include a computed [qualityScore].
 *
 * The quality score is a composite metric derived from:
 *   1. Satellite count  — more satellites → better geometry → higher score
 *   2. Accuracy         — lower reported accuracy radius → higher score
 *   3. Fix age          — staler fixes → lower score
 *
 * This feeds Member 4's Mode Manager directly (§6.5). The Mode Manager
 * consumes the quality score for millisecond-scale GNSS-aided-INS ↔ pure-DR
 * transitions — it's NOT just a GPS pass-through.
 *
 * Bearing convention note: Android provides bearing in degrees, clockwise
 * from North.  The frozen imu_types.h (line 76-79) specifies bearing_rad
 * in ENU convention (0=East, CCW positive).  The Kotlin-side data class
 * keeps bearing in degrees (per api-contracts.md §1 Kotlin mirror), and the
 * conversion to radians + ENU is performed when marshaling to C++ via the
 * JNI bridge layer (Member 4's FusionBridge / Member 1's ModeManager).
 *
 * Threading: location callbacks fire on the main looper by default; an
 * explicit Looper is used if available.  The user-provided [onFix] callback
 * is invoked on the callback thread.
 *
 * References:
 *   PRD §2.1   — canonical data flow
 *   PRD §6.5   — mode manager consumes quality_score
 *   PRD §3.1   — play-services-location
 *   api-contracts.md §5  — exact interface surface
 *   imu_types.h lines 56-84  — C++ GnssFix struct contract
 */
package com.sih.deadreckoning.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssStatus
import android.location.LocationManager
import android.os.Looper
import android.util.Log
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

/* ─────────────────────────────────────────────────────────────────────────────
 * GnssFix — Kotlin-side mirror of core-engine GnssFix (imu_types.h)
 *
 * Defined here per the PRD folder structure and api-contracts.md §1.
 *
 * All fields match api-contracts.md §1 Kotlin-side mirror exactly:
 *   timestampMs     — epoch milliseconds
 *   latDeg, lonDeg  — WGS-84 degrees
 *   speedMps        — ground speed, m/s
 *   bearingDeg      — course over ground, degrees (Android native convention:
 *                     CW from North).
 *   bearingEnuRad   — course over ground, radians, ENU convention (0=East,
 *                     CCW positive).  Converted from bearingDeg at construction
 *                     time per imu_types.h line 77-79 which explicitly assigns
 *                     this conversion to GnssQualityMonitor.
 *   accuracyM       — estimated horizontal accuracy, meters
 *   satelliteCount  — satellites used in fix
 *   qualityScore    — [0.0, 1.0] composite quality metric
 * ───────────────────────────────────────────────────────────────────────────── */
data class GnssFix(
    val timestampMs: Long,
    val latDeg: Double, val lonDeg: Double,
    val speedMps: Float, val bearingDeg: Float,
    val bearingEnuRad: Float,
    val accuracyM: Float, val satelliteCount: Int,
    val qualityScore: Float
)

/**
 * GNSS location monitor with quality scoring.
 *
 * Usage:
 * ```
 * val monitor = GnssQualityMonitor(context)
 * monitor.start { fix -> fusionBridge.updateGnss(fix) }
 * // ... later
 * monitor.stop()
 * ```
 *
 * @param context  Android [Context] for location services access.
 *                 Must have ACCESS_FINE_LOCATION permission granted at runtime.
 */
class GnssQualityMonitor(private val context: Context) {

    companion object {
        private const val TAG = "GnssQualityMonitor"

        /* ── Quality scoring parameters ────────────────────────────────────────
         * These match the ⚠️ ASSUMPTION defaults in api-contracts.md §7:
         *   satelliteCount >= 4  AND  accuracyM <= 15.0  AND  fixAge <= 2000ms
         * → healthy GNSS.  We use the same thresholds for the continuous [0,1]
         * quality score so that score > ~0.5 roughly aligns with the
         * mode-manager's aided-INS threshold.
         * ─────────────────────────────────────────────────────────────────── */

        /** Satellite count at or above this is "ideal." */
        private const val SAT_COUNT_IDEAL = 8
        /** Minimum usable satellite count.  Below this, quality drops steeply. */
        private const val SAT_COUNT_MIN = 4

        /** Accuracy radius (m) at or below this is "ideal." */
        private const val ACCURACY_IDEAL_M = 5.0f
        /** Accuracy radius (m) above this is considered useless. */
        private const val ACCURACY_MAX_M = 30.0f

        /** Fix age (ms) at or below this is "ideal" (fresh). */
        private const val FIX_AGE_IDEAL_MS = 500L
        /** Fix age (ms) above this is considered stale / lost. */
        private const val FIX_AGE_MAX_MS = 3000L

        /** Weights for the three sub-scores (sum to 1.0). */
        private const val W_SATELLITES = 0.30f
        private const val W_ACCURACY   = 0.45f
        private const val W_FIX_AGE    = 0.25f

        /**
         * Convert Android's bearing (degrees, CW from North) to ENU radians
         * (0 = East, CCW positive).
         *
         * Formula: ENU_rad = (90° - bearing_deg) converted to radians,
         * normalized to [-π, π].
         *
         * This conversion is explicitly assigned to GnssQualityMonitor by
         * imu_types.h lines 77-79: "Converted from Android's CW-from-North at
         * ingress by GnssQualityMonitor."
         */
        fun bearingDegToEnuRad(bearingDeg: Float): Float {
            val enuDeg = 90.0f - bearingDeg
            val enuRad = Math.toRadians(enuDeg.toDouble()).toFloat()
            // Normalize to [-π, π]
            return ((enuRad + Math.PI.toFloat()) % (2f * Math.PI.toFloat()) + (2f * Math.PI.toFloat())) %
                    (2f * Math.PI.toFloat()) - Math.PI.toFloat()
        }

        /**
         * Compute a composite quality score ∈ [0.0, 1.0] from three sub-scores.
         *
         * Each sub-score is linearly mapped from its raw input to [0, 1] between
         * the "useless" and "ideal" bounds, then clamped.  The three sub-scores
         * are combined via a weighted sum.
         *
         * The formula is intentionally simple and transparent so that tuning
         * against real-world data (different cities, highway vs. urban canyon) is
         * straightforward — there's no hidden nonlinearity to debug.
         *
         * Placed in companion object (not instance method) so unit tests can call
         * the REAL production formula without needing a Context-dependent
         * GnssQualityMonitor instance — avoids §7 guardrail #1 ("never maintain
         * two implementations of the same formula").
         *
         * @param satelliteCount  satellites used in the fix
         * @param accuracyM       reported horizontal accuracy, meters
         * @param fixAgeMs        age of the fix in milliseconds (nowMs − fixTimeMs)
         * @return quality score ∈ [0.0, 1.0]
         */
        fun computeQualityScore(
            satelliteCount: Int,
            accuracyM: Float,
            fixAgeMs: Long
        ): Float {
            // Sub-score 1: satellite count (more is better)
            val satScore = when {
                satelliteCount >= SAT_COUNT_IDEAL -> 1.0f
                satelliteCount <= 0               -> 0.0f
                satelliteCount < SAT_COUNT_MIN    ->
                    satelliteCount.toFloat() / SAT_COUNT_MIN * 0.3f  // below-minimum: heavily penalized
                else ->
                    0.3f + 0.7f * (satelliteCount - SAT_COUNT_MIN).toFloat() /
                            (SAT_COUNT_IDEAL - SAT_COUNT_MIN).toFloat()
            }

            // Sub-score 2: horizontal accuracy (lower is better)
            val accScore = when {
                accuracyM <= ACCURACY_IDEAL_M -> 1.0f
                accuracyM >= ACCURACY_MAX_M   -> 0.0f
                else ->
                    1.0f - (accuracyM - ACCURACY_IDEAL_M) /
                            (ACCURACY_MAX_M - ACCURACY_IDEAL_M)
            }

            // Sub-score 3: fix age (fresher is better)
            val ageScore = when {
                fixAgeMs <= FIX_AGE_IDEAL_MS -> 1.0f
                fixAgeMs >= FIX_AGE_MAX_MS   -> 0.0f
                else ->
                    1.0f - (fixAgeMs - FIX_AGE_IDEAL_MS).toFloat() /
                            (FIX_AGE_MAX_MS - FIX_AGE_IDEAL_MS).toFloat()
            }

            val raw = W_SATELLITES * satScore +
                      W_ACCURACY   * accScore +
                      W_FIX_AGE    * ageScore

            return raw.coerceIn(0.0f, 1.0f)
        }

        /** Location request interval — ~1 Hz per PRD / IO-VNBD real GPS rate. */
        private const val LOCATION_INTERVAL_MS = 1000L
    }

    private val fusedClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val locationManager: LocationManager? =
        context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager

    private var onFixCallback: ((GnssFix) -> Unit)? = null
    private var locationCallback: LocationCallback? = null
    private var gnssStatusCallback: GnssStatus.Callback? = null
    private var running = false

    /** Most recently observed satellite count from [GnssStatus]. */
    @Volatile private var lastSatelliteCount: Int = 0

    /** Timestamp of the last received fix — used for fix-age calculation. */
    @Volatile private var lastFixTimeMs: Long = 0L

    /* ── Public API (matches api-contracts.md §5) ──────────────────────────── */

    /**
     * Start requesting GNSS fixes at ~1 Hz.
     *
     * @param onFix  invoked for each valid location fix, with a fully-populated
     *               [GnssFix] including the computed [GnssFix.qualityScore].
     * @throws IllegalStateException  if already started.
     * @throws SecurityException      if location permission is not granted.
     */
    @SuppressLint("MissingPermission")  // Caller is responsible for runtime permission check.
    fun start(onFix: (GnssFix) -> Unit) {
        check(!running) { "GnssQualityMonitor is already running. Call stop() first." }

        onFixCallback = onFix

        // ── GnssStatus listener for satellite count ──────────────────────────
        // Android's Location object has `extras` with satellite info on some
        // OEMs, but GnssStatus.Callback is the reliable cross-device API.
        gnssStatusCallback = object : GnssStatus.Callback() {
            override fun onSatelliteStatusChanged(status: GnssStatus) {
                var usedCount = 0
                for (i in 0 until status.satelliteCount) {
                    if (status.usedInFix(i)) usedCount++
                }
                lastSatelliteCount = usedCount
            }
        }

        locationManager?.registerGnssStatusCallback(
            gnssStatusCallback!!, android.os.Handler(Looper.getMainLooper())
        )

        // ── Fused location request ───────────────────────────────────────────
        val request = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            LOCATION_INTERVAL_MS
        ).apply {
            setMinUpdateIntervalMillis(LOCATION_INTERVAL_MS / 2)
            setWaitForAccurateLocation(false)
        }.build()

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val location = result.lastLocation ?: return
                val nowMs = System.currentTimeMillis()

                val fixTimeMs = location.time  // epoch ms
                val fixAgeMs = (nowMs - fixTimeMs).coerceAtLeast(0)

                val satCount = lastSatelliteCount
                val accuracyM = if (location.hasAccuracy()) location.accuracy else ACCURACY_MAX_M
                val speedMps = if (location.hasSpeed()) location.speed else 0f
                val bearingDeg = if (location.hasBearing()) location.bearing else 0f
                val bearingEnuRad = bearingDegToEnuRad(bearingDeg)

                val qualityScore = computeQualityScore(
                    satelliteCount = satCount,
                    accuracyM = accuracyM,
                    fixAgeMs = fixAgeMs
                )

                lastFixTimeMs = fixTimeMs

                val fix = GnssFix(
                    timestampMs = fixTimeMs,
                    latDeg = location.latitude,
                    lonDeg = location.longitude,
                    speedMps = speedMps,
                    bearingDeg = bearingDeg,
                    bearingEnuRad = bearingEnuRad,
                    accuracyM = accuracyM,
                    satelliteCount = satCount,
                    qualityScore = qualityScore
                )

                Log.d(TAG, "Fix: lat=${fix.latDeg}, lon=${fix.lonDeg}, " +
                        "acc=${fix.accuracyM}m, sats=${fix.satelliteCount}, " +
                        "quality=${fix.qualityScore}")

                onFixCallback?.invoke(fix)
            }
        }

        fusedClient.requestLocationUpdates(
            request,
            locationCallback!!,
            Looper.getMainLooper()
        )

        running = true
        Log.i(TAG, "Started GNSS monitoring at ~${1000 / LOCATION_INTERVAL_MS} Hz.")
    }

    /**
     * Stop GNSS monitoring and release resources.
     *
     * Safe to call multiple times or when not started (no-op).
     */
    fun stop() {
        if (!running) return

        locationCallback?.let { fusedClient.removeLocationUpdates(it) }
        gnssStatusCallback?.let { locationManager?.unregisterGnssStatusCallback(it) }

        locationCallback = null
        gnssStatusCallback = null
        onFixCallback = null
        running = false

        Log.i(TAG, "Stopped GNSS monitoring.")
    }

    /**
     * @return `true` if the monitor is actively requesting location updates.
     */
    fun isRunning(): Boolean = running

    /**
     * Returns the timestamp of the last fix received, or 0 if none yet.
     *
     * Useful for external fix-age checks (e.g. Member 4's mode manager can
     * compute `currentTimeMs - lastFixTimeMs` without waiting for the next
     * callback to fire).
     */
    fun getLastFixTimeMs(): Long = lastFixTimeMs

}
