/**
 * GnssQualityScoreTest.kt — Unit tests for quality score computation
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 2 (Sensor & GNSS Data Acquisition)
 *
 * Tests the REAL [GnssQualityMonitor.computeQualityScore] companion object
 * method and [GnssQualityMonitor.bearingDegToEnuRad] — no local formula
 * copies, per §7 guardrail #1 ("never maintain two implementations of the
 * same formula").
 *
 * Covers:
 *   - Perfect conditions → score ≈ 1.0
 *   - Worst conditions → score ≈ 0.0
 *   - Each sub-score axis independently (satellites, accuracy, fix-age)
 *   - Edge cases: zero satellites, very large accuracy, negative fix-age
 *   - Boundary values at ideal/min/max thresholds
 *   - Bearing conversion: known angles, edge cases
 *
 * NOTE: GnssQualityMonitor requires Android context for the location
 * provider — that part is tested via instrumented (Espresso) tests on a
 * device.  These JUnit tests exercise the scoring and bearing math only,
 * which is pure Kotlin in the companion object and runs on the JVM.
 */
package com.sih.deadreckoning.sensor

import org.junit.Assert.*
import org.junit.Test

class GnssQualityScoreTest {

    /*
     * All tests call GnssQualityMonitor.computeQualityScore() directly —
     * the companion object method — so we are testing the actual production
     * code, not a local copy.
     */

    /* ── Test: perfect conditions ──────────────────────────────────────────── */

    @Test
    fun `perfect conditions yield score near 1_0`() {
        val score = GnssQualityMonitor.computeQualityScore(
            satelliteCount = 12,   // well above ideal
            accuracyM = 2.0f,      // below ideal threshold
            fixAgeMs = 100          // very fresh
        )
        assertEquals(1.0f, score, 0.001f)
    }

    /* ── Test: worst conditions ────────────────────────────────────────────── */

    @Test
    fun `worst conditions yield score 0_0`() {
        val score = GnssQualityMonitor.computeQualityScore(
            satelliteCount = 0,
            accuracyM = 50.0f,    // above max
            fixAgeMs = 5000        // very stale
        )
        assertEquals(0.0f, score, 0.001f)
    }

    /* ── Test: satellite axis ──────────────────────────────────────────────── */

    @Test
    fun `score improves with more satellites`() {
        val low  = GnssQualityMonitor.computeQualityScore(satelliteCount = 2, accuracyM = 5.0f, fixAgeMs = 300)
        val mid  = GnssQualityMonitor.computeQualityScore(satelliteCount = 5, accuracyM = 5.0f, fixAgeMs = 300)
        val high = GnssQualityMonitor.computeQualityScore(satelliteCount = 10, accuracyM = 5.0f, fixAgeMs = 300)

        assertTrue("mid > low", mid > low)
        assertTrue("high > mid", high > mid)
    }

    @Test
    fun `zero satellites gives satellite sub-score 0`() {
        // With perfect accuracy and fix-age, score should be < 1.0
        // because sat sub-score is 0 (weighted 0.30)
        val score = GnssQualityMonitor.computeQualityScore(satelliteCount = 0, accuracyM = 2.0f, fixAgeMs = 100)
        // Expected: 0.0 * 0.30 + 1.0 * 0.45 + 1.0 * 0.25 = 0.70
        assertEquals(0.70f, score, 0.001f)
        assertTrue(score < 1.0f)
    }

    @Test
    fun `exactly 8 satellites gives satellite sub-score 1`() {
        val scoreAtIdeal = GnssQualityMonitor.computeQualityScore(satelliteCount = 8, accuracyM = 2.0f, fixAgeMs = 100)
        val scoreAboveIdeal = GnssQualityMonitor.computeQualityScore(satelliteCount = 13, accuracyM = 2.0f, fixAgeMs = 100)
        assertEquals(scoreAtIdeal, scoreAboveIdeal, 0.001f)  // both should be 1.0
    }

    /* ── Test: accuracy axis ───────────────────────────────────────────────── */

    @Test
    fun `score degrades with worse accuracy`() {
        val good = GnssQualityMonitor.computeQualityScore(satelliteCount = 8, accuracyM = 3.0f, fixAgeMs = 300)
        val ok   = GnssQualityMonitor.computeQualityScore(satelliteCount = 8, accuracyM = 15.0f, fixAgeMs = 300)
        val bad  = GnssQualityMonitor.computeQualityScore(satelliteCount = 8, accuracyM = 28.0f, fixAgeMs = 300)

        assertTrue("good > ok", good > ok)
        assertTrue("ok > bad", ok > bad)
    }

    @Test
    fun `accuracy at max gives accuracy sub-score 0`() {
        // With perfect sats and fix-age, missing accuracy contribution only
        val score = GnssQualityMonitor.computeQualityScore(satelliteCount = 10, accuracyM = 30.0f, fixAgeMs = 100)
        // Expected: 1.0 * 0.30 + 0.0 * 0.45 + 1.0 * 0.25 = 0.55
        assertEquals(0.55f, score, 0.001f)
    }

    /* ── Test: fix-age axis ────────────────────────────────────────────────── */

    @Test
    fun `score degrades with staler fixes`() {
        val fresh = GnssQualityMonitor.computeQualityScore(satelliteCount = 8, accuracyM = 5.0f, fixAgeMs = 200)
        val stale = GnssQualityMonitor.computeQualityScore(satelliteCount = 8, accuracyM = 5.0f, fixAgeMs = 2000)
        val dead  = GnssQualityMonitor.computeQualityScore(satelliteCount = 8, accuracyM = 5.0f, fixAgeMs = 5000)

        assertTrue("fresh > stale", fresh > stale)
        assertTrue("stale > dead", stale > dead)
    }

    @Test
    fun `fix-age at max gives age sub-score 0`() {
        val score = GnssQualityMonitor.computeQualityScore(satelliteCount = 10, accuracyM = 2.0f, fixAgeMs = 3000)
        // Expected: 1.0 * 0.30 + 1.0 * 0.45 + 0.0 * 0.25 = 0.75
        assertEquals(0.75f, score, 0.001f)
    }

    /* ── Boundary values ───────────────────────────────────────────────────── */

    @Test
    fun `score is always in 0 to 1 range for extreme inputs`() {
        val extremes = listOf(
            GnssQualityMonitor.computeQualityScore(satelliteCount = -1, accuracyM = -10.0f, fixAgeMs = -1000),
            GnssQualityMonitor.computeQualityScore(satelliteCount = 100, accuracyM = 0.001f, fixAgeMs = 0),
            GnssQualityMonitor.computeQualityScore(satelliteCount = 0, accuracyM = 9999.0f, fixAgeMs = 999999),
            GnssQualityMonitor.computeQualityScore(satelliteCount = Int.MAX_VALUE, accuracyM = Float.MAX_VALUE, fixAgeMs = Long.MAX_VALUE),
        )

        for (score in extremes) {
            assertTrue("Score $score should be in [0,1]", score in 0.0f..1.0f)
        }
    }

    /* ── Bearing conversion tests ──────────────────────────────────────────── */

    @Test
    fun `bearing 0 deg (North) converts to pi_over_2 rad (East+90)`() {
        // North (0° CW from North) → 90° in ENU → π/2 rad
        val enuRad = GnssQualityMonitor.bearingDegToEnuRad(0f)
        assertEquals(Math.PI.toFloat() / 2f, enuRad, 0.001f)
    }

    @Test
    fun `bearing 90 deg (East) converts to 0 rad`() {
        // East (90° CW from North) → 0° in ENU → 0 rad
        val enuRad = GnssQualityMonitor.bearingDegToEnuRad(90f)
        assertEquals(0f, enuRad, 0.001f)
    }

    @Test
    fun `bearing 180 deg (South) converts to minus_pi_over_2 rad`() {
        // South (180° CW from North) → -90° in ENU → -π/2 rad
        val enuRad = GnssQualityMonitor.bearingDegToEnuRad(180f)
        assertEquals(-Math.PI.toFloat() / 2f, enuRad, 0.001f)
    }

    @Test
    fun `bearing 270 deg (West) converts to pi or minus_pi`() {
        // West (270° CW from North) → -180° in ENU → ±π rad
        val enuRad = GnssQualityMonitor.bearingDegToEnuRad(270f)
        // Both π and -π are valid normalizations; check magnitude
        assertEquals(Math.PI.toFloat(), Math.abs(enuRad), 0.01f)
    }

    @Test
    fun `bearing 45 deg (NE) converts to pi_over_4 rad`() {
        // NE (45° CW from North) → 45° in ENU → π/4 rad
        val enuRad = GnssQualityMonitor.bearingDegToEnuRad(45f)
        assertEquals(Math.PI.toFloat() / 4f, enuRad, 0.001f)
    }

    @Test
    fun `bearing 360 deg wraps correctly`() {
        // 360° = 0° (North) → π/2 rad
        val enuRad = GnssQualityMonitor.bearingDegToEnuRad(360f)
        assertEquals(Math.PI.toFloat() / 2f, enuRad, 0.01f)
    }

    /* ── Test: GnssFix data class ──────────────────────────────────────────── */

    @Test
    fun `GnssFix data class stores all fields correctly`() {
        val fix = GnssFix(
            timestampMs = 1695000000000L,
            latDeg = 28.6139,
            lonDeg = 77.2090,
            speedMps = 16.67f,
            bearingDeg = 45.0f,
            bearingEnuRad = GnssQualityMonitor.bearingDegToEnuRad(45.0f),
            accuracyM = 3.5f,
            satelliteCount = 12,
            qualityScore = 0.95f
        )

        assertEquals(1695000000000L, fix.timestampMs)
        assertEquals(28.6139, fix.latDeg, 0.0001)
        assertEquals(77.2090, fix.lonDeg, 0.0001)
        assertEquals(16.67f, fix.speedMps, 0.01f)
        assertEquals(45.0f, fix.bearingDeg, 0.01f)
        assertEquals(Math.PI.toFloat() / 4f, fix.bearingEnuRad, 0.01f)
        assertEquals(3.5f, fix.accuracyM, 0.01f)
        assertEquals(12, fix.satelliteCount)
        assertEquals(0.95f, fix.qualityScore, 0.01f)
    }

    /* ── Test: ImuSample data class ────────────────────────────────────────── */

    @Test
    fun `ImuSample data class stores all fields correctly`() {
        val sample = ImuSample(
            timestampMs = 1695000000100L,
            ax = 0.5f, ay = -0.3f, az = 9.81f,
            gx = 0.01f, gy = -0.02f, gz = 0.005f
        )

        assertEquals(1695000000100L, sample.timestampMs)
        assertEquals(0.5f, sample.ax, 0.001f)
        assertEquals(-0.3f, sample.ay, 0.001f)
        assertEquals(9.81f, sample.az, 0.001f)
        assertEquals(0.01f, sample.gx, 0.001f)
        assertEquals(-0.02f, sample.gy, 0.001f)
        assertEquals(0.005f, sample.gz, 0.001f)
    }
}
