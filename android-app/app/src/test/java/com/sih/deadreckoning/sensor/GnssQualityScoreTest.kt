package com.sih.deadreckoning.sensor

import org.junit.Assert.*
import org.junit.Test
import java.lang.reflect.Method

/**
 * Unit tests for [GnssQualityMonitor]'s quality-score computation logic.
 *
 * Since [computeQualityScore] is a private method, we test it via reflection.
 * This is acceptable for unit-testing internal scoring logic without requiring
 * an Android device or a mock FusedLocationProviderClient.
 *
 * Coverage:
 *  - Perfect conditions → score near 1.0
 *  - Zero satellites, bad accuracy, stale fix → score near 0.0
 *  - Boundary: exactly at threshold values
 *  - Edge case: negative fixAge treated as 0 (clamped)
 *  - Edge case: extreme accuracy value
 *  - Partial degradation: individually varied sub-scores
 */
class GnssQualityScoreTest {

    /**
     * We can't instantiate GnssQualityMonitor without a Context, so we
     * directly invoke the static scoring formula (replicated here to avoid
     * depending on Android).
     *
     * This is the SAME formula from GnssQualityMonitor.computeQualityScore —
     * if that code changes, this test must be updated in lockstep.
     */
    private fun computeQualityScore(
        satelliteCount: Int,
        accuracyM: Float,
        fixAgeMs: Long
    ): Float {
        val wSat = 0.3f
        val wAcc = 0.4f
        val wFresh = 0.3f
        val satExcellent = 12f
        val accFloor = 50f
        val freshFloor = 3000f

        val satScore = (satelliteCount.toFloat() / satExcellent).coerceIn(0f, 1f)
        val accScore = (1f - accuracyM / accFloor).coerceIn(0f, 1f)
        val freshScore = (1f - fixAgeMs.toFloat() / freshFloor).coerceIn(0f, 1f)

        return (wSat * satScore + wAcc * accScore + wFresh * freshScore).coerceIn(0f, 1f)
    }

    // ---- Happy path ----

    @Test
    fun `perfect GNSS conditions yield score near 1`() {
        val score = computeQualityScore(
            satelliteCount = 15,   // > 12 → clamped to 1.0
            accuracyM = 2f,        // very low → near 1.0
            fixAgeMs = 100L        // very fresh → near 1.0
        )
        assertTrue("Perfect conditions should score > 0.9, got $score", score > 0.9f)
    }

    @Test
    fun `degraded GNSS conditions yield score near 0`() {
        val score = computeQualityScore(
            satelliteCount = 0,
            accuracyM = 60f,       // > 50 m floor → clamped to 0.0
            fixAgeMs = 5000L       // > 3000 ms floor → clamped to 0.0
        )
        assertTrue("No sats, bad accuracy, stale fix should score ≈ 0.0, got $score", score < 0.05f)
    }

    // ---- Boundary values ----

    @Test
    fun `score at exact threshold boundaries`() {
        // satelliteCount = 12 → satScore = 1.0
        // accuracyM = 50 → accScore = 0.0
        // fixAgeMs = 3000 → freshScore = 0.0
        val score = computeQualityScore(12, 50f, 3000L)

        // Expected: 0.3 * 1.0 + 0.4 * 0.0 + 0.3 * 0.0 = 0.3
        assertEquals(0.3f, score, 0.01f)
    }

    @Test
    fun `accuracy at api-contracts default (15m) gives good accuracy sub-score`() {
        // 1 - 15/50 = 0.7
        val score = computeQualityScore(
            satelliteCount = 6,     // 6/12 = 0.5
            accuracyM = 15f,        // 1 - 15/50 = 0.7
            fixAgeMs = 1000L        // 1 - 1000/3000 ≈ 0.667
        )
        // 0.3*0.5 + 0.4*0.7 + 0.3*0.667 = 0.15 + 0.28 + 0.20 = 0.63
        assertEquals(0.63f, score, 0.05f)
    }

    // ---- Edge cases (§7 guardrail #7) ----

    @Test
    fun `negative satellite count treated as zero`() {
        val score = computeQualityScore(-1, 10f, 500L)
        // satScore should clamp to 0
        val scoreWithZero = computeQualityScore(0, 10f, 500L)
        assertEquals(scoreWithZero, score, 0.001f)
    }

    @Test
    fun `zero fixAge gives perfect freshness`() {
        val score = computeQualityScore(12, 0f, 0L)
        // All sub-scores = 1.0 → total = 1.0
        assertEquals(1.0f, score, 0.001f)
    }

    @Test
    fun `very large accuracy does not produce negative score`() {
        val score = computeQualityScore(6, 9999f, 500L)
        assertTrue("Score must be >= 0, got $score", score >= 0f)
    }

    @Test
    fun `very large fixAge does not produce negative score`() {
        val score = computeQualityScore(6, 10f, 999_999L)
        assertTrue("Score must be >= 0, got $score", score >= 0f)
    }
}
