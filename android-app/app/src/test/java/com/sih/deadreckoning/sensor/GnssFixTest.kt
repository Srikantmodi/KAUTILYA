package com.sih.deadreckoning.sensor

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for the [GnssFix] data class.
 *
 * Since [GnssQualityMonitor] requires Android Context/FusedLocationProviderClient
 * and needs androidTest for integration, these tests validate:
 *  - The data class shape matches api-contracts.md §1 exactly.
 *  - Edge-case values (zeros, extremes, boundary coordinates) don't crash.
 *  - Data class equality/copy semantics.
 *
 * Coverage:
 *  - Happy path: typical GNSS fix.
 *  - Edge case: no fix (zero/null-equivalent values).
 *  - Edge case: boundary coordinates (poles, antimeridian).
 *  - Edge case: quality score boundaries.
 *  - Data class contract verification.
 */
class GnssFixTest {

    // ---- Happy path ----

    @Test
    fun `typical GNSS fix constructs correctly`() {
        val fix = GnssFix(
            timestampMs = 1695100000000L,
            latDeg = 28.6139,       // Delhi
            lonDeg = 77.2090,
            speedMps = 16.67f,      // ~60 km/h
            bearingDeg = 45.0f,     // NE
            accuracyM = 5.0f,
            satelliteCount = 10,
            qualityScore = 0.85f
        )

        assertEquals(1695100000000L, fix.timestampMs)
        assertEquals(28.6139, fix.latDeg, 1e-4)
        assertEquals(77.2090, fix.lonDeg, 1e-4)
        assertEquals(16.67f, fix.speedMps, 1e-2f)
        assertEquals(45.0f, fix.bearingDeg, 1e-2f)
        assertEquals(5.0f, fix.accuracyM, 1e-2f)
        assertEquals(10, fix.satelliteCount)
        assertEquals(0.85f, fix.qualityScore, 1e-2f)
    }

    // ---- Edge cases (§7 guardrail #7) ----

    @Test
    fun `no-fix state with zeros does not crash`() {
        // Before the first GNSS fix arrives, all values are zero/default.
        val fix = GnssFix(
            timestampMs = 0L,
            latDeg = 0.0, lonDeg = 0.0,
            speedMps = 0f, bearingDeg = 0f,
            accuracyM = Float.MAX_VALUE,
            satelliteCount = 0,
            qualityScore = 0f
        )
        assertEquals(0, fix.satelliteCount)
        assertEquals(0f, fix.qualityScore, 0f)
    }

    @Test
    fun `boundary coordinates at poles and antimeridian`() {
        // North pole
        val northPole = GnssFix(
            timestampMs = 1000L,
            latDeg = 90.0, lonDeg = 0.0,
            speedMps = 0f, bearingDeg = 0f,
            accuracyM = 10f, satelliteCount = 6,
            qualityScore = 0.5f
        )
        assertEquals(90.0, northPole.latDeg, 0.0)

        // Antimeridian crossing
        val antiMeridian = GnssFix(
            timestampMs = 1000L,
            latDeg = 0.0, lonDeg = -180.0,
            speedMps = 0f, bearingDeg = 0f,
            accuracyM = 10f, satelliteCount = 6,
            qualityScore = 0.5f
        )
        assertEquals(-180.0, antiMeridian.lonDeg, 0.0)
    }

    @Test
    fun `quality score at exact boundaries`() {
        val fixLow = GnssFix(
            timestampMs = 1000L,
            latDeg = 0.0, lonDeg = 0.0,
            speedMps = 0f, bearingDeg = 0f,
            accuracyM = 10f, satelliteCount = 4,
            qualityScore = 0.0f  // minimum
        )
        assertEquals(0f, fixLow.qualityScore, 0f)

        val fixHigh = fixLow.copy(qualityScore = 1.0f) // maximum
        assertEquals(1.0f, fixHigh.qualityScore, 0f)
    }

    @Test
    fun `negative speed does not crash`() {
        // Some devices may report negative speed briefly on GNSS glitches.
        val fix = GnssFix(
            timestampMs = 1000L,
            latDeg = 28.0, lonDeg = 77.0,
            speedMps = -1.0f,
            bearingDeg = 0f,
            accuracyM = 10f, satelliteCount = 8,
            qualityScore = 0.7f
        )
        assertTrue(fix.speedMps < 0f)
    }

    @Test
    fun `bearing wraps at 360 degrees`() {
        // Android reports bearing in [0, 360). Ensure 359.9 is storable.
        val fix = GnssFix(
            timestampMs = 1000L,
            latDeg = 28.0, lonDeg = 77.0,
            speedMps = 10f,
            bearingDeg = 359.9f,
            accuracyM = 5f, satelliteCount = 10,
            qualityScore = 0.9f
        )
        assertTrue(fix.bearingDeg > 359f)
    }

    // ---- Data class contract: matches api-contracts.md §1 ----

    @Test
    fun `GnssFix has exactly 8 fields matching api-contracts`() {
        // api-contracts.md §1: timestampMs, latDeg, lonDeg, speedMps,
        //                      bearingDeg, accuracyM, satelliteCount, qualityScore
        val fix = GnssFix(1000L, 28.0, 77.0, 10f, 45f, 5f, 10, 0.8f)
        val (ts, lat, lon, speed, bearing, acc, sats, quality) = fix
        assertEquals(1000L, ts)
        assertEquals(28.0, lat, 0.0)
        assertEquals(10, sats)
        assertEquals(0.8f, quality, 0f)
    }

    @Test
    fun `data class equality for identical fixes`() {
        val a = GnssFix(1000L, 28.0, 77.0, 10f, 45f, 5f, 10, 0.8f)
        val b = GnssFix(1000L, 28.0, 77.0, 10f, 45f, 5f, 10, 0.8f)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
    }

    @Test
    fun `copy preserves unmodified fields`() {
        val original = GnssFix(1000L, 28.0, 77.0, 10f, 45f, 5f, 10, 0.8f)
        val modified = original.copy(qualityScore = 0.3f)
        assertEquals(28.0, modified.latDeg, 0.0)
        assertEquals(0.3f, modified.qualityScore, 0f)
    }
}
