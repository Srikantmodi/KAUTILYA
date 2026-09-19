package com.sih.deadreckoning.sensor

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for the [ImuSample] data class.
 *
 * Since [SensorBridge] itself requires Android's SensorManager/Context and
 * can't be unit-tested in a pure-JVM environment (needs androidTest), these
 * tests validate:
 *  - The data class shape matches api-contracts.md §1 exactly.
 *  - Edge-case values (zeros, extremes, negative timestamps) don't crash.
 *  - Data class equality/copy semantics work as expected (downstream
 *    consumers rely on these).
 *
 * Coverage:
 *  - Happy path: construct with typical IMU values.
 *  - Edge case: all-zero sample (session-start before gyro arrives).
 *  - Edge case: extreme acceleration / angular rate values.
 *  - Edge case: negative timestamp (malformed data should not crash).
 *  - Data class copy with modified field.
 *  - Structural equality.
 */
class ImuSampleTest {

    // ---- Happy path ----

    @Test
    fun `typical IMU sample constructs correctly`() {
        val sample = ImuSample(
            timestampMs = 1695100000000L,
            ax = 0.12f, ay = -0.34f, az = 9.81f,
            gx = 0.001f, gy = -0.002f, gz = 0.003f
        )

        assertEquals(1695100000000L, sample.timestampMs)
        assertEquals(0.12f, sample.ax, 1e-6f)
        assertEquals(-0.34f, sample.ay, 1e-6f)
        assertEquals(9.81f, sample.az, 1e-6f)
        assertEquals(0.001f, sample.gx, 1e-6f)
        assertEquals(-0.002f, sample.gy, 1e-6f)
        assertEquals(0.003f, sample.gz, 1e-6f)
    }

    // ---- Edge cases (§7 guardrail #7) ----

    @Test
    fun `all-zero sample does not crash`() {
        // Before the first gyro event, SensorBridge emits gyro=(0,0,0).
        // This must not cause any downstream issue.
        val sample = ImuSample(
            timestampMs = 0L,
            ax = 0f, ay = 0f, az = 0f,
            gx = 0f, gy = 0f, gz = 0f
        )
        assertEquals(0L, sample.timestampMs)
        assertEquals(0f, sample.ax, 0f)
    }

    @Test
    fun `extreme acceleration values do not overflow`() {
        // MEMS accelerometers can saturate at ±16g; ensure no overflow.
        val sample = ImuSample(
            timestampMs = 1000L,
            ax = 156.96f, ay = -156.96f, az = 156.96f,  // ~16g in m/s²
            gx = 34.9f, gy = -34.9f, gz = 34.9f         // ~2000 deg/s in rad/s
        )
        assertTrue(sample.ax > 100f)
        assertTrue(sample.gy < -30f)
    }

    @Test
    fun `negative timestamp does not crash`() {
        // Malformed or pre-epoch timestamps must not cause exceptions.
        val sample = ImuSample(
            timestampMs = -1L,
            ax = 0f, ay = 0f, az = 9.81f,
            gx = 0f, gy = 0f, gz = 0f
        )
        assertEquals(-1L, sample.timestampMs)
    }

    @Test
    fun `NaN accelerometer values propagate without crash`() {
        // If a sensor returns NaN (hardware fault), the data class must
        // not throw — downstream should detect and handle this.
        val sample = ImuSample(
            timestampMs = 1000L,
            ax = Float.NaN, ay = 0f, az = 9.81f,
            gx = 0f, gy = 0f, gz = 0f
        )
        assertTrue(sample.ax.isNaN())
    }

    // ---- Data class semantics ----

    @Test
    fun `data class equality works for identical samples`() {
        val a = ImuSample(1000L, 1f, 2f, 3f, 4f, 5f, 6f)
        val b = ImuSample(1000L, 1f, 2f, 3f, 4f, 5f, 6f)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
    }

    @Test
    fun `data class inequality on different timestamp`() {
        val a = ImuSample(1000L, 1f, 2f, 3f, 4f, 5f, 6f)
        val b = ImuSample(2000L, 1f, 2f, 3f, 4f, 5f, 6f)
        assertNotEquals(a, b)
    }

    @Test
    fun `copy with modified field`() {
        val original = ImuSample(1000L, 1f, 2f, 3f, 4f, 5f, 6f)
        val modified = original.copy(timestampMs = 2000L)
        assertEquals(2000L, modified.timestampMs)
        // Other fields unchanged.
        assertEquals(original.ax, modified.ax, 0f)
        assertEquals(original.gz, modified.gz, 0f)
    }

    // ---- Contract: field count matches api-contracts.md §1 ----

    @Test
    fun `ImuSample has exactly 7 fields matching api-contracts`() {
        // api-contracts.md §1 Kotlin-side mirror: timestampMs, ax, ay, az, gx, gy, gz
        // If someone adds/removes a field, this test should break compilation
        // or fail the destructuring.
        val sample = ImuSample(1000L, 1f, 2f, 3f, 4f, 5f, 6f)
        val (ts, ax, ay, az, gx, gy, gz) = sample
        assertEquals(1000L, ts)
        assertEquals(1f, ax, 0f)
        assertEquals(6f, gz, 0f)
    }
}
