/**
 * SensorDataBufferTest.kt — Unit tests for the ring buffer
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 2 (Sensor & GNSS Data Acquisition)
 *
 * Covers:
 *   - Happy path (push + getWindow retrieval)
 *   - Ring wraparound (capacity overflow)
 *   - Empty buffer (edge case: §7 guardrail #7)
 *   - Single-sample buffer
 *   - Window larger than buffer contents
 *   - Count-based retrieval (getLatest)
 *   - Clear + re-use
 *   - Zero / negative windowMs (bad input)
 */
package com.sih.deadreckoning.sensor

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class SensorDataBufferTest {

    /** Helper to create a sample with a given timestamp and a recognizable
     *  accel-x value (used for ordering verification). */
    private fun sample(timestampMs: Long, ax: Float = timestampMs.toFloat()): ImuSample =
        ImuSample(
            timestampMs = timestampMs,
            ax = ax, ay = 0f, az = 9.81f,
            gx = 0f, gy = 0f, gz = 0f
        )

    private lateinit var buffer: SensorDataBuffer

    @Before
    fun setUp() {
        buffer = SensorDataBuffer(capacity = 8)
    }

    /* ── Empty buffer (edge case) ──────────────────────────────────────────── */

    @Test
    fun `getWindow on empty buffer returns empty list`() {
        assertTrue(buffer.isEmpty())
        assertEquals(0, buffer.size())
        assertTrue(buffer.getWindow(1000).isEmpty())
    }

    @Test
    fun `peekLatest on empty buffer returns null`() {
        assertNull(buffer.peekLatest())
    }

    @Test
    fun `getLatest with n=0 returns empty list`() {
        buffer.push(sample(100))
        assertTrue(buffer.getLatest(0).isEmpty())
    }

    @Test
    fun `getLatest with n negative returns empty list`() {
        buffer.push(sample(100))
        assertTrue(buffer.getLatest(-5).isEmpty())
    }

    /* ── Happy path ────────────────────────────────────────────────────────── */

    @Test
    fun `push and getWindow retrieves samples in chronological order`() {
        buffer.push(sample(1000))
        buffer.push(sample(1050))
        buffer.push(sample(1100))
        buffer.push(sample(1150))

        val window = buffer.getWindow(200)  // all within 200ms of newest (1150)
        assertEquals(4, window.size)
        assertEquals(1000L, window.first().timestampMs)
        assertEquals(1150L, window.last().timestampMs)
    }

    @Test
    fun `getWindow returns only samples within the time window`() {
        buffer.push(sample(1000))  // outside 100ms window of 1300
        buffer.push(sample(1100))  // outside
        buffer.push(sample(1200))  // inside (1300 - 100 = 1200)
        buffer.push(sample(1250))  // inside
        buffer.push(sample(1300))  // inside (newest)

        val window = buffer.getWindow(100)
        assertEquals(3, window.size)
        assertEquals(1200L, window.first().timestampMs)
        assertEquals(1300L, window.last().timestampMs)
    }

    @Test
    fun `peekLatest returns the most recent sample`() {
        buffer.push(sample(100))
        buffer.push(sample(200))
        buffer.push(sample(300))

        assertEquals(300L, buffer.peekLatest()!!.timestampMs)
    }

    @Test
    fun `size reflects actual count`() {
        assertEquals(0, buffer.size())
        buffer.push(sample(100))
        assertEquals(1, buffer.size())
        buffer.push(sample(200))
        assertEquals(2, buffer.size())
    }

    /* ── Ring wraparound ───────────────────────────────────────────────────── */

    @Test
    fun `buffer wraps around at capacity and retains newest samples`() {
        // capacity = 8, push 12 samples → oldest 4 evicted
        for (i in 1..12) {
            buffer.push(sample(i * 100L))
        }

        assertEquals(8, buffer.size())
        assertEquals(1200L, buffer.peekLatest()!!.timestampMs)

        // All 8 should be the most recent: 500..1200
        val all = buffer.getWindow(Long.MAX_VALUE / 2)
        assertEquals(8, all.size)
        assertEquals(500L, all.first().timestampMs)
        assertEquals(1200L, all.last().timestampMs)
    }

    @Test
    fun `getWindow works correctly after wraparound`() {
        // Fill past capacity
        for (i in 1..12) {
            buffer.push(sample(i * 100L))
        }

        // Window of 300ms from newest (1200): should get 1000, 1100, 1200
        // Actually: 1200 - 300 = 900, so >= 900: 900 is NOT in buffer (evicted).
        // Samples 500-1200 are in buffer. Those >= 900: 900, 1000, 1100, 1200.
        // Wait, 900 IS in buffer (it's the 9th sample, but with capacity=8 and 12 pushed,
        // samples 5-12 are in buffer → timestamps 500, 600, 700, 800, 900, 1000, 1100, 1200).
        val window = buffer.getWindow(300)
        assertEquals(4, window.size)  // 900, 1000, 1100, 1200
        assertEquals(900L, window.first().timestampMs)
        assertEquals(1200L, window.last().timestampMs)
    }

    /* ── Single-sample ─────────────────────────────────────────────────────── */

    @Test
    fun `single sample in buffer is retrievable`() {
        buffer.push(sample(42))

        assertEquals(1, buffer.size())
        assertEquals(42L, buffer.peekLatest()!!.timestampMs)

        val window = buffer.getWindow(1000)
        assertEquals(1, window.size)
        assertEquals(42L, window[0].timestampMs)
    }

    /* ── Window larger than buffer contents ─────────────────────────────────── */

    @Test
    fun `getWindow with very large windowMs returns all samples`() {
        buffer.push(sample(100))
        buffer.push(sample(200))
        buffer.push(sample(300))

        val window = buffer.getWindow(999999)
        assertEquals(3, window.size)
    }

    /* ── Count-based retrieval ─────────────────────────────────────────────── */

    @Test
    fun `getLatest returns correct count in order`() {
        for (i in 1..5) buffer.push(sample(i * 100L))

        val latest3 = buffer.getLatest(3)
        assertEquals(3, latest3.size)
        assertEquals(300L, latest3[0].timestampMs)
        assertEquals(400L, latest3[1].timestampMs)
        assertEquals(500L, latest3[2].timestampMs)
    }

    @Test
    fun `getLatest with n greater than count returns all`() {
        buffer.push(sample(100))
        buffer.push(sample(200))

        val result = buffer.getLatest(50)
        assertEquals(2, result.size)
    }

    /* ── Clear ─────────────────────────────────────────────────────────────── */

    @Test
    fun `clear empties the buffer`() {
        buffer.push(sample(100))
        buffer.push(sample(200))
        buffer.clear()

        assertEquals(0, buffer.size())
        assertTrue(buffer.isEmpty())
        assertNull(buffer.peekLatest())
        assertTrue(buffer.getWindow(1000).isEmpty())
    }

    @Test
    fun `buffer is reusable after clear`() {
        buffer.push(sample(100))
        buffer.clear()
        buffer.push(sample(999))

        assertEquals(1, buffer.size())
        assertEquals(999L, buffer.peekLatest()!!.timestampMs)
    }

    /* ── Bad inputs ────────────────────────────────────────────────────────── */

    @Test(expected = IllegalArgumentException::class)
    fun `getWindow with zero windowMs throws`() {
        buffer.push(sample(100))
        buffer.getWindow(0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `getWindow with negative windowMs throws`() {
        buffer.push(sample(100))
        buffer.getWindow(-100)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `constructor with zero capacity throws`() {
        SensorDataBuffer(capacity = 0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `constructor with negative capacity throws`() {
        SensorDataBuffer(capacity = -1)
    }

    /* ── Capacity-1 edge case ──────────────────────────────────────────────── */

    @Test
    fun `buffer with capacity 1 retains only latest`() {
        val tiny = SensorDataBuffer(capacity = 1)
        tiny.push(sample(100))
        tiny.push(sample(200))
        tiny.push(sample(300))

        assertEquals(1, tiny.size())
        assertEquals(300L, tiny.peekLatest()!!.timestampMs)
    }
}
