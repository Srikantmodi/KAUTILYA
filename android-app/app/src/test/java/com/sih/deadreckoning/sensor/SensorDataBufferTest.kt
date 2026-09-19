package com.sih.deadreckoning.sensor

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [SensorDataBuffer].
 *
 * These are pure-JVM tests (no Android framework needed) because
 * SensorDataBuffer has no Android dependencies — it operates purely
 * on [ImuSample] data classes and an ArrayDeque.
 *
 * Coverage:
 *  - Happy path: push samples, retrieve window.
 *  - Decimation: verify ~10 Hz gate drops rapid-fire samples.
 *  - Ring-buffer eviction: verify capacity limit is enforced.
 *  - Edge case: empty buffer returns empty window.
 *  - Edge case: getWindow with windowMs = 0.
 *  - Edge case: push with non-monotonic timestamps.
 *  - Edge case: negative windowMs throws.
 *  - Edge case: push callback notification.
 *  - Diagnostic counters.
 */
class SensorDataBufferTest {

    private lateinit var buffer: SensorDataBuffer

    @Before
    fun setUp() {
        buffer = SensorDataBuffer(capacity = 100)
    }

    // ---- Helpers ----

    private fun makeSample(timestampMs: Long): ImuSample =
        ImuSample(
            timestampMs = timestampMs,
            ax = 0.1f, ay = 0.2f, az = 9.81f,
            gx = 0.01f, gy = 0.02f, gz = 0.03f
        )

    // ---- Happy path ----

    @Test
    fun `push and getWindow returns samples within window`() {
        // Push 10 samples at 100 ms intervals (10 Hz) → all pass decimation.
        for (i in 0 until 10) {
            buffer.push(makeSample(timestampMs = 1000L + i * 100L))
        }

        assertEquals(10, buffer.size())

        // Request a 500 ms window from the latest sample (t=1900).
        // Should include samples at t=1400..1900 → 6 samples.
        val window = buffer.getWindow(500L)
        assertEquals(6, window.size)
        assertEquals(1400L, window.first().timestampMs)
        assertEquals(1900L, window.last().timestampMs)
    }

    @Test
    fun `getWindow returns all samples when window covers entire buffer`() {
        for (i in 0 until 5) {
            buffer.push(makeSample(timestampMs = 1000L + i * 100L))
        }

        // Window larger than the buffer span.
        val window = buffer.getWindow(10000L)
        assertEquals(5, window.size)
    }

    // ---- Decimation ----

    @Test
    fun `decimation drops samples arriving faster than 10 Hz`() {
        // Push 50 samples at 20 ms intervals (50 Hz) over 1 second.
        for (i in 0 until 50) {
            buffer.push(makeSample(timestampMs = 1000L + i * 20L))
        }

        // At 100 ms decimation interval, expect ~10 samples from a 1-second span.
        // First sample at t=1000, then t=1100, t=1200, ... t=1900 → 10 samples.
        val stored = buffer.size()
        assertTrue(
            "Expected ~10 decimated samples from 50 raw, got $stored",
            stored in 9..11
        )

        assertEquals(50L, buffer.rawSampleCount())
        assertEquals(stored.toLong(), buffer.decimatedSampleCount())
    }

    @Test
    fun `decimation passes samples exactly at 100 ms intervals`() {
        buffer.push(makeSample(1000L))
        buffer.push(makeSample(1050L))  // too soon — dropped
        buffer.push(makeSample(1099L))  // still too soon — dropped
        buffer.push(makeSample(1100L))  // exactly 100 ms later — accepted

        assertEquals(2, buffer.size())
    }

    // ---- Ring buffer capacity ----

    @Test
    fun `buffer evicts oldest samples when capacity exceeded`() {
        val smallBuffer = SensorDataBuffer(capacity = 5)

        for (i in 0 until 10) {
            smallBuffer.push(makeSample(timestampMs = 1000L + i * 100L))
        }

        assertEquals(5, smallBuffer.size())

        // Oldest retained should be sample at t=1500 (samples 0–4 evicted).
        val window = smallBuffer.getWindow(10000L)
        assertEquals(1500L, window.first().timestampMs)
        assertEquals(1900L, window.last().timestampMs)
    }

    // ---- Edge cases (§7 guardrail #7) ----

    @Test
    fun `getWindow on empty buffer returns empty list`() {
        val window = buffer.getWindow(1000L)
        assertTrue(window.isEmpty())
    }

    @Test
    fun `getLatest on empty buffer returns null`() {
        assertNull(buffer.getLatest())
    }

    @Test
    fun `getWindow with windowMs zero returns only the latest sample`() {
        buffer.push(makeSample(1000L))
        buffer.push(makeSample(1100L))
        buffer.push(makeSample(1200L))

        val window = buffer.getWindow(0L)
        assertEquals(1, window.size)
        assertEquals(1200L, window.first().timestampMs)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `getWindow with negative windowMs throws`() {
        buffer.getWindow(-100L)
    }

    @Test
    fun `non-monotonic timestamps are handled gracefully`() {
        // In real life timestamps might jitter slightly or a stale event
        // could arrive. The buffer should not crash — it just applies the
        // decimation gate as normal (stale timestamps won't pass the gate).
        buffer.push(makeSample(1000L))
        buffer.push(makeSample(1100L))
        buffer.push(makeSample(1050L))  // backwards — will NOT pass gate (1050 < 1100 + 100)
        buffer.push(makeSample(1200L))

        assertEquals(3, buffer.size())
    }

    @Test
    fun `clear resets buffer and counters`() {
        buffer.push(makeSample(1000L))
        buffer.push(makeSample(1100L))
        assertEquals(2, buffer.size())

        buffer.clear()
        assertEquals(0, buffer.size())
        assertEquals(0L, buffer.rawSampleCount())
        assertEquals(0L, buffer.decimatedSampleCount())
        assertNull(buffer.getLatest())
    }

    // ---- Push callback ----

    @Test
    fun `onDecimatedSample callback fires for accepted samples only`() {
        val received = mutableListOf<Long>()
        val callbackBuffer = SensorDataBuffer(
            capacity = 100,
            onDecimatedSample = { received.add(it.timestampMs) }
        )

        callbackBuffer.push(makeSample(1000L))   // accepted → callback
        callbackBuffer.push(makeSample(1050L))   // dropped → no callback
        callbackBuffer.push(makeSample(1100L))   // accepted → callback

        assertEquals(listOf(1000L, 1100L), received)
    }

    // ---- getLatest ----

    @Test
    fun `getLatest returns the most recent decimated sample`() {
        buffer.push(makeSample(1000L))
        buffer.push(makeSample(1100L))
        buffer.push(makeSample(1200L))

        val latest = buffer.getLatest()
        assertNotNull(latest)
        assertEquals(1200L, latest!!.timestampMs)
    }

    // ---- Window ordering ----

    @Test
    fun `getWindow returns samples in chronological order`() {
        for (i in 0 until 20) {
            buffer.push(makeSample(timestampMs = 1000L + i * 100L))
        }

        val window = buffer.getWindow(500L)
        for (i in 1 until window.size) {
            assertTrue(
                "Window must be chronologically ordered",
                window[i].timestampMs >= window[i - 1].timestampMs
            )
        }
    }
}
