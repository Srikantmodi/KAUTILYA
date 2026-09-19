/**
 * SensorDataBuffer.kt — 10 Hz timestamped ring buffer
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 2 (Sensor & GNSS Data Acquisition)
 *
 * This is the "10Hz timestamped ring buffer" hop named explicitly in the PRD's
 * canonical data flow (§2.1): it sits between [SensorBridge] (which fires at
 * the device's native sensor rate, often 50–200 Hz) and everything downstream
 * (calibration, inference, fusion).
 *
 * Downstream consumers pull windows of samples via [getWindow] — a pull-based
 * API matching the ⚠️ ASSUMPTION in api-contracts.md §5.  If Member 6 later
 * prefers push, we add a push-overlay; the core ring buffer stays the same.
 *
 * Thread safety: all public methods are safe to call from any thread.
 * Internally, access to the ring buffer is guarded by a lock-free approach:
 * the buffer itself is a fixed-size array written by a single producer
 * (sensor thread) and read by one or more consumers.  We use a
 * [synchronized] block on a dedicated lock object for simplicity and
 * correctness — the critical section is trivially small (array copy) so
 * contention is negligible relative to the 100ms inference interval.
 *
 * References:
 *   PRD §2.1   — "10Hz timestamped ring buffer"
 *   api-contracts.md §5  — push / getWindow interface
 */
package com.sih.deadreckoning.sensor

/**
 * Fixed-capacity circular buffer of [ImuSample]s.
 *
 * @param capacity  maximum number of samples to retain.  Default 512 is
 *                  comfortably larger than the ~200 samples a 2-second window
 *                  at 100 Hz would require, providing headroom for bursty
 *                  sensor delivery without silently dropping data that a
 *                  downstream consumer hasn't read yet.
 */
class SensorDataBuffer(private val capacity: Int = 512) {

    init {
        require(capacity > 0) { "Buffer capacity must be > 0, was $capacity" }
    }

    /*
     * Ring buffer internals.  `buffer` is pre-allocated at construction.
     * `head` is the index of the NEXT write position.  `count` tracks how
     * many valid entries exist (≤ capacity).
     *
     * Invariants:
     *   - buffer[(head - count + capacity) % capacity] is the oldest entry
     *   - buffer[(head - 1 + capacity) % capacity] is the newest entry
     *   - 0 <= count <= capacity
     */
    private val buffer: Array<ImuSample?> = arrayOfNulls(capacity)
    private var head: Int = 0
    private var count: Int = 0
    private val lock = Any()

    /* ── Public API (matches api-contracts.md §5) ──────────────────────────── */

    /**
     * Append a new sample to the ring buffer.
     *
     * If the buffer is at capacity, the oldest sample is silently overwritten.
     * This is the intended behaviour — the PRD does not require back-pressure
     * or loss-signalling from the sensor buffer.
     *
     * @param sample  raw [ImuSample] from [SensorBridge].
     */
    fun push(sample: ImuSample) {
        synchronized(lock) {
            buffer[head] = sample
            head = (head + 1) % capacity
            if (count < capacity) count++
        }
    }

    /**
     * Retrieve the most recent samples spanning up to [windowMs] milliseconds.
     *
     * Walks backward from the newest sample, collecting all samples whose
     * timestamp falls within `[newest.timestampMs - windowMs, newest.timestampMs]`.
     *
     * @param windowMs  window duration in milliseconds (e.g. 1000 for 1 s,
     *                  2000 for 2 s).  Must be > 0.
     * @return  an ordered list (oldest-first) of samples within the window.
     *          Returns an empty list if the buffer is empty.
     */
    fun getWindow(windowMs: Long): List<ImuSample> {
        require(windowMs > 0) { "windowMs must be > 0, was $windowMs" }

        synchronized(lock) {
            if (count == 0) return emptyList()

            val result = mutableListOf<ImuSample>()

            // Index of the newest sample
            val newestIdx = (head - 1 + capacity) % capacity
            val newest = buffer[newestIdx]
                ?: return emptyList()  // defensive — should never happen when count > 0

            val cutoffMs = newest.timestampMs - windowMs

            // Walk backward from newest
            for (i in 0 until count) {
                val idx = (head - 1 - i + capacity * 2) % capacity
                val sample = buffer[idx] ?: break

                if (sample.timestampMs < cutoffMs) break

                result.add(sample)
            }

            // Reverse so the list is oldest-first (chronological order)
            result.reverse()
            return result
        }
    }

    /**
     * Retrieve the N most recent samples, regardless of time span.
     *
     * Useful for fixed-size-window inference models or for diagnostics.
     *
     * @param n  number of samples to retrieve.  Clamped to [0, count].
     * @return  an ordered list (oldest-first) of the most recent [n] samples.
     */
    fun getLatest(n: Int): List<ImuSample> {
        synchronized(lock) {
            val take = n.coerceIn(0, count)
            if (take == 0) return emptyList()

            val result = mutableListOf<ImuSample>()
            for (i in 0 until take) {
                val idx = (head - take + i + capacity) % capacity
                val sample = buffer[idx] ?: continue
                result.add(sample)
            }
            return result
        }
    }

    /**
     * @return the most recent sample, or `null` if the buffer is empty.
     */
    fun peekLatest(): ImuSample? {
        synchronized(lock) {
            if (count == 0) return null
            return buffer[(head - 1 + capacity) % capacity]
        }
    }

    /**
     * @return the number of valid samples currently in the buffer.
     */
    fun size(): Int = synchronized(lock) { count }

    /**
     * @return `true` if the buffer contains no samples.
     */
    fun isEmpty(): Boolean = synchronized(lock) { count == 0 }

    /**
     * Remove all samples from the buffer.
     */
    fun clear() {
        synchronized(lock) {
            buffer.fill(null)
            head = 0
            count = 0
        }
    }
}
