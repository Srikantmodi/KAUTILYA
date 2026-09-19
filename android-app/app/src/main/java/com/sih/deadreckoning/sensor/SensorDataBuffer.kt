package com.sih.deadreckoning.sensor

import androidx.annotation.GuardedBy

/**
 * SensorDataBuffer — the "10Hz timestamped ring buffer" from PRD §2.1.
 *
 * Sits between [SensorBridge] (which fires at hardware rate, ~50–100 Hz)
 * and everything downstream (AlignmentManager, TFLiteInferenceEngine, etc.).
 *
 * Responsibilities:
 *  1. Accept every raw [ImuSample] via [push].
 *  2. Decimate to ~10 Hz so downstream consumers see a consistent rate
 *     regardless of device-specific accelerometer frequency.
 *  3. Store the decimated samples in a fixed-capacity ring buffer.
 *  4. Expose a pull-based [getWindow] API that returns the last N ms of
 *     decimated samples — consumed by Member 6's TFLiteInferenceEngine
 *     (api-contracts.md §5).
 *  5. Optionally notify a listener on each decimated sample for push-based
 *     consumers (Member 3 AlignmentManager, Member 4 FusionBridge).
 *
 * Design decisions:
 *  - TARGET_INTERVAL_MS = 100 ms → 10 Hz, matching PRD §1.2 phone update rate.
 *  - Ring buffer capacity defaults to 2000 samples (200 seconds at 10 Hz),
 *    comfortably covering the "1–2 second trailing window" the ML model needs
 *    plus headroom for longer replay/debug windows.
 *  - Decimation is simple "take the latest sample whose timestamp is ≥
 *    lastEmittedTimestamp + interval" — no interpolation, no filtering.
 *    Filtering/leveling is Member 3's job.
 *  - Thread-safe: [push] may be called from SensorBridge's HandlerThread
 *    while [getWindow] / [getLatest] are called from inference/fusion coroutines.
 */
class SensorDataBuffer(
    /**
     * Maximum number of decimated samples retained. Oldest samples are
     * silently dropped when this limit is reached.
     */
    private val capacity: Int = DEFAULT_CAPACITY,

    /**
     * Optional callback invoked for each decimated (10 Hz) sample.
     * Useful for push-based consumers that don't want to poll [getWindow].
     */
    private val onDecimatedSample: ((ImuSample) -> Unit)? = null
) {

    companion object {
        /** Target interval between decimated samples, in milliseconds. */
        const val TARGET_INTERVAL_MS: Long = 100L   // 10 Hz

        /** Default ring-buffer capacity (200 s at 10 Hz). */
        const val DEFAULT_CAPACITY: Int = 2000
    }

    // ---- Ring buffer internals ----

    private val lock = Any()

    @GuardedBy("lock")
    private val buffer = ArrayDeque<ImuSample>(capacity)

    /** Timestamp (ms) of the last sample that passed the decimation gate. */
    @GuardedBy("lock")
    private var lastEmittedMs: Long = Long.MIN_VALUE

    /** Count of raw samples received (diagnostic). */
    @GuardedBy("lock")
    private var rawCount: Long = 0

    /** Count of decimated samples stored (diagnostic). */
    @GuardedBy("lock")
    private var decimatedCount: Long = 0

    // ---- Public API (matches api-contracts.md §5) ----

    /**
     * Accept a raw [ImuSample] from [SensorBridge].
     *
     * The sample is stored in the ring buffer only if at least
     * [TARGET_INTERVAL_MS] has elapsed since the last stored sample,
     * effectively decimating the ~50–100 Hz hardware stream to ~10 Hz.
     */
    fun push(sample: ImuSample) {
        val decimated: Boolean
        synchronized(lock) {
            rawCount++

            if (sample.timestampMs - lastEmittedMs < TARGET_INTERVAL_MS) {
                return // drop — too soon since last decimated sample
            }

            lastEmittedMs = sample.timestampMs

            if (buffer.size >= capacity) {
                buffer.removeFirst() // evict oldest
            }
            buffer.addLast(sample)
            decimatedCount++
            decimated = true
        }

        // Notify outside the lock.
        if (decimated) {
            onDecimatedSample?.invoke(sample)
        }
    }

    /**
     * Return all decimated samples whose timestamps fall within
     * `[now - windowMs, now]`, where `now` is the timestamp of the most
     * recent sample in the buffer.
     *
     * Returns an empty list if the buffer is empty.
     *
     * @param windowMs trailing window duration in milliseconds.
     *                 Example: 2000 for a 2-second trailing window.
     */
    fun getWindow(windowMs: Long): List<ImuSample> {
        require(windowMs >= 0) { "windowMs must be non-negative, got $windowMs" }

        synchronized(lock) {
            if (buffer.isEmpty()) return emptyList()

            val latestMs = buffer.last().timestampMs
            val cutoffMs = latestMs - windowMs

            // Fast path: if the entire buffer fits inside the window, return a copy.
            if (buffer.first().timestampMs >= cutoffMs) {
                return buffer.toList()
            }

            // Binary-search-style linear scan from the back (buffer is sorted by time).
            // For typical window sizes (1–2 s → 10–20 samples) this is fast enough.
            val result = mutableListOf<ImuSample>()
            val iter = buffer.descendingIterator()
            while (iter.hasNext()) {
                val s = iter.next()
                if (s.timestampMs < cutoffMs) break
                result.add(s)
            }
            result.reverse() // restore chronological order
            return result
        }
    }

    /**
     * Return the most recent decimated sample, or `null` if the buffer is empty.
     */
    fun getLatest(): ImuSample? {
        synchronized(lock) {
            return buffer.lastOrNull()
        }
    }

    /**
     * Return the number of decimated samples currently in the buffer.
     */
    fun size(): Int {
        synchronized(lock) {
            return buffer.size
        }
    }

    /**
     * Remove all samples and reset decimation state.
     * Useful on session-start or recalibration.
     */
    fun clear() {
        synchronized(lock) {
            buffer.clear()
            lastEmittedMs = Long.MIN_VALUE
            rawCount = 0
            decimatedCount = 0
        }
    }

    /**
     * Diagnostic: total raw samples received via [push] since creation or last [clear].
     */
    fun rawSampleCount(): Long = synchronized(lock) { rawCount }

    /**
     * Diagnostic: total samples that passed decimation and were stored.
     */
    fun decimatedSampleCount(): Long = synchronized(lock) { decimatedCount }

    // ---- Private helpers ----

    /**
     * Descending iterator over the ring buffer (newest-first).
     * ArrayDeque doesn't have a built-in descendingIterator, so we iterate
     * the backing structure in reverse.
     */
    private fun <T> ArrayDeque<T>.descendingIterator(): Iterator<T> {
        val snapshot = this.toList()
        return object : Iterator<T> {
            private var index = snapshot.size - 1
            override fun hasNext() = index >= 0
            override fun next(): T = snapshot[index--]
        }
    }
}
