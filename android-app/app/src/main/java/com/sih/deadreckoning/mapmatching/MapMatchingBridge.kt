package com.sih.deadreckoning.mapmatching

import android.util.Log

/**
 * MapMatchingBridge — Thin JNI wrapper for the C++ HMM map-matching engine.
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 5 (Map-Matching & Map Data Pipeline)
 * Ref:   api-contracts.md §9 — "Thin JNI wrappers only — one Kotlin method
 *        per exposed C++ function, no business logic."
 *        PRD §7 guardrail #1 — no reimplementation of C++ math in Kotlin.
 *
 * This class does NOTHING beyond marshaling Kotlin types to/from JNI.
 * All map-matching logic lives in core-engine/src/map_matching.cpp.
 *
 * If you find yourself writing HMM, Viterbi, emission/transition probability,
 * or confidence-scoring logic here — STOP. That belongs in C++ (§7 #1).
 *
 * Usage:
 *   val bridge = MapMatchingBridge()
 *   val result = bridge.match(floatArrayOf(east, north, up), headingRad)
 *   if (result != null && result.confidence > 0.7f) {
 *       fusionBridge.updateMapMatch(result.pseudoMeasurement, result.confidence)
 *   }
 */
class MapMatchingBridge {

    companion object {
        private const val TAG = "MapMatchingBridge"

        init {
            System.loadLibrary("core_engine")
        }
    }

    /**
     * Kotlin-side mirror of C++ MapMatchResult (api-contracts.md §6).
     *
     * pseudo_measurement[0] = lateral_offset_m
     * pseudo_measurement[1] = heading_offset_rad
     */
    data class MatchResult(
        val matchedSegmentId: Int,
        val pseudoMeasurement: FloatArray,   // [lateral_offset_m, heading_offset_rad]
        val confidence: Float                // [0.0, 1.0]
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is MatchResult) return false
            return matchedSegmentId == other.matchedSegmentId &&
                   pseudoMeasurement.contentEquals(other.pseudoMeasurement) &&
                   confidence == other.confidence
        }

        override fun hashCode(): Int {
            var result = matchedSegmentId
            result = 31 * result + pseudoMeasurement.contentHashCode()
            result = 31 * result + confidence.hashCode()
            return result
        }
    }

    /**
     * Match a trajectory point to the road graph.
     *
     * @param trajectoryPoint [east, north, up] in ENU meters (from NavState).
     * @param headingRad      current heading, ENU convention (0=East, CCW+).
     * @return MatchResult, or null if no match / graph not loaded.
     *
     * The returned pseudo-measurement feeds directly into
     * FusionBridge.updateMapMatch() — ONLY when confidence exceeds the
     * threshold (default 0.7, per api-contracts.md §4).
     */
    fun match(trajectoryPoint: FloatArray, headingRad: Float): MatchResult? {
        if (trajectoryPoint.size < 3) {
            Log.w(TAG, "trajectoryPoint must have 3 elements")
            return null
        }

        // Call native — returns [segment_id, lateral_offset, heading_offset, confidence]
        val raw = nativeMapMatch(trajectoryPoint, headingRad) ?: return null

        if (raw.size < 4) {
            Log.w(TAG, "Unexpected native result size: ${raw.size}")
            return null
        }

        val segmentId = raw[0].toInt()
        if (segmentId < 0) {
            // No match found (segment_id = -1)
            return null
        }

        return MatchResult(
            matchedSegmentId = segmentId,
            pseudoMeasurement = floatArrayOf(raw[1], raw[2]),
            confidence = raw[3]
        )
    }

    /**
     * Reset the HMM Viterbi state — call at session start / recalibration.
     */
    fun reset() {
        nativeResetMatcher()
    }

    // ── JNI native methods (implemented in C++ jni_bridge.cpp, Member 4) ──

    /**
     * Native map-match call.
     * @return float[4]: [segment_id, lateral_offset_m, heading_offset_rad, confidence]
     *         or null on error.
     */
    private external fun nativeMapMatch(
        trajectoryPoint: FloatArray,
        headingRad: Float
    ): FloatArray?

    /** Reset the C++ MapMatcher's Viterbi state. */
    private external fun nativeResetMatcher()
}
