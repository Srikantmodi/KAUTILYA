/**
 * ModeManager.kt — Pure JNI wrapper for NavigationMode enum
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 1 (App Shell, UI & Navigation Frontend)
 *
 * This is a PURE PASS-THROUGH to the C++ NavigationMode enum and
 * mode_manager.cpp's evaluate_mode() function (§6.5). All GNSS-quality-
 * scoring, satellite-count thresholds, transition-debounce, and fix-age
 * logic lives in C++ (Member 4). If you find yourself writing scoring
 * logic here, STOP — that violates §7 guardrail #1.
 *
 * The only Kotlin-side additions are display helpers (name strings,
 * colors) which are purely presentational per §2.2.
 *
 * References:
 *   api-contracts.md §7  — ModeManager contract
 *   api-contracts.md §1  — NavigationMode enum ordinals
 *   PRD §6.5, §7 guardrail #1
 */
package com.sih.deadreckoning.mode

import android.util.Log

/**
 * Kotlin mirror of the C++ `NavigationMode` enum (imu_types.h).
 *
 * Ordinal values are LOCKED to match the C++ enum:
 *   GNSS_AIDED_INS = 0
 *   PURE_DR        = 1
 *   RECOVERING     = 2
 *
 * @property ordinalValue  the integer ordinal matching the C++ enum,
 *                         used for JNI marshaling (booleans cross JNI as
 *                         int 0/1 per §0, enums likewise as int ordinals).
 * @property displayName   human-readable label for UI display.
 * @property colorHex      hex color for UI badges/trails (§2.1):
 *                         green = GNSS aided, amber = pure DR, red = recovering.
 */
enum class NavigationMode(
    val ordinalValue: Int,
    val displayName: String,
    val colorHex: Long
) {
    /** Full GNSS-aided INS — satellite fixes are healthy and fused. */
    GNSS_AIDED_INS(0, "GNSS AIDED", 0xFF4CAF50),

    /** Pure dead reckoning — GNSS lost, navigating on IMU + ML only. */
    PURE_DR(1, "DEAD RECKONING", 0xFFFF9800),

    /** Recovering — re-acquiring GNSS confidence after an outage. */
    RECOVERING(2, "RECOVERING", 0xFFF44336);

    companion object {
        /**
         * Map a C++ ordinal to the Kotlin enum.
         *
         * Returns [PURE_DR] as a safe default for unknown ordinals —
         * if the C++ side returns garbage, the UI should assume worst-case
         * (no GNSS) rather than falsely displaying healthy state.
         *
         * @param ordinal  integer from JNI `getCurrentMode()`.
         * @return the corresponding [NavigationMode], or [PURE_DR] if unknown.
         */
        fun fromOrdinal(ordinal: Int): NavigationMode {
            return entries.firstOrNull { it.ordinalValue == ordinal } ?: run {
                Log.w(TAG, "Unknown NavigationMode ordinal: $ordinal — defaulting to PURE_DR")
                PURE_DR
            }
        }

        private const val TAG = "ModeManager"
    }
}

/**
 * Thin JNI bridge to the C++ mode manager (mode_manager.cpp).
 *
 * Usage:
 * ```
 * val modeManager = ModeManager()
 * val mode = modeManager.getCurrentMode()  // NavigationMode enum
 * overlay.setModeBadge(mode.displayName, mode.colorHex)
 * ```
 *
 * This class does NOTHING beyond calling into C++ and mapping the
 * returned int to the Kotlin [NavigationMode] enum. No scoring, no
 * debounce, no threshold checks — all of that is in C++.
 */
class ModeManager {

    companion object {
        private const val TAG = "ModeManager"

        init {
            try {
                System.loadLibrary("core_engine")
                Log.i(TAG, "core_engine native library loaded for ModeManager")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load core_engine: ${e.message}")
            }
        }
    }

    /* ── JNI native declaration ────────────────────────────────────────────
     * Returns the NavigationMode ordinal (0, 1, or 2) as determined by
     * the C++ mode_manager.cpp's evaluate_mode() scoring.
     *
     * Matches JniBridge.h: Java_..._getCurrentMode(JNIEnv*, jobject) → jint
     * ─────────────────────────────────────────────────────────────────────── */
    private external fun nativeGetCurrentMode(): Int

    /**
     * Get the current navigation mode from the C++ engine.
     *
     * @return [NavigationMode] enum reflecting the C++ mode state machine's
     *         current evaluation. Safe for unknown ordinals (defaults to PURE_DR).
     */
    fun getCurrentMode(): NavigationMode {
        return try {
            val ordinal = nativeGetCurrentMode()
            NavigationMode.fromOrdinal(ordinal)
        } catch (e: UnsatisfiedLinkError) {
            // Native library not yet linked (early development / unit testing).
            // Default to PURE_DR — safe fallback per §7 guardrail #1.
            Log.w(TAG, "JNI not linked — returning PURE_DR: ${e.message}")
            NavigationMode.PURE_DR
        }
    }
}
