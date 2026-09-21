/**
 * mode_manager.h — GNSS/DR Mode State Machine
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §7, Master PRD §6.5
 *
 * Owner:  Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Purpose: Continuous GNSS quality scoring that determines whether the
 * system operates in GNSS-aided INS mode or pure dead reckoning mode.
 * Transitions are millisecond-scale — no multi-second debounce (§6.5).
 *
 * Thresholds (from api-contracts.md §7):
 *   satellite_count >= 4  AND
 *   accuracy_m <= 15.0    AND
 *   fix_age_ms <= 2000    → GNSS_AIDED_INS
 *   Otherwise             → PURE_DR
 *
 * Member 1's ModeManager.kt is a pure pass-through — no scoring logic
 * duplicated in Kotlin (§7 guardrail #1).
 */

#ifndef CORE_ENGINE_MODE_MANAGER_H
#define CORE_ENGINE_MODE_MANAGER_H

#include "imu_types.h"
#include <cstdint>

/* ─────────────────────────────────────────────────────────────────────────────
 * Mode Manager thresholds (tune against real data before finals)
 * ───────────────────────────────────────────────────────────────────────────── */
static constexpr int   MM_MIN_SATELLITES    = 4;
static constexpr float MM_MAX_ACCURACY_M    = 15.0f;
static constexpr int64_t MM_MAX_FIX_AGE_MS  = 2000;

/* ─────────────────────────────────────────────────────────────────────────────
 * ModeManager class
 * ───────────────────────────────────────────────────────────────────────────── */
class ModeManager {
public:
    ModeManager();

    /**
     * Feed a GNSS fix for quality scoring.
     * Must be called with every GnssFix from GnssQualityMonitor.
     */
    void update_gnss_fix(int64_t timestamp_ms, int satellite_count,
                         float accuracy_m, float quality_score);

    /**
     * Evaluate the current navigation mode.
     * Uses the latest fix data and current time to score GNSS quality.
     *
     * @param now_ms  current time in epoch milliseconds
     * @return NavigationMode enum value
     */
    NavigationMode evaluate_mode(int64_t now_ms) const;

    /**
     * Get the current mode (using internally tracked time from last update).
     * @return NavigationMode enum ordinal
     */
    NavigationMode get_current_mode() const;

    /** Reset to initial state. */
    void reset();

private:
    int64_t last_fix_ts_ms_;
    int     last_satellite_count_;
    float   last_accuracy_m_;
    float   last_quality_score_;
    int64_t last_query_ts_ms_;
};

/* ─────────────────────────────────────────────────────────────────────────────
 * Free functions (extern "C")
 * ───────────────────────────────────────────────────────────────────────────── */
#ifdef __cplusplus
extern "C" {
#endif

/** Global ModeManager instance. */
ModeManager& get_mode_manager_instance();

#ifdef __cplusplus
}
#endif

#endif /* CORE_ENGINE_MODE_MANAGER_H */
