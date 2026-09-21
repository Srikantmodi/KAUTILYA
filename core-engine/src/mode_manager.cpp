/**
 * mode_manager.cpp — GNSS/DR Mode State Machine
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Implements GNSS quality scoring and mode transitions.
 * Also implements the JNI functions for ModeManager declared in JniBridge.h.
 *
 * References:
 *   api-contracts.md §7, PRD §6.5, §7 guardrail #1
 */

#include "mode_manager.h"
#include <cstring>

/* ═════════════════════════════════════════════════════════════════════════════
 * Global instance
 * ═════════════════════════════════════════════════════════════════════════════ */

static ModeManager g_mode_manager;

ModeManager& get_mode_manager_instance() { return g_mode_manager; }


/* ═════════════════════════════════════════════════════════════════════════════
 * ModeManager implementation
 * ═════════════════════════════════════════════════════════════════════════════ */

ModeManager::ModeManager() {
    reset();
}

void ModeManager::reset() {
    last_fix_ts_ms_       = 0;
    last_satellite_count_ = 0;
    last_accuracy_m_      = 999.0f;  /* default to poor accuracy */
    last_quality_score_   = 0.0f;
    last_query_ts_ms_     = 0;
}

void ModeManager::update_gnss_fix(int64_t timestamp_ms, int satellite_count,
                                   float accuracy_m, float quality_score) {
    last_fix_ts_ms_       = timestamp_ms;
    last_satellite_count_ = satellite_count;
    last_accuracy_m_      = accuracy_m;
    last_quality_score_   = quality_score;
    last_query_ts_ms_     = timestamp_ms; /* track latest known time */
}

NavigationMode ModeManager::evaluate_mode(int64_t now_ms) const {
    /* Compute fix age */
    int64_t fix_age_ms = now_ms - last_fix_ts_ms_;

    /* If no fix has ever been received, default to PURE_DR */
    if (last_fix_ts_ms_ == 0) {
        return NAVIGATION_MODE_PURE_DR;
    }

    /* Scoring (api-contracts.md §7, no debounce — millisecond-scale):
     *   satellite_count >= 4  AND
     *   accuracy_m <= 15.0    AND
     *   fix_age_ms <= 2000    → GNSS_AIDED_INS
     *   Otherwise             → PURE_DR */
    if (last_satellite_count_ >= MM_MIN_SATELLITES &&
        last_accuracy_m_ <= MM_MAX_ACCURACY_M &&
        fix_age_ms <= MM_MAX_FIX_AGE_MS) {
        return NAVIGATION_MODE_GNSS_AIDED_INS;
    }

    return NAVIGATION_MODE_PURE_DR;
}

NavigationMode ModeManager::get_current_mode() const {
    /* Use the latest known time. In production, this would use
     * a real-time clock query. For now, use the last known timestamp. */
    return evaluate_mode(last_query_ts_ms_);
}


/* ═════════════════════════════════════════════════════════════════════════════
 * JNI bridge implementations — Mode Manager
 *
 * Implements declarations from JniBridge.h for ModeManager.
 * ═════════════════════════════════════════════════════════════════════════════ */

#ifdef __ANDROID__
#include <jni.h>
#include "../../bindings/java/JniBridge.h"

extern "C" {

JNIEXPORT jint JNICALL
Java_com_sih_deadreckoning_mode_ModeManager_getCurrentMode(
    JNIEnv *env, jobject thiz) {
    return static_cast<jint>(get_mode_manager_instance().get_current_mode());
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_mode_ModeManager_updateGnssFix(
    JNIEnv *env, jobject thiz,
    jlong ts, jint satelliteCount, jfloat accuracyM, jfloat qualityScore) {
    get_mode_manager_instance().update_gnss_fix(
        static_cast<int64_t>(ts),
        static_cast<int>(satelliteCount),
        accuracyM,
        qualityScore
    );
}

} /* extern "C" */
#endif /* __ANDROID__ */
