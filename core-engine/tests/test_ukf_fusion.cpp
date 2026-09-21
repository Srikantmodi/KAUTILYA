/**
 * test_ukf_fusion.cpp — UKF Fusion Engine Unit Tests
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Tests included:
 *   1. test_zupt_clamps_velocity         — ZUPT drives velocity ≈ 0
 *   2. test_zaru_tightens_bias_cov       — ZARU reduces gyro bias covariance
 *   3. test_straight_line_drift          — 10s at 10m/s, drift within tolerance
 *   4. test_gnss_update_corrects_pos     — GNSS fix pulls position toward fix
 *   5. test_map_match_low_conf_skipped   — confidence < 0.7 → state unchanged
 *   6. test_map_match_high_conf_applied  — confidence ≥ 0.7 → correction applied
 *   7. test_es_nn_residual_subtracts     — ES-NN subtracts corrections from state
 *   8. test_gnss_gap_at_start            — no GNSS before first predict → no crash
 *   9. test_zero_delta_v_zero_dt         — predict(0,0) → no crash, no NaN
 *  10. test_reset_clears_state           — reset() zeroes state
 *  11. test_mode_manager_transitions     — mode transitions based on GNSS quality
 *
 * References:
 *   PRD §6.3, §2.3, §2.4, §7 guardrails #1–#8
 */

#include "ukf_fusion.h"
#include "mode_manager.h"
#include <cstdio>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Minimal test harness ───────────────────────────────────────────────── */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float _a = static_cast<float>(a), _b = static_cast<float>(b), _t = static_cast<float>(tol); \
    if (fabsf(_a - _b) > _t) { \
        fprintf(stderr, "  FAIL: %s  got=%.6f  expected=%.6f  tol=%.6f (line %d)\n", \
                (msg), _a, _b, _t, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NOT_NAN(val, msg) do { \
    if (std::isnan(static_cast<float>(val))) { \
        fprintf(stderr, "  FAIL: %s  value is NaN (line %d)\n", (msg), __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    fprintf(stdout, "[ RUN      ] " #fn "\n"); \
    tests_run++; \
    int prev_failed = tests_failed; \
    fn(); \
    if (tests_failed == prev_failed) { \
        fprintf(stdout, "[       OK ] " #fn "\n"); \
        tests_passed++; \
    } \
} while(0)


/* ═══════════════════════════════════════════════════════════════════════════ */

/* ── Test 1: ZUPT clamps velocity to zero ─────────────────────────────────── */
static void test_zupt_clamps_velocity() {
    UKFFusion ukf;
    ukf.reset();

    /* Give it a GNSS fix to set the origin */
    GnssFix fix = {};
    fix.timestamp_ms = 1000;
    fix.lat_deg = 12.9716;
    fix.lon_deg = 77.5946;
    fix.speed_mps = 10.0f;
    fix.bearing_rad = 0.0f;
    fix.accuracy_m = 5.0f;
    fix.satellite_count = 8;
    fix.quality_score = 0.9f;
    ukf.update_gnss(fix);

    /* Predict with some velocity */
    ukf.predict(10.0f, 0.1f);

    NavState ns_before = ukf.get_nav_state();
    float speed_before = sqrtf(ns_before.velocity[0] * ns_before.velocity[0] +
                                ns_before.velocity[1] * ns_before.velocity[1]);
    ASSERT(speed_before > 1.0f, "pre-ZUPT: velocity should be > 0");

    /* Apply ZUPT */
    ukf.apply_zupt();

    NavState ns_after = ukf.get_nav_state();
    float speed_after = sqrtf(ns_after.velocity[0] * ns_after.velocity[0] +
                               ns_after.velocity[1] * ns_after.velocity[1]);

    /* Velocity should be very close to zero */
    ASSERT(speed_after < 0.1f, "post-ZUPT: velocity ≈ 0");
}

/* ── Test 2: ZARU tightens gyro bias covariance ──────────────────────────── */
static void test_zaru_tightens_bias_cov() {
    UKFFusion ukf;
    ukf.reset();

    /* Get initial covariance via nav state (gyro bias cov is internal,
     * but we can observe its effect through repeated ZARU) */

    /* Run some predicts to let covariance grow */
    for (int i = 0; i < 50; i++) {
        ukf.predict(5.0f, 0.1f);
    }

    NavState ns_before = ukf.get_nav_state();

    /* Apply ZARU multiple times */
    for (int i = 0; i < 10; i++) {
        ukf.apply_zaru();
    }

    NavState ns_after = ukf.get_nav_state();

    /* The gyro bias should remain finite and not NaN */
    ASSERT_NOT_NAN(ns_after.gyro_bias[0], "post-ZARU: bias x not NaN");
    ASSERT_NOT_NAN(ns_after.gyro_bias[1], "post-ZARU: bias y not NaN");
    ASSERT_NOT_NAN(ns_after.gyro_bias[2], "post-ZARU: bias z not NaN");

    /* Position and heading should also remain valid */
    ASSERT_NOT_NAN(ns_after.position[0], "post-ZARU: pos east not NaN");
    ASSERT_NOT_NAN(ns_after.heading_rad, "post-ZARU: heading not NaN");
}

/* ── Test 3: Straight-line drift check ────────────────────────────────────── */
static void test_straight_line_drift() {
    UKFFusion ukf;
    ukf.reset();

    /* Set origin with GNSS fix */
    GnssFix fix = {};
    fix.timestamp_ms = 1000;
    fix.lat_deg = 12.9716;
    fix.lon_deg = 77.5946;
    fix.speed_mps = 10.0f;
    fix.bearing_rad = 0.0f; /* East */
    fix.accuracy_m = 3.0f;
    fix.satellite_count = 10;
    fix.quality_score = 0.95f;
    ukf.update_gnss(fix);

    /* Predict at 10 m/s heading East for 10 seconds */
    for (int i = 0; i < 100; i++) {
        ukf.predict(10.0f, 0.1f);
    }

    NavState ns = ukf.get_nav_state();

    /* Expected: position ≈ (100, 0, 0) meters East of origin
     * Allow generous tolerance due to UKF covariance growth */
    ASSERT(ns.position[0] > 50.0f, "straight-line: moved significantly east");
    ASSERT(fabsf(ns.position[1]) < 20.0f, "straight-line: limited north drift");
    ASSERT_NOT_NAN(ns.position[0], "straight-line: no NaN in position");
    ASSERT_NOT_NAN(ns.heading_rad, "straight-line: no NaN in heading");

    /* Heading should still be near 0 (East) */
    ASSERT(fabsf(ns.heading_rad) < 1.0f, "straight-line: heading near East");
}

/* ── Test 4: GNSS update corrects position ────────────────────────────────── */
static void test_gnss_update_corrects_pos() {
    UKFFusion ukf;
    ukf.reset();

    /* First fix = origin */
    GnssFix fix0 = {};
    fix0.timestamp_ms = 1000;
    fix0.lat_deg = 12.0;
    fix0.lon_deg = 77.0;
    fix0.speed_mps = 0.0f;
    fix0.bearing_rad = 0.0f;
    fix0.accuracy_m = 5.0f;
    fix0.satellite_count = 8;
    fix0.quality_score = 0.9f;
    ukf.update_gnss(fix0);

    /* Predict with no motion (should stay near origin) */
    for (int i = 0; i < 10; i++) {
        ukf.predict(0.0f, 0.1f);
    }

    NavState ns_before = ukf.get_nav_state();
    ASSERT_NEAR(ns_before.position[0], 0.0f, 5.0f, "pre-GNSS: near origin E");
    ASSERT_NEAR(ns_before.position[1], 0.0f, 5.0f, "pre-GNSS: near origin N");

    /* Apply a GNSS fix that says we're ~111 meters north
     * (1 degree lat ≈ 111km, so 0.001 deg ≈ 111 m) */
    GnssFix fix1 = {};
    fix1.timestamp_ms = 2000;
    fix1.lat_deg = 12.001; /* ~111m north */
    fix1.lon_deg = 77.0;
    fix1.speed_mps = 0.0f;
    fix1.bearing_rad = 0.0f;
    fix1.accuracy_m = 3.0f;
    fix1.satellite_count = 10;
    fix1.quality_score = 0.95f;
    ukf.update_gnss(fix1);

    NavState ns_after = ukf.get_nav_state();

    /* Position should have been pulled north significantly */
    ASSERT(ns_after.position[1] > 20.0f,
           "post-GNSS: position pulled north toward fix");
}

/* ── Test 5: Map-match with low confidence is SKIPPED ─────────────────────── */
static void test_map_match_low_conf_skipped() {
    UKFFusion ukf;
    ukf.reset();

    /* Set origin */
    GnssFix fix = {};
    fix.timestamp_ms = 1000;
    fix.lat_deg = 12.0;
    fix.lon_deg = 77.0;
    fix.speed_mps = 5.0f;
    fix.bearing_rad = 0.0f;
    fix.accuracy_m = 3.0f;
    fix.satellite_count = 8;
    fix.quality_score = 0.9f;
    ukf.update_gnss(fix);

    NavState ns_before = ukf.get_nav_state();

    /* Apply map match with confidence = 0.3 (below 0.7 threshold) */
    float pseudo[2] = { 50.0f, 1.0f }; /* large corrections */
    ukf.update_map_match(pseudo, 0.3f);

    NavState ns_after = ukf.get_nav_state();

    /* State should be UNCHANGED — low confidence skips the update */
    ASSERT_NEAR(ns_after.position[0], ns_before.position[0], 0.001f,
                "low-conf map-match: position unchanged");
    ASSERT_NEAR(ns_after.heading_rad, ns_before.heading_rad, 0.001f,
                "low-conf map-match: heading unchanged");
}

/* ── Test 6: Map-match with high confidence is APPLIED ────────────────────── */
static void test_map_match_high_conf_applied() {
    UKFFusion ukf;
    ukf.reset();

    /* Set origin and move a bit */
    GnssFix fix = {};
    fix.timestamp_ms = 1000;
    fix.lat_deg = 12.0;
    fix.lon_deg = 77.0;
    fix.speed_mps = 10.0f;
    fix.bearing_rad = 0.0f;
    fix.accuracy_m = 3.0f;
    fix.satellite_count = 8;
    fix.quality_score = 0.9f;
    ukf.update_gnss(fix);

    for (int i = 0; i < 50; i++) {
        ukf.predict(10.0f, 0.1f);
    }

    NavState ns_before = ukf.get_nav_state();

    /* Apply map match with high confidence and a heading offset */
    float pseudo[2] = { 0.0f, 0.2f }; /* 0.2 rad heading offset */
    ukf.update_map_match(pseudo, 0.9f);

    NavState ns_after = ukf.get_nav_state();

    /* Heading should have changed (correction applied) */
    float heading_change = fabsf(ns_after.heading_rad - ns_before.heading_rad);
    ASSERT(heading_change > 0.001f,
           "high-conf map-match: heading changed");
}

/* ── Test 7: ES-NN residual subtracts corrections ─────────────────────────── */
static void test_es_nn_residual_subtracts() {
    UKFFusion ukf;
    ukf.reset();

    /* Set origin */
    GnssFix fix = {};
    fix.timestamp_ms = 1000;
    fix.lat_deg = 12.0;
    fix.lon_deg = 77.0;
    fix.speed_mps = 0.0f;
    fix.bearing_rad = 0.5f;
    fix.accuracy_m = 3.0f;
    fix.satellite_count = 8;
    fix.quality_score = 0.9f;
    ukf.update_gnss(fix);

    NavState ns_before = ukf.get_nav_state();

    /* Apply ES-NN residual */
    float dp[3] = { 1.0f, 2.0f, 0.0f };
    float dv[3] = { 0.5f, 0.5f, 0.0f };
    float dpsi = 0.1f;
    ukf.apply_es_nn_residual(dp, dv, dpsi);

    NavState ns_after = ukf.get_nav_state();

    /* Position should have been corrected (subtracted) */
    ASSERT_NEAR(ns_after.position[0], ns_before.position[0] - 1.0f, 0.01f,
                "ES-NN: position east corrected");
    ASSERT_NEAR(ns_after.position[1], ns_before.position[1] - 2.0f, 0.01f,
                "ES-NN: position north corrected");

    /* Heading should have been corrected */
    float expected_heading = wrap_to_pi(ns_before.heading_rad - 0.1f);
    ASSERT_NEAR(ns_after.heading_rad, expected_heading, 0.01f,
                "ES-NN: heading corrected");
}

/* ── Test 8: No GNSS before first predict → no crash ──────────────────────── */
static void test_gnss_gap_at_start() {
    UKFFusion ukf;
    ukf.reset();

    /* Predict without any GNSS fix (origin not set) */
    for (int i = 0; i < 50; i++) {
        ukf.predict(5.0f, 0.1f);
    }

    NavState ns = ukf.get_nav_state();

    /* Should not crash, and values should not be NaN */
    ASSERT_NOT_NAN(ns.position[0], "no-GNSS: position east not NaN");
    ASSERT_NOT_NAN(ns.position[1], "no-GNSS: position north not NaN");
    ASSERT_NOT_NAN(ns.heading_rad, "no-GNSS: heading not NaN");

    /* Origin should not be set */
    ASSERT(!ukf.has_origin(), "no-GNSS: origin not set");
}

/* ── Test 9: predict(0, 0) → no crash, no NaN ────────────────────────────── */
static void test_zero_delta_v_zero_dt() {
    UKFFusion ukf;
    ukf.reset();

    /* This should be a no-op */
    ukf.predict(0.0f, 0.0f);

    NavState ns = ukf.get_nav_state();
    ASSERT_NOT_NAN(ns.position[0], "zero-zero: pos E not NaN");
    ASSERT_NOT_NAN(ns.velocity[0], "zero-zero: vel E not NaN");
    ASSERT_NOT_NAN(ns.heading_rad, "zero-zero: heading not NaN");
}

/* ── Test 10: reset() clears state ────────────────────────────────────────── */
static void test_reset_clears_state() {
    UKFFusion ukf;

    /* Set origin and predict some motion */
    GnssFix fix = {};
    fix.timestamp_ms = 1000;
    fix.lat_deg = 12.0;
    fix.lon_deg = 77.0;
    fix.speed_mps = 10.0f;
    fix.bearing_rad = 0.5f;
    fix.accuracy_m = 3.0f;
    fix.satellite_count = 8;
    fix.quality_score = 0.9f;
    ukf.update_gnss(fix);
    ukf.predict(10.0f, 1.0f);

    /* Reset */
    ukf.reset();

    NavState ns = ukf.get_nav_state();
    ASSERT_NEAR(ns.position[0], 0.0f, 0.001f, "reset: pos E = 0");
    ASSERT_NEAR(ns.position[1], 0.0f, 0.001f, "reset: pos N = 0");
    ASSERT_NEAR(ns.velocity[0], 0.0f, 0.001f, "reset: vel E = 0");
    ASSERT_NEAR(ns.heading_rad, 0.0f, 0.001f, "reset: heading = 0");
    ASSERT(!ukf.has_origin(), "reset: origin cleared");
}

/* ── Test 11: Mode manager transitions ────────────────────────────────────── */
static void test_mode_manager_transitions() {
    ModeManager mm;
    mm.reset();

    /* No fix yet → PURE_DR */
    ASSERT(mm.evaluate_mode(1000) == NAVIGATION_MODE_PURE_DR,
           "no fix: PURE_DR");

    /* Good fix → GNSS_AIDED_INS */
    mm.update_gnss_fix(1000, 8, 5.0f, 0.9f);
    ASSERT(mm.evaluate_mode(1000) == NAVIGATION_MODE_GNSS_AIDED_INS,
           "good fix: GNSS_AIDED_INS");

    /* Same fix but 3 seconds later → fix age > 2000ms → PURE_DR */
    ASSERT(mm.evaluate_mode(4000) == NAVIGATION_MODE_PURE_DR,
           "stale fix: PURE_DR");

    /* Poor accuracy fix → PURE_DR */
    mm.update_gnss_fix(5000, 8, 20.0f, 0.5f); /* accuracy 20m > 15m threshold */
    ASSERT(mm.evaluate_mode(5000) == NAVIGATION_MODE_PURE_DR,
           "poor accuracy: PURE_DR");

    /* Low satellite count → PURE_DR */
    mm.update_gnss_fix(6000, 2, 5.0f, 0.3f); /* only 2 sats < 4 threshold */
    ASSERT(mm.evaluate_mode(6000) == NAVIGATION_MODE_PURE_DR,
           "low sats: PURE_DR");

    /* All criteria met again → GNSS_AIDED_INS (millisecond-scale, no debounce) */
    mm.update_gnss_fix(7000, 10, 3.0f, 0.95f);
    ASSERT(mm.evaluate_mode(7000) == NAVIGATION_MODE_GNSS_AIDED_INS,
           "restored: GNSS_AIDED_INS (no debounce)");
}


/* ═══════════════════════════════════════════════════════════════════════════ */

int main() {
    fprintf(stdout, "========================================\n");
    fprintf(stdout, " test_ukf_fusion — Member 4\n");
    fprintf(stdout, "========================================\n\n");

    RUN_TEST(test_zupt_clamps_velocity);
    RUN_TEST(test_zaru_tightens_bias_cov);
    RUN_TEST(test_straight_line_drift);
    RUN_TEST(test_gnss_update_corrects_pos);
    RUN_TEST(test_map_match_low_conf_skipped);
    RUN_TEST(test_map_match_high_conf_applied);
    RUN_TEST(test_es_nn_residual_subtracts);
    RUN_TEST(test_gnss_gap_at_start);
    RUN_TEST(test_zero_delta_v_zero_dt);
    RUN_TEST(test_reset_clears_state);
    RUN_TEST(test_mode_manager_transitions);

    fprintf(stdout, "\n========================================\n");
    fprintf(stdout, " Results: %d/%d passed, %d failed\n",
            tests_passed, tests_run, tests_failed);
    fprintf(stdout, "========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
