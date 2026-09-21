/**
 * test_ins_mechanization.cpp — INS Mechanization Unit Tests
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Tests included:
 *   1. test_straight_line_east    — 10m/s east for 10s → position ≈ (100, 0, 0)
 *   2. test_straight_line_north   — 10m/s north for 10s → position ≈ (0, 100, 0)
 *   3. test_stationary            — zero delta_v for 10s → position stays at origin
 *   4. test_zero_dt               — dt=0 → no state change, no crash
 *   5. test_negative_dt           — dt<0 → no state change, no crash
 *   6. test_heading_wrap          — heading wraps to [-π, π] correctly
 *   7. test_45deg_heading         — diagonal motion decomposes correctly
 *   8. test_state_reset           — reset zeroes all state
 *   9. test_navstate_conversion   — state array → NavState struct mapping
 *
 * References:
 *   PRD §7 guardrail #7 (deliberately bad test cases)
 */

#include "ins_mechanization.h"
#include "imu_types.h"
#include <cstdio>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Minimal self-contained test harness (matches Member 3's pattern) ───── */
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
    if (fabsf(static_cast<float>(a) - static_cast<float>(b)) > static_cast<float>(tol)) { \
        fprintf(stderr, "  FAIL: %s  got=%.6f  expected=%.6f  tol=%.6f (line %d)\n", \
                (msg), static_cast<float>(a), static_cast<float>(b), static_cast<float>(tol), __LINE__); \
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

static INSMechanization ins;

/* ── Test 1: Straight line heading East (ψ=0 in ENU) ─────────────────────── */
static void test_straight_line_east() {
    float state[N_STATE];
    INSMechanization::reset_state(state);

    /* Heading = 0 rad (East). delta_v = 10 m/s. dt = 0.1s. 100 steps = 10s. */
    state[ST_PSI] = 0.0f;
    float delta_v = 10.0f;
    float dt = 0.1f;

    for (int i = 0; i < 100; i++) {
        ins.propagate_state(state, delta_v, dt);
    }

    /* After 10s at 10 m/s heading East:
     * position should be ≈ (100, 0, 0) meters
     * velocity should be ≈ (10, 0, 0) m/s */
    ASSERT_NEAR(state[ST_PE], 100.0f, 1.0f, "position east ≈ 100m");
    ASSERT_NEAR(state[ST_PN],   0.0f, 0.1f, "position north ≈ 0m");
    ASSERT_NEAR(state[ST_PU],   0.0f, 0.1f, "position up ≈ 0m");
    ASSERT_NEAR(state[ST_VE],  10.0f, 0.1f, "velocity east ≈ 10 m/s");
    ASSERT_NEAR(state[ST_VN],   0.0f, 0.1f, "velocity north ≈ 0 m/s");
}

/* ── Test 2: Straight line heading North (ψ=π/2 in ENU) ──────────────────── */
static void test_straight_line_north() {
    float state[N_STATE];
    INSMechanization::reset_state(state);

    /* Heading = π/2 rad (North in ENU). */
    state[ST_PSI] = static_cast<float>(M_PI / 2.0);
    float delta_v = 10.0f;
    float dt = 0.1f;

    for (int i = 0; i < 100; i++) {
        ins.propagate_state(state, delta_v, dt);
    }

    ASSERT_NEAR(state[ST_PE],   0.0f, 1.0f, "position east ≈ 0m");
    ASSERT_NEAR(state[ST_PN], 100.0f, 1.0f, "position north ≈ 100m");
}

/* ── Test 3: Stationary (zero delta_v) ────────────────────────────────────── */
static void test_stationary() {
    float state[N_STATE];
    INSMechanization::reset_state(state);

    float delta_v = 0.0f;
    float dt = 0.1f;

    for (int i = 0; i < 100; i++) {
        ins.propagate_state(state, delta_v, dt);
    }

    ASSERT_NEAR(state[ST_PE], 0.0f, 0.001f, "stationary: pe stays 0");
    ASSERT_NEAR(state[ST_PN], 0.0f, 0.001f, "stationary: pn stays 0");
    ASSERT_NEAR(state[ST_VE], 0.0f, 0.001f, "stationary: ve stays 0");
    ASSERT_NEAR(state[ST_VN], 0.0f, 0.001f, "stationary: vn stays 0");
}

/* ── Test 4: Zero dt → no-op, no crash ────────────────────────────────────── */
static void test_zero_dt() {
    float state[N_STATE];
    INSMechanization::reset_state(state);
    state[ST_PE] = 42.0f;

    ins.propagate_state(state, 10.0f, 0.0f);

    ASSERT_NEAR(state[ST_PE], 42.0f, 0.001f, "zero dt: position unchanged");
    ASSERT_NEAR(state[ST_VE],  0.0f, 0.001f, "zero dt: velocity unchanged");
}

/* ── Test 5: Negative dt → no-op, no crash ────────────────────────────────── */
static void test_negative_dt() {
    float state[N_STATE];
    INSMechanization::reset_state(state);
    state[ST_PE] = 42.0f;

    ins.propagate_state(state, 10.0f, -1.0f);

    ASSERT_NEAR(state[ST_PE], 42.0f, 0.001f, "negative dt: position unchanged");
}

/* ── Test 6: Heading wraps to [-π, π] ─────────────────────────────────────── */
static void test_heading_wrap() {
    float state[N_STATE];
    INSMechanization::reset_state(state);

    /* Set heading beyond π */
    state[ST_PSI] = 3.5f; /* > π ≈ 3.14159 */
    ins.propagate_state(state, 1.0f, 0.1f);

    float heading = state[ST_PSI];
    ASSERT(heading >= -static_cast<float>(M_PI) &&
           heading <= static_cast<float>(M_PI),
           "heading wraps to [-π, π]");

    /* Expected wrapped value: 3.5 - 2π ≈ -2.783 */
    float expected = 3.5f - 2.0f * static_cast<float>(M_PI);
    ASSERT_NEAR(heading, expected, 0.01f, "heading wrap value correct");
}

/* ── Test 7: 45° heading → diagonal motion ────────────────────────────────── */
static void test_45deg_heading() {
    float state[N_STATE];
    INSMechanization::reset_state(state);

    state[ST_PSI] = static_cast<float>(M_PI / 4.0); /* 45° NE */
    float delta_v = 10.0f;
    float dt = 0.1f;

    for (int i = 0; i < 100; i++) {
        ins.propagate_state(state, delta_v, dt);
    }

    /* At 45°, cos(45°) = sin(45°) ≈ 0.7071 */
    float expected_pos = 10.0f * 10.0f * 0.7071f; /* ≈ 70.71 m */
    ASSERT_NEAR(state[ST_PE], expected_pos, 2.0f, "45° heading: east component");
    ASSERT_NEAR(state[ST_PN], expected_pos, 2.0f, "45° heading: north component");
}

/* ── Test 8: State reset zeroes everything ────────────────────────────────── */
static void test_state_reset() {
    float state[N_STATE];
    for (int i = 0; i < N_STATE; i++) state[i] = 99.0f;

    INSMechanization::reset_state(state);

    for (int i = 0; i < N_STATE; i++) {
        ASSERT_NEAR(state[i], 0.0f, 0.001f, "reset: all states zero");
    }
}

/* ── Test 9: NavState conversion ──────────────────────────────────────────── */
static void test_navstate_conversion() {
    float state[N_STATE] = { 1.0f, 2.0f, 3.0f,   /* pos */
                              4.0f, 5.0f, 6.0f,   /* vel */
                              0.5f,                /* heading */
                              0.01f, 0.02f, 0.03f  /* gyro bias */ };
    float cov[16];
    for (int i = 0; i < 16; i++) cov[i] = static_cast<float>(i);

    NavState ns;
    INSMechanization::state_to_navstate(state, cov, &ns);

    ASSERT_NEAR(ns.position[0], 1.0f, 0.001f, "navstate pos[0]");
    ASSERT_NEAR(ns.position[1], 2.0f, 0.001f, "navstate pos[1]");
    ASSERT_NEAR(ns.position[2], 3.0f, 0.001f, "navstate pos[2]");
    ASSERT_NEAR(ns.velocity[0], 4.0f, 0.001f, "navstate vel[0]");
    ASSERT_NEAR(ns.heading_rad, 0.5f, 0.001f, "navstate heading");
    ASSERT_NEAR(ns.gyro_bias[0], 0.01f, 0.001f, "navstate gyro_bias[0]");
    ASSERT_NEAR(ns.covariance[0], 0.0f, 0.001f, "navstate cov[0]");
    ASSERT_NEAR(ns.covariance[15], 15.0f, 0.001f, "navstate cov[15]");
}


/* ═══════════════════════════════════════════════════════════════════════════ */

int main() {
    fprintf(stdout, "========================================\n");
    fprintf(stdout, " test_ins_mechanization — Member 4\n");
    fprintf(stdout, "========================================\n\n");

    RUN_TEST(test_straight_line_east);
    RUN_TEST(test_straight_line_north);
    RUN_TEST(test_stationary);
    RUN_TEST(test_zero_dt);
    RUN_TEST(test_negative_dt);
    RUN_TEST(test_heading_wrap);
    RUN_TEST(test_45deg_heading);
    RUN_TEST(test_state_reset);
    RUN_TEST(test_navstate_conversion);

    fprintf(stdout, "\n========================================\n");
    fprintf(stdout, " Results: %d/%d passed, %d failed\n",
            tests_passed, tests_run, tests_failed);
    fprintf(stdout, "========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
