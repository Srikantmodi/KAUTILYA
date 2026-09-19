/**
 * test_calibration.cpp — Calibration & Device Alignment Unit Tests
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §3, Master PRD §6.2, §7 guardrail #2
 *
 * Owner:  Member 3 (Calibration & Device Alignment)
 * Status: FULL TEST SUITE
 *
 * Tests included:
 *   1. test_known_angle_pitch_30deg (Hand-computed ground truth §5.2)
 *   2. test_angle_wrap_long_gap (1000s gyro integration angle wrap endurance §5.3)
 *   3. test_noise_inflation_suppresses_braking (5s braking noise inflation check §5.4)
 *   4. test_edge_zero_dt (dt=0 division-by-zero check §5.5)
 *   5. test_edge_session_start_no_static_init (Start without static init §5.5)
 *   6. test_edge_nan_accel_rejected (Corrupted sensor data NaN rejection §5.5)
 *   7. test_level_accelerometer_flat_and_tilted (Leveling transform verification)
 *   8. test_pooled_yaw_optimizer (Closed-form yaw solve verification)
 */

#include "calibration.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Minimal self-contained test harness ────────────────────────────────────
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


// ─── Test 1: Hand-computed known angle ground truth (§5.2) ──────────────────
static void test_known_angle_pitch_30deg() {
    calibration_init();

    // Hand-computed gravity vector for phone pitched +30° forward:
    // θ = π/6 rad (0.5235987756 rad), φ = 0, ψ = 0
    // ax = +g * sin(30°) = +4.903325 m/s²
    // ay = 0.0 m/s²
    // az = -g * cos(30°) = -8.492806 m/s² (using g = 9.80665)
    const float pitch_rad = static_cast<float>(M_PI / 6.0);
    const float ax = G_MS2 * sinf(pitch_rad);
    const float ay = 0.0f;
    const float az = -G_MS2 * cosf(pitch_rad);

    // Provide a sample during stationary condition
    ImuSample s;
    s.timestamp_ms = 1700000000000LL;
    s.accel[0] = ax; s.accel[1] = ay; s.accel[2] = az;
    s.gyro[0] = 0.0f; s.gyro[1] = 0.0f; s.gyro[2] = 0.0f;

    calibration_update(s);
    CalibrationState st = get_current_rotation();

    ASSERT_NEAR(st.pitch_rad, pitch_rad, 0.005f, "Pitch from known 30 deg gravity vector");
    ASSERT_NEAR(st.roll_rad,  0.0f,      0.001f, "Roll should be near 0 for pure pitch scenario");
}


// ─── Test 2: Angle wrap endurance over long gap (§5.3) ──────────────────────
static void test_angle_wrap_long_gap() {
    calibration_init();

    // Initialize flat
    ImuSample init_s;
    init_s.timestamp_ms = 1700000000000LL;
    init_s.accel[0] = 0.0f; init_s.accel[1] = 0.0f; init_s.accel[2] = -G_MS2;
    init_s.gyro[0] = 0.0f; init_s.gyro[1] = 0.0f; init_s.gyro[2] = 0.0f;
    calibration_update(init_s);

    // 1000 seconds at 100 Hz = 100,000 steps with pure gyro pitch rate = 5.0 rad/s
    // Without wrap_pi, angle would reach 5000 rad and cause precision failure.
    const int STEPS = 100000;
    const int64_t dt_ms = 10;
    int64_t ts = 1700000000010LL;

    for (int i = 0; i < STEPS; ++i) {
        ImuSample s;
        s.timestamp_ms = ts;
        // Accelerometer reads invalid or dynamic values so gate does NOT trigger correct()
        s.accel[0] = 1.5f; s.accel[1] = 0.0f; s.accel[2] = -G_MS2;
        s.gyro[0]  = 0.0f; s.gyro[1]  = 5.0f; s.gyro[2]  = 0.0f; // 5 rad/s
        ts += dt_ms;
        calibration_update(s);
    }

    CalibrationState st = get_current_rotation();

    ASSERT(st.pitch_rad >= -static_cast<float>(M_PI) && st.pitch_rad <= static_cast<float>(M_PI),
           "Pitch must remain wrapped in [-pi, pi] after 1000s gap");
    ASSERT(st.roll_rad >= -static_cast<float>(M_PI) && st.roll_rad <= static_cast<float>(M_PI),
           "Roll must remain wrapped in [-pi, pi] after 1000s gap");

    ASSERT(!std::isnan(st.pitch_rad) && !std::isinf(st.pitch_rad), "Pitch must not be NaN/Inf");
    ASSERT(!std::isnan(st.roll_rad)  && !std::isinf(st.roll_rad),  "Roll must not be NaN/Inf");
}


// ─── Test 3: Noise inflation suppresses bad gravity during braking (§5.4) ───
static void test_noise_inflation_suppresses_braking() {
    calibration_init();

    // Start with phone flat: θ=0, φ=0
    int64_t ts = 1700000000000LL;
    ImuSample init_s;
    init_s.timestamp_ms = ts;
    init_s.accel[0] = 0.0f; init_s.accel[1] = 0.0f; init_s.accel[2] = -G_MS2;
    init_s.gyro[0] = 0.0f; init_s.gyro[1] = 0.0f; init_s.gyro[2] = 0.0f;
    calibration_update(init_s);

    // Sustained hard braking at 4 m/s² for 5 seconds (500 steps at 100 Hz)
    // ‖a‖ = sqrt(4² + 9.80665²) ≈ 10.59 m/s²
    const int STEPS = 500;
    const int64_t dt_ms = 10;
    const float brake_ax = 4.0f;

    for (int i = 0; i < STEPS; ++i) {
        ts += dt_ms;
        ImuSample s;
        s.timestamp_ms = ts;
        s.accel[0] = brake_ax;
        s.accel[1] = 0.0f;
        s.accel[2] = -G_MS2;
        s.gyro[0]  = 0.0f; s.gyro[1] = 0.0f; s.gyro[2] = 0.0f;
        calibration_update(s);
    }

    CalibrationState st = get_current_rotation();

    // Noise inflation and gate check must protect pitch from false tilt: drift < 2° (0.035 rad)
    ASSERT(fabsf(st.pitch_rad) < 0.035f,
           ("Noise inflation must suppress pitch drift during braking: got " +
            std::to_string(st.pitch_rad) + " rad").c_str());
}


// ─── Test 4: Edge case — Zero dt (§5.5) ─────────────────────────────────────
static void test_edge_zero_dt() {
    calibration_init();

    ImuSample s;
    s.timestamp_ms = 1700000000000LL;
    s.accel[0] = 0.0f; s.accel[1] = 0.0f; s.accel[2] = -G_MS2;
    s.gyro[0]  = 0.0f; s.gyro[1]  = 0.0f; s.gyro[2]  = 0.0f;

    calibration_update(s);
    // Identical timestamp → dt = 0
    calibration_update(s);

    CalibrationState st = get_current_rotation();
    ASSERT(!std::isnan(st.pitch_rad) && !std::isinf(st.pitch_rad), "Must handle dt=0 without NaN/Inf");
    ASSERT(!std::isnan(st.roll_rad)  && !std::isinf(st.roll_rad),  "Must handle dt=0 without NaN/Inf");
}


// ─── Test 5: Edge case — Session start without static init (§5.5) ───────────
static void test_edge_session_start_no_static_init() {
    calibration_init();

    ImuSample s;
    s.timestamp_ms = 1700000000000LL;
    s.accel[0] = 0.0f; s.accel[1] = 0.0f; s.accel[2] = -G_MS2;
    s.gyro[0]  = 0.1f; s.gyro[1]  = 0.0f; s.gyro[2]  = 0.0f;

    for (int i = 0; i < 10; ++i) {
        s.timestamp_ms += 10;
        calibration_update(s);
    }

    CalibrationState st = get_current_rotation();
    ASSERT(!std::isnan(st.pitch_rad), "Must handle start without prior static init");
    ASSERT(!std::isnan(st.roll_rad),  "Must handle start without prior static init");
}


// ─── Test 6: Edge case — NaN accel input rejected (§5.5) ───────────────────
static void test_edge_nan_accel_rejected() {
    calibration_init();

    ImuSample good;
    good.timestamp_ms = 1700000000000LL;
    good.accel[0] = 0.0f; good.accel[1] = 0.0f; good.accel[2] = -G_MS2;
    good.gyro[0]  = 0.0f; good.gyro[1]  = 0.0f; good.gyro[2]  = 0.0f;
    calibration_update(good);

    ImuSample bad;
    bad.timestamp_ms = 1700000000010LL;
    bad.accel[0] = NAN;  bad.accel[1] = 0.0f; bad.accel[2] = -G_MS2;
    bad.gyro[0]  = 0.0f; bad.gyro[1]  = 0.0f; bad.gyro[2]  = 0.0f;
    calibration_update(bad); // Guard must skip corrupted sample

    CalibrationState st = get_current_rotation();
    ASSERT(!std::isnan(st.pitch_rad), "NaN accel input must not corrupt pitch");
    ASSERT(!std::isnan(st.roll_rad),  "NaN accel input must not corrupt roll");
}


// ─── Test 7: Level accelerometer check ──────────────────────────────────────
static void test_level_accelerometer_flat_and_tilted() {
    calibration_init();

    // Flat phone: raw accel [0, 0, -g]
    // Leveled accel should be [0, 0, 0] (gravity removed)
    float raw_flat[3] = { 0.0f, 0.0f, -G_MS2 };
    float leveled_flat[3];
    level_accelerometer(raw_flat, leveled_flat);

    ASSERT_NEAR(leveled_flat[0], 0.0f, 0.001f, "Leveled ax flat must be 0");
    ASSERT_NEAR(leveled_flat[1], 0.0f, 0.001f, "Leveled ay flat must be 0");
    ASSERT_NEAR(leveled_flat[2], 0.0f, 0.001f, "Leveled az flat must be 0 (gravity removed)");

    // Add forward acceleration of 2.0 m/s²: raw = [2.0, 0, -g]
    float raw_accel[3] = { 2.0f, 0.0f, -G_MS2 };
    float leveled_accel[3];
    level_accelerometer(raw_accel, leveled_accel);

    ASSERT_NEAR(leveled_accel[0], 2.0f, 0.001f, "Leveled ax forward must be 2.0");
    ASSERT_NEAR(leveled_accel[1], 0.0f, 0.001f, "Leveled ay forward must be 0.0");
    ASSERT_NEAR(leveled_accel[2], 0.0f, 0.001f, "Leveled az forward must be 0.0");
}


// ─── Test 8: Pooled yaw optimizer verification ──────────────────────────────
static void test_pooled_yaw_optimizer() {
    PooledYawOptimizer opt;
    ASSERT(!opt.ready(), "Optimizer must not be ready initially");

    // True heading misalignment ψ = 30° = π/6 rad (0.523599 rad)
    // Vehicle longitudinal braking of 3.0 m/s² appears rotated by ψ:
    // ax_leveled = a_fwd * cos(ψ)
    // ay_leveled = a_fwd * sin(ψ)
    const float true_psi = static_cast<float>(M_PI / 6.0);
    const int N_PER_WIN = 60;
    std::vector<float> win_ax(N_PER_WIN);
    std::vector<float> win_ay(N_PER_WIN);

    for (int i = 0; i < N_PER_WIN; ++i) {
        float a_fwd = 2.5f + 0.5f * sinf(static_cast<float>(i));
        win_ax[i] = a_fwd * cosf(true_psi);
        win_ay[i] = a_fwd * sinf(true_psi);
    }

    // Add 3 windows (exceeds YAW_MIN_WINDOWS=3 and YAW_MIN_SAMPLES=150)
    opt.add_window(win_ax.data(), win_ay.data(), N_PER_WIN);
    opt.add_window(win_ax.data(), win_ay.data(), N_PER_WIN);
    opt.add_window(win_ax.data(), win_ay.data(), N_PER_WIN);

    ASSERT(opt.ready(), "Optimizer should be ready after 3 windows of 60 samples");
    float solved_psi = opt.solve();

    ASSERT_NEAR(solved_psi, true_psi, 0.01f, "Pooled yaw solve must recover true ψ within 0.01 rad");
}


// ─── Main test entry point ──────────────────────────────────────────────────
int main() {
    fprintf(stdout, "\n========================================\n");
    fprintf(stdout, "   Member 3: Calibration & Alignment Tests\n");
    fprintf(stdout, "========================================\n");

    RUN_TEST(test_known_angle_pitch_30deg);
    RUN_TEST(test_angle_wrap_long_gap);
    RUN_TEST(test_noise_inflation_suppresses_braking);
    RUN_TEST(test_edge_zero_dt);
    RUN_TEST(test_edge_session_start_no_static_init);
    RUN_TEST(test_edge_nan_accel_rejected);
    RUN_TEST(test_level_accelerometer_flat_and_tilted);
    RUN_TEST(test_pooled_yaw_optimizer);

    fprintf(stdout, "========================================\n");
    fprintf(stdout, "Results: %d / %d tests passed (%d failed)\n", tests_passed, tests_run, tests_failed);
    fprintf(stdout, "========================================\n\n");

    return (tests_failed == 0 ? 0 : 1);
}
