/**
 * calibration.cpp — Device Alignment & Gravity Leveling Implementation
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §3, Master PRD §6.2
 *
 * Owner:  Member 3 (Calibration & Device Alignment)
 * Status: FULL IMPLEMENTATION
 *
 * Implements:
 *   - Static gravity alignment initialization (§4.1.2)
 *   - 2-state pitch/roll Extended Kalman Filter with continuous noise inflation (§4.1.3 - §4.1.5)
 *   - Pooled nonlinear least-squares yaw optimizer (§4.1.6)
 *   - Gravity removal and vehicle frame leveling (§4.1.7)
 *   - JNI bindings matching JniBridge.h (§4.2)
 */

#include "calibration.h"
#include <cstring>
#include <cmath>
#include <atomic>
#if defined(__has_include)
  #if __has_include(<jni.h>)
    #define HAVE_JNI 1
    #include <jni.h>
  #endif
#elif defined(__ANDROID__)
  #define HAVE_JNI 1
  #include <jni.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Diagnostic Logging Macros ──────────────────────────────────────────────
#if defined(__ANDROID__)
  #include <android/log.h>
  #define CALIB_LOG_TAG "CalibCore"
  #define CALIB_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, CALIB_LOG_TAG, __VA_ARGS__)
  #define CALIB_LOGI(...) __android_log_print(ANDROID_LOG_INFO,  CALIB_LOG_TAG, __VA_ARGS__)
  #define CALIB_LOGW(...) __android_log_print(ANDROID_LOG_WARN,  CALIB_LOG_TAG, __VA_ARGS__)
  #define CALIB_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, CALIB_LOG_TAG, __VA_ARGS__)
#elif defined(CALIB_DEBUG_ENABLE)
  #include <cstdio>
  #define CALIB_LOGD(...) do { fprintf(stdout, "[CalibCore DEBUG] " __VA_ARGS__); fprintf(stdout, "\n"); } while(0)
  #define CALIB_LOGI(...) do { fprintf(stdout, "[CalibCore INFO]  " __VA_ARGS__); fprintf(stdout, "\n"); } while(0)
  #define CALIB_LOGW(...) do { fprintf(stderr, "[CalibCore WARN]  " __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
  #define CALIB_LOGE(...) do { fprintf(stderr, "[CalibCore ERROR] " __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#else
  #define CALIB_LOGD(...) ((void)0)
  #define CALIB_LOGI(...) ((void)0)
  #define CALIB_LOGW(...) ((void)0)
  #define CALIB_LOGE(...) ((void)0)
#endif

// ─── Module singletons & thread safety ──────────────────────────────────────
static DynamicAlignmentEKF g_ekf;
static PooledYawOptimizer  g_yaw_opt;
static CalibrationState    g_state = {0.0f, 0.0f, 0.0f, 0};

// Spinlock for JNI and cross-thread concurrency
static std::atomic_flag g_lock = ATOMIC_FLAG_INIT;

static inline void acquire_lock() {
    while (g_lock.test_and_set(std::memory_order_acquire)) {
        // spin
    }
}

static inline void release_lock() {
    g_lock.clear(std::memory_order_release);
}

// Window accumulator for pooled yaw optimizer
static constexpr int WINDOW_BUFFER_CAPACITY = 100;
static float s_win_ax[WINDOW_BUFFER_CAPACITY];
static float s_win_ay[WINDOW_BUFFER_CAPACITY];
static int   s_win_count = 0;
static bool  s_in_window = false;


/* ─────────────────────────────────────────────────────────────────────────────
 * DynamicAlignmentEKF Implementation
 * ───────────────────────────────────────────────────────────────────────────── */

DynamicAlignmentEKF::DynamicAlignmentEKF() {
    reset();
}

void DynamicAlignmentEKF::reset() {
    // x = [pitch θ, roll φ]
    x_[0] = 0.0f;
    x_[1] = 0.0f;

    // Initial covariance P: 2x2 row-major, initial variance 0.01 rad²
    P_[0] = 0.01f; P_[1] = 0.0f;
    P_[2] = 0.0f;  P_[3] = 0.01f;

    // Process noise Q (gyro variance): Q = diag(σ_gyro², σ_gyro²)
    Q_[0] = SIGMA_GYRO * SIGMA_GYRO; Q_[1] = 0.0f;
    Q_[2] = 0.0f;                    Q_[3] = SIGMA_GYRO * SIGMA_GYRO;

    // Base measurement noise R0 (accel angle variance)
    R0_[0] = SIGMA_ACCEL * SIGMA_ACCEL; R0_[1] = 0.0f;
    R0_[2] = 0.0f;                      R0_[3] = SIGMA_ACCEL * SIGMA_ACCEL;

    last_ts_ms_ = -1;
    gate_elapsed_ms_ = 0;
    gate_sustained_ = false;
    initialized_ = false;
}

/**
 * §4.1.2 Static Initialization from Gravity Vector
 * θ0 = atan2(-ax, sqrt(ay² + az²))
 * φ0 = atan2(ay, az)
 */
void DynamicAlignmentEKF::initialize_from_gravity(float ax, float ay, float az) {
    float norm_yz = sqrtf(ay * ay + az * az);
    float theta0 = atan2f(ax, norm_yz);
    float phi0 = atan2f(ay, -az);

    x_[0] = wrap_pi(theta0);
    x_[1] = wrap_pi(phi0);

    P_[0] = SIGMA_ACCEL * SIGMA_ACCEL; P_[1] = 0.0f;
    P_[2] = 0.0f;                      P_[3] = SIGMA_ACCEL * SIGMA_ACCEL;

    gate_elapsed_ms_ = 0;
    gate_sustained_ = false;
    initialized_ = true;

    CALIB_LOGI("Static gravity alignment locked: pitch=%.3f deg (%.4f rad), roll=%.3f deg (%.4f rad)",
               x_[0] * 180.0f / static_cast<float>(M_PI), x_[0],
               x_[1] * 180.0f / static_cast<float>(M_PI), x_[1]);
}

float DynamicAlignmentEKF::pitch() const {
    return wrap_pi(x_[0]);
}

float DynamicAlignmentEKF::roll() const {
    return wrap_pi(x_[1]);
}

/**
 * §4.1.3 EKF Process Model — Nonlinear Kinematics
 *
 * Continuous-time rate equations:
 *   θ_dot = ω_y * cos(φ) - ω_z * sin(φ)
 *   φ_dot = ω_x + (ω_y * sin(φ) + ω_z * cos(φ)) * tan(θ)
 *
 * Discrete integration:
 *   θ_{k+1} = wrap_pi(θ_k + θ_dot * dt)
 *   φ_{k+1} = wrap_pi(φ_k + φ_dot * dt)
 */
void DynamicAlignmentEKF::predict(float gx, float gy, float gz, float dt_s) {
    if (dt_s <= 0.0f) {
        return; // Guard against zero or negative dt
    }

    const float theta = x_[0];
    const float phi   = x_[1];

    const float sin_phi = sinf(phi);
    const float cos_phi = cosf(phi);

    // Singularity protection for tan(θ) and sec²(θ) near ±π/2
    float cos_theta = cosf(theta);
    if (fabsf(cos_theta) < 1e-4f) {
        cos_theta = (cos_theta >= 0.0f ? 1e-4f : -1e-4f);
    }
    const float tan_theta = sinf(theta) / cos_theta;
    const float sec2_theta = 1.0f / (cos_theta * cos_theta);

    // Kinematic rates
    const float theta_dot = gy * cos_phi - gz * sin_phi;
    const float phi_dot   = gx + (gy * sin_phi + gz * cos_phi) * tan_theta;

    // Propagate state with MANDATORY wrap_pi
    x_[0] = wrap_pi(theta + theta_dot * dt_s);
    x_[1] = wrap_pi(phi   + phi_dot   * dt_s);

    // Linearized Jacobian F = ∂f/∂x:
    // F00 = 0
    // F01 = -ω_y * sin(φ) - ω_z * cos(φ)
    // F10 = (ω_y * sin(φ) + ω_z * cos(φ)) * sec²(θ)
    // F11 = (ω_y * cos(φ) - ω_z * sin(φ)) * tan(θ)
    const float F01 = -gy * sin_phi - gz * cos_phi;
    const float F10 = (gy * sin_phi + gz * cos_phi) * sec2_theta;
    const float F11 = (gy * cos_phi - gz * sin_phi) * tan_theta;

    // Discrete transition matrix F_d = I + F * dt
    const float Fd[4] = {
        1.0f,          F01 * dt_s,
        F10 * dt_s,    1.0f + F11 * dt_s
    };

    // Covariance propagation: P = F_d * P * F_d^T + Q * dt_s
    // Step 1: FP = F_d * P
    float FP[4];
    FP[0] = Fd[0] * P_[0] + Fd[1] * P_[2];
    FP[1] = Fd[0] * P_[1] + Fd[1] * P_[3];
    FP[2] = Fd[2] * P_[0] + Fd[3] * P_[2];
    FP[3] = Fd[2] * P_[1] + Fd[3] * P_[3];

    // Step 2: P_new = FP * F_d^T + Q * dt_s
    const float q_dt = Q_[0] * dt_s;
    float P_new[4];
    P_new[0] = FP[0] * Fd[0] + FP[1] * Fd[1] + q_dt;
    P_new[1] = FP[0] * Fd[2] + FP[1] * Fd[3];
    P_new[2] = FP[2] * Fd[0] + FP[3] * Fd[1];
    P_new[3] = FP[2] * Fd[2] + FP[3] * Fd[3] + q_dt;

    // Enforce symmetry
    const float p_cross = 0.5f * (P_new[1] + P_new[2]);
    P_[0] = P_new[0];
    P_[1] = p_cross;
    P_[2] = p_cross;
    P_[3] = P_new[3];
}

/**
 * §4.1.4 Continuous Noise Inflation
 * R_k = R0 * (1 + κ * (|‖a‖ - g|)^2)
 */
void DynamicAlignmentEKF::compute_inflated_R(float accel_mag, float R_out[4]) const {
    const float diff = accel_mag - G_MS2;
    const float inflation_factor = 1.0f + KAPPA * (diff * diff);
    const float r_val = R0_[0] * inflation_factor;

    R_out[0] = r_val; R_out[1] = 0.0f;
    R_out[2] = 0.0f;  R_out[3] = r_val;
}

/**
 * §4.1.4 EKF Measurement Model — Gravity Angles with Innovation Wrap
 *
 * z_k = [atan2(-ax, sqrt(ay² + az²)), atan2(ay, az)]^T
 * ν = wrap_pi(z_k - x_{k|k-1})
 */
void DynamicAlignmentEKF::correct(float ax, float ay, float az) {
    const float norm_yz = sqrtf(ay * ay + az * az);
    const float theta_acc = atan2f(ax, norm_yz);
    const float phi_acc   = atan2f(ay, -az);

    // Innovation residual — MANDATORY wrap_pi
    const float nu0 = wrap_pi(theta_acc - x_[0]);
    const float nu1 = wrap_pi(phi_acc   - x_[1]);

    const float accel_mag = sqrtf(ax * ax + ay * ay + az * az);
    float R[4];
    compute_inflated_R(accel_mag, R);

    // Innovation covariance: S = P + R (since H = I_2x2)
    const float S00 = P_[0] + R[0];
    const float S01 = P_[1];
    const float S10 = P_[2];
    const float S11 = P_[3] + R[3];

    // Invert 2x2 matrix S
    const float det = S00 * S11 - S01 * S10;
    if (fabsf(det) < 1e-9f) {
        return; // Ill-conditioned, skip correction
    }
    const float inv_det = 1.0f / det;
    const float invS00 =  S11 * inv_det;
    const float invS01 = -S01 * inv_det;
    const float invS10 = -S10 * inv_det;
    const float invS11 =  S00 * inv_det;

    // Kalman Gain K = P * S^{-1}
    const float K00 = P_[0] * invS00 + P_[1] * invS10;
    const float K01 = P_[0] * invS01 + P_[1] * invS11;
    const float K10 = P_[2] * invS00 + P_[3] * invS10;
    const float K11 = P_[2] * invS01 + P_[3] * invS11;

    // State update with MANDATORY wrap_pi
    x_[0] = wrap_pi(x_[0] + K00 * nu0 + K01 * nu1);
    x_[1] = wrap_pi(x_[1] + K10 * nu0 + K11 * nu1);

    // Covariance update: P = (I - K) * P
    const float I_K00 = 1.0f - K00;
    const float I_K01 = -K01;
    const float I_K10 = -K10;
    const float I_K11 = 1.0f - K11;

    float P_post[4];
    P_post[0] = I_K00 * P_[0] + I_K01 * P_[2];
    P_post[1] = I_K00 * P_[1] + I_K01 * P_[3];
    P_post[2] = I_K10 * P_[0] + I_K11 * P_[2];
    P_post[3] = I_K10 * P_[1] + I_K11 * P_[3];

    const float p_post_cross = 0.5f * (P_post[1] + P_post[2]);
    P_[0] = P_post[0];
    P_[1] = p_post_cross;
    P_[2] = p_post_cross;
    P_[3] = P_post[3];
}

/**
 * §4.1.5 Sustained Gate Logic & Update Loop
 */
void DynamicAlignmentEKF::update(const ImuSample& sample) {
    // NaN / Inf guard (PRD §7 guardrail)
    if (std::isnan(sample.accel[0]) || std::isnan(sample.accel[1]) || std::isnan(sample.accel[2]) ||
        std::isnan(sample.gyro[0])  || std::isnan(sample.gyro[1])  || std::isnan(sample.gyro[2]) ||
        std::isinf(sample.accel[0]) || std::isinf(sample.accel[1]) || std::isinf(sample.accel[2]) ||
        std::isinf(sample.gyro[0])  || std::isinf(sample.gyro[1])  || std::isinf(sample.gyro[2])) {
        CALIB_LOGW("update: Corrupted IMU sample rejected (contains NaN or Inf)");
        return;
    }

    if (!initialized_) {
        initialize_from_gravity(sample.accel[0], sample.accel[1], sample.accel[2]);
    }

    if (last_ts_ms_ < 0) {
        last_ts_ms_ = sample.timestamp_ms;
        return;
    }

    int64_t dt_ms = sample.timestamp_ms - last_ts_ms_;
    if (dt_ms < 0) {
        dt_ms = 0;
    }
    last_ts_ms_ = sample.timestamp_ms;
    const float dt_s = static_cast<float>(dt_ms) * 0.001f;

    // Process model predict
    predict(sample.gyro[0], sample.gyro[1], sample.gyro[2], dt_s);

    // Sustained gate evaluation (§4.1.5)
    const float accel_mag = sqrtf(sample.accel[0] * sample.accel[0] +
                                  sample.accel[1] * sample.accel[1] +
                                  sample.accel[2] * sample.accel[2]);
    const float gyro_mag  = sqrtf(sample.gyro[0] * sample.gyro[0] +
                                  sample.gyro[1] * sample.gyro[1] +
                                  sample.gyro[2] * sample.gyro[2]);

    const bool accel_ok = fabsf(accel_mag - G_MS2) < ACCEL_GATE_MS2;
    const bool gyro_ok  = gyro_mag < GYRO_GATE_RADS;

    if (accel_ok && gyro_ok) {
        gate_elapsed_ms_ += dt_ms;
        // G6 fix: cap counter at threshold to prevent unbounded growth on long
        // stationary sessions (int64_t won't overflow but the value is semantically
        // meaningless beyond SUSTAINED_GATE_SECS).
        const int64_t gate_cap_ms = static_cast<int64_t>(SUSTAINED_GATE_SECS * 1000.0f);
        if (gate_elapsed_ms_ >= gate_cap_ms) {
            gate_elapsed_ms_ = gate_cap_ms;   // cap — do not grow unbounded
            gate_sustained_ = true;
            correct(sample.accel[0], sample.accel[1], sample.accel[2]);
        }
    } else {
        // CONTINUOUS gate requirement: reset if broken
        gate_elapsed_ms_ = 0;
        gate_sustained_ = false;
    }
}


/* ─────────────────────────────────────────────────────────────────────────────
 * PooledYawOptimizer Implementation
 * ───────────────────────────────────────────────────────────────────────────── */

PooledYawOptimizer::PooledYawOptimizer() {
    reset();
}

void PooledYawOptimizer::reset() {
    sum_ax2_   = 0.0;
    sum_ay2_   = 0.0;
    sum_axay_  = 0.0;
    sum_ax_    = 0.0;   // G5: first-order longitudinal sum for quadrant disambiguation
    sum_ay_    = 0.0;   // G5: first-order lateral sum for quadrant disambiguation
    n_windows_ = 0;
    n_samples_ = 0;
}

void PooledYawOptimizer::add_window(const float* leveled_ax_samples,
                                    const float* leveled_ay_samples,
                                    int n_samples) {
    if (n_samples <= 0 || !leveled_ax_samples || !leveled_ay_samples) {
        return;
    }

    for (int i = 0; i < n_samples; ++i) {
        const double ax = static_cast<double>(leveled_ax_samples[i]);
        const double ay = static_cast<double>(leveled_ay_samples[i]);

        sum_ax2_  += ax * ax;
        sum_ay2_  += ay * ay;
        sum_axay_ += ax * ay;
        sum_ax_   += ax;   // G5: accumulate first-order for quadrant disambiguation
        sum_ay_   += ay;   // G5: accumulate first-order for quadrant disambiguation
    }

    n_windows_++;
    n_samples_ += n_samples;
}

bool PooledYawOptimizer::ready() const {
    return (n_windows_ >= YAW_MIN_WINDOWS && n_samples_ >= YAW_MIN_SAMPLES);
}

/**
 * §4.1.6 Pooled Yaw Solve — Closed-Form with Quadrant Disambiguation
 *
 * Step 1: ψ* = 0.5 * atan2(2 * Σ(ax*ay), Σ(ax²) - Σ(ay²))
 * Step 2: Quadrant disambiguation — atan2/2 returns one of two energy minima
 *         (ψ* and ψ* + π). The correct one is where dominant longitudinal
 *         motion projects positively onto the solved x_vehicle direction:
 *         Σax·cos(ψ) + Σay·sin(ψ) > 0 → keep ψ; < 0 → flip by π.
 */
float PooledYawOptimizer::solve() const {
    const double diff  = sum_ax2_ - sum_ay2_;
    const double cross = 2.0 * sum_axay_;

    // Raw closed-form solution (one of two minima)
    float psi = 0.5f * atan2f(static_cast<float>(cross), static_cast<float>(diff));
    const float psi_raw = psi;

    // Quadrant disambiguation (G5 fix — §4.1.6):
    // Check if the solved ψ aligns with the dominant direction of accumulated
    // longitudinal acceleration. If alignment < 0, the phone is facing backward
    // relative to the solved ψ — flip by π to get the forward-facing solution.
    const double alignment = sum_ax_ * static_cast<double>(cosf(psi))
                           + sum_ay_ * static_cast<double>(sinf(psi));
    if (alignment < 0.0) {
        psi = wrap_pi(psi + static_cast<float>(M_PI));
        CALIB_LOGI("Yaw quadrant flip applied: raw=%.4f rad -> flipped=%.4f rad (alignment=%.3f)",
                   psi_raw, psi, alignment);
    } else {
        CALIB_LOGD("Yaw quadrant alignment positive (%.3f): keeping raw psi=%.4f rad",
                   alignment, psi);
    }

    return wrap_pi(psi);
}

int PooledYawOptimizer::window_count() const {
    return n_windows_;
}

int PooledYawOptimizer::total_samples() const {
    return n_samples_;
}


/* ─────────────────────────────────────────────────────────────────────────────
 * Level Accelerometer Implementation
 * ───────────────────────────────────────────────────────────────────────────── */

/**
 * §4.1.1 & §4.1.7 Direction Cosine Matrix and Leveling
 *
 * R_{V←B} = R_z(ψ) * R_y(θ) * R_x(φ)
 *
 * Leveled accel = R_{V←B} * raw_accel + [0, 0, g]^T
 */
void level_accelerometer(const float raw_accel[3], float out_accel[3]) {
    acquire_lock();
    const float theta = g_state.pitch_rad;
    const float phi   = g_state.roll_rad;
    const float psi   = g_state.yaw_calibrated ? g_state.yaw_rad : 0.0f;
    release_lock();

    const float cx = cosf(phi),   sx = sinf(phi);
    const float cy = cosf(theta), sy = sinf(theta);
    const float cz = cosf(psi),   sz = sinf(psi);

    // Direction Cosine Matrix R_{V←B} = R_z * R_y * R_x
    // Row 0:
    const float R00 = cz * cy;
    const float R01 = cz * sy * sx - sz * cx;
    const float R02 = cz * sy * cx + sz * sx;

    // Row 1:
    const float R10 = sz * cy;
    const float R11 = sz * sy * sx + cz * cx;
    const float R12 = sz * sy * cx - cz * sx;

    // Row 2:
    const float R20 = -sy;
    const float R21 = cy * sx;
    const float R22 = cy * cx;

    const float ax = raw_accel[0];
    const float ay = raw_accel[1];
    const float az = raw_accel[2];

    // Vehicle-frame acceleration (still includes gravity along -z_V)
    const float v_ax = R00 * ax + R01 * ay + R02 * az;
    const float v_ay = R10 * ax + R11 * ay + R12 * az;
    const float v_az = R20 * ax + R21 * ay + R22 * az;

    // Subtract vehicle gravity vector [0, 0, -g]^T (add +g along vehicle vertical axis)
    out_accel[0] = v_ax;
    out_accel[1] = v_ay;
    out_accel[2] = v_az + G_MS2;
}


/* ─────────────────────────────────────────────────────────────────────────────
 * Module Free Functions
 * ───────────────────────────────────────────────────────────────────────────── */

void calibration_init() {
    acquire_lock();
    g_ekf.reset();
    g_yaw_opt.reset();
    g_state = {0.0f, 0.0f, 0.0f, 0};
    s_win_count = 0;
    s_in_window = false;
    release_lock();
    CALIB_LOGI("calibration_init: Engine state reset to identity.");
}

void calibration_update(const ImuSample& sample) {
    acquire_lock();

    // 1. Advance dynamic EKF for pitch and roll
    g_ekf.update(sample);
    g_state.pitch_rad = g_ekf.pitch();
    g_state.roll_rad  = g_ekf.roll();

    // 2. Candidate window accumulation for pooled yaw optimizer
    // Check if vehicle is in a straight braking/acceleration maneuver
    const float gyro_mag = sqrtf(sample.gyro[0] * sample.gyro[0] +
                                 sample.gyro[1] * sample.gyro[1] +
                                 sample.gyro[2] * sample.gyro[2]);

    if (!g_state.yaw_calibrated) {
        // Level sample acceleration using pitch/roll only (yaw=0)
        float unleveled_leveled[3];
        const float theta = g_state.pitch_rad;
        const float phi   = g_state.roll_rad;

        const float cx = cosf(phi),   sx = sinf(phi);
        const float cy = cosf(theta), sy = sinf(theta);

        const float R00 = cy;
        const float R01 = sy * sx;
        const float R02 = sy * cx;
        const float R10 = 0.0f;
        const float R11 = cx;
        const float R12 = -sx;
        const float R20 = -sy;
        const float R21 = cy * sx;
        const float R22 = cy * cx;

        const float ax = sample.accel[0];
        const float ay = sample.accel[1];
        const float az = sample.accel[2];

        unleveled_leveled[0] = R00 * ax + R01 * ay + R02 * az;
        unleveled_leveled[1] = R10 * ax + R11 * ay + R12 * az;
        unleveled_leveled[2] = R20 * ax + R21 * ay + R22 * az + G_MS2;

        // Use rotation-invariant horizontal acceleration magnitude (Issue #1 fix)
        // so braking windows qualify reliably regardless of phone mounting yaw.
        const float horiz_accel_mag = sqrtf(unleveled_leveled[0] * unleveled_leveled[0] +
                                            unleveled_leveled[1] * unleveled_leveled[1]);
        const bool is_braking_accel = (horiz_accel_mag >= YAW_WINDOW_ACCEL_THRESHOLD) &&
                                      (gyro_mag < GYRO_GATE_RADS);

        if (is_braking_accel) {
            if (!s_in_window) {
                CALIB_LOGD("Yaw window #%d started (horiz_accel=%.3f m/s²)",
                           g_yaw_opt.window_count() + 1, horiz_accel_mag);
            }
            s_in_window = true;
            if (s_win_count < WINDOW_BUFFER_CAPACITY) {
                s_win_ax[s_win_count] = unleveled_leveled[0];
                s_win_ay[s_win_count] = unleveled_leveled[1];
                s_win_count++;
            }
        } else {
            if (s_in_window) {
                // Event ended: commit accumulated window to pooled optimizer
                if (s_win_count >= 10) {
                    g_yaw_opt.add_window(s_win_ax, s_win_ay, s_win_count);
                    CALIB_LOGD("Yaw window committed: %d samples. Total: %d samples, %d windows",
                               s_win_count, g_yaw_opt.total_samples(), g_yaw_opt.window_count());
                }
                s_win_count = 0;
                s_in_window = false;

                // Check if ready to solve
                if (g_yaw_opt.ready()) {
                    g_state.yaw_rad = g_yaw_opt.solve();
                    g_state.yaw_calibrated = 1;
                    CALIB_LOGI("Yaw calibration COMPLETE: yaw=%.4f rad (%.2f deg)",
                               g_state.yaw_rad, g_state.yaw_rad * 180.0f / static_cast<float>(M_PI));
                }
            }
        }
    }

    release_lock();
}

CalibrationState get_current_rotation() {
    acquire_lock();
    CalibrationState copy = g_state;
    release_lock();
    return copy;
}

void reset_calibration() {
    calibration_init();
}


#if defined(HAVE_JNI)
/* ─────────────────────────────────────────────────────────────────────────────
 * JNI Bindings (matching JniBridge.h)
 * ───────────────────────────────────────────────────────────────────────────── */

extern "C" {

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_calibrationUpdate(
    JNIEnv *env, jobject /* thiz */,
    jlong ts,
    jfloatArray accel,
    jfloatArray gyro) {
    if (!accel || !gyro) return;

    jfloat accel_buf[3];
    jfloat gyro_buf[3];

    env->GetFloatArrayRegion(accel, 0, 3, accel_buf);
    env->GetFloatArrayRegion(gyro, 0, 3, gyro_buf);

    ImuSample sample;
    sample.timestamp_ms = ts;
    sample.accel[0] = accel_buf[0];
    sample.accel[1] = accel_buf[1];
    sample.accel[2] = accel_buf[2];
    sample.gyro[0]  = gyro_buf[0];
    sample.gyro[1]  = gyro_buf[1];
    sample.gyro[2]  = gyro_buf[2];

    calibration_update(sample);
}

JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_levelAccelerometer(
    JNIEnv *env, jobject /* thiz */,
    jfloatArray rawAccel) {
    if (!rawAccel) return nullptr;

    jfloat raw_buf[3];
    jfloat leveled_buf[3];

    env->GetFloatArrayRegion(rawAccel, 0, 3, raw_buf);
    level_accelerometer(raw_buf, leveled_buf);

    jfloatArray result = env->NewFloatArray(3);
    if (result) {
        env->SetFloatArrayRegion(result, 0, 3, leveled_buf);
    }
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_getCurrentRotation(
    JNIEnv *env, jobject /* thiz */) {
    CalibrationState st = get_current_rotation();

    // 4-element return: [pitch_rad, roll_rad, yaw_rad, yaw_calibrated_float]
    // See Implementation Guide §5.3 / Gap 1: preserves yaw_calibrated for Kotlin
    jfloat values[4] = {
        st.pitch_rad,
        st.roll_rad,
        st.yaw_rad,
        static_cast<jfloat>(st.yaw_calibrated)
    };

    jfloatArray result = env->NewFloatArray(4);
    if (result) {
        env->SetFloatArrayRegion(result, 0, 4, values);
    }
    return result;
}

/**
 * Native alias for AlignmentManager.kt private external fun getCurrentRotationNative
 */
JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_getCurrentRotationNative(
    JNIEnv *env, jobject thiz) {
    return Java_com_sih_deadreckoning_calibration_AlignmentManager_getCurrentRotation(env, thiz);
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_resetCalibration(
    JNIEnv * /* env */, jobject /* thiz */) {
    reset_calibration();
}

} // extern "C"
#endif // HAVE_JNI
