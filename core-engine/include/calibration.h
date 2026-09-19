/**
 * calibration.h — Device Alignment & Gravity Leveling
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §3, Master PRD §6.2
 *
 * Owner:  Member 3 (Calibration & Device Alignment)
 * Status: IMPLEMENTATION REQUIRED
 *
 * Purpose: Continuously estimate the physical mounting orientation of the
 * phone relative to the vehicle (pitch θ, roll φ, yaw ψ) using:
 *   - Static gravity alignment (stationary window → initial θ, φ)
 *   - Dynamic 2-state EKF (gyro-driven predict + gravity-gated correct → θ, φ)
 *   - Pooled nonlinear least-squares optimization → ψ (yaw, computed once)
 *
 * Exposes level_accelerometer() which ALL downstream consumers call on every
 * raw ImuSample to remove gravity and rotate into vehicle frame.
 *
 * Conventions (api-contracts.md §0):
 *   - Angles: radians. Timestamps: epoch ms. Accel: m/s². Gyro: rad/s.
 *   - g = 9.80665 m/s² (exact).
 *   - All internal angles MUST stay in [-π, π]. Use wrap_pi() everywhere.
 */

#ifndef CORE_ENGINE_CALIBRATION_H
#define CORE_ENGINE_CALIBRATION_H

#include <cstdint>
#include <cmath>
#include "imu_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * CONSTANTS  (constexpr — no extern "C" needed)
 * ───────────────────────────────────────────────────────────────────────── */

/** Standard gravity constant — use this everywhere, never 9.81 */
static constexpr float G_MS2 = 9.80665f;

/** Accel magnitude gate half-width (m/s²): |‖a‖ - g| < ACCEL_GATE_MS2 */
static constexpr float ACCEL_GATE_MS2 = 0.35f;

/** Gyro rate gate (rad/s): ‖ω‖ < GYRO_GATE_RADS to allow EKF correction */
static constexpr float GYRO_GATE_RADS = 0.15f;

/** Minimum sustained gate duration before triggering EKF correct() (seconds) */
static constexpr float SUSTAINED_GATE_SECS = 0.5f;

/** Noise inflation gain κ: R_k = R0 * (1 + KAPPA * (‖a‖ - g)²) */
static constexpr float KAPPA = 50.0f;

/** Baseline gyro process noise std-dev (rad/s) — tunes Q */
static constexpr float SIGMA_GYRO = 0.01f;

/** Baseline accel measurement noise std-dev (m/s²) — tunes R0 */
static constexpr float SIGMA_ACCEL = 0.1f;

/** Min braking/accel windows required before pooled yaw solve */
static constexpr int   YAW_MIN_WINDOWS = 3;

/** Min total samples (across all windows) required for yaw solve */
static constexpr int   YAW_MIN_SAMPLES = 150;

/** Longitudinal accel threshold to count a window as braking/accel (m/s²) */
static constexpr float YAW_WINDOW_ACCEL_THRESHOLD = 1.5f;


/* ─────────────────────────────────────────────────────────────────────────
 * Plain-C structs — JNI-marshalable; placed in extern "C" for ABI safety
 * ───────────────────────────────────────────────────────────────────────── */
#ifdef __cplusplus
extern "C" {
#endif

/* CalibrationState — the output struct (api-contracts.md §3) */
struct CalibrationState {
    float pitch_rad;       /**< θ — phone tilt fore/aft relative to vehicle, radians */
    float roll_rad;        /**< φ — phone tilt left/right relative to vehicle, radians */
    float yaw_rad;         /**< ψ — phone heading misalignment, radians              */
    int   yaw_calibrated;  /**< 1 when pooled optimizer has solved ψ; 0 until then   */
                           /**< NOTE: int (not bool) — safe for JNI marshaling       */
};

/* Vector3 — lightweight 3-vector (used for leveled accel return) */
struct Vector3 {
    float x, y, z;
};

#ifdef __cplusplus
} /* extern "C" */
#endif


/* ─────────────────────────────────────────────────────────────────────────
 * Math utility: angle wrapping
 * MUST be outside extern "C" — it is an inline C++ function.
 * ───────────────────────────────────────────────────────────────────────── */
/**
 * Wrap angle to [-π, π].
 * MANDATORY: call this on every EKF state component and every innovation.
 * Without this, predict() accumulates unbounded angle drift over long gaps.
 */
inline float wrap_pi(float angle) {
    angle = fmodf(angle + static_cast<float>(M_PI), 2.0f * static_cast<float>(M_PI));
    if (angle < 0.0f) angle += 2.0f * static_cast<float>(M_PI);
    return angle - static_cast<float>(M_PI);
}

#ifdef __cplusplus
/* ─────────────────────────────────────────────────────────────────────────
 * DynamicAlignmentEKF — 2-state pitch/roll Extended Kalman Filter
 *
 * State vector: x = [θ (pitch), φ (roll)]
 * Yaw is NOT part of this EKF — it is not gravity-observable.
 * ───────────────────────────────────────────────────────────────────────── */
class DynamicAlignmentEKF {
public:
    DynamicAlignmentEKF();

    /**
     * Feed a raw IMU sample and advance the EKF.
     * Internally calls predict() then conditionally correct().
     *
     * @param sample  Raw ImuSample from SensorBridge (phone frame, m/s², rad/s)
     */
    void update(const ImuSample& sample);

    /** Get the current pitch estimate in radians, wrapped to [-π, π]. */
    float pitch() const;

    /** Get the current roll estimate in radians, wrapped to [-π, π]. */
    float roll() const;

    /** Reset EKF state — call on session-start or phone remount. */
    void reset();

    /**
     * Initialize pitch/roll from a gravity vector measured during a
     * stationary window. Must be called before the first update().
     */
    void initialize_from_gravity(float ax, float ay, float az);

private:
    /* EKF State [θ, φ] */
    float x_[2];   // [pitch_rad, roll_rad]

    /* EKF Covariance — 2x2, row-major: [P00, P01, P10, P11] */
    float P_[4];

    /* Process noise covariance Q — 2x2 diagonal */
    float Q_[4];

    /* Base measurement noise covariance R0 — 2x2 diagonal */
    float R0_[4];

    /* Last timestamp seen, epoch ms. Used to compute dt. */
    int64_t last_ts_ms_;

    /* Sustained gate: time elapsed (ms) with gate conditions met */
    int64_t gate_elapsed_ms_;

    /* Whether gate conditions have been met for >= SUSTAINED_GATE_SECS */
    bool gate_sustained_;

    /* Whether the EKF has been initialized with a gravity vector */
    bool initialized_;

    /**
     * Process model: integrate gyro rates over dt to propagate [θ, φ].
     * Also propagates covariance P via linearized Jacobian F.
     * Wraps both state components to [-π, π] before returning.
     *
     * @param gx, gy, gz  Phone-frame angular rates, rad/s
     * @param dt_s         Time step in SECONDS
     */
    void predict(float gx, float gy, float gz, float dt_s);

    /**
     * Measurement model: correct [θ, φ] using gravity vector from accelerometer.
     * Only called when gate conditions are sustained (|‖a‖-g|<threshold AND
     * ‖ω‖<threshold AND sustained for >=0.5s).
     *
     * Inflates measurement noise R proportional to (‖a‖ - g)² — this is the
     * mandatory guard against accepting bad gravity reads during braking.
     *
     * Wraps innovation AND posterior state to [-π, π].
     *
     * @param ax, ay, az  Phone-frame accelerometer, m/s²
     */
    void correct(float ax, float ay, float az);

    /**
     * Compute inflated measurement noise: R = R0 * (1 + KAPPA*(‖a‖-g)²)
     * Returns R as [r00, r01, r10, r11] (diagonal → r01=r10=0).
     */
    void compute_inflated_R(float accel_mag, float R_out[4]) const;
};


/* ─────────────────────────────────────────────────────────────────────────
 * PooledYawOptimizer — one-shot closed-form yaw estimator
 *
 * Collects leveled acceleration samples from distinct qualifying
 * braking/acceleration events, then solves a single least-squares problem
 * over all of them to find the phone's heading misalignment ψ.
 *
 * PRD §6.2: "Never fit yaw per-tiny-window — it overfits noise."
 * ───────────────────────────────────────────────────────────────────────── */
class PooledYawOptimizer {
public:
    PooledYawOptimizer();

    /**
     * Add a batch of leveled acceleration samples from a qualifying window.
     * A "qualifying window" is a sustained braking or acceleration event
     * (detected as: pitch/roll rate low, ‖a_leveled_longitudinal‖ > threshold).
     *
     * @param leveled_ax_samples  Array of longitudinal (x_V) leveled accelerations
     * @param leveled_ay_samples  Array of lateral (y_V) leveled accelerations
     * @param n_samples           Number of samples in the arrays
     */
    void add_window(const float* leveled_ax_samples,
                    const float* leveled_ay_samples,
                    int n_samples);

    /**
     * Returns true when enough distinct windows and samples have been
     * accumulated to run the pooled solve.
     */
    bool ready() const;

    /**
     * Run the pooled closed-form least-squares yaw solve.
     * Should only be called when ready() returns true.
     *
     * Math: minimize sum_i (a_lateral_i(ψ))² over all pooled samples
     *       = minimize sum_i (-a_xi*sin(ψ) + a_yi*cos(ψ))²
     *
     * Closed-form:
     *   ψ* = 0.5 * atan2(2 * Σ(ax*ay), Σ(ax²) - Σ(ay²))
     *
     * @return Optimal yaw ψ in radians, wrapped to [-π, π]
     */
    float solve() const;

    /** Reset optimizer — call on session-start or phone remount. */
    void reset();

    /** Number of distinct qualifying windows accumulated so far. */
    int window_count() const;

    /** Total samples accumulated across all windows. */
    int total_samples() const;

private:
    /* Accumulated second-moment sums for the closed-form solve */
    double sum_ax2_;      // Σ ax²
    double sum_ay2_;      // Σ ay²
    double sum_axay_;     // Σ ax*ay
    /* First-order sums — needed for quadrant disambiguation (§4.1.6 Gap-G5) */
    double sum_ax_;       // Σ ax  (net longitudinal)
    double sum_ay_;       // Σ ay  (net lateral)
    int    n_windows_;    // distinct qualifying window count
    int    n_samples_;    // total sample count
};

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Module-level C API (api-contracts.md §3) — these are safe in extern "C"
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * Initialize the calibration module. Call once at session start.
 * Sets all state to identity rotation, zeros covariances.
 */
void calibration_init();

/**
 * Process one raw ImuSample. Updates EKF, checks yaw optimizer windows.
 * This is the main per-sample call — invoked by AlignmentManager.kt
 * via the JNI calibrationUpdate() function.
 *
 * @param sample  Raw ImuSample (phone frame, unleveled, includes gravity)
 */
void calibration_update(const ImuSample& sample);

/**
 * Get the current rotation estimate.
 * Thread-safety: caller must ensure no concurrent calibration_update().
 *
 * @return CalibrationState with pitch, roll, yaw (all radians),
 *         and yaw_calibrated flag.
 */
CalibrationState get_current_rotation();

/**
 * Remove gravity and rotate a raw phone-frame acceleration vector into
 * the vehicle frame. Uses the current CalibrationState internally.
 *
 * This is called on EVERY raw sample before ML inference windowing
 * (Member 6) and before UKF predict (Member 4). Signature is frozen.
 *
 * @param raw_accel  Input [ax, ay, az] phone frame, m/s² (includes gravity)
 * @param out_accel  Output [a_fwd, a_lat, a_vert] vehicle frame, m/s²
 *                   (gravity removed; a_vert ≈ 0 on flat road)
 */
void level_accelerometer(const float raw_accel[3], float out_accel[3]);

/**
 * Reset all calibration state. Call on session-start or phone remount.
 */
void reset_calibration();

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CORE_ENGINE_CALIBRATION_H */
