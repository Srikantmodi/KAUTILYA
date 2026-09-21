/**
 * ukf_fusion.h — Unscented Kalman Filter Fusion Engine
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §4, Master PRD §6.3
 *
 * Owner:  Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Purpose: The UKF maintains a 10-dimensional state vector and fuses:
 *   - ML model delta_v (predict)
 *   - GNSS fixes (loosely-coupled update)
 *   - Map-matching pseudo-measurements (closed-loop)
 *   - ZUPT (zero velocity when stationary)
 *   - ZARU (zero angular rate for gyro bias correction)
 *   - ES-NN residual corrections
 *
 * State vector x[10]:
 *   [pe, pn, pu, ve, vn, vu, heading, gyro_bias_x, gyro_bias_y, gyro_bias_z]
 *
 * Conventions (api-contracts.md §0):
 *   - ENU frame, heading 0=East CCW+ (matching imu_types.h)
 *   - Angles: radians.  Distances: meters.  Speed: m/s.
 *   - Timestamps: epoch milliseconds (int64_t).
 */

#ifndef CORE_ENGINE_UKF_FUSION_H
#define CORE_ENGINE_UKF_FUSION_H

#include "imu_types.h"
#include "ins_mechanization.h"

/* ─────────────────────────────────────────────────────────────────────────────
 * UKF tuning constants
 * ───────────────────────────────────────────────────────────────────────────── */

/** Map-match confidence threshold — skip update below this (§2.4, §7 #5).
 *  Default 0.7 per api-contracts.md §4. */
static constexpr float MAP_MATCH_CONFIDENCE_THRESHOLD = 0.7f;

/** Earth radius in meters (WGS-84 mean) */
static constexpr double EARTH_RADIUS_M = 6371000.0;

/** Degrees to radians */
static constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;

/* UKF sigma point parameters — chosen for numerical stability with n=10 */
static constexpr float UKF_ALPHA = 1.0f;     /* spread — 1.0 gives all-positive weights */
static constexpr float UKF_BETA  = 2.0f;     /* optimal for Gaussian priors */
static constexpr float UKF_KAPPA = 0.0f;
/* lambda = alpha^2 * (n + kappa) - n = 1.0 * 10 - 10 = 0 */
static constexpr float UKF_LAMBDA = 0.0f;
static constexpr int   N_SIGMA = 2 * N_STATE + 1;  /* 21 sigma points */

/* ─────────────────────────────────────────────────────────────────────────────
 * UKFFusion class
 * ───────────────────────────────────────────────────────────────────────────── */
class UKFFusion {
public:
    UKFFusion();

    /* ── API matching api-contracts.md §4 ────────────────────────────────── */

    /**
     * UKF predict step — strapdown integration using ML-derived delta-v.
     * @param delta_v     velocity (m/s) from ML model output
     * @param dt_seconds  time step since last predict (seconds)
     */
    void predict(float delta_v, float dt_seconds);

    /**
     * GNSS measurement update — loosely-coupled correction.
     * Converts lat/lon to ENU relative to session origin.
     * @param fix  GNSS fix from GnssQualityMonitor (Member 2)
     */
    void update_gnss(const GnssFix& fix);

    /**
     * Map-match measurement update — closed-loop correction from HMM.
     * Skipped when confidence < MAP_MATCH_CONFIDENCE_THRESHOLD.
     * @param pseudo_measurement  [lateral_offset_m, heading_offset_rad]
     * @param confidence          match confidence [0.0, 1.0]
     */
    void update_map_match(const float pseudo_measurement[2], float confidence);

    /**
     * Zero Angular Rate Update — correct gyro bias during straight-line.
     * Tightens gyro bias covariance based on pseudo-measurement of zero turn rate.
     */
    void apply_zaru();

    /**
     * Zero velocity Update — hard-clamp velocity when stationary.
     * Called when stationary_probability > 0.95 (from Member 6 model).
     */
    void apply_zupt();

    /**
     * Apply error-state neural network residual correction (§2.5).
     * Subtracts the fusion-corrector network's output from the state.
     * @param delta_p    position correction [de, dn, du] meters
     * @param delta_v    velocity correction [dve, dvn, dvu] m/s
     * @param delta_psi  heading correction, radians
     */
    void apply_es_nn_residual(const float delta_p[3],
                              const float delta_v[3],
                              float delta_psi);

    /**
     * Get the current fused navigation state.
     * @return NavState struct (from imu_types.h)
     */
    NavState get_nav_state() const;

    /**
     * Reset all state to zero, reinitialize covariance.
     */
    void reset();

    /**
     * Check if the session origin has been set (first GNSS fix received).
     */
    bool has_origin() const { return origin_set_; }

private:
    /* ── State and covariance ──────────────────────────────────────────── */
    float x_[N_STATE];                 /* state vector */
    float P_[N_STATE][N_STATE];        /* covariance matrix */

    /* ── Process noise ─────────────────────────────────────────────────── */
    float Q_[N_STATE][N_STATE];        /* process noise covariance */

    /* ── INS mechanization (process model) ─────────────────────────────── */
    INSMechanization ins_;

    /* ── Session origin (first GNSS fix → ENU reference) ───────────────── */
    double origin_lat_rad_;
    double origin_lon_rad_;
    bool   origin_set_;

    /* ── UKF sigma point weights ───────────────────────────────────────── */
    float Wm_[N_SIGMA];   /* weights for mean reconstruction */
    float Wc_[N_SIGMA];   /* weights for covariance reconstruction */

    /* ── Internal helpers ──────────────────────────────────────────────── */

    /** Compute sigma points from (x_, P_) → sigma_points[N_SIGMA][N_STATE] */
    void generate_sigma_points(float sigma[N_SIGMA][N_STATE]) const;

    /** Reconstruct mean and covariance from propagated sigma points. */
    void reconstruct_mean_cov(const float sigma[N_SIGMA][N_STATE],
                              float mean[N_STATE],
                              float cov[N_STATE][N_STATE]) const;

    /** Convert WGS-84 (lat, lon) to local ENU (east, north) meters. */
    void latlon_to_enu(double lat_deg, double lon_deg,
                       float& east_m, float& north_m) const;

    /** Extract the 4×4 position+heading covariance block for NavState. */
    void extract_pos_heading_cov(float out[16]) const;

    /** Initialize UKF weights based on alpha/beta/kappa. */
    void init_weights();

    /** Initialize process noise Q. */
    void init_process_noise();

    /** Initialize covariance P with default uncertainties. */
    void init_covariance();

    /**
     * Generic UKF measurement update for an m-dimensional measurement.
     * @param z          measurement vector [m]
     * @param z_pred     predicted measurement from sigma points [N_SIGMA][m]
     * @param m_dim      measurement dimension
     * @param R          measurement noise [m][m] (row-major flat)
     * @param sigma      sigma points [N_SIGMA][N_STATE]
     * @param x_pred     predicted state mean [N_STATE]
     */
    void ukf_measurement_update(const float* z,
                                const float z_pred_sigma[][8],
                                int m_dim,
                                const float* R,
                                const float sigma[N_SIGMA][N_STATE],
                                const float x_pred[N_STATE]);
};

/* ─────────────────────────────────────────────────────────────────────────────
 * Free functions (extern "C" for JNI bridge implementation)
 * ───────────────────────────────────────────────────────────────────────────── */
#ifdef __cplusplus
extern "C" {
#endif

/** Global UKF instance — accessed from JNI and Python bindings. */
UKFFusion& get_ukf_instance();

/** Reset the global UKF instance. */
void fusion_reset();

#ifdef __cplusplus
}
#endif

#endif /* CORE_ENGINE_UKF_FUSION_H */
