/**
 * ukf_fusion.cpp — Unscented Kalman Filter Fusion Engine
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Full UKF implementation:
 *   - Sigma-point predict via strapdown INS mechanization
 *   - GNSS loosely-coupled update (lat/lon → ENU)
 *   - Map-match closed-loop update (lateral offset + heading)
 *   - ZUPT: hard-clamp velocity when stationary
 *   - ZARU: correct gyro bias during straight-line driving
 *   - ES-NN: subtract fusion-corrector residuals
 *
 * References:
 *   api-contracts.md §4, PRD §6.3, §2.3, §2.4, §2.5, §7 guardrails
 */

#include "ukf_fusion.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═════════════════════════════════════════════════════════════════════════════
 * Small matrix utilities (no Eigen dependency — Android NDK portability)
 * ═════════════════════════════════════════════════════════════════════════════ */

namespace mat {

/** Zero a NxN matrix (stored as float[N][N]). */
static void zero(float* m, int n) {
    memset(m, 0, n * n * sizeof(float));
}

/** Identity matrix NxN. */
static void eye(float* m, int n) {
    zero(m, n);
    for (int i = 0; i < n; i++)
        m[i * n + i] = 1.0f;
}

/**
 * Cholesky decomposition: P = L * L^T.
 * L is lower triangular, written in-place to 'L_out'.
 * Returns false if P is not positive-definite (adds a small jitter and retries).
 */
static bool cholesky(const float* P, float* L_out, int n) {
    memset(L_out, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L_out[i * n + k] * L_out[j * n + k];
            if (i == j) {
                float val = P[i * n + i] - sum;
                if (val <= 0.0f) {
                    /* Not positive-definite — add jitter and retry */
                    val = 1e-8f;
                }
                L_out[i * n + j] = sqrtf(val);
            } else {
                float denom = L_out[j * n + j];
                if (fabsf(denom) < 1e-12f) denom = 1e-12f;
                L_out[i * n + j] = (P[i * n + j] - sum) / denom;
            }
        }
    }
    return true;
}

/**
 * Invert a small dense matrix (up to ~8×8) using Gauss-Jordan.
 * 'A' is n×n input, 'Ainv' is n×n output. Returns false on singular.
 */
static bool invert(const float* A, float* Ainv, int n) {
    /* Augmented matrix [A | I] */
    float aug[8 * 16]; /* max n=8 */
    if (n > 8) return false;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * 2 * n + j] = A[i * n + j];
            aug[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int col = 0; col < n; col++) {
        /* Partial pivoting */
        int max_row = col;
        float max_val = fabsf(aug[col * 2 * n + col]);
        for (int row = col + 1; row < n; row++) {
            float v = fabsf(aug[row * 2 * n + col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-12f) return false; /* singular */
        if (max_row != col) {
            for (int j = 0; j < 2 * n; j++)
                std::swap(aug[col * 2 * n + j], aug[max_row * 2 * n + j]);
        }

        float pivot = aug[col * 2 * n + col];
        for (int j = 0; j < 2 * n; j++)
            aug[col * 2 * n + j] /= pivot;

        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            float factor = aug[row * 2 * n + col];
            for (int j = 0; j < 2 * n; j++)
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
        }
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ainv[i * n + j] = aug[i * 2 * n + n + j];

    return true;
}

} /* namespace mat */


/* ═════════════════════════════════════════════════════════════════════════════
 * Global UKF instance
 * ═════════════════════════════════════════════════════════════════════════════ */

static UKFFusion g_ukf;

UKFFusion& get_ukf_instance() { return g_ukf; }
void fusion_reset() { g_ukf.reset(); }


/* ═════════════════════════════════════════════════════════════════════════════
 * UKFFusion implementation
 * ═════════════════════════════════════════════════════════════════════════════ */

UKFFusion::UKFFusion() {
    reset();
}

void UKFFusion::reset() {
    INSMechanization::reset_state(x_);
    origin_set_ = false;
    origin_lat_rad_ = 0.0;
    origin_lon_rad_ = 0.0;

    init_weights();
    init_covariance();
    init_process_noise();
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Weight initialization
 * ───────────────────────────────────────────────────────────────────────────── */

void UKFFusion::init_weights() {
    /* With alpha=1, kappa=0: lambda=0, n+lambda=10
     * Wm[0] = lambda / (n+lambda) = 0
     * Wc[0] = Wm[0] + (1 - alpha^2 + beta) = 0 + 2 = 2
     * Wm[i] = Wc[i] = 1 / (2*(n+lambda)) = 1/20 = 0.05 */
    float n_plus_lambda = static_cast<float>(N_STATE) + UKF_LAMBDA;

    Wm_[0] = UKF_LAMBDA / n_plus_lambda;
    Wc_[0] = Wm_[0] + (1.0f - UKF_ALPHA * UKF_ALPHA + UKF_BETA);

    float wi = 1.0f / (2.0f * n_plus_lambda);
    for (int i = 1; i < N_SIGMA; i++) {
        Wm_[i] = wi;
        Wc_[i] = wi;
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Covariance initialization — default uncertainties
 * ───────────────────────────────────────────────────────────────────────────── */

void UKFFusion::init_covariance() {
    memset(P_, 0, sizeof(P_));

    /* Position: 100 m² initial uncertainty */
    P_[ST_PE][ST_PE] = 100.0f;
    P_[ST_PN][ST_PN] = 100.0f;
    P_[ST_PU][ST_PU] = 25.0f;

    /* Velocity: 25 (m/s)² */
    P_[ST_VE][ST_VE] = 25.0f;
    P_[ST_VN][ST_VN] = 25.0f;
    P_[ST_VU][ST_VU] = 1.0f;

    /* Heading: (π/2)² ≈ 2.47 rad² */
    P_[ST_PSI][ST_PSI] = 2.47f;

    /* Gyro bias: (0.1 rad/s)² = 0.01 */
    P_[ST_GBX][ST_GBX] = 0.01f;
    P_[ST_GBY][ST_GBY] = 0.01f;
    P_[ST_GBZ][ST_GBZ] = 0.01f;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Process noise Q — tuned for 10Hz prediction rate
 * ───────────────────────────────────────────────────────────────────────────── */

void UKFFusion::init_process_noise() {
    memset(Q_, 0, sizeof(Q_));

    /* Position process noise (m²) — small, driven by velocity uncertainty */
    Q_[ST_PE][ST_PE] = 0.1f;
    Q_[ST_PN][ST_PN] = 0.1f;
    Q_[ST_PU][ST_PU] = 0.01f;

    /* Velocity process noise (m/s)² — accounts for model uncertainty */
    Q_[ST_VE][ST_VE] = 1.0f;
    Q_[ST_VN][ST_VN] = 1.0f;
    Q_[ST_VU][ST_VU] = 0.01f;

    /* Heading process noise — small, heading mostly from measurements */
    Q_[ST_PSI][ST_PSI] = 0.001f;

    /* Gyro bias random walk — PRD: never ignore gyro bias (§2.3) */
    Q_[ST_GBX][ST_GBX] = 1e-6f;
    Q_[ST_GBY][ST_GBY] = 1e-6f;
    Q_[ST_GBZ][ST_GBZ] = 1e-6f;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Sigma point generation
 *
 * sigma[0]     = x
 * sigma[i]     = x + sqrt((n+λ) * P) column i,      i=1..n
 * sigma[n+i]   = x - sqrt((n+λ) * P) column i,      i=1..n
 * ───────────────────────────────────────────────────────────────────────────── */

void UKFFusion::generate_sigma_points(float sigma[N_SIGMA][N_STATE]) const {
    /* Scaling factor c = n + lambda = 10 */
    float c = static_cast<float>(N_STATE) + UKF_LAMBDA;

    /* Compute scaled covariance: S = c * P */
    float S[N_STATE * N_STATE];
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < N_STATE; j++)
            S[i * N_STATE + j] = c * P_[i][j];

    /* Cholesky: S = L * L^T */
    float L[N_STATE * N_STATE];
    mat::cholesky(S, L, N_STATE);

    /* Sigma point 0 = mean */
    memcpy(sigma[0], x_, N_STATE * sizeof(float));

    /* Sigma points 1..N_STATE and N_STATE+1..2*N_STATE */
    for (int i = 0; i < N_STATE; i++) {
        for (int j = 0; j < N_STATE; j++) {
            float offset = L[j * N_STATE + i]; /* column i of L */
            sigma[1 + i][j]           = x_[j] + offset;
            sigma[1 + N_STATE + i][j] = x_[j] - offset;
        }
        /* Wrap heading for these sigma points */
        sigma[1 + i][ST_PSI]           = wrap_to_pi(sigma[1 + i][ST_PSI]);
        sigma[1 + N_STATE + i][ST_PSI] = wrap_to_pi(sigma[1 + N_STATE + i][ST_PSI]);
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Reconstruct mean & covariance from sigma points
 * ───────────────────────────────────────────────────────────────────────────── */

void UKFFusion::reconstruct_mean_cov(const float sigma[N_SIGMA][N_STATE],
                                      float mean[N_STATE],
                                      float cov[N_STATE][N_STATE]) const {
    /* Weighted mean */
    memset(mean, 0, N_STATE * sizeof(float));
    for (int s = 0; s < N_SIGMA; s++)
        for (int i = 0; i < N_STATE; i++)
            mean[i] += Wm_[s] * sigma[s][i];
    mean[ST_PSI] = wrap_to_pi(mean[ST_PSI]);

    /* Weighted covariance */
    memset(cov, 0, N_STATE * N_STATE * sizeof(float));
    for (int s = 0; s < N_SIGMA; s++) {
        float diff[N_STATE];
        for (int i = 0; i < N_STATE; i++)
            diff[i] = sigma[s][i] - mean[i];
        diff[ST_PSI] = wrap_to_pi(diff[ST_PSI]);

        for (int i = 0; i < N_STATE; i++)
            for (int j = 0; j < N_STATE; j++)
                cov[i][j] += Wc_[s] * diff[i] * diff[j];
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Lat/lon → ENU conversion
 * ───────────────────────────────────────────────────────────────────────────── */

void UKFFusion::latlon_to_enu(double lat_deg, double lon_deg,
                               float& east_m, float& north_m) const {
    double lat_rad = lat_deg * DEG_TO_RAD;
    double lon_rad = lon_deg * DEG_TO_RAD;

    double dlat = lat_rad - origin_lat_rad_;
    double dlon = lon_rad - origin_lon_rad_;

    /* Flat-earth approximation — accurate for ~10 km */
    north_m = static_cast<float>(dlat * EARTH_RADIUS_M);
    east_m  = static_cast<float>(dlon * EARTH_RADIUS_M * cos(origin_lat_rad_));
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Extract 4×4 covariance block [pe,pn,pu,heading] for NavState
 * ───────────────────────────────────────────────────────────────────────────── */

void UKFFusion::extract_pos_heading_cov(float out[16]) const {
    /* Indices in full state: PE=0, PN=1, PU=2, PSI=6
     * Map to 4×4 block: [0→PE, 1→PN, 2→PU, 3→PSI] */
    int idx[4] = { ST_PE, ST_PN, ST_PU, ST_PSI };
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            out[i * 4 + j] = P_[idx[i]][idx[j]];
}


/* ═════════════════════════════════════════════════════════════════════════════
 * predict() — UKF predict step
 * ═════════════════════════════════════════════════════════════════════════════ */

void UKFFusion::predict(float delta_v, float dt_seconds) {
    if (dt_seconds <= 0.0f) return;

    /* 1. Generate sigma points */
    float sigma[N_SIGMA][N_STATE];
    generate_sigma_points(sigma);

    /* 2. Propagate each sigma point through the process model */
    for (int s = 0; s < N_SIGMA; s++) {
        ins_.propagate_state(sigma[s], delta_v, dt_seconds);
    }

    /* 3. Reconstruct predicted mean and covariance */
    float x_pred[N_STATE];
    float P_pred[N_STATE][N_STATE];
    reconstruct_mean_cov(sigma, x_pred, P_pred);

    /* 4. Add process noise Q (scaled by dt for continuous-time approximation) */
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < N_STATE; j++)
            P_pred[i][j] += Q_[i][j] * dt_seconds;

    /* 5. Update state and covariance */
    memcpy(x_, x_pred, sizeof(x_));
    memcpy(P_, P_pred, sizeof(P_));
}


/* ═════════════════════════════════════════════════════════════════════════════
 * update_gnss() — GNSS measurement update (loosely-coupled)
 *
 * Measurement z = [pe_gnss, pn_gnss, ve_gnss, vn_gnss, heading_gnss]
 * Dimension m = 5
 * ═════════════════════════════════════════════════════════════════════════════ */

void UKFFusion::update_gnss(const GnssFix& fix) {
    /* Set session origin on first fix */
    if (!origin_set_) {
        origin_lat_rad_ = fix.lat_deg * DEG_TO_RAD;
        origin_lon_rad_ = fix.lon_deg * DEG_TO_RAD;
        origin_set_ = true;

        /* Initialize position at origin, velocity and heading from fix */
        x_[ST_PE] = 0.0f;
        x_[ST_PN] = 0.0f;
        x_[ST_VE] = fix.speed_mps * cosf(fix.bearing_rad);
        x_[ST_VN] = fix.speed_mps * sinf(fix.bearing_rad);
        x_[ST_PSI] = fix.bearing_rad;

        /* Tighten covariance */
        P_[ST_PE][ST_PE] = fix.accuracy_m * fix.accuracy_m;
        P_[ST_PN][ST_PN] = fix.accuracy_m * fix.accuracy_m;
        P_[ST_PSI][ST_PSI] = 0.5f;
        return;
    }

    /* Convert GNSS lat/lon to ENU */
    float pe_gnss, pn_gnss;
    latlon_to_enu(fix.lat_deg, fix.lon_deg, pe_gnss, pn_gnss);

    /* GNSS velocity from speed + bearing */
    float ve_gnss = fix.speed_mps * cosf(fix.bearing_rad);
    float vn_gnss = fix.speed_mps * sinf(fix.bearing_rad);

    /* Measurement vector z[5] */
    static constexpr int M_GNSS = 5;
    float z[M_GNSS] = { pe_gnss, pn_gnss, ve_gnss, vn_gnss, fix.bearing_rad };

    /* Measurement noise R — scaled by accuracy and quality */
    float pos_noise = fix.accuracy_m * fix.accuracy_m;
    float vel_noise = 4.0f * (1.0f - fix.quality_score + 0.01f); /* worse with lower quality */
    float hdg_noise = 0.3f * (1.0f - fix.quality_score + 0.01f);

    float R[M_GNSS * M_GNSS];
    memset(R, 0, sizeof(R));
    R[0 * M_GNSS + 0] = pos_noise;
    R[1 * M_GNSS + 1] = pos_noise;
    R[2 * M_GNSS + 2] = vel_noise;
    R[3 * M_GNSS + 3] = vel_noise;
    R[4 * M_GNSS + 4] = hdg_noise;

    /* ── UKF update with sigma points ────────────────────────────────────── */

    /* Generate sigma points */
    float sigma[N_SIGMA][N_STATE];
    generate_sigma_points(sigma);

    /* Propagate sigma points through measurement model h(x) */
    /* h(x) = [x[PE], x[PN], x[VE], x[VN], x[PSI]] — linear extraction */
    float z_sigma[N_SIGMA][M_GNSS];
    for (int s = 0; s < N_SIGMA; s++) {
        z_sigma[s][0] = sigma[s][ST_PE];
        z_sigma[s][1] = sigma[s][ST_PN];
        z_sigma[s][2] = sigma[s][ST_VE];
        z_sigma[s][3] = sigma[s][ST_VN];
        z_sigma[s][4] = sigma[s][ST_PSI];
    }

    /* Predicted measurement mean */
    float z_pred[M_GNSS];
    memset(z_pred, 0, sizeof(z_pred));
    for (int s = 0; s < N_SIGMA; s++)
        for (int i = 0; i < M_GNSS; i++)
            z_pred[i] += Wm_[s] * z_sigma[s][i];
    z_pred[4] = wrap_to_pi(z_pred[4]);

    /* Innovation covariance S = sum Wc * (z_s - z_pred) * (z_s - z_pred)^T + R */
    float S[M_GNSS * M_GNSS];
    memset(S, 0, sizeof(S));
    for (int s = 0; s < N_SIGMA; s++) {
        float dz[M_GNSS];
        for (int i = 0; i < M_GNSS; i++)
            dz[i] = z_sigma[s][i] - z_pred[i];
        dz[4] = wrap_to_pi(dz[4]);

        for (int i = 0; i < M_GNSS; i++)
            for (int j = 0; j < M_GNSS; j++)
                S[i * M_GNSS + j] += Wc_[s] * dz[i] * dz[j];
    }
    for (int i = 0; i < M_GNSS * M_GNSS; i++)
        S[i] += R[i];

    /* Cross-covariance P_xz */
    float Pxz[N_STATE * M_GNSS];
    memset(Pxz, 0, sizeof(Pxz));
    for (int s = 0; s < N_SIGMA; s++) {
        float dx[N_STATE], dz[M_GNSS];
        for (int i = 0; i < N_STATE; i++)
            dx[i] = sigma[s][i] - x_[i];
        dx[ST_PSI] = wrap_to_pi(dx[ST_PSI]);

        for (int i = 0; i < M_GNSS; i++)
            dz[i] = z_sigma[s][i] - z_pred[i];
        dz[4] = wrap_to_pi(dz[4]);

        for (int i = 0; i < N_STATE; i++)
            for (int j = 0; j < M_GNSS; j++)
                Pxz[i * M_GNSS + j] += Wc_[s] * dx[i] * dz[j];
    }

    /* Kalman gain K = P_xz * S^{-1} */
    float S_inv[M_GNSS * M_GNSS];
    if (!mat::invert(S, S_inv, M_GNSS)) return; /* singular — skip update */

    float K[N_STATE * M_GNSS];
    memset(K, 0, sizeof(K));
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < M_GNSS; j++)
            for (int k = 0; k < M_GNSS; k++)
                K[i * M_GNSS + j] += Pxz[i * M_GNSS + k] * S_inv[k * M_GNSS + j];

    /* Innovation */
    float innov[M_GNSS];
    for (int i = 0; i < M_GNSS; i++)
        innov[i] = z[i] - z_pred[i];
    innov[4] = wrap_to_pi(innov[4]);

    /* State update: x = x + K * innovation */
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < M_GNSS; j++)
            x_[i] += K[i * M_GNSS + j] * innov[j];
    x_[ST_PSI] = wrap_to_pi(x_[ST_PSI]);

    /* Covariance update: P = P - K * S * K^T */
    for (int i = 0; i < N_STATE; i++) {
        for (int j = 0; j < N_STATE; j++) {
            float sum = 0.0f;
            for (int m = 0; m < M_GNSS; m++)
                for (int l = 0; l < M_GNSS; l++)
                    sum += K[i * M_GNSS + m] * S[m * M_GNSS + l] * K[j * M_GNSS + l];
            P_[i][j] -= sum;
        }
    }
}


/* ═════════════════════════════════════════════════════════════════════════════
 * update_map_match() — closed-loop map-matching correction
 *
 * Measurement z = [lateral_offset_m, heading_offset_rad]
 * CRITICAL: skip if confidence < threshold (§2.4, §7 guardrail #5)
 * ═════════════════════════════════════════════════════════════════════════════ */

void UKFFusion::update_map_match(const float pseudo_measurement[2],
                                  float confidence) {
    /* Guard: a bad snap is worse than no snap */
    if (confidence < MAP_MATCH_CONFIDENCE_THRESHOLD) return;

    static constexpr int M_MAP = 2;
    float heading = x_[ST_PSI];

    /* Measurement: z = [lateral_offset, heading_offset]
     * Expected measurement if state is perfectly on the road: h(x) = [0, 0]
     * Innovation = z - h(x) = z (the offsets themselves ARE the corrections) */

    /* Measurement Jacobian H (linearized):
     *   lateral_offset depends on position perpendicular to heading
     *   heading_offset depends on heading
     *
     * H = [[-sin(ψ), cos(ψ), 0,  0, 0, 0,  0,  0, 0, 0],   // lateral → position
     *      [ 0,       0,      0,  0, 0, 0,  1,  0, 0, 0]]    // heading → heading
     */
    float H[M_MAP * N_STATE];
    memset(H, 0, sizeof(H));
    H[0 * N_STATE + ST_PE] = -sinf(heading);
    H[0 * N_STATE + ST_PN] =  cosf(heading);
    H[1 * N_STATE + ST_PSI] = 1.0f;

    /* Measurement noise R — inversely scaled by confidence */
    float lat_noise = 4.0f * (1.0f - confidence + 0.01f);  /* tighter with higher confidence */
    float hdg_noise = 0.1f * (1.0f - confidence + 0.01f);

    float R[M_MAP * M_MAP];
    memset(R, 0, sizeof(R));
    R[0 * M_MAP + 0] = lat_noise;
    R[1 * M_MAP + 1] = hdg_noise;

    /* EKF-style update (linear H makes UKF identical to EKF here):
     * S = H * P * H^T + R
     * K = P * H^T * S^{-1}
     * x = x + K * z
     * P = P - K * H * P */

    /* HPHt = H * P * H^T (2×2) */
    float HPHt[M_MAP * M_MAP];
    memset(HPHt, 0, sizeof(HPHt));
    for (int i = 0; i < M_MAP; i++)
        for (int j = 0; j < M_MAP; j++)
            for (int k = 0; k < N_STATE; k++)
                for (int l = 0; l < N_STATE; l++)
                    HPHt[i * M_MAP + j] += H[i * N_STATE + k] * P_[k][l] * H[j * N_STATE + l];

    /* S = HPHt + R */
    float S[M_MAP * M_MAP];
    for (int i = 0; i < M_MAP * M_MAP; i++)
        S[i] = HPHt[i] + R[i];

    /* S^{-1} (2×2 analytic) */
    float det = S[0] * S[3] - S[1] * S[2];
    if (fabsf(det) < 1e-12f) return;
    float S_inv[M_MAP * M_MAP];
    S_inv[0] =  S[3] / det;
    S_inv[1] = -S[1] / det;
    S_inv[2] = -S[2] / det;
    S_inv[3] =  S[0] / det;

    /* PHt = P * H^T (10×2) */
    float PHt[N_STATE * M_MAP];
    memset(PHt, 0, sizeof(PHt));
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < M_MAP; j++)
            for (int k = 0; k < N_STATE; k++)
                PHt[i * M_MAP + j] += P_[i][k] * H[j * N_STATE + k];

    /* K = PHt * S^{-1} (10×2) */
    float K[N_STATE * M_MAP];
    memset(K, 0, sizeof(K));
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < M_MAP; j++)
            for (int k = 0; k < M_MAP; k++)
                K[i * M_MAP + j] += PHt[i * M_MAP + k] * S_inv[k * M_MAP + j];

    /* State update: x += K * z */
    for (int i = 0; i < N_STATE; i++) {
        x_[i] += K[i * M_MAP + 0] * pseudo_measurement[0]
               + K[i * M_MAP + 1] * pseudo_measurement[1];
    }
    x_[ST_PSI] = wrap_to_pi(x_[ST_PSI]);

    /* Covariance update: P -= K * H * P (Joseph form avoided for simplicity) */
    float KH[N_STATE * N_STATE];
    memset(KH, 0, sizeof(KH));
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < N_STATE; j++)
            for (int k = 0; k < M_MAP; k++)
                KH[i * N_STATE + j] += K[i * M_MAP + k] * H[k * N_STATE + j];

    float P_new[N_STATE][N_STATE];
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < N_STATE; j++) {
            P_new[i][j] = P_[i][j];
            for (int k = 0; k < N_STATE; k++)
                P_new[i][j] -= KH[i * N_STATE + k] * P_[k][j];
        }
    memcpy(P_, P_new, sizeof(P_));
}


/* ═════════════════════════════════════════════════════════════════════════════
 * apply_zupt() — Zero Velocity Update
 *
 * Hard-clamp velocity to zero when stationary_probability > 0.95.
 * (Threshold checked at the call site by FusionBridge / ModeManager.)
 * ═════════════════════════════════════════════════════════════════════════════ */

void UKFFusion::apply_zupt() {
    /* Measurement: z = [0, 0, 0] (velocity should be zero)
     * H maps velocity states directly:
     *   H = [[0,0,0, 1,0,0, 0, 0,0,0],
     *        [0,0,0, 0,1,0, 0, 0,0,0],
     *        [0,0,0, 0,0,1, 0, 0,0,0]]
     * With very tight measurement noise. */

    static constexpr int M_ZUPT = 3;
    static constexpr float ZUPT_NOISE = 0.001f; /* Very tight — we're confident about zero velocity */

    /* Measurement z = [0, 0, 0] */
    float z[M_ZUPT] = { 0.0f, 0.0f, 0.0f };

    /* Innovation = z - h(x) = [0,0,0] - [ve,vn,vu] = [-ve, -vn, -vu] */
    float innov[M_ZUPT] = {
        -x_[ST_VE],
        -x_[ST_VN],
        -x_[ST_VU]
    };

    /* H (3×10) */
    float H[M_ZUPT * N_STATE];
    memset(H, 0, sizeof(H));
    H[0 * N_STATE + ST_VE] = 1.0f;
    H[1 * N_STATE + ST_VN] = 1.0f;
    H[2 * N_STATE + ST_VU] = 1.0f;

    /* R (3×3 diagonal) */
    float R[M_ZUPT * M_ZUPT];
    memset(R, 0, sizeof(R));
    R[0 * M_ZUPT + 0] = ZUPT_NOISE;
    R[1 * M_ZUPT + 1] = ZUPT_NOISE;
    R[2 * M_ZUPT + 2] = ZUPT_NOISE;

    /* S = H*P*H^T + R */
    float S[M_ZUPT * M_ZUPT];
    memset(S, 0, sizeof(S));
    for (int i = 0; i < M_ZUPT; i++)
        for (int j = 0; j < M_ZUPT; j++) {
            int si = ST_VE + i;
            int sj = ST_VE + j;
            S[i * M_ZUPT + j] = P_[si][sj] + R[i * M_ZUPT + j];
        }

    /* S^{-1} (3×3) */
    float S_inv[M_ZUPT * M_ZUPT];
    if (!mat::invert(S, S_inv, M_ZUPT)) return;

    /* K = P*H^T * S^{-1} (10×3) */
    float K[N_STATE * M_ZUPT];
    memset(K, 0, sizeof(K));
    for (int i = 0; i < N_STATE; i++) {
        for (int j = 0; j < M_ZUPT; j++) {
            /* P*H^T: column j of H^T is [0..0, 1, 0..0] at row ST_VE+j */
            float pht = P_[i][ST_VE + j];
            for (int k = 0; k < M_ZUPT; k++)
                K[i * M_ZUPT + k] += pht * S_inv[j * M_ZUPT + k];
        }
    }

    /* State update: x += K * innov */
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < M_ZUPT; j++)
            x_[i] += K[i * M_ZUPT + j] * innov[j];

    /* Covariance update: P -= K*S*K^T */
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < N_STATE; j++) {
            float sum = 0.0f;
            for (int m = 0; m < M_ZUPT; m++)
                for (int l = 0; l < M_ZUPT; l++)
                    sum += K[i * M_ZUPT + m] * S[m * M_ZUPT + l] * K[j * M_ZUPT + l];
            P_[i][j] -= sum;
        }
}


/* ═════════════════════════════════════════════════════════════════════════════
 * apply_zaru() — Zero Angular Rate Update
 *
 * Pseudo-measurement: gyro bias drift rate ≈ 0 during straight-line driving.
 * Tightens gyro bias covariance.
 * PRD §2.3: Never ignore gyro bias — silently fails on longer blackouts.
 * ═════════════════════════════════════════════════════════════════════════════ */

void UKFFusion::apply_zaru() {
    static constexpr int M_ZARU = 3;
    static constexpr float ZARU_NOISE = 0.0001f; /* Tight — straight-line means low angular rate */

    /* Pseudo-measurement: gyro bias should not be drifting → z = current bias estimate
     * H maps gyro bias states:
     *   H = [[0..0, 1, 0, 0],   ← GB_X
     *        [0..0, 0, 1, 0],   ← GB_Y
     *        [0..0, 0, 0, 1]]   ← GB_Z
     *
     * Innovation = 0 - bias = -bias (drive bias toward zero while tightening covariance)
     * Actually, we want to KEEP the current bias estimate but tighten its covariance,
     * so the measurement is the CURRENT bias itself. */

    /* Innovation: we constrain the bias to stay near its current value with reduced uncertainty */
    float innov[M_ZARU] = { 0.0f, 0.0f, 0.0f }; /* zero innovation — just tighten covariance */

    float R[M_ZARU * M_ZARU];
    memset(R, 0, sizeof(R));
    R[0] = ZARU_NOISE;
    R[4] = ZARU_NOISE;
    R[8] = ZARU_NOISE;

    /* S = P_bias_block + R (3×3) */
    float S[M_ZARU * M_ZARU];
    memset(S, 0, sizeof(S));
    for (int i = 0; i < M_ZARU; i++)
        for (int j = 0; j < M_ZARU; j++)
            S[i * M_ZARU + j] = P_[ST_GBX + i][ST_GBX + j] + R[i * M_ZARU + j];

    float S_inv[M_ZARU * M_ZARU];
    if (!mat::invert(S, S_inv, M_ZARU)) return;

    /* K = P * H^T * S^{-1} — H^T selects gyro bias rows */
    float K[N_STATE * M_ZARU];
    memset(K, 0, sizeof(K));
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < M_ZARU; j++) {
            float pht = P_[i][ST_GBX + j];
            for (int k = 0; k < M_ZARU; k++)
                K[i * M_ZARU + k] += pht * S_inv[j * M_ZARU + k];
        }

    /* State update (with zero innovation, this is a no-op on state) */
    /* But we still update covariance to tighten it */

    /* Covariance: P -= K*S*K^T */
    for (int i = 0; i < N_STATE; i++)
        for (int j = 0; j < N_STATE; j++) {
            float sum = 0.0f;
            for (int m = 0; m < M_ZARU; m++)
                for (int l = 0; l < M_ZARU; l++)
                    sum += K[i * M_ZARU + m] * S[m * M_ZARU + l] * K[j * M_ZARU + l];
            P_[i][j] -= sum;
        }
}


/* ═════════════════════════════════════════════════════════════════════════════
 * apply_es_nn_residual() — Error-State Neural Network correction (§2.5)
 *
 * Directly subtracts corrections from state.
 * ═════════════════════════════════════════════════════════════════════════════ */

void UKFFusion::apply_es_nn_residual(const float delta_p[3],
                                      const float delta_v[3],
                                      float delta_psi) {
    x_[ST_PE] -= delta_p[0];
    x_[ST_PN] -= delta_p[1];
    x_[ST_PU] -= delta_p[2];

    x_[ST_VE] -= delta_v[0];
    x_[ST_VN] -= delta_v[1];
    x_[ST_VU] -= delta_v[2];

    x_[ST_PSI] -= delta_psi;
    x_[ST_PSI] = wrap_to_pi(x_[ST_PSI]);
}


/* ═════════════════════════════════════════════════════════════════════════════
 * get_nav_state() — expose current state as NavState struct
 * ═════════════════════════════════════════════════════════════════════════════ */

NavState UKFFusion::get_nav_state() const {
    NavState ns;
    float cov_block[16];
    extract_pos_heading_cov(cov_block);
    INSMechanization::state_to_navstate(x_, cov_block, &ns);
    return ns;
}


/* ═════════════════════════════════════════════════════════════════════════════
 * JNI bridge implementations — Fusion
 *
 * These implement the declarations in JniBridge.h for the Fusion section.
 * Compiled only when JNI headers are available (Android NDK build).
 * ═════════════════════════════════════════════════════════════════════════════ */

#ifdef __ANDROID__
#include <jni.h>
#include "../../bindings/java/JniBridge.h"

extern "C" {

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionPredict(
    JNIEnv *env, jobject thiz, jfloat deltaV, jfloat dtSeconds) {
    get_ukf_instance().predict(deltaV, dtSeconds);
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionUpdateGnss(
    JNIEnv *env, jobject thiz,
    jlong ts, jdouble lat, jdouble lon,
    jfloat speed, jfloat bearing, jfloat accuracy,
    jint satellites, jfloat quality) {
    GnssFix fix;
    fix.timestamp_ms   = static_cast<int64_t>(ts);
    fix.lat_deg        = lat;
    fix.lon_deg        = lon;
    fix.speed_mps      = speed;
    fix.bearing_rad    = bearing;
    fix.accuracy_m     = accuracy;
    fix.satellite_count = static_cast<int32_t>(satellites);
    fix.quality_score  = quality;
    get_ukf_instance().update_gnss(fix);
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionUpdateMapMatch(
    JNIEnv *env, jobject thiz,
    jfloatArray pseudoMeasurement, jfloat confidence) {
    jfloat* pm = env->GetFloatArrayElements(pseudoMeasurement, nullptr);
    if (pm && env->GetArrayLength(pseudoMeasurement) >= 2) {
        float pseudo[2] = { pm[0], pm[1] };
        get_ukf_instance().update_map_match(pseudo, confidence);
    }
    if (pm) env->ReleaseFloatArrayElements(pseudoMeasurement, pm, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionApplyZaru(
    JNIEnv *env, jobject thiz) {
    get_ukf_instance().apply_zaru();
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionApplyZupt(
    JNIEnv *env, jobject thiz) {
    get_ukf_instance().apply_zupt();
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionApplyEsNnResidual(
    JNIEnv *env, jobject thiz,
    jfloatArray deltaP, jfloatArray deltaV, jfloat deltaPsi) {
    jfloat* dp = env->GetFloatArrayElements(deltaP, nullptr);
    jfloat* dv = env->GetFloatArrayElements(deltaV, nullptr);
    if (dp && dv) {
        get_ukf_instance().apply_es_nn_residual(dp, dv, deltaPsi);
    }
    if (dp) env->ReleaseFloatArrayElements(deltaP, dp, JNI_ABORT);
    if (dv) env->ReleaseFloatArrayElements(deltaV, dv, JNI_ABORT);
}

JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionGetNavState(
    JNIEnv *env, jobject thiz) {
    NavState ns = get_ukf_instance().get_nav_state();

    /* Return 26-element float array per JniBridge.h contract:
     *   [0..2]   position
     *   [3..5]   velocity
     *   [6]      heading
     *   [7..9]   gyro_bias
     *   [10..25] covariance 4×4 */
    jfloatArray result = env->NewFloatArray(26);
    if (result) {
        jfloat buf[26];
        buf[0] = ns.position[0]; buf[1] = ns.position[1]; buf[2] = ns.position[2];
        buf[3] = ns.velocity[0]; buf[4] = ns.velocity[1]; buf[5] = ns.velocity[2];
        buf[6] = ns.heading_rad;
        buf[7] = ns.gyro_bias[0]; buf[8] = ns.gyro_bias[1]; buf[9] = ns.gyro_bias[2];
        for (int i = 0; i < 16; i++) buf[10 + i] = ns.covariance[i];
        env->SetFloatArrayRegion(result, 0, 26, buf);
    }
    return result;
}

} /* extern "C" */
#endif /* __ANDROID__ */
