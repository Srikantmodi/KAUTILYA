/**
 * ins_mechanization.cpp — Strapdown INS Mechanization
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Implementation of the strapdown process model.
 * See ins_mechanization.h for detailed documentation.
 */

#include "ins_mechanization.h"
#include <cstring>
#include <cmath>

/* ─────────────────────────────────────────────────────────────────────────── */

INSMechanization::INSMechanization() {
    /* No state held in the mechanization object itself — it operates
     * purely on the state array passed to propagate_state(). */
}

void INSMechanization::propagate_state(float state[N_STATE],
                                        float delta_v,
                                        float dt_s) const {
    if (dt_s <= 0.0f) return;  /* Guard: zero/negative dt → no-op */

    /* 1. Propagate position using CURRENT velocity (before velocity update).
     *    This is the standard discrete-time approach: x_{k+1} = x_k + v_k * dt */
    state[ST_PE] += state[ST_VE] * dt_s;
    state[ST_PN] += state[ST_VN] * dt_s;
    state[ST_PU] += state[ST_VU] * dt_s;

    /* 2. Update velocity from ML model's delta_v.
     *    delta_v is treated as the current forward speed estimate (m/s).
     *    Decompose along heading into ENU velocity components.
     *
     *    ENU convention: heading 0 = East, CCW positive.
     *      ve = speed × cos(heading)
     *      vn = speed × sin(heading) */
    float heading = state[ST_PSI];
    state[ST_VE] = delta_v * cosf(heading);
    state[ST_VN] = delta_v * sinf(heading);
    state[ST_VU] = 0.0f;   /* Ground vehicle — negligible vertical velocity */

    /* 3. Heading: constant in the process model.
     *    Corrections come from GNSS bearing updates (update_gnss) and
     *    map-matching heading offsets (update_map_match). */

    /* 4. Gyro bias: constant in the process model.
     *    Modeled as a random walk — the UKF's process noise Q handles
     *    the covariance growth. ZARU corrects it during straight-line. */

    /* 5. Wrap heading to [-π, π] to prevent accumulation. */
    state[ST_PSI] = wrap_to_pi(state[ST_PSI]);
}

/* ─────────────────────────────────────────────────────────────────────────── */

void INSMechanization::reset_state(float state[N_STATE]) {
    memset(state, 0, N_STATE * sizeof(float));
}

/* ─────────────────────────────────────────────────────────────────────────── */

void INSMechanization::state_to_navstate(const float state[N_STATE],
                                          const float cov_pos_heading[16],
                                          NavState* out) {
    out->position[0] = state[ST_PE];
    out->position[1] = state[ST_PN];
    out->position[2] = state[ST_PU];

    out->velocity[0] = state[ST_VE];
    out->velocity[1] = state[ST_VN];
    out->velocity[2] = state[ST_VU];

    out->heading_rad = state[ST_PSI];

    out->gyro_bias[0] = state[ST_GBX];
    out->gyro_bias[1] = state[ST_GBY];
    out->gyro_bias[2] = state[ST_GBZ];

    /* Copy the 4×4 position+heading covariance block */
    if (cov_pos_heading) {
        memcpy(out->covariance, cov_pos_heading, 16 * sizeof(float));
    } else {
        memset(out->covariance, 0, 16 * sizeof(float));
    }
}
