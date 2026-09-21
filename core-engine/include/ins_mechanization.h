/**
 * ins_mechanization.h — Strapdown INS Mechanization
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §4, Master PRD §6.3
 *
 * Owner:  Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Purpose: Simplified 2D strapdown integration that serves as the UKF's
 * process model.  Takes the ML model's delta_v output and propagates
 * position/velocity/heading in a local ENU frame.
 *
 * The ML model (Member 6) produces a scalar delta_v each inference step.
 * This module decomposes it along the current heading to obtain velocity
 * components, then propagates position.
 *
 * Conventions (api-contracts.md §0):
 *   - ENU frame, heading 0=East CCW+ (matching imu_types.h)
 *   - All angles in radians, distances in meters, speed in m/s
 *   - Timestamps in epoch milliseconds
 */

#ifndef CORE_ENGINE_INS_MECHANIZATION_H
#define CORE_ENGINE_INS_MECHANIZATION_H

#include <cmath>
#include "imu_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─────────────────────────────────────────────────────────────────────────────
 * State index constants — shared between INS and UKF
 * ───────────────────────────────────────────────────────────────────────────── */
static constexpr int ST_PE  = 0;  /* position east  (m)   */
static constexpr int ST_PN  = 1;  /* position north (m)   */
static constexpr int ST_PU  = 2;  /* position up    (m)   */
static constexpr int ST_VE  = 3;  /* velocity east  (m/s) */
static constexpr int ST_VN  = 4;  /* velocity north (m/s) */
static constexpr int ST_VU  = 5;  /* velocity up    (m/s) */
static constexpr int ST_PSI = 6;  /* heading (rad, 0=E CCW+) */
static constexpr int ST_GBX = 7;  /* gyro bias x (rad/s)  */
static constexpr int ST_GBY = 8;  /* gyro bias y (rad/s)  */
static constexpr int ST_GBZ = 9;  /* gyro bias z (rad/s)  */
static constexpr int N_STATE = 10;

/* ─────────────────────────────────────────────────────────────────────────────
 * Angle wrapping utility (shared)
 * ───────────────────────────────────────────────────────────────────────────── */
inline float wrap_to_pi(float angle) {
    angle = fmodf(angle + static_cast<float>(M_PI), 2.0f * static_cast<float>(M_PI));
    if (angle < 0.0f) angle += 2.0f * static_cast<float>(M_PI);
    return angle - static_cast<float>(M_PI);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * INSMechanization — strapdown process model
 *
 * Operates on a raw state array float[N_STATE]. Designed to be called
 * per-sigma-point by the UKF.
 * ───────────────────────────────────────────────────────────────────────────── */
class INSMechanization {
public:
    INSMechanization();

    /**
     * Propagate a state vector forward by one timestep.
     *
     * Process model:
     *   1. Position += velocity × dt
     *   2. Forward speed = delta_v (ML model output, m/s)
     *   3. ve = forward_speed × cos(heading)
     *   4. vn = forward_speed × sin(heading)
     *   5. vu = 0 (2D ground vehicle assumption)
     *   6. Heading constant (corrected by measurements)
     *   7. Gyro bias constant (random walk via Q in UKF)
     *
     * @param state     In/out: float[N_STATE] state vector
     * @param delta_v   ML model's velocity output (m/s)
     * @param dt_s      Time step in seconds
     */
    void propagate_state(float state[N_STATE], float delta_v, float dt_s) const;

    /**
     * Initialize a state vector to all zeros.
     */
    static void reset_state(float state[N_STATE]);

    /**
     * Fill a NavState struct from a raw state array.
     */
    static void state_to_navstate(const float state[N_STATE],
                                  const float cov_pos_heading[16],
                                  NavState* out);
};

#endif /* CORE_ENGINE_INS_MECHANIZATION_H */
