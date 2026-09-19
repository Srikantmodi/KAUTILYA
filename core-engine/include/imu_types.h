/**
 * imu_types.h — Day-0 shared data contracts
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §1, Master PRD §6.1
 *
 * Owner:  Member 4 (Fusion Core)
 * Status: FROZEN after Day-0 team sign-off — any field change is a breaking
 *         change across Kotlin, C++, and Python bindings simultaneously (§6.1).
 *
 * Rules:
 *   - Header-only, no logic.
 *   - extern "C"-compatible / trivially JNI-marshalable.
 *   - Fixed-size arrays only — no STL containers crossing the JNI boundary.
 *   - Booleans crossing JNI are int (0/1), never bool (safe fixed-layout).
 *
 * Global conventions (apply everywhere, no exceptions):
 *   Angles        — radians
 *   Timestamps    — epoch milliseconds, int64_t
 *   Distance      — meters
 *   Speed         — m/s
 *   Acceleration  — m/s²
 *   Angular rate  — rad/s
 *   Coord frame   — local ENU (East-North-Up) relative to session-start origin
 *   Heading       — radians, 0 = East, increasing counter-clockwise (ENU math)
 *                   UI layer converts to navigation convention at display time
 *   Confidence    — float [0.0, 1.0]
 */

#ifndef CORE_ENGINE_IMU_TYPES_H
#define CORE_ENGINE_IMU_TYPES_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────────────
 * ImuSample — raw inertial measurement, phone frame
 *
 * Emitted by SensorBridge.kt at sensor rate (~100–200 Hz).
 * Contains RAW values only — no leveling, filtering, or bias correction.
 * Calibration (Member 3) and inference (Member 6) apply their own transforms
 * downstream; do NOT pre-process here.
 * ───────────────────────────────────────────────────────────────────────────── */
struct ImuSample {
    int64_t timestamp_ms;       /* epoch milliseconds                          */

    float   accel[3];           /* [ax, ay, az] — raw, phone frame, m/s²       *
                                 * Includes gravity component.                 */

    float   gyro[3];            /* [gx, gy, gz] — raw, phone frame, rad/s      */
};

/* ─────────────────────────────────────────────────────────────────────────────
 * GnssFix — single GNSS position fix with quality metadata
 *
 * Produced by GnssQualityMonitor.kt at ~1 Hz (matches IO-VNBD real GPS rate).
 * quality_score is computed at ingress (satellite count + accuracy + fix-age);
 * it feeds ModeManager directly (§6.5), not just a GPS pass-through.
 *
 * NOTE: lat_deg / lon_deg are the ONLY fields in degrees — they exist at the
 * GNSS ingress/egress boundary only. Everything internal is ENU meters.
 * bearing_rad is converted to radians at ingress (Android provides degrees
 * natively; conversion happens in GnssQualityMonitor before populating this
 * struct).
 * ───────────────────────────────────────────────────────────────────────────── */
struct GnssFix {
    int64_t timestamp_ms;       /* epoch milliseconds                          */

    double  lat_deg;            /* WGS-84 latitude, degrees                    */
    double  lon_deg;            /* WGS-84 longitude, degrees                   */

    float   speed_mps;          /* ground speed, m/s                           */
    float   bearing_rad;        /* course over ground, radians, ENU convention  *
                                 * (0 = East, CCW positive).                   *
                                 * Converted from Android's CW-from-North at   *
                                 * ingress by GnssQualityMonitor.              */

    float   accuracy_m;         /* estimated horizontal accuracy, meters       */
    int32_t satellite_count;    /* visible satellites used in fix               */
    float   quality_score;      /* [0.0, 1.0] — composite quality metric       */
};

/* ─────────────────────────────────────────────────────────────────────────────
 * NavState — fused navigation solution output
 *
 * Produced by ukf_fusion.cpp (Member 4).
 * Consumed by:
 *   - Member 1 (UI) for vehicle icon position / heading display
 *   - Member 5 (Map-Matching) as input to the HMM
 *   - Member 6 (Benchmark) for drift calculation
 *
 * Coordinate frame: local ENU, meters, relative to session-start origin.
 * ───────────────────────────────────────────────────────────────────────────── */
struct NavState {
    float   position[3];        /* [east, north, up] ENU, meters               */
    float   velocity[3];        /* [ve, vn, vu] ENU, m/s                       */
    float   heading_rad;        /* radians, ENU convention (0=East, CCW+)      */
    float   gyro_bias[3];       /* [bx, by, bz] rad/s, per-axis estimated bias */

    float   covariance[16];     /* flattened 4×4, row-major                    *
                                 * Covers the position(3) + heading(1) block   *
                                 * of the full UKF covariance matrix.          *
                                 * Layout:                                     *
                                 *   [ Pee  Pen  Peu  Peh  ]                   *
                                 *   [ Pne  Pnn  Pnu  Pnh  ]                   *
                                 *   [ Pue  Pun  Puu  Puh  ]                   *
                                 *   [ Phe  Phn  Phu  Phh  ]                   */
};

/* ─────────────────────────────────────────────────────────────────────────────
 * NavigationMode — system operating mode
 *
 * Determined by mode_manager.cpp (Member 4) based on continuous GNSS quality
 * scoring from GnssQualityMonitor (Member 2). Millisecond-scale transitions,
 * no multi-second debounce (§6.5).
 *
 * Exposed to Kotlin UI via JNI as int ordinal. Member 1's ModeManager.kt is
 * a pure pass-through — no scoring logic duplicated in Kotlin (§7 #1).
 * ───────────────────────────────────────────────────────────────────────────── */
enum NavigationMode {
    NAVIGATION_MODE_GNSS_AIDED_INS = 0,   /* GNSS fix available and trusted   */
    NAVIGATION_MODE_PURE_DR        = 1    /* GNSS unavailable or untrusted —  *
                                           * running on INS + ML + map-match  */
};

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CORE_ENGINE_IMU_TYPES_H */
