/**
 * JniBridge.h — JNI method declarations (Day-0 shared contract)
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §2, Master PRD §6.1 / §6.7
 *
 * Owner:  Member 4 (Fusion Core)
 * Status: FROZEN after Day-0 team sign-off — changing a signature here is a
 *         breaking change across Kotlin, C++, and Python bindings (§6.1).
 *
 * This file declares the C++ functions exposed to Kotlin via JNI. Each
 * function name is mangled according to the actual package paths in the repo:
 *
 *   Calibration  → com.sih.deadreckoning.calibration.AlignmentManager
 *   Fusion       → com.sih.deadreckoning.fusion.FusionBridge
 *   Map matching → com.sih.deadreckoning.mapmatching.MapMatchingBridge
 *   Mode         → com.sih.deadreckoning.mode.ModeManager
 *
 * Implementation lives in the respective .cpp files owned by each member;
 * this header is declarations only.
 *
 * Conventions (from api-contracts.md §0):
 *   - Booleans crossing JNI: jint (0/1), never jboolean, for safe layout.
 *   - Angles: radians everywhere (including bearing — converted at ingress).
 *   - Timestamps: epoch milliseconds (jlong).
 *   - Confidence/probability: jfloat [0.0, 1.0].
 */

#ifndef CORE_ENGINE_JNI_BRIDGE_H
#define CORE_ENGINE_JNI_BRIDGE_H

#include <jni.h>
#include "imu_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CALIBRATION — com.sih.deadreckoning.calibration.AlignmentManager
 *
 * Owner: Member 3 (Calibration & Device Alignment)
 * Kotlin side: AlignmentManager.kt calls these via external fun declarations.
 * C++ impl:   calibration.cpp
 *
 * Data flow: raw ImuSample → update() → EKF tracks pitch/roll;
 *            levelAccelerometer() called on every sample before inference
 *            windowing (Member 6) and before UKF predict (Member 4).
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Feed a raw IMU sample into the calibration EKF.
 *
 * @param ts        timestamp_ms (epoch milliseconds)
 * @param accel     jfloatArray[3]: [ax, ay, az] raw phone-frame m/s²
 * @param gyro      jfloatArray[3]: [gx, gy, gz] raw phone-frame rad/s
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_calibrationUpdate(
    JNIEnv *env, jobject thiz,
    jlong ts,
    jfloatArray accel,
    jfloatArray gyro);

/**
 * Remove gravity and rotate a raw accelerometer reading into the vehicle frame.
 *
 * @param rawAccel  jfloatArray[3]: [ax, ay, az] raw phone-frame m/s²
 * @return          jfloatArray[3]: leveled vehicle-frame acceleration m/s²
 *
 * NOTE: This is called on EVERY raw sample before it enters the ML inference
 * window (Member 6) or the UKF predict step (Member 4). Signature is frozen —
 * do not change without coordinating with Members 4 and 6.
 */
JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_levelAccelerometer(
    JNIEnv *env, jobject thiz,
    jfloatArray rawAccel);

/**
 * Get the current calibration rotation estimate.
 *
 * @return  jfloatArray[3]: [pitch_rad, roll_rad, yaw_rad]
 *          yaw_rad is valid only after sufficient straight-line-braking windows
 *          have been observed for the pooled nonlinear optimization (§6.2).
 */
JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_calibration_AlignmentManager_getCurrentRotation(
    JNIEnv *env, jobject thiz);


/* ═══════════════════════════════════════════════════════════════════════════════
 * FUSION — com.sih.deadreckoning.fusion.FusionBridge
 *
 * Owner: Member 4 (Fusion Core, INS Mechanization & UKF)
 * Kotlin side: FusionBridge.kt
 * C++ impl:   ukf_fusion.cpp, ins_mechanization.cpp
 *
 * Data flow:
 *   predict()         ← ML model's delta_v output (Member 6)
 *   updateGnss()      ← GnssFix from GnssQualityMonitor (Member 2)
 *   updateMapMatch()  ← pseudo-measurement from map_matching.cpp (Member 5)
 *   applyZupt()       ← stationary_probability > 0.95 (Member 6 model output)
 *   applyZaru()       ← straight-line driving detected
 *   applyEsNnResidual() ← fusion-corrector network output (§2.5)
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * UKF predict step — strapdown integration using ML-derived delta-v.
 *
 * @param deltaV      velocity change (m/s) over the last inference interval
 * @param dtSeconds   time step duration (seconds) since last predict
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionPredict(
    JNIEnv *env, jobject thiz,
    jfloat deltaV,
    jfloat dtSeconds);

/**
 * UKF GNSS measurement update — loosely-coupled correction.
 *
 * @param ts        GnssFix timestamp_ms
 * @param lat       latitude, degrees (WGS-84)
 * @param lon       longitude, degrees (WGS-84)
 * @param speed     ground speed, m/s
 * @param bearing   course over ground, radians (ENU: 0=East, CCW+)
 * @param accuracy  horizontal accuracy, meters
 * @param satellites  satellite count
 * @param quality   quality score [0.0, 1.0]
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionUpdateGnss(
    JNIEnv *env, jobject thiz,
    jlong ts,
    jdouble lat, jdouble lon,
    jfloat speed,
    jfloat bearing,
    jfloat accuracy,
    jint satellites,
    jfloat quality);

/**
 * UKF map-match measurement update — closed-loop correction from HMM.
 *
 * Applied ONLY when confidence exceeds tunable threshold (§2.4, §7 #5).
 * Skipping this wiring is the documented most likely cause of benchmark
 * failure that looks fine in a demo.
 *
 * @param pseudoMeasurement  jfloatArray[2]: [lateral_offset_m, heading_offset_rad]
 * @param confidence         match confidence [0.0, 1.0]
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionUpdateMapMatch(
    JNIEnv *env, jobject thiz,
    jfloatArray pseudoMeasurement,
    jfloat confidence);

/**
 * Zero Angular Rate Update — correct gyro bias during straight-line driving.
 *
 * Called when straight-line driving is detected. Never ignore gyro bias as
 * negligible — silently fails on longer blackouts (§2.3).
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionApplyZaru(
    JNIEnv *env, jobject thiz);

/**
 * Zero velocity UPdaTe — hard-clamp velocity when stationary.
 *
 * Called when stationary_probability > 0.95 (from Member 6's model output).
 * Threshold is explicit in the PRD, not tunable.
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionApplyZupt(
    JNIEnv *env, jobject thiz);

/**
 * Apply error-state neural network residual correction (§2.5).
 *
 * Subtracts the fusion-corrector network's output each step.
 *
 * @param deltaP    jfloatArray[3]: position correction [de, dn, du] meters
 * @param deltaV    jfloatArray[3]: velocity correction [dve, dvn, dvu] m/s
 * @param deltaPsi  heading correction, radians
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionApplyEsNnResidual(
    JNIEnv *env, jobject thiz,
    jfloatArray deltaP,
    jfloatArray deltaV,
    jfloat deltaPsi);

/**
 * Retrieve the current fused navigation state.
 *
 * @return  jfloatArray[26]:
 *            [0..2]   position   — east, north, up (meters, ENU)
 *            [3..5]   velocity   — ve, vn, vu (m/s, ENU)
 *            [6]      heading    — radians (ENU: 0=East, CCW+)
 *            [7..9]   gyro_bias  — bx, by, bz (rad/s)
 *            [10..25] covariance — 4×4, row-major (position+heading block)
 */
JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_fusion_FusionBridge_fusionGetNavState(
    JNIEnv *env, jobject thiz);


/* ═══════════════════════════════════════════════════════════════════════════════
 * MAP MATCHING — com.sih.deadreckoning.mapmatching.MapMatchingBridge
 *
 * Owner: Member 5 (Map-Matching & Map Data Pipeline)
 * Kotlin side: MapMatchingBridge.kt
 * C++ impl:   map_matching.cpp
 *
 * Data flow: position estimate (from NavState) + heading → HMM Viterbi →
 *            MapMatchResult → feeds back into fusionUpdateMapMatch() above.
 *            NOT display-only — must feed the UKF closed-loop (§2.4).
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Run HMM map-matching for a trajectory point.
 *
 * @param posEast      east position, meters (ENU)
 * @param posNorth     north position, meters (ENU)
 * @param posUp        up position, meters (ENU)
 * @param headingRad   heading, radians (ENU: 0=East, CCW+)
 *
 * @return  jfloatArray[4]:
 *            [0]  matched_segment_id  (cast to float for uniform array return)
 *            [1]  lateral_offset_m    (pseudo-measurement component 1)
 *            [2]  heading_offset_rad  (pseudo-measurement component 2)
 *            [3]  confidence          [0.0, 1.0]
 *
 * NOTE: confidence below the tunable threshold means Member 4's
 * fusionUpdateMapMatch() must SKIP the update — a bad snap is worse than
 * no snap (§7 guardrail #5).
 */
JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_mapmatching_MapMatchingBridge_mapMatch(
    JNIEnv *env, jobject thiz,
    jfloat posEast, jfloat posNorth, jfloat posUp,
    jfloat headingRad);

/**
 * Load the pre-converted road graph asset into memory.
 *
 * Called once at app startup. The graph file is produced offline by
 * convert_to_graph.py (Member 5) — core-engine never parses raw OSM at
 * runtime.
 *
 * @param graphPath  absolute filesystem path to the binary graph asset
 * @return           0 on success, negative error code on failure
 */
JNIEXPORT jint JNICALL
Java_com_sih_deadreckoning_mapmatching_MapMatchingBridge_loadGraph(
    JNIEnv *env, jobject thiz,
    jstring graphPath);


/* ═══════════════════════════════════════════════════════════════════════════════
 * MODE MANAGER — com.sih.deadreckoning.mode.ModeManager
 *
 * Owner: Member 4 (C++ logic) / Member 1 (Kotlin wrapper — pure pass-through)
 * Kotlin side: ModeManager.kt — calls getCurrentMode() via JNI, exposes
 *              NavigationMode ordinal to UI. NO scoring logic in Kotlin (§7 #1).
 * C++ impl:   mode_manager.cpp
 *
 * Scoring inputs: satellite_count, accuracy_m, fix_age_ms.
 * Transitions are millisecond-scale — no multi-second debounce (§6.5).
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Get the current navigation mode based on GNSS quality scoring.
 *
 * @return  int ordinal of NavigationMode enum:
 *            0 = NAVIGATION_MODE_GNSS_AIDED_INS
 *            1 = NAVIGATION_MODE_PURE_DR
 */
JNIEXPORT jint JNICALL
Java_com_sih_deadreckoning_mode_ModeManager_getCurrentMode(
    JNIEnv *env, jobject thiz);

/**
 * Feed a GNSS fix into the mode manager's quality scorer.
 *
 * Must be called with every GnssFix from GnssQualityMonitor so the mode
 * manager has up-to-date fix-age and quality information.
 *
 * @param ts            GnssFix timestamp_ms
 * @param satelliteCount  number of satellites
 * @param accuracyM     horizontal accuracy, meters
 * @param qualityScore  composite quality [0.0, 1.0]
 */
JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_mode_ModeManager_updateGnssFix(
    JNIEnv *env, jobject thiz,
    jlong ts,
    jint satelliteCount,
    jfloat accuracyM,
    jfloat qualityScore);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CORE_ENGINE_JNI_BRIDGE_H */
