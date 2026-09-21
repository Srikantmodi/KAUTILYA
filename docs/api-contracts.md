# API Contracts — SIH PS 26168 (Intelligent Dead Reckoning)

Derived from the Master PRD, Section 6 (file-by-file implementation guide) and
Section 2.1 (canonical data flow). This document is **binding**: if your own
judgment conflicts with what's written here, this file wins — everyone else's
code is being written against these exact shapes. If you think something here
is wrong, raise it with the team and get this file updated; don't quietly
deviate in your own module.

Items marked **⚠️ ASSUMPTION** are not explicitly specified in the PRD. A
reasonable default has been chosen so work isn't blocked — confirm or
override these as a team before individual coding starts, then delete the
⚠️ marker once confirmed.

---

## 0. Global Conventions (apply everywhere, no exceptions)

| Quantity | Convention |
|---|---|
| Angles | **radians**, always — never degrees, anywhere in `core-engine` or Kotlin |
| Timestamps | **epoch milliseconds**, `int64_t` (C++) / `Long` (Kotlin) / `int` (Python, ms since epoch) |
| Distance | **meters** |
| Speed | **meters/second (m/s)** |
| Acceleration | **meters/second² (m/s²)** |
| Angular rate | **radians/second (rad/s)** |
| Coordinate frame | position/velocity in a local ENU (East-North-Up) tangent frame relative to session start; lat/lon only at the GNSS ingress/egress boundary |
| Heading | radians, 0 = North, increasing clockwise ⚠️ ASSUMPTION — confirm convention matches whatever `ins_mechanization.cpp` and the UI compass actually use |
| Booleans crossing JNI | `int` (0/1), never `bool`, for safe fixed-layout marshaling |
| Confidence / probability scores | `float`, range `[0.0, 1.0]` |

---

## 1. `core-engine/include/imu_types.h` (§6.1 — foundation, build first)

Shared plain-data structs. **Header-only, no logic.** Must be `extern "C"`-compatible
or trivially JNI-marshalable: fixed-size arrays only, no STL containers crossing
the JNI boundary. Owner: Member 4. Requires sign-off from all 6 before anyone
codes against it.

```cpp
struct ImuSample {
    int64_t timestamp_ms;
    float accel[3];   // ax, ay, az — raw, phone frame, m/s^2 (includes gravity)
    float gyro[3];    // gx, gy, gz — raw, phone frame, rad/s
    float mag[3];     // mx, my, mz — raw, phone frame, microtesla. ⚠️ ASSUMPTION:
                       // zero-filled if magnetometer unavailable/unused for a sample
};

struct GnssFix {
    int64_t timestamp_ms;
    double lat_deg;
    double lon_deg;
    float speed_mps;
    float bearing_deg;      // ⚠️ ASSUMPTION: degrees here specifically, since this
                             // mirrors Android's native GNSS bearing field — converted
                             // to radians internally by anything that consumes it
    float accuracy_m;
    int satellite_count;
    float quality_score;    // [0,1], computed by GnssQualityMonitor — see §5 below
};

struct NavState {
    float position[3];      // ENU, meters, relative to session-start origin
    float velocity[3];      // ENU, m/s
    float heading_rad;
    float gyro_bias[3];     // rad/s, per-axis
    float covariance[16];   // ⚠️ ASSUMPTION: flattened 4x4 (position/heading block)
                             // row-major — CONFIRM actual UKF state dimensionality
                             // with Member 4 before relying on this shape
};

enum class NavigationMode : int {
    GNSS_AIDED_INS = 0,
    PURE_DR = 1,
    RECOVERING = 2   // ⚠️ ASSUMPTION: transitional state while re-acquiring GNSS
                     // confidence before fully trusting fixes again — confirm this
                     // 3rd state is wanted, or simplify to just the first two
};
```

**Kotlin-side mirror** (no separate file — defined inline in the Kotlin classes
that use them, per the folder tree in §5 not listing standalone data-class files):

```kotlin
// inside SensorBridge.kt
data class ImuSample(
    val timestampMs: Long,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float
)

// inside GnssQualityMonitor.kt
data class GnssFix(
    val timestampMs: Long,
    val latDeg: Double, val lonDeg: Double,
    val speedMps: Float, val bearingDeg: Float,
    val accuracyM: Float, val satelliteCount: Int,
    val qualityScore: Float
)
```

---

## 2. `core-engine/bindings/java/JniBridge.h` (§6.1 — foundation, build first)

Declares the C++ API surface exposed to Kotlin. Owner: Member 4, same Day-0
sign-off requirement as `imu_types.h`.

```cpp
extern "C" {
    // Calibration
    JNIEXPORT void JNICALL Java_..._calibrationUpdate(JNIEnv*, jobject, jlong ts, jfloatArray accel, jfloatArray gyro);
    JNIEXPORT jfloatArray JNICALL Java_..._levelAccelerometer(JNIEnv*, jobject, jfloatArray rawAccel);
    JNIEXPORT jfloatArray JNICALL Java_..._getCurrentRotation(JNIEnv*, jobject); // returns [pitch, roll, yaw] rad

    // Fusion
    JNIEXPORT void JNICALL Java_..._fusionPredict(JNIEnv*, jobject, jfloat deltaV);
    JNIEXPORT void JNICALL Java_..._fusionUpdateGnss(JNIEnv*, jobject, jlong ts, jdouble lat, jdouble lon, jfloat speed, jfloat accuracy);
    JNIEXPORT void JNICALL Java_..._fusionUpdateMapMatch(JNIEnv*, jobject, jfloatArray pseudoMeasurement, jfloat confidence);
    JNIEXPORT jfloatArray JNICALL Java_..._fusionGetNavState(JNIEnv*, jobject); // [px,py,pz,vx,vy,vz,heading]

    // Map matching
    JNIEXPORT jobject JNICALL Java_..._mapMatch(JNIEnv*, jobject, jfloatArray trajectoryPoint); // returns MatchResult

    // Mode manager
    JNIEXPORT jint JNICALL Java_..._getCurrentMode(JNIEnv*, jobject); // NavigationMode ordinal
}
```

⚠️ ASSUMPTION — the exact JNI method names/package prefix (`Java_com_sih_deadreckoning_..._`)
depend on final package structure; Member 4 fills in the real mangled names once
`fusion/FusionBridge.kt`, `mapmatching/MapMatchingBridge.kt`, `mode/ModeManager.kt`,
and `calibration/AlignmentManager.kt` package paths are finalized. Treat the
above as the **method surface contract** (names, params, return shape), not
final mangled symbols.

---

## 3. `core-engine/src/calibration.cpp` + Kotlin `calibration/` (§6.2) — Member 3

```cpp
struct CalibrationState {
    float pitch_rad;
    float roll_rad;
    float yaw_rad;          // from pooled nonlinear optimization, not per-window
    bool yaw_calibrated;    // false until enough straight-line-braking windows collected
};

CalibrationState get_current_rotation();
void update(ImuSample sample);
Vector3 level_accelerometer(Vector3 raw_accel);  // Vector3 = float[3], returns
                                                   // gravity-removed, vehicle-frame accel
```

**Handoff downstream:** `level_accelerometer()`'s output feeds directly into
Member 6's `TFLiteInferenceEngine` windowing (every raw sample, before window
construction — not batched/deferred) and into Member 4's `ukf_fusion.cpp`
`predict()` step.

**Gate parameters (from PRD, not ambiguous — do not change):**
- Static correction gate: `abs(accel_magnitude - 9.81) < threshold` ⚠️ ASSUMPTION
  — PRD doesn't give exact threshold value, only says "near 9.81." Recommend
  starting at `0.3 m/s²` and tuning against real data.
- Sustained duration: `>= 0.5s` (explicit in PRD).
- Measurement noise inflation: proportional to `(accel_magnitude - 9.81)^2` (explicit).

---

## 4. `core-engine/src/ukf_fusion.cpp` + `ins_mechanization.cpp` (§6.3) — Member 4

```cpp
void predict(float delta_v, float dt_seconds);
void update_gnss(GnssFix fix);
void update_map_match(float pseudo_measurement[2], float confidence); // [lateral_offset_m, heading_offset_rad]
void apply_zaru();  // called when straight-line driving detected
void apply_zupt();  // called when stationary_probability > 0.95
void apply_es_nn_residual(float delta_p[3], float delta_v[3], float delta_psi);
NavState get_nav_state();
```

**Confidence threshold for `update_map_match`:** ⚠️ ASSUMPTION — PRD says
"tunable threshold" but doesn't give a starting value. Recommend `0.7`
as the initial default (Member 5 and Member 4 agree on tuning this together
once real map-match confidence scores are observed).

**ZUPT threshold:** `stationary_probability > 0.95` (explicit in PRD, not ambiguous).

**Handoff downstream:** `get_nav_state()`'s `NavState` struct is what Member 1's
UI and Member 6's `test-harness` both read for position/drift display and
benchmark scoring respectively.

---

## 5. `sensor/SensorBridge.kt`, `sensor/SensorDataBuffer.kt`, `sensor/GnssQualityMonitor.kt` (§5, §2.1) — Member 2

```kotlin
class SensorBridge {
    fun start(callback: (ImuSample) -> Unit)
    fun stop()
}

class SensorDataBuffer {
    fun push(sample: ImuSample)
    fun getWindow(windowMs: Long): List<ImuSample>  // ⚠️ ASSUMPTION: pull-based API;
                                                       // confirm with Member 6 whether
                                                       // TFLiteInferenceEngine wants
                                                       // push (callback) or pull instead
}

class GnssQualityMonitor {
    fun start(onFix: (GnssFix) -> Unit)
    fun stop()
    // quality_score computed internally from satelliteCount + accuracyM + fix-age,
    // exposed as part of GnssFix (see struct above) — not a separate method
}
```

**Handoff downstream:** `ImuSample` stream → `AlignmentManager` (Member 3) and
`TFLiteInferenceEngine` (Member 6) both consume raw samples directly.
`GnssFix` (with `qualityScore`) → `ModeManager` (Member 4/1) and `FusionBridge`
(Member 4).

---

## 6. `core-engine/src/map_matching.cpp` + `mapmatching/` (§6.4) — Member 5

```cpp
struct MapMatchCandidate {
    int road_segment_id;
    float perpendicular_distance_m;
    float emission_probability;
};

struct MapMatchResult {
    int matched_segment_id;
    float pseudo_measurement[2];  // [lateral_offset_m, heading_offset_rad] — same
                                    // shape consumed by ukf_fusion::update_map_match
    float confidence;             // [0,1]
};

MapMatchResult match(float trajectory_point[3], float heading_rad);
```

**Handoff downstream:** `MapMatchResult.pseudo_measurement` + `.confidence` feed
directly into Member 4's `update_map_match()` — field names/order must match
exactly what's declared here.

---

## 7. `core-engine/src/mode_manager.cpp` + `mode/ModeManager.kt` (§6.5) — Member 4 (C++) / Member 1 (Kotlin wrapper)

```cpp
NavigationMode evaluate_mode(GnssFix latest_fix, int64_t now_ms);
// scoring inputs: satellite_count, accuracy_m, fix_age_ms = now_ms - latest_fix.timestamp_ms
```

⚠️ ASSUMPTION — exact scoring thresholds (satellite count cutoff, accuracy_m
cutoff, max fix-age before declaring GNSS lost) aren't specified in the PRD.
Recommend as a starting point: `satellite_count >= 4 AND accuracy_m <= 15.0 AND
fix_age_ms <= 2000` → `GNSS_AIDED_INS`; otherwise `PURE_DR`. Tune against real
data before finals.

**Kotlin side (`ModeManager.kt`) is a pure pass-through** — calls
`getCurrentMode()` via JNI, exposes the `NavigationMode` enum to the UI. No
scoring logic duplicated in Kotlin (§7 guardrail #1).

---

## 8. `inference/TFLiteInferenceEngine.kt` + `OnnxInferenceEngine.kt` (§6.6) — Member 6

**LOCKED as of placeholder model v1.0 — confirmed by ML team, shape/plumbing
final, only accuracy pending. No `⚠️ ASSUMPTION` markers remain in this
section.**

**Input — sliding window:**
- Window length: exactly **20 samples** = 2 seconds of history at 10Hz.
  Maintain a FIFO buffer in `SensorDataBuffer`: push 1 new sample, pop 1 old
  sample every 100ms.
- 6 features per timestep, in this **exact order**:
  `[leveled_ax, leveled_ay, leveled_az, gyro_yaw, gyro_pitch, gyro_roll]`
  — `leveled_a*` is `calibration.cpp`'s `level_accelerometer()` output
  (Section 3), never raw accel.
- Normalization: `x_norm = (x_raw - mean_i) / std_i` per feature, applied in
  Kotlin using the constants in `normalization_stats_v2.npz`, **before** the
  window is fed to the model. Not baked into the model graph.

**Hidden state (GRU) — model is structurally stateless, state managed externally:**
- `h_in` tensor: shape `[1, 1, 64]`, Float32
- `h_out` tensor: shape `[1, 1, 64]`, Float32
- First inference call of a drive/session: `h_in` = all zeros
- Every subsequent call: `h_in` = the `h_out` returned by the *previous* call
- Reset to all-zeros **only** at new-session start — never mid-session, never
  every call. This is the exact silent-degradation trap called out in the
  PRD (§7 guardrail #6) — write an explicit test proving persistence across
  calls and reset only at session boundaries.

**Outputs:**
- Output 1 `speed_metric`: shape `[1,1]`, Float32. Placeholder model outputs
  relative Δv; the final model will output absolute speed (m/s) through the
  same tensor. Only the downstream *meaning* in fusion's `predict()` call
  changes later, not the plumbing.
- Output 2 `stationary_logit`: shape `[1,1]`, Float32, a **raw logit** —
  apply sigmoid (`1 / (1 + exp(-x))`) in Kotlin before comparing against the
  ZUPT threshold (`stationary_probability > 0.95`, Section 4).

```kotlin
class TFLiteInferenceEngine {
    fun initialize(context: Context)  // loads converted .tflite once
    fun infer(window: List<ImuSample>): InferenceResult
    // h_in/h_out cached as a persistent private FloatArray member,
    // fed forward across calls, reset ONLY at session start
}

data class InferenceResult(
    val deltaV: Float,               // raw speed_metric output (see note above
                                       // on placeholder-vs-final meaning)
    val stationaryProbability: Float // [0,1], AFTER sigmoid applied
)
```

**Handoff downstream:** `InferenceResult` feeds Member 4's `predict(delta_v, dt)`
and the ZUPT check in `apply_zupt()`.

---

## 9. `fusion/FusionBridge.kt`, `mapmatching/MapMatchingBridge.kt` (§6.7) — Member 4 / Member 5

Thin JNI wrappers only — one Kotlin method per exposed C++ function in
`JniBridge.h` (Section 2 above), no business logic. If a "quick fix" appears
here, it's a sign the C++ core is being bypassed (§7 guardrail #1).

```kotlin
class FusionBridge {
    fun predict(deltaV: Float, dtSeconds: Float)
    fun updateGnss(fix: GnssFix)
    fun updateMapMatch(pseudoMeasurement: FloatArray, confidence: Float)
    fun getNavState(): NavState  // Kotlin mirror of the C++ struct
}
```

---

## 10. `ml-pipeline/models/velocity_regressor.py` (§6.8) — parallel ML team, referenced for interface only

```python
def forward(self, x: Tensor, h0: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """
    x: (batch, seq_len, n_features) -- leveled accel + raw gyro window
    h0: GRU hidden state from previous call, or None at session start
    Returns: (delta_v_pred, stationary_logit, h_out)
    """
```

Architecture: causal 1D-CNN (LEFT-only padding, verified via unit test that a
sample after time t never affects output at time t) → stateful GRU (hidden
state exposed for explicit external management) → dual heads (Huber loss for
Δv, BCE loss for stationary classification). This is the frozen reference the
exported `velocity_model.tflite` implements — Member 6's `TFLiteInferenceEngine`
input/output contract (Section 8 above) must match this exactly.

---

## 11. `ml-pipeline/training/evaluate.py` + `test-harness/drift_calculator.py` (§6.10) — Member 6

```python
def compute_drift_pct(
    final_position_error_m: float,
    true_distance_travelled_m: float
) -> float:
    return (final_position_error_m / true_distance_travelled_m) * 100.0

def compute_drift_report(sessions: list[SessionResult]) -> dict:
    """
    Returns drift % broken out by (blackout_duration_bucket, speed_regime) —
    NEVER a single pooled number (§7 guardrail #3). RMSE may be included as a
    diagnostic field but must never be the pass/fail criterion.
    """
```

Both `evaluate.py` and `drift_calculator.py` must call the **same**
underlying implementation via `core_bindings.cpp` (Member 4) — not two
independent reimplementations of drift math (§7 guardrail #1).

---

## 12. `docs/dataset-notes.md` (§6.11) — must capture verbatim (Member 6)

- CSV column names carry inconsistent leading/trailing whitespace — always
  `.strip()` column names immediately after reading.
- S-(phone) files are `latin-1`, not UTF-8 — try UTF-8 first, fall back to `latin-1`.
- V-(vehicle) time column is in **seconds**; S-(phone) time column is in
  **milliseconds** — convert before any time-alignment logic.
- GPS-outage index file location unconfirmed — if found, wire into
  `real_outage_evaluator.py`; otherwise `gnss_outage_simulator.py` is the
  documented fallback, and any resulting numbers must be labeled as simulated.

---

## 13. `map-data/scripts/convert_to_graph.py` (§6.12) — Member 5

```python
def convert_osm_to_graph(pbf_path: str, output_path: str) -> None:
    """
    Extracts drivable-road ways only (filters footpaths/buildings). Output
    format: ⚠️ ASSUMPTION — PRD says "serializable format core-engine can load
    without parsing OSM XML/PBF at runtime" but doesn't specify the format.
    Recommend a simple flatbuffer or custom binary node/edge format; confirm
    with Member 4 (who writes the C++ loader) before finalizing the schema.
    """
```

**Handoff downstream:** the converted graph file is loaded once at app
startup by `core-engine/src/map_matching.cpp` (Member 5's own C++ file) —
schema must match between this script's output and that loader's input.

---

## 14. `test-harness/run_benchmark.py` + `report_generator.py` (§6.13) — Member 6

```python
def run_benchmark(drive_log_path: str, inject_blackout: bool = True) -> BenchmarkReport:
    """
    Replays drive log through calibration -> inference -> fusion -> map-matching,
    injects a GNSS blackout (real or simulated per dataset-notes.md), computes
    drift % via drift_calculator.compute_drift_report(), and outputs:
      - a CSV of per-segment results
      - a position-plot PNG (true vs. estimated trajectory)
    """
```

---

## Open items requiring team confirmation before Day-0 sign-off

1. Exact `NavState.covariance` dimensionality (Section 1) — depends on final UKF state vector size.
2. Static calibration gate threshold value (Section 3).
3. Mode-manager quality-scoring thresholds (Section 7).
4. ~~Inference window size in samples (Section 8).~~ **RESOLVED** — locked
   per ML team's placeholder model v1.0 confirmation (20 samples, 6 features,
   exact order and hidden-state shapes now specified in Section 8).
5. Map-graph serialization format (Section 13).
6. `update_map_match` confidence threshold default (Section 4).

Resolve these in the Day-0 sync — once resolved, delete the ⚠️ markers and
this "Open items" section from the file.