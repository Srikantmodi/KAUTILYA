/**
 * core_bindings.cpp — Python bindings via pybind11
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 4 (Fusion Core, INS Mechanization & Mode State Machine)
 *
 * Exposes core-engine types and the UKF fusion engine to Python so that
 * Member 6's test-harness (drift_calculator.py, run_benchmark.py) can call
 * the same C++ implementation used on-device — ONE implementation of drift
 * math, not two (PRD §7 guardrail #1).
 *
 * Build:
 *   pip install pybind11
 *   c++ -O2 -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       core_bindings.cpp ../src/ukf_fusion.cpp ../src/ins_mechanization.cpp \
 *       ../src/mode_manager.cpp \
 *       -I../include -o core_engine$(python3-config --extension-suffix)
 *
 * Usage from Python:
 *   import core_engine
 *   ukf = core_engine.UKFFusion()
 *   ukf.predict(delta_v=5.0, dt_seconds=0.1)
 *   ns = ukf.get_nav_state()
 *   print(ns.position)  # [pe, pn, pu]
 *
 * References:
 *   api-contracts.md §11, PRD §7 guardrail #1
 */

#ifdef ENABLE_PYBIND11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

#include "imu_types.h"
#include "ins_mechanization.h"
#include "ukf_fusion.h"
#include "mode_manager.h"

namespace py = pybind11;

/* ═════════════════════════════════════════════════════════════════════════════
 * Drift calculation — single implementation for §7 guardrail #1
 *
 * This is the CANONICAL compute_drift_pct. Both Python evaluate.py and
 * drift_calculator.py should call this through the bindings rather than
 * maintaining independent Python reimplementations.
 * ═════════════════════════════════════════════════════════════════════════════ */

static float compute_drift_pct(float final_position_error_m,
                                float true_distance_travelled_m) {
    if (true_distance_travelled_m <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    return (final_position_error_m / true_distance_travelled_m) * 100.0f;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Haversine distance (meters) — matches drift_calculator.py
 * ═════════════════════════════════════════════════════════════════════════════ */

static double haversine_m(double lat1, double lon1, double lat2, double lon2) {
    constexpr double R = 6371000.0;
    constexpr double DEG2RAD = M_PI / 180.0;

    double phi1 = lat1 * DEG2RAD;
    double phi2 = lat2 * DEG2RAD;
    double dphi = (lat2 - lat1) * DEG2RAD;
    double dlam = (lon2 - lon1) * DEG2RAD;

    double a = sin(dphi / 2.0) * sin(dphi / 2.0) +
               cos(phi1) * cos(phi2) * sin(dlam / 2.0) * sin(dlam / 2.0);
    return R * 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
}


/* ═════════════════════════════════════════════════════════════════════════════
 * pybind11 module definition
 * ═════════════════════════════════════════════════════════════════════════════ */

PYBIND11_MODULE(core_engine, m) {
    m.doc() = "SIH PS-26168 Core Engine — Python bindings for UKF fusion, "
              "INS mechanization, mode management, and drift calculation.";

    /* ── ImuSample struct ────────────────────────────────────────────────── */
    py::class_<ImuSample>(m, "ImuSample")
        .def(py::init<>())
        .def_readwrite("timestamp_ms", &ImuSample::timestamp_ms)
        .def_property("accel",
            [](const ImuSample& s) {
                return std::vector<float>{s.accel[0], s.accel[1], s.accel[2]};
            },
            [](ImuSample& s, const std::vector<float>& v) {
                for (int i = 0; i < 3 && i < (int)v.size(); i++) s.accel[i] = v[i];
            })
        .def_property("gyro",
            [](const ImuSample& s) {
                return std::vector<float>{s.gyro[0], s.gyro[1], s.gyro[2]};
            },
            [](ImuSample& s, const std::vector<float>& v) {
                for (int i = 0; i < 3 && i < (int)v.size(); i++) s.gyro[i] = v[i];
            });

    /* ── GnssFix struct ──────────────────────────────────────────────────── */
    py::class_<GnssFix>(m, "GnssFix")
        .def(py::init<>())
        .def_readwrite("timestamp_ms",   &GnssFix::timestamp_ms)
        .def_readwrite("lat_deg",        &GnssFix::lat_deg)
        .def_readwrite("lon_deg",        &GnssFix::lon_deg)
        .def_readwrite("speed_mps",      &GnssFix::speed_mps)
        .def_readwrite("bearing_rad",    &GnssFix::bearing_rad)
        .def_readwrite("accuracy_m",     &GnssFix::accuracy_m)
        .def_readwrite("satellite_count",&GnssFix::satellite_count)
        .def_readwrite("quality_score",  &GnssFix::quality_score);

    /* ── NavState struct ─────────────────────────────────────────────────── */
    py::class_<NavState>(m, "NavState")
        .def(py::init<>())
        .def_property_readonly("position",
            [](const NavState& ns) {
                return std::vector<float>{ns.position[0], ns.position[1], ns.position[2]};
            })
        .def_property_readonly("velocity",
            [](const NavState& ns) {
                return std::vector<float>{ns.velocity[0], ns.velocity[1], ns.velocity[2]};
            })
        .def_readwrite("heading_rad", &NavState::heading_rad)
        .def_property_readonly("gyro_bias",
            [](const NavState& ns) {
                return std::vector<float>{ns.gyro_bias[0], ns.gyro_bias[1], ns.gyro_bias[2]};
            })
        .def_property_readonly("covariance",
            [](const NavState& ns) {
                return std::vector<float>(ns.covariance, ns.covariance + 16);
            });

    /* ── NavigationMode enum ─────────────────────────────────────────────── */
    py::enum_<NavigationMode>(m, "NavigationMode")
        .value("GNSS_AIDED_INS", NAVIGATION_MODE_GNSS_AIDED_INS)
        .value("PURE_DR",        NAVIGATION_MODE_PURE_DR)
        .export_values();

    /* ── UKFFusion class ─────────────────────────────────────────────────── */
    py::class_<UKFFusion>(m, "UKFFusion")
        .def(py::init<>())
        .def("predict",             &UKFFusion::predict,
             py::arg("delta_v"), py::arg("dt_seconds"))
        .def("update_gnss",         &UKFFusion::update_gnss,
             py::arg("fix"))
        .def("update_map_match",
            [](UKFFusion& self, const std::vector<float>& pm, float conf) {
                float arr[2] = { pm.size() > 0 ? pm[0] : 0.0f,
                                 pm.size() > 1 ? pm[1] : 0.0f };
                self.update_map_match(arr, conf);
            },
            py::arg("pseudo_measurement"), py::arg("confidence"))
        .def("apply_zaru",          &UKFFusion::apply_zaru)
        .def("apply_zupt",          &UKFFusion::apply_zupt)
        .def("apply_es_nn_residual",
            [](UKFFusion& self, const std::vector<float>& dp,
               const std::vector<float>& dv, float dpsi) {
                float p[3] = {0}, v[3] = {0};
                for (int i = 0; i < 3; i++) {
                    if (i < (int)dp.size()) p[i] = dp[i];
                    if (i < (int)dv.size()) v[i] = dv[i];
                }
                self.apply_es_nn_residual(p, v, dpsi);
            },
            py::arg("delta_p"), py::arg("delta_v"), py::arg("delta_psi"))
        .def("get_nav_state",       &UKFFusion::get_nav_state)
        .def("reset",               &UKFFusion::reset)
        .def("has_origin",          &UKFFusion::has_origin);

    /* ── ModeManager class ───────────────────────────────────────────────── */
    py::class_<ModeManager>(m, "ModeManager")
        .def(py::init<>())
        .def("update_gnss_fix",  &ModeManager::update_gnss_fix,
             py::arg("timestamp_ms"), py::arg("satellite_count"),
             py::arg("accuracy_m"), py::arg("quality_score"))
        .def("evaluate_mode",    &ModeManager::evaluate_mode,
             py::arg("now_ms"))
        .def("get_current_mode", &ModeManager::get_current_mode)
        .def("reset",            &ModeManager::reset);

    /* ── Standalone functions ────────────────────────────────────────────── */
    m.def("compute_drift_pct", &compute_drift_pct,
          "Compute drift percentage: (error / distance) * 100. "
          "Returns inf if distance <= 0.",
          py::arg("final_position_error_m"),
          py::arg("true_distance_travelled_m"));

    m.def("haversine_m", &haversine_m,
          "Haversine distance between two WGS-84 points, in meters.",
          py::arg("lat1"), py::arg("lon1"),
          py::arg("lat2"), py::arg("lon2"));
}

#endif /* ENABLE_PYBIND11 */
