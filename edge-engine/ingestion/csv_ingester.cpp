/**
 * csv_ingester.cpp — Parse CSV drive logs into ImuSample/GnssFix vectors
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)
 *
 * Handles the dataset-notes.md gotchas:
 *   - Strip column name whitespace
 *   - Detect seconds vs milliseconds timestamps
 *   - Parse both S-(phone) and V-(vehicle) CSV formats
 *
 * Uses only standard C++ (no external CSV library) for portability.
 *
 * References:
 *   PRD §6.11 (dataset gotchas), api-contracts.md §1 (struct contracts)
 */

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>

// Include the frozen shared structs
#include "../../core-engine/include/imu_types.h"

namespace edge {

/* ── String utilities ──────────────────────────────────────────────────────── */

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(trim(field));
    }
    return fields;
}

static float safe_float(const std::string& s, float default_val = 0.0f) {
    if (s.empty()) return default_val;
    try {
        return std::stof(s);
    } catch (...) {
        return default_val;
    }
}

static double safe_double(const std::string& s, double default_val = 0.0) {
    if (s.empty()) return default_val;
    try {
        return std::stod(s);
    } catch (...) {
        return default_val;
    }
}

static int64_t safe_int64(const std::string& s, int64_t default_val = 0) {
    if (s.empty()) return default_val;
    try {
        return std::stoll(s);
    } catch (...) {
        return default_val;
    }
}

/* ── Column index mapping ──────────────────────────────────────────────────── */

struct ColumnMap {
    int timestamp = -1;
    int ax = -1, ay = -1, az = -1;
    int gx = -1, gy = -1, gz = -1;
    int lat = -1, lon = -1;
    int speed = -1, bearing = -1;
    int accuracy = -1, satellites = -1;
    bool is_vehicle_file = false;  // V-file timestamps are in seconds
};

static ColumnMap build_column_map(const std::vector<std::string>& headers) {
    ColumnMap cm;
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = headers[i];
        // Lowercase for matching
        std::transform(h.begin(), h.end(), h.begin(), ::tolower);

        if (h == "timestamp_ms" || h == "time")   cm.timestamp = static_cast<int>(i);
        else if (h == "ax")                        cm.ax = static_cast<int>(i);
        else if (h == "ay")                        cm.ay = static_cast<int>(i);
        else if (h == "az")                        cm.az = static_cast<int>(i);
        else if (h == "gx")                        cm.gx = static_cast<int>(i);
        else if (h == "gy")                        cm.gy = static_cast<int>(i);
        else if (h == "gz")                        cm.gz = static_cast<int>(i);
        else if (h == "lat" || h == "latitude")    cm.lat = static_cast<int>(i);
        else if (h == "lon" || h == "longitude")   cm.lon = static_cast<int>(i);
        else if (h == "speed" || h == "speed_mps") cm.speed = static_cast<int>(i);
        else if (h == "bearing" || h == "bearing_deg" || h == "heading")
                                                   cm.bearing = static_cast<int>(i);
        else if (h == "accuracy" || h == "accuracy_m")
                                                   cm.accuracy = static_cast<int>(i);
        else if (h == "satellites" || h == "satellite_count")
                                                   cm.satellites = static_cast<int>(i);
    }
    return cm;
}

/* ── Ingestion result ──────────────────────────────────────────────────────── */

struct IngestionResult {
    std::vector<ImuSample> imu_samples;
    std::vector<GnssFix> gnss_fixes;
    int total_rows = 0;
    int parsed_imu = 0;
    int parsed_gnss = 0;
    int skipped_rows = 0;
    std::string error;
};

/* ── Main ingestion function ───────────────────────────────────────────────── */

IngestionResult ingest_csv(const std::string& filepath) {
    IngestionResult result;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        result.error = "Cannot open file: " + filepath;
        return result;
    }

    // Read header line
    std::string header_line;
    if (!std::getline(file, header_line)) {
        result.error = "Empty file: " + filepath;
        return result;
    }

    std::vector<std::string> headers = split_csv_line(header_line);
    ColumnMap cm = build_column_map(headers);

    if (cm.timestamp < 0) {
        result.error = "No timestamp column found in: " + filepath;
        return result;
    }

    // Detect if this is a V-file (timestamps in seconds)
    // by reading the first data row
    std::streampos data_start = file.tellg();
    std::string peek_line;
    if (std::getline(file, peek_line)) {
        auto peek_fields = split_csv_line(peek_line);
        if (cm.timestamp < static_cast<int>(peek_fields.size())) {
            double ts_val = safe_double(peek_fields[cm.timestamp]);
            // If timestamp < 2e10, it's in seconds (epoch seconds ~1.7e9)
            cm.is_vehicle_file = (ts_val < 2e10 && ts_val > 1e8);
        }
    }
    // Reset to start of data
    file.seekg(data_start);

    // Parse all data rows
    std::string line;
    while (std::getline(file, line)) {
        result.total_rows++;

        if (line.empty() || line[0] == '#') {
            result.skipped_rows++;
            continue;
        }

        auto fields = split_csv_line(line);
        if (fields.size() <= static_cast<size_t>(cm.timestamp)) {
            result.skipped_rows++;
            continue;
        }

        // Parse timestamp (handle seconds vs milliseconds)
        int64_t timestamp_ms;
        if (cm.is_vehicle_file) {
            double ts_sec = safe_double(fields[cm.timestamp]);
            timestamp_ms = static_cast<int64_t>(ts_sec * 1000.0);
        } else {
            timestamp_ms = safe_int64(fields[cm.timestamp]);
        }

        // Parse IMU data if columns exist
        if (cm.ax >= 0 && cm.ay >= 0 && cm.az >= 0 &&
            cm.gx >= 0 && cm.gy >= 0 && cm.gz >= 0 &&
            static_cast<size_t>(cm.gz) < fields.size()) {

            ImuSample sample;
            sample.timestamp_ms = timestamp_ms;
            sample.accel[0] = safe_float(fields[cm.ax]);
            sample.accel[1] = safe_float(fields[cm.ay]);
            sample.accel[2] = safe_float(fields[cm.az]);
            sample.gyro[0] = safe_float(fields[cm.gx]);
            sample.gyro[1] = safe_float(fields[cm.gy]);
            sample.gyro[2] = safe_float(fields[cm.gz]);

            result.imu_samples.push_back(sample);
            result.parsed_imu++;
        }

        // Parse GNSS data if columns exist and values are non-empty
        if (cm.lat >= 0 && cm.lon >= 0 &&
            static_cast<size_t>(cm.lat) < fields.size() &&
            static_cast<size_t>(cm.lon) < fields.size() &&
            !fields[cm.lat].empty() && !fields[cm.lon].empty()) {

            GnssFix fix;
            fix.timestamp_ms = timestamp_ms;
            fix.lat_deg = safe_double(fields[cm.lat]);
            fix.lon_deg = safe_double(fields[cm.lon]);
            fix.speed_mps = (cm.speed >= 0 &&
                             static_cast<size_t>(cm.speed) < fields.size())
                            ? safe_float(fields[cm.speed]) : 0.0f;
            fix.bearing_rad = 0.0f;  // Converted from degrees at usage site
            if (cm.bearing >= 0 &&
                static_cast<size_t>(cm.bearing) < fields.size()) {
                float bearing_deg = safe_float(fields[cm.bearing]);
                // Convert CW-from-North degrees to ENU radians (0=East, CCW+)
                fix.bearing_rad = static_cast<float>(
                    (90.0 - bearing_deg) * M_PI / 180.0);
            }
            fix.accuracy_m = (cm.accuracy >= 0 &&
                              static_cast<size_t>(cm.accuracy) < fields.size())
                             ? safe_float(fields[cm.accuracy]) : 30.0f;
            fix.satellite_count = (cm.satellites >= 0 &&
                                   static_cast<size_t>(cm.satellites) < fields.size())
                                  ? static_cast<int32_t>(safe_float(fields[cm.satellites]))
                                  : 0;
            fix.quality_score = 0.5f;  // Placeholder — real scoring is in M2's code

            result.gnss_fixes.push_back(fix);
            result.parsed_gnss++;
        }
    }

    return result;
}

} // namespace edge
