/**
 * stream_ingester.cpp — Real-time streaming IMU/GNSS ingestion
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)
 *
 * Reads line-delimited JSON or CSV from stdin or a TCP socket, parsing each
 * line into ImuSample or GnssFix and dispatching via callbacks.
 *
 * Designed for the edge-engine's real-time mode: connect a FOG/IMU device's
 * serial output → pipe through a simple JSON wrapper → feed into this ingester.
 *
 * JSON format (one per line):
 *   {"type":"imu","ts":1700000000000,"ax":0.1,"ay":-0.05,"az":9.81,"gx":0.01,"gy":0.0,"gz":-0.001}
 *   {"type":"gnss","ts":1700000000000,"lat":12.97,"lon":77.59,"speed":8.4,"acc":2.0,"sats":10}
 *
 * References:
 *   PRD §2.1 (edge-engine swaps SensorBridge for stream_ingester),
 *   api-contracts.md §1 (struct contracts)
 */

#include <string>
#include <functional>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <cmath>
#include <cstring>

// Include the frozen shared structs
#include "../../core-engine/include/imu_types.h"

namespace edge {

/* ── Callbacks ─────────────────────────────────────────────────────────────── */

using ImuCallback = std::function<void(const ImuSample&)>;
using GnssCallback = std::function<void(const GnssFix&)>;

/* ── Simple JSON field extractor (no external JSON library) ────────────────── */

/**
 * Extract a string value for a given key from a JSON-like string.
 * This is intentionally minimal — handles flat key-value JSON only,
 * which is sufficient for the streaming format.
 */
static std::string json_get_str(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;

    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.size()) return "";

    // Check if value is quoted
    if (json[pos] == '"') {
        size_t end = json.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    }

    // Numeric value — read until comma, }, or end
    size_t end = json.find_first_of(",}", pos);
    if (end == std::string::npos) end = json.size();
    std::string val = json.substr(pos, end - pos);

    // Trim whitespace
    size_t s = val.find_first_not_of(" \t\r\n");
    size_t e = val.find_last_not_of(" \t\r\n");
    if (s == std::string::npos) return "";
    return val.substr(s, e - s + 1);
}

static float json_get_float(const std::string& json, const std::string& key,
                            float default_val = 0.0f) {
    std::string val = json_get_str(json, key);
    if (val.empty()) return default_val;
    try { return std::stof(val); } catch (...) { return default_val; }
}

static double json_get_double(const std::string& json, const std::string& key,
                              double default_val = 0.0) {
    std::string val = json_get_str(json, key);
    if (val.empty()) return default_val;
    try { return std::stod(val); } catch (...) { return default_val; }
}

static int64_t json_get_int64(const std::string& json, const std::string& key,
                              int64_t default_val = 0) {
    std::string val = json_get_str(json, key);
    if (val.empty()) return default_val;
    try { return std::stoll(val); } catch (...) { return default_val; }
}

/* ── Parse functions ───────────────────────────────────────────────────────── */

static bool parse_imu_json(const std::string& line, ImuSample& out) {
    out.timestamp_ms = json_get_int64(line, "ts");
    out.accel[0] = json_get_float(line, "ax");
    out.accel[1] = json_get_float(line, "ay");
    out.accel[2] = json_get_float(line, "az");
    out.gyro[0] = json_get_float(line, "gx");
    out.gyro[1] = json_get_float(line, "gy");
    out.gyro[2] = json_get_float(line, "gz");
    return out.timestamp_ms > 0;
}

static bool parse_gnss_json(const std::string& line, GnssFix& out) {
    out.timestamp_ms = json_get_int64(line, "ts");
    out.lat_deg = json_get_double(line, "lat");
    out.lon_deg = json_get_double(line, "lon");
    out.speed_mps = json_get_float(line, "speed");
    out.bearing_rad = 0.0f;
    float bearing_deg = json_get_float(line, "bearing");
    if (bearing_deg != 0.0f) {
        out.bearing_rad = static_cast<float>((90.0 - bearing_deg) * M_PI / 180.0);
    }
    out.accuracy_m = json_get_float(line, "acc", 30.0f);
    out.satellite_count = static_cast<int32_t>(json_get_float(line, "sats"));
    out.quality_score = 0.5f;  // Placeholder
    return out.timestamp_ms > 0 && out.lat_deg != 0.0;
}

/* ── Stream ingestion ──────────────────────────────────────────────────────── */

/**
 * Read line-delimited JSON from an input stream, dispatching each parsed
 * sample/fix to the appropriate callback.
 *
 * Reads until EOF or an error. For stdin usage: pipe data in and close.
 * For a TCP socket: wrap the socket fd in an istream.
 *
 * @param input      Input stream (e.g. std::cin for stdin)
 * @param on_imu     Callback for each parsed ImuSample
 * @param on_gnss    Callback for each parsed GnssFix
 * @return           Total number of lines processed
 */
int ingest_stream(
    std::istream& input,
    ImuCallback on_imu,
    GnssCallback on_gnss
) {
    std::string line;
    int count = 0;
    int imu_count = 0;
    int gnss_count = 0;
    int error_count = 0;

    while (std::getline(input, line)) {
        count++;

        if (line.empty() || line[0] == '#') continue;

        // Determine message type
        std::string msg_type = json_get_str(line, "type");

        if (msg_type == "imu") {
            ImuSample sample;
            if (parse_imu_json(line, sample)) {
                on_imu(sample);
                imu_count++;
            } else {
                error_count++;
            }
        } else if (msg_type == "gnss" || msg_type == "gps") {
            GnssFix fix;
            if (parse_gnss_json(line, fix)) {
                on_gnss(fix);
                gnss_count++;
            } else {
                error_count++;
            }
        } else {
            // Try auto-detection based on field presence
            if (line.find("\"ax\"") != std::string::npos) {
                ImuSample sample;
                if (parse_imu_json(line, sample)) {
                    on_imu(sample);
                    imu_count++;
                }
            } else if (line.find("\"lat\"") != std::string::npos) {
                GnssFix fix;
                if (parse_gnss_json(line, fix)) {
                    on_gnss(fix);
                    gnss_count++;
                }
            } else {
                error_count++;
            }
        }
    }

    std::cerr << "[stream_ingester] Processed " << count << " lines: "
              << imu_count << " IMU, " << gnss_count << " GNSS, "
              << error_count << " errors" << std::endl;

    return count;
}

} // namespace edge
