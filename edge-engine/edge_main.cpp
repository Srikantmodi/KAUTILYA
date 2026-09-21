/**
 * edge_main.cpp — Edge engine CLI entry point
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)
 *
 * Standalone CLI binary that proves the same core inference runs outside
 * the Android app.  Wires csv_ingester → normalization → stdout output.
 *
 * The ONNX inference is handled by the Python edge_runner.py for now,
 * since ONNX Runtime C++ requires a build-system integration that depends
 * on the platform.  This C++ binary focuses on the ingestion + struct
 * pipeline, proving the same ImuSample/GnssFix structs flow correctly.
 *
 * Usage:
 *   ./edge_engine --csv <drive_log.csv> [--stream]
 *
 * In --csv mode: ingests the CSV, prints parsed sample counts + first/last
 * samples as validation.
 *
 * In --stream mode: reads line-delimited JSON from stdin and prints parsed
 * samples in real time.
 *
 * References:
 *   PRD §2.1 (edge-engine is identical core with swapped I/O adapters),
 *   PRD §3 (ONNX Runtime for edge inference)
 */

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>

// Include the frozen shared structs
#include "../core-engine/include/imu_types.h"

// Forward declarations from ingestion files
// (In a real build these would be headers; for now we declare the interface)
namespace edge {
    struct IngestionResult {
        std::vector<ImuSample> imu_samples;
        std::vector<GnssFix> gnss_fixes;
        int total_rows;
        int parsed_imu;
        int parsed_gnss;
        int skipped_rows;
        std::string error;
    };
    IngestionResult ingest_csv(const std::string& filepath);

    using ImuCallback = std::function<void(const ImuSample&)>;
    using GnssCallback = std::function<void(const GnssFix&)>;
    int ingest_stream(std::istream& input, ImuCallback on_imu, GnssCallback on_gnss);
}


static void print_usage(const char* prog) {
    std::cout << "SIH PS-26168 — Edge Engine CLI\n"
              << "\nUsage:\n"
              << "  " << prog << " --csv <drive_log.csv>\n"
              << "  " << prog << " --stream\n"
              << "\nOptions:\n"
              << "  --csv <path>   Ingest a CSV drive log file\n"
              << "  --stream       Read line-delimited JSON from stdin\n"
              << "  --help         Show this help message\n"
              << "\nThe Python edge_runner.py handles ONNX inference;\n"
              << "this binary validates the C++ ingestion pipeline.\n";
}


static void print_imu_sample(const ImuSample& s) {
    std::cout << "  IMU t=" << s.timestamp_ms
              << " accel=[" << s.accel[0] << ", " << s.accel[1]
              << ", " << s.accel[2] << "]"
              << " gyro=[" << s.gyro[0] << ", " << s.gyro[1]
              << ", " << s.gyro[2] << "]\n";
}


static void print_gnss_fix(const GnssFix& f) {
    std::cout << "  GNSS t=" << f.timestamp_ms
              << " lat=" << f.lat_deg << " lon=" << f.lon_deg
              << " speed=" << f.speed_mps << " m/s"
              << " acc=" << f.accuracy_m << "m"
              << " sats=" << f.satellite_count << "\n";
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            mode = "csv";
            csv_path = argv[++i];
        }
        if (std::strcmp(argv[i], "--stream") == 0) {
            mode = "stream";
        }
    }

    if (mode == "csv") {
        std::cout << "=== Edge Engine — CSV Ingestion ===\n";
        std::cout << "File: " << csv_path << "\n\n";

        auto result = edge::ingest_csv(csv_path);

        if (!result.error.empty()) {
            std::cerr << "ERROR: " << result.error << "\n";
            return 1;
        }

        std::cout << "Parsed:\n"
                  << "  Total rows:    " << result.total_rows << "\n"
                  << "  IMU samples:   " << result.parsed_imu << "\n"
                  << "  GNSS fixes:    " << result.parsed_gnss << "\n"
                  << "  Skipped rows:  " << result.skipped_rows << "\n\n";

        if (!result.imu_samples.empty()) {
            std::cout << "First IMU sample:\n";
            print_imu_sample(result.imu_samples.front());
            std::cout << "Last IMU sample:\n";
            print_imu_sample(result.imu_samples.back());
        }

        if (!result.gnss_fixes.empty()) {
            std::cout << "\nFirst GNSS fix:\n";
            print_gnss_fix(result.gnss_fixes.front());
            std::cout << "Last GNSS fix:\n";
            print_gnss_fix(result.gnss_fixes.back());
        }

        // Output all IMU samples as CSV (for piping to Python edge_runner)
        std::cout << "\n--- IMU Output (CSV) ---\n";
        std::cout << "timestamp_ms,ax,ay,az,gx,gy,gz\n";
        for (const auto& s : result.imu_samples) {
            std::cout << s.timestamp_ms << ","
                      << s.accel[0] << "," << s.accel[1] << "," << s.accel[2] << ","
                      << s.gyro[0] << "," << s.gyro[1] << "," << s.gyro[2] << "\n";
        }

        return 0;

    } else if (mode == "stream") {
        std::cout << "=== Edge Engine — Stream Mode ===\n";
        std::cout << "Reading line-delimited JSON from stdin...\n";
        std::cout << "(Send EOF / Ctrl+D to stop)\n\n";

        int count = edge::ingest_stream(
            std::cin,
            [](const ImuSample& s) { print_imu_sample(s); },
            [](const GnssFix& f) { print_gnss_fix(f); }
        );

        std::cout << "\nProcessed " << count << " lines total.\n";
        return 0;

    } else {
        std::cerr << "ERROR: specify --csv <path> or --stream\n";
        print_usage(argv[0]);
        return 1;
    }
}
