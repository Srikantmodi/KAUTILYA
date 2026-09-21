#!/usr/bin/env bash
# push_model_to_device.sh — Deploy model artifacts to Android device
#
# SIH PS-26168  Intelligent Dead Reckoning
# Owner: Member 6
#
# Pushes the ML model files and normalization stats to the device's
# app-private storage via adb.  Supports both .tflite and .onnx formats.
#
# Usage:
#   ./scripts/push_model_to_device.sh [--tflite <path>] [--onnx <path>] [--stats <path>]
#
# Defaults:
#   --tflite  android-app/app/src/main/res/raw/velocity_model.tflite
#   --onnx    edge-engine/python/placeholder_model.onnx
#   --stats   edge-engine/python/normalization_stats_v2.npz

set -euo pipefail

PACKAGE="com.sih.deadreckoning"
DEVICE_DIR="/data/local/tmp/deadreckoning"

# Defaults
TFLITE_PATH="${TFLITE_PATH:-android-app/app/src/main/res/raw/velocity_model.tflite}"
ONNX_PATH="${ONNX_PATH:-edge-engine/python/placeholder_model.onnx}"
STATS_PATH="${STATS_PATH:-edge-engine/python/normalization_stats_v2.npz}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tflite) TFLITE_PATH="$2"; shift 2 ;;
        --onnx)   ONNX_PATH="$2";   shift 2 ;;
        --stats)  STATS_PATH="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--tflite <path>] [--onnx <path>] [--stats <path>]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== SIH Dead Reckoning — Model Push ==="
echo ""

# Check adb
if ! command -v adb &>/dev/null; then
    echo "ERROR: adb not found in PATH. Install Android SDK Platform Tools."
    exit 1
fi

# Check device connection
DEVICE_COUNT=$(adb devices | grep -c 'device$' || true)
if [[ "$DEVICE_COUNT" -eq 0 ]]; then
    echo "ERROR: No Android device connected. Connect via USB or start an emulator."
    exit 1
fi
echo "Device connected."

# Create target directory on device
adb shell "mkdir -p ${DEVICE_DIR}"

# Push TFLite model
if [[ -f "$TFLITE_PATH" ]]; then
    echo "Pushing TFLite model: ${TFLITE_PATH}"
    adb push "$TFLITE_PATH" "${DEVICE_DIR}/velocity_model.tflite"
    echo "  → ${DEVICE_DIR}/velocity_model.tflite"
else
    echo "SKIP: TFLite model not found at ${TFLITE_PATH}"
fi

# Push ONNX model
if [[ -f "$ONNX_PATH" ]]; then
    echo "Pushing ONNX model: ${ONNX_PATH}"
    adb push "$ONNX_PATH" "${DEVICE_DIR}/placeholder_model.onnx"
    echo "  → ${DEVICE_DIR}/placeholder_model.onnx"
else
    echo "SKIP: ONNX model not found at ${ONNX_PATH}"
fi

# Push normalization stats
if [[ -f "$STATS_PATH" ]]; then
    echo "Pushing normalization stats: ${STATS_PATH}"
    adb push "$STATS_PATH" "${DEVICE_DIR}/normalization_stats_v2.npz"
    echo "  → ${DEVICE_DIR}/normalization_stats_v2.npz"
else
    echo "SKIP: Normalization stats not found at ${STATS_PATH}"
fi

echo ""
echo "=== Push complete ==="
echo "Files on device:"
adb shell "ls -la ${DEVICE_DIR}/"
