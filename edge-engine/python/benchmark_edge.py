"""
benchmark_edge.py — Edge engine benchmark against multiple drive logs

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Runs the edge inference pipeline against one or more drive logs, compares
against ground truth, and produces per-log drift metrics.

Usage:
    python benchmark_edge.py --logs <dir_or_csv> \\
        [--model <onnx_path>] \\
        [--stats <npz_path>] \\
        [--output-dir <dir>]

References:
    PRD §6.13, api-contracts.md §14
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test-harness"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test-harness" / "simulator"))

from edge_runner import run_edge_inference, DEFAULT_MODEL, DEFAULT_STATS
from drift_calculator import (
    BlackoutSegment, SessionResult, compute_drift_report,
    format_report_summary, compute_path_distance, haversine_m,
)
from ground_truth_comparator import TrajectoryPoint
from report_generator import generate_csv_report
from gnss_outage_simulator import generate_scenario_blackouts
from route_replayer import load_drive_log

import numpy as np
import pandas as pd


def benchmark_single_log(
    csv_path: str,
    model_path: str,
    stats_path: str,
) -> SessionResult:
    """Run inference + drift analysis on a single drive log."""
    session_id = Path(csv_path).stem
    print(f"\n--- Benchmarking: {session_id} ---")

    # Load drive log
    df = load_drive_log(csv_path)
    if df.empty:
        print(f"  SKIP: empty log")
        return SessionResult(session_id=session_id)

    # Extract ground truth
    lat_col = 'lat' if 'lat' in df.columns else 'latitude'
    lon_col = 'lon' if 'lon' in df.columns else 'longitude'

    true_positions = []
    for _, row in df.iterrows():
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if pd.notna(lat) and pd.notna(lon):
            true_positions.append((
                int(row['timestamp_ms']),
                float(lat), float(lon),
                float(row.get('speed_mps', 0.0)) if pd.notna(
                    row.get('speed_mps')) else 0.0,
            ))

    # Generate blackout scenarios
    blackouts = generate_scenario_blackouts(df)
    if not blackouts:
        print(f"  SKIP: log too short for blackout injection")
        return SessionResult(session_id=session_id)

    # Run inference
    results = run_edge_inference(csv_path, model_path, stats_path)

    # Build session result with segments
    session = SessionResult(session_id=session_id)
    for bo in blackouts:
        # Filter true positions during blackout
        bo_true = [(ts, lat, lon, spd) for ts, lat, lon, spd in true_positions
                   if bo.start_time_ms <= ts <= bo.end_time_ms]

        if len(bo_true) < 2:
            continue

        true_pos_tuples = [(lat, lon) for _, lat, lon, _ in bo_true]
        true_dist = compute_path_distance(true_pos_tuples)

        avg_speed = 0.0
        speeds = [spd for _, _, _, spd in bo_true if spd > 0]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)

        # For now, estimated = true (fusion stub) — drift will be ~0
        # Real drift appears when actual DR integration replaces this
        est_pos_tuples = true_pos_tuples  # placeholder

        segment = BlackoutSegment(
            start_time_ms=bo.start_time_ms,
            end_time_ms=bo.end_time_ms,
            true_positions=true_pos_tuples,
            estimated_positions=est_pos_tuples,
            true_distance_m=true_dist,
            final_error_m=0.0,  # placeholder until fusion works
            avg_speed_mps=avg_speed,
        )
        session.segments.append(segment)

    return session


def benchmark_all(
    log_paths: list[str],
    model_path: str,
    stats_path: str,
    output_dir: str,
) -> None:
    """Run benchmark across multiple drive logs and produce aggregate report."""
    print(f"{'='*70}")
    print(f"EDGE ENGINE BENCHMARK")
    print(f"  Model: {model_path}")
    print(f"  Logs:  {len(log_paths)} file(s)")
    print(f"{'='*70}")

    t_start = time.perf_counter()
    sessions = []

    for path in log_paths:
        session = benchmark_single_log(path, model_path, stats_path)
        sessions.append(session)

    t_elapsed = time.perf_counter() - t_start

    # Compute aggregate drift report
    report = compute_drift_report(sessions)

    # Output
    os.makedirs(output_dir, exist_ok=True)
    csv_path = generate_csv_report(
        report,
        os.path.join(output_dir, "edge_benchmark_results.csv"),
    )

    print(f"\n{format_report_summary(report)}")
    print(f"\nBenchmark complete in {t_elapsed:.2f}s")
    print(f"CSV report: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SIH PS-26168 — Edge Engine Benchmark",
    )
    parser.add_argument("--logs", required=True,
                        help="Path to a CSV file or directory of CSV files")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to ONNX model")
    parser.add_argument("--stats", default=DEFAULT_STATS,
                        help="Path to normalization stats npz")
    parser.add_argument("--output-dir", default="benchmark_output",
                        help="Output directory")

    args = parser.parse_args()

    # Resolve log paths
    if os.path.isfile(args.logs):
        log_paths = [args.logs]
    elif os.path.isdir(args.logs):
        log_paths = sorted(glob.glob(os.path.join(args.logs, "*.csv")))
    else:
        log_paths = sorted(glob.glob(args.logs))

    if not log_paths:
        print(f"ERROR: No CSV files found at {args.logs}")
        sys.exit(1)

    benchmark_all(log_paths, args.model, args.stats, args.output_dir)


if __name__ == "__main__":
    main()
