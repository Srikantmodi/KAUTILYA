"""
run_benchmark.py — Single-command benchmark runner

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

THE single command that produces proposal-ready evidence.

Usage:
    python run_benchmark.py --drive-log <path> \\
        [--model <onnx_path>] \\
        [--stats <npz_path>] \\
        [--inject-blackout] \\
        [--output-dir <dir>]

Orchestrates:
    route_replayer → gnss_outage_simulator → drift_calculator → report_generator

Outputs:
    - CSV of per-segment drift results (broken out by duration + speed regime)
    - Position-plot PNG (true vs. estimated trajectory, blackout regions shaded)

References:
    PRD §6.13, api-contracts.md §14
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local imports work regardless of cwd
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from drift_calculator import (
    BlackoutSegment, SessionResult, compute_drift_report, format_report_summary,
)
from ground_truth_comparator import TrajectoryPoint, compare_trajectories
from report_generator import generate_full_report
from simulator.gnss_outage_simulator import (
    BlackoutWindow, generate_scenario_blackouts, inject_blackout,
    inject_multiple_blackouts,
)
from simulator.route_replayer import (
    RouteReplayer, ReplayConfig, load_drive_log, extract_features_from_row,
)


# ─── Default paths ──────────────────────────────────────────────────────────

DEFAULT_MODEL = str(Path(__file__).parent.parent /
                    "edge-engine" / "python" / "placeholder_model.onnx")
DEFAULT_STATS = str(Path(__file__).parent.parent /
                    "edge-engine" / "python" / "normalization_stats_v2.npz")
DEFAULT_OUTPUT_DIR = str(Path(__file__).parent / "benchmark_output")


def run_benchmark(
    drive_log_path: str,
    model_path: str = DEFAULT_MODEL,
    stats_path: str = DEFAULT_STATS,
    inject_blackout: bool = True,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Run the full benchmark pipeline:
        1. Load drive log
        2. Optionally inject GNSS blackouts
        3. Replay through inference model
        4. Compare against ground truth
        5. Compute drift metrics
        6. Generate CSV + PNG report

    Args:
        drive_log_path: Path to the drive log CSV.
        model_path: Path to the ONNX model file.
        stats_path: Path to normalization_stats_v2.npz.
        inject_blackout: Whether to inject simulated GNSS blackouts.
        output_dir: Output directory for CSV + PNG.

    Returns:
        Dict with 'csv_path', 'png_path', and 'report' keys.
    """
    session_name = Path(drive_log_path).stem
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {session_name}")
    print(f"{'='*70}")

    # ── 1. Load drive log ─────────────────────────────────────────────────
    print(f"\n[1/6] Loading drive log: {drive_log_path}")
    df = load_drive_log(drive_log_path)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    if df.empty:
        print("ERROR: Drive log is empty. Nothing to benchmark.")
        return {"csv_path": None, "png_path": None, "report": None}

    # ── 2. Extract ground truth (from the unmodified GPS data) ────────────
    print("\n[2/6] Extracting ground truth trajectory...")

    # Normalize column names for GPS
    lat_col = 'lat' if 'lat' in df.columns else 'latitude'
    lon_col = 'lon' if 'lon' in df.columns else 'longitude'

    true_positions = []
    for _, row in df.iterrows():
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if pd.notna(lat) and pd.notna(lon):
            true_positions.append(TrajectoryPoint(
                timestamp_ms=int(row['timestamp_ms']),
                lat=float(lat),
                lon=float(lon),
                speed_mps=float(row.get('speed_mps', 0.0)) if pd.notna(
                    row.get('speed_mps')) else 0.0,
            ))

    print(f"  Found {len(true_positions)} ground truth positions")

    # ── 3. Inject blackouts ───────────────────────────────────────────────
    blackout_windows: list[BlackoutWindow] = []
    if inject_blackout:
        print("\n[3/6] Generating simulated GNSS blackouts...")
        blackout_windows = generate_scenario_blackouts(df)
        if blackout_windows:
            df_with_blackout = inject_multiple_blackouts(df.copy(), blackout_windows)
            for bo in blackout_windows:
                duration_s = bo.duration_ms / 1000.0
                print(f"  Injected blackout: {bo.label} "
                      f"(start={bo.start_time_ms}, duration={duration_s:.1f}s)")
        else:
            print("  WARNING: Drive log too short for meaningful blackout injection")
            df_with_blackout = df.copy()
    else:
        print("\n[3/6] Skipping blackout injection (--no-inject-blackout)")
        df_with_blackout = df.copy()

    # ── 4. Run inference replay ───────────────────────────────────────────
    print(f"\n[4/6] Replaying through model: {model_path}")

    config = ReplayConfig(
        model_path=model_path,
        stats_path=stats_path,
    )
    replayer = RouteReplayer(config)
    replay_result = replayer.replay(df_with_blackout)

    print(f"  Generated {len(replay_result.inference_results)} inference results")

    # Build estimated trajectory from replay results
    # In the full pipeline, positions would come from the UKF fusion.
    # For now, we use available GPS (passthrough when available, dead-reckoned
    # positions when GPS is NaN would come from fusion).
    est_positions = []
    for ep in replay_result.estimated_positions:
        if ep['lat'] is not None and ep['lon'] is not None:
            est_positions.append(TrajectoryPoint(
                timestamp_ms=ep['timestamp_ms'],
                lat=ep['lat'],
                lon=ep['lon'],
                speed_mps=abs(ep.get('speed_metric', 0.0)),
            ))

    # ── 5. Compute drift metrics ─────────────────────────────────────────
    print("\n[5/6] Computing drift metrics...")

    session_result = SessionResult(session_id=session_name)

    for bo in blackout_windows:
        # Extract ground truth during this blackout
        bo_true = [p for p in true_positions
                   if bo.start_time_ms <= p.timestamp_ms <= bo.end_time_ms]
        bo_est = [p for p in est_positions
                  if bo.start_time_ms <= p.timestamp_ms <= bo.end_time_ms]

        true_pos_tuples = [(p.lat, p.lon) for p in bo_true]
        est_pos_tuples = [(p.lat, p.lon) for p in bo_est]

        # Compute true distance
        from drift_calculator import compute_path_distance
        true_dist = compute_path_distance(true_pos_tuples)

        # Compute final error
        final_error = 0.0
        if true_pos_tuples and est_pos_tuples:
            from drift_calculator import haversine_m
            final_error = haversine_m(
                true_pos_tuples[-1][0], true_pos_tuples[-1][1],
                est_pos_tuples[-1][0], est_pos_tuples[-1][1],
            )

        # Average speed
        avg_speed = 0.0
        speeds = [p.speed_mps for p in bo_true if p.speed_mps > 0]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)

        segment = BlackoutSegment(
            start_time_ms=bo.start_time_ms,
            end_time_ms=bo.end_time_ms,
            true_positions=true_pos_tuples,
            estimated_positions=est_pos_tuples,
            true_distance_m=true_dist,
            final_error_m=final_error,
            avg_speed_mps=avg_speed,
        )
        session_result.segments.append(segment)

    report = compute_drift_report([session_result])

    # ── 6. Generate reports ──────────────────────────────────────────────
    print(f"\n[6/6] Generating reports to: {output_dir}")

    true_pos_dicts = [
        {"timestamp_ms": p.timestamp_ms, "lat": p.lat, "lon": p.lon}
        for p in true_positions
    ]
    est_pos_dicts = [
        {"timestamp_ms": p.timestamp_ms, "lat": p.lat, "lon": p.lon}
        for p in est_positions
    ]
    bo_dicts = [
        {"start_time_ms": bo.start_time_ms, "end_time_ms": bo.end_time_ms}
        for bo in blackout_windows
    ]

    output_files = generate_full_report(
        report=report,
        true_positions=true_pos_dicts,
        estimated_positions=est_pos_dicts,
        blackout_windows=bo_dicts,
        output_dir=output_dir,
        session_name=session_name,
    )

    print(f"\n  CSV report: {output_files.get('csv_path', 'N/A')}")
    print(f"  PNG plot:   {output_files.get('png_path', 'N/A')}")
    print(f"{'='*70}\n")

    return {
        "csv_path": output_files.get("csv_path"),
        "png_path": output_files.get("png_path"),
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SIH PS-26168 — Dead Reckoning Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --drive-log fixtures/sample_drive_log.csv
  python run_benchmark.py --drive-log session.csv --model final_model.onnx --no-inject-blackout
        """,
    )
    parser.add_argument(
        "--drive-log", required=True,
        help="Path to drive log CSV file",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to ONNX model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--stats", default=DEFAULT_STATS,
        help=f"Path to normalization stats npz (default: {DEFAULT_STATS})",
    )
    parser.add_argument(
        "--inject-blackout", action="store_true", default=True,
        help="Inject simulated GNSS blackouts (default: True)",
    )
    parser.add_argument(
        "--no-inject-blackout", action="store_false", dest="inject_blackout",
        help="Skip blackout injection (use natural GPS gaps in the log)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    result = run_benchmark(
        drive_log_path=args.drive_log,
        model_path=args.model,
        stats_path=args.stats,
        inject_blackout=args.inject_blackout,
        output_dir=args.output_dir,
    )

    # Exit with 0 if benchmark ran (even if drift is bad — that's informational)
    sys.exit(0 if result["report"] is not None else 1)


if __name__ == "__main__":
    main()
