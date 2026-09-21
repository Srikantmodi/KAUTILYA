"""
report_generator.py — Generate benchmark reports (CSV + PNG)

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Takes a DriftReport from drift_calculator and generates:
    1. CSV of per-segment results
    2. Position-plot PNG (true vs. estimated trajectory, blackout regions shaded)
    3. Summary table to stdout

References:
    PRD §6.13, api-contracts.md §14
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

# Attempt matplotlib import — fail gracefully if not installed
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for PNG generation
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from drift_calculator import DriftReport, format_report_summary


def generate_csv_report(
    report: DriftReport,
    output_path: str,
) -> str:
    """
    Write per-segment drift results to a CSV file.

    Columns:
        session_id, start_time_ms, end_time_ms, duration_ms,
        duration_bucket, speed_regime, avg_speed_mps,
        true_distance_m, final_error_m, drift_pct, pass_10pct

    Args:
        report: DriftReport from drift_calculator.compute_drift_report()
        output_path: Path for the output CSV file.

    Returns:
        Absolute path of the written CSV.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "session_id", "start_time_ms", "end_time_ms", "duration_ms",
        "duration_bucket", "speed_regime", "avg_speed_mps",
        "true_distance_m", "final_error_m", "drift_pct", "pass_10pct",
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in report.all_segments:
            row = {k: seg.get(k, "") for k in fieldnames}
            writer.writerow(row)

    return os.path.abspath(output_path)


def generate_position_plot(
    true_positions: list[dict],
    estimated_positions: list[dict],
    blackout_windows: Optional[list[dict]] = None,
    output_path: str = "trajectory_plot.png",
    title: str = "True vs Estimated Trajectory",
) -> Optional[str]:
    """
    Generate a position-plot PNG overlaying true vs. estimated trajectory.

    Blackout regions are shaded in red to visually indicate GNSS-denied
    segments.

    Args:
        true_positions: List of dicts with 'lat', 'lon', 'timestamp_ms'.
        estimated_positions: Same format as true_positions.
        blackout_windows: List of dicts with 'start_time_ms', 'end_time_ms'.
        output_path: Path for the output PNG.
        title: Plot title.

    Returns:
        Absolute path of the PNG, or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not installed — skipping position plot. "
              "Install with: pip install matplotlib")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left panel: spatial trajectory ──────────────────────────────────────
    ax1 = axes[0]

    # Filter out None/NaN positions
    true_lats = [p['lat'] for p in true_positions
                 if p.get('lat') is not None]
    true_lons = [p['lon'] for p in true_positions
                 if p.get('lon') is not None]
    est_lats = [p['lat'] for p in estimated_positions
                if p.get('lat') is not None]
    est_lons = [p['lon'] for p in estimated_positions
                if p.get('lon') is not None]

    if true_lats and true_lons:
        ax1.plot(true_lons, true_lats, 'b-', linewidth=2, label='Ground Truth',
                 alpha=0.8)
    if est_lats and est_lons:
        ax1.plot(est_lons, est_lats, 'r--', linewidth=1.5, label='Estimated (DR)',
                 alpha=0.8)

    # Mark blackout start/end on spatial plot
    if blackout_windows and true_positions:
        ts_to_pos = {p['timestamp_ms']: (p.get('lat'), p.get('lon'))
                     for p in true_positions
                     if p.get('lat') is not None}

        for bo in blackout_windows:
            start_ts = bo['start_time_ms']
            end_ts = bo['end_time_ms']

            # Find nearest positions to blackout boundaries
            for ts, marker, color in [(start_ts, 'v', 'red'), (end_ts, '^', 'green')]:
                nearest_ts = min(ts_to_pos.keys(), key=lambda t: abs(t - ts),
                                 default=None)
                if nearest_ts is not None:
                    pos = ts_to_pos[nearest_ts]
                    ax1.plot(pos[1], pos[0], marker=marker, color=color,
                             markersize=10, zorder=5)

    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title('Spatial Trajectory')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ── Right panel: position error over time ──────────────────────────────
    ax2 = axes[1]

    # Compute per-point error if both trajectories have timestamps
    from drift_calculator import haversine_m

    true_by_ts = {p['timestamp_ms']: p for p in true_positions
                  if p.get('lat') is not None}
    est_by_ts = {p['timestamp_ms']: p for p in estimated_positions
                 if p.get('lat') is not None}

    common_ts = sorted(set(true_by_ts.keys()) & set(est_by_ts.keys()))
    if common_ts:
        errors = []
        times_sec = []
        t0 = common_ts[0]
        for ts in common_ts:
            tp = true_by_ts[ts]
            ep = est_by_ts[ts]
            err = haversine_m(tp['lat'], tp['lon'], ep['lat'], ep['lon'])
            errors.append(err)
            times_sec.append((ts - t0) / 1000.0)

        ax2.plot(times_sec, errors, 'k-', linewidth=1.5)
        ax2.set_ylabel('Position Error (m)')
    else:
        ax2.text(0.5, 0.5, 'No overlapping timestamps',
                 transform=ax2.transAxes, ha='center', va='center')

    # Shade blackout regions
    if blackout_windows and common_ts:
        t0 = common_ts[0]
        for bo in blackout_windows:
            start_s = (bo['start_time_ms'] - t0) / 1000.0
            end_s = (bo['end_time_ms'] - t0) / 1000.0
            ax2.axvspan(start_s, end_s, alpha=0.2, color='red',
                        label='GNSS Blackout')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Position Error During Drive')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return os.path.abspath(output_path)


def generate_full_report(
    report: DriftReport,
    true_positions: list[dict],
    estimated_positions: list[dict],
    blackout_windows: Optional[list[dict]] = None,
    output_dir: str = "benchmark_output",
    session_name: str = "benchmark",
) -> dict[str, Optional[str]]:
    """
    Generate the complete benchmark report package:
        1. CSV of per-segment drift results
        2. Position-plot PNG
        3. Print summary to stdout

    Args:
        report: DriftReport from drift_calculator
        true_positions: Ground truth positions [{timestamp_ms, lat, lon}, ...]
        estimated_positions: DR estimated positions
        blackout_windows: [{start_time_ms, end_time_ms}, ...]
        output_dir: Directory for output files
        session_name: Prefix for output filenames

    Returns:
        Dict with 'csv_path' and 'png_path' (None if matplotlib unavailable)
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = generate_csv_report(
        report,
        os.path.join(output_dir, f"{session_name}_drift_results.csv"),
    )

    png_path = generate_position_plot(
        true_positions,
        estimated_positions,
        blackout_windows,
        os.path.join(output_dir, f"{session_name}_trajectory.png"),
        title=f"Trajectory: {session_name}",
    )

    # Print human-readable summary
    print(format_report_summary(report))

    return {"csv_path": csv_path, "png_path": png_path}
