"""
real_outage_evaluator.py — Evaluate against real GNSS outage windows

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Uses the GPS-outage index file from the IO-VNBD dataset (if found) to
evaluate drift against real, labeled GNSS outage windows.  Falls back
to simulated outages with clear labeling if the index isn't available.

References:
    PRD §6.11, api-contracts.md §12
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "simulator"))

from drift_calculator import (
    BlackoutSegment, SessionResult, compute_drift_report,
    DriftReport, haversine_m,
)
from ground_truth_comparator import TrajectoryPoint, extract_blackout_comparison
from gnss_outage_simulator import (
    BlackoutWindow, generate_scenario_blackouts, inject_blackout,
)

# Expected path for the GPS outage index file (location unconfirmed — PRD §6.11)
DEFAULT_OUTAGE_INDEX_PATH = os.path.join(
    "ml-pipeline", "data", "raw", "iovnbd",
    "synchronised", "categorised", "gps_outages.csv"
)


def load_outage_index(
    index_path: str = DEFAULT_OUTAGE_INDEX_PATH,
) -> Optional[pd.DataFrame]:
    """
    Attempt to load the GPS-outage index file.

    Returns None if the file doesn't exist (location unconfirmed per §6.11).
    """
    if not os.path.isfile(index_path):
        return None

    try:
        df = pd.read_csv(index_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(index_path, encoding='latin-1')

    df.columns = df.columns.str.strip()
    return df


def extract_real_outage_windows(
    outage_df: pd.DataFrame,
    session_id: str,
) -> list[BlackoutWindow]:
    """
    Extract real outage windows for a specific session from the index file.

    The exact column names depend on the dataset's format (unconfirmed).
    This implements best-effort parsing based on the dataset paper's
    description.
    """
    windows = []

    # Try different possible column name conventions
    session_col = None
    for candidate in ['session_id', 'session', 'file', 'filename']:
        if candidate in outage_df.columns:
            session_col = candidate
            break

    if session_col is None:
        return windows

    session_rows = outage_df[outage_df[session_col].astype(str).str.contains(
        session_id, case=False, na=False
    )]

    start_col = None
    for candidate in ['start_time', 'start_ms', 'outage_start', 'start']:
        if candidate in outage_df.columns:
            start_col = candidate
            break

    end_col = None
    for candidate in ['end_time', 'end_ms', 'outage_end', 'end', 'duration']:
        if candidate in outage_df.columns:
            end_col = candidate
            break

    if start_col is None:
        return windows

    for _, row in session_rows.iterrows():
        start = int(row[start_col])
        if end_col and end_col != 'duration':
            end = int(row[end_col])
            duration = end - start
        elif end_col == 'duration':
            duration = int(row[end_col])
        else:
            duration = 60_000  # default 60s if no end/duration

        windows.append(BlackoutWindow(
            start_time_ms=start,
            duration_ms=duration,
            label="real_from_index",
        ))

    return windows


def evaluate_session(
    drive_log_path: str,
    true_positions: list[TrajectoryPoint],
    estimated_positions: list[TrajectoryPoint],
    outage_index_path: str = DEFAULT_OUTAGE_INDEX_PATH,
    session_id: Optional[str] = None,
) -> tuple[SessionResult, str]:
    """
    Evaluate drift for a single drive session.

    Tries real outage windows first; falls back to simulated.

    Args:
        drive_log_path: Path to the drive log CSV.
        true_positions: Ground truth trajectory.
        estimated_positions: DR-estimated trajectory.
        outage_index_path: Path to the GPS outage index file.
        session_id: Session identifier for matching in the index.

    Returns:
        (SessionResult, source_label) where source_label is
        "real_from_index" or "simulated".
    """
    sid = session_id or Path(drive_log_path).stem
    result = SessionResult(session_id=sid)

    # Try loading real outage index
    outage_df = load_outage_index(outage_index_path)
    source_label = "simulated"
    blackout_windows: list[BlackoutWindow] = []

    if outage_df is not None:
        blackout_windows = extract_real_outage_windows(outage_df, sid)
        if blackout_windows:
            source_label = "real_from_index"
            print(f"  Using {len(blackout_windows)} REAL outage window(s) "
                  f"from index for session '{sid}'")

    # Fall back to simulated if no real outages found
    if not blackout_windows:
        try:
            df = pd.read_csv(drive_log_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(drive_log_path, encoding='latin-1')
        df.columns = df.columns.str.strip()

        if 'timestamp_ms' not in df.columns and 'time' in df.columns:
            if df['time'].max() < 2e10:
                df['timestamp_ms'] = (df['time'] * 1000).astype(int)
            else:
                df['timestamp_ms'] = df['time'].astype(int)

        blackout_windows = generate_scenario_blackouts(df)
        source_label = "simulated"
        print(f"  Using {len(blackout_windows)} SIMULATED outage window(s) "
              f"for session '{sid}' (real index not found)")

    # Evaluate each blackout window
    for bo in blackout_windows:
        comparison = extract_blackout_comparison(
            true_positions, estimated_positions,
            bo.start_time_ms, bo.end_time_ms,
        )

        # Build true/estimated position tuples for the segment
        true_pos_tuples = [
            (p.true_lat, p.true_lon)
            for p in comparison.aligned_pairs
        ]
        est_pos_tuples = [
            (p.est_lat, p.est_lon)
            for p in comparison.aligned_pairs
        ]

        avg_speed = 0.0
        speed_points = [p.true_speed_mps for p in comparison.aligned_pairs
                        if p.true_speed_mps > 0]
        if speed_points:
            avg_speed = sum(speed_points) / len(speed_points)

        segment = BlackoutSegment(
            start_time_ms=bo.start_time_ms,
            end_time_ms=bo.end_time_ms,
            true_positions=true_pos_tuples,
            estimated_positions=est_pos_tuples,
            true_distance_m=comparison.true_path_distance_m,
            final_error_m=comparison.final_error_m,
            avg_speed_mps=avg_speed,
        )
        result.segments.append(segment)

    return result, source_label
