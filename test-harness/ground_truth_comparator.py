"""
ground_truth_comparator.py — Align and compare estimated vs true trajectories

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Aligns an estimated trajectory against ground truth by timestamp, computes
per-point position errors, and extracts reacquisition-point errors for
drift_calculator.py to compute the PS benchmark metric.

References:
    PRD §6.10, api-contracts.md §11
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryPoint:
    """A single point in a trajectory (ground truth or estimated)."""
    timestamp_ms: int
    lat: float
    lon: float
    speed_mps: float = 0.0
    heading_deg: float = 0.0


@dataclass
class AlignedPair:
    """A matched pair of ground-truth and estimated points at close timestamps."""
    timestamp_ms: int
    true_lat: float
    true_lon: float
    est_lat: float
    est_lon: float
    error_m: float
    true_speed_mps: float = 0.0


@dataclass
class ComparisonResult:
    """Result of comparing two trajectories over a time interval."""
    aligned_pairs: list[AlignedPair]
    final_error_m: float
    max_error_m: float
    mean_error_m: float
    true_path_distance_m: float
    start_time_ms: int
    end_time_ms: int


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two WGS-84 points, in meters."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def align_trajectories(
    true_traj: list[TrajectoryPoint],
    est_traj: list[TrajectoryPoint],
    max_time_diff_ms: int = 200,
) -> list[AlignedPair]:
    """
    Align estimated trajectory points to ground truth by nearest timestamp.

    For each ground truth point, find the nearest estimated point within
    max_time_diff_ms. Uses a two-pointer sweep for O(n+m) performance.

    Args:
        true_traj: Ground truth trajectory, sorted by timestamp.
        est_traj: Estimated trajectory, sorted by timestamp.
        max_time_diff_ms: Maximum allowable timestamp difference for a match.

    Returns:
        List of AlignedPair, sorted by timestamp.
    """
    if not true_traj or not est_traj:
        return []

    pairs = []
    j = 0  # pointer into est_traj

    for tp in true_traj:
        # Advance j to the closest estimated point
        while (j + 1 < len(est_traj) and
               abs(est_traj[j + 1].timestamp_ms - tp.timestamp_ms) <=
               abs(est_traj[j].timestamp_ms - tp.timestamp_ms)):
            j += 1

        time_diff = abs(est_traj[j].timestamp_ms - tp.timestamp_ms)
        if time_diff > max_time_diff_ms:
            continue

        error = haversine_m(tp.lat, tp.lon, est_traj[j].lat, est_traj[j].lon)
        pairs.append(AlignedPair(
            timestamp_ms=tp.timestamp_ms,
            true_lat=tp.lat,
            true_lon=tp.lon,
            est_lat=est_traj[j].lat,
            est_lon=est_traj[j].lon,
            error_m=error,
            true_speed_mps=tp.speed_mps,
        ))

    return pairs


def compute_path_distance(points: list[TrajectoryPoint]) -> float:
    """Compute total path distance from a sequence of TrajectoryPoints."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        total += haversine_m(
            points[i - 1].lat, points[i - 1].lon,
            points[i].lat, points[i].lon
        )
    return total


def compare_trajectories(
    true_traj: list[TrajectoryPoint],
    est_traj: list[TrajectoryPoint],
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    max_time_diff_ms: int = 200,
) -> ComparisonResult:
    """
    Compare estimated trajectory against ground truth over a time window.

    Args:
        true_traj: Full ground truth trajectory.
        est_traj: Full estimated trajectory.
        start_time_ms: Optional start of comparison window (inclusive).
        end_time_ms: Optional end of comparison window (inclusive).
        max_time_diff_ms: Max timestamp difference for alignment.

    Returns:
        ComparisonResult with per-point errors and summary statistics.
    """
    # Filter to time window if specified
    if start_time_ms is not None:
        true_traj = [p for p in true_traj if p.timestamp_ms >= start_time_ms]
        est_traj = [p for p in est_traj if p.timestamp_ms >= start_time_ms]
    if end_time_ms is not None:
        true_traj = [p for p in true_traj if p.timestamp_ms <= end_time_ms]
        est_traj = [p for p in est_traj if p.timestamp_ms <= end_time_ms]

    # Sort by time
    true_traj = sorted(true_traj, key=lambda p: p.timestamp_ms)
    est_traj = sorted(est_traj, key=lambda p: p.timestamp_ms)

    if not true_traj or not est_traj:
        return ComparisonResult(
            aligned_pairs=[],
            final_error_m=0.0,
            max_error_m=0.0,
            mean_error_m=0.0,
            true_path_distance_m=0.0,
            start_time_ms=start_time_ms or 0,
            end_time_ms=end_time_ms or 0,
        )

    pairs = align_trajectories(true_traj, est_traj, max_time_diff_ms)

    if not pairs:
        return ComparisonResult(
            aligned_pairs=[],
            final_error_m=0.0,
            max_error_m=0.0,
            mean_error_m=0.0,
            true_path_distance_m=compute_path_distance(true_traj),
            start_time_ms=true_traj[0].timestamp_ms,
            end_time_ms=true_traj[-1].timestamp_ms,
        )

    errors = [p.error_m for p in pairs]

    return ComparisonResult(
        aligned_pairs=pairs,
        final_error_m=pairs[-1].error_m,
        max_error_m=max(errors),
        mean_error_m=sum(errors) / len(errors),
        true_path_distance_m=compute_path_distance(true_traj),
        start_time_ms=pairs[0].timestamp_ms,
        end_time_ms=pairs[-1].timestamp_ms,
    )


def extract_blackout_comparison(
    true_traj: list[TrajectoryPoint],
    est_traj: list[TrajectoryPoint],
    blackout_start_ms: int,
    blackout_end_ms: int,
) -> ComparisonResult:
    """
    Extract the comparison specifically over a GNSS blackout window.

    This is the key input to drift_calculator.compute_drift_pct():
    - final_error_m at reacquisition
    - true_path_distance_m during the blackout

    Args:
        true_traj: Full ground truth (has positions even during blackout).
        est_traj: Estimated trajectory (DR-only during blackout).
        blackout_start_ms: Start of GNSS outage.
        blackout_end_ms: End of GNSS outage (reacquisition point).
    """
    return compare_trajectories(
        true_traj, est_traj,
        start_time_ms=blackout_start_ms,
        end_time_ms=blackout_end_ms,
    )
