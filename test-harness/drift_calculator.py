"""
drift_calculator.py — PS-26168 drift metric computation

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Implements the exact drift metric from the PS benchmarks:
    drift_pct = (final_position_error_at_reacquisition /
                 true_distance_travelled_during_blackout) * 100

Results are ALWAYS broken out by (blackout_duration_bucket, speed_regime) —
never a single pooled number (PRD §7 guardrail #3).

RMSE is included as a diagnostic field but NEVER as the pass/fail criterion.

TODO: Once Member 4 ships core_bindings.cpp, route the drift math through
pybind11 bindings so there is exactly ONE implementation (PRD §7 guardrail #1).
For now, the Python implementation is the single source.

References:
    PRD §6.10, api-contracts.md §11, PRD §7 guardrail #3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class BlackoutSegment:
    """A single GNSS blackout episode within a drive session."""
    start_time_ms: int
    end_time_ms: int
    true_positions: list[tuple[float, float]]  # [(lat, lon), ...] ground truth
    estimated_positions: list[tuple[float, float]]  # [(lat, lon), ...] DR output
    true_distance_m: float = 0.0   # total ground truth distance during blackout
    final_error_m: float = 0.0     # position error at reacquisition point
    avg_speed_mps: float = 0.0     # average true speed during blackout


@dataclass
class SessionResult:
    """Aggregated result for one drive session."""
    session_id: str
    segments: list[BlackoutSegment] = field(default_factory=list)


@dataclass
class DriftReport:
    """Drift report broken out by duration bucket and speed regime."""
    by_duration: dict[str, list[float]] = field(default_factory=dict)
    by_speed: dict[str, list[float]] = field(default_factory=dict)
    by_duration_and_speed: dict[str, list[float]] = field(default_factory=dict)
    all_segments: list[dict] = field(default_factory=list)
    rmse_diagnostic: Optional[float] = None


# ─── Duration buckets and speed regimes (match PS benchmark scales) ──────────

DURATION_BUCKETS = {
    "short_<30s": (0, 30_000),
    "medium_30s-2min": (30_000, 120_000),
    "long_2min-5min": (120_000, 300_000),
    "very_long_>5min": (300_000, float('inf')),
}

SPEED_REGIMES = {
    "stationary_<1mps": (0.0, 1.0),
    "low_1-10mps": (1.0, 10.0),
    "medium_10-20mps": (10.0, 20.0),        # ~36-72 km/h
    "high_>20mps": (20.0, float('inf')),     # >72 km/h  (60 km/h benchmark)
}


# ─── Core functions ──────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance between two WGS-84 points, in meters.

    Used for position error and distance computation when coordinates
    are in (lat, lon) degrees.
    """
    R = 6_371_000.0  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_drift_pct(final_position_error_m: float,
                      true_distance_travelled_m: float) -> float:
    """
    Compute drift percentage exactly as the PS defines it.

    drift_pct = (final_position_error / true_distance_travelled) * 100

    Returns float('inf') if true_distance is zero (stationary — no meaningful
    drift percentage can be computed).
    """
    if true_distance_travelled_m <= 0.0:
        return float('inf')
    return (final_position_error_m / true_distance_travelled_m) * 100.0


def compute_path_distance(positions: list[tuple[float, float]]) -> float:
    """
    Compute total path distance (sum of consecutive haversine segments)
    from a list of (lat, lon) positions.
    """
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        total += haversine_m(
            positions[i - 1][0], positions[i - 1][1],
            positions[i][0], positions[i][1]
        )
    return total


def compute_rmse(true_positions: list[tuple[float, float]],
                 estimated_positions: list[tuple[float, float]]) -> float:
    """
    Root-Mean-Square Error across aligned position pairs.

    This is a DIAGNOSTIC metric only — never the pass/fail criterion
    (PRD §7 guardrail #3).
    """
    if not true_positions or not estimated_positions:
        return 0.0

    n = min(len(true_positions), len(estimated_positions))
    if n == 0:
        return 0.0

    sum_sq = 0.0
    for i in range(n):
        err = haversine_m(
            true_positions[i][0], true_positions[i][1],
            estimated_positions[i][0], estimated_positions[i][1]
        )
        sum_sq += err ** 2

    return math.sqrt(sum_sq / n)


def _classify_duration(duration_ms: int) -> str:
    """Map a blackout duration to its bucket name."""
    for name, (lo, hi) in DURATION_BUCKETS.items():
        if lo <= duration_ms < hi:
            return name
    return "very_long_>5min"


def _classify_speed(avg_speed_mps: float) -> str:
    """Map an average speed to its regime name."""
    for name, (lo, hi) in SPEED_REGIMES.items():
        if lo <= avg_speed_mps < hi:
            return name
    return "high_>20mps"


def analyze_segment(segment: BlackoutSegment) -> dict:
    """
    Analyze a single blackout segment and populate its computed fields.

    Returns a dict with all metrics for this segment.
    """
    # Compute true distance if not already set
    if segment.true_distance_m <= 0.0 and len(segment.true_positions) >= 2:
        segment.true_distance_m = compute_path_distance(segment.true_positions)

    # Compute final position error if not already set
    if segment.final_error_m <= 0.0 and segment.true_positions and segment.estimated_positions:
        segment.final_error_m = haversine_m(
            segment.true_positions[-1][0], segment.true_positions[-1][1],
            segment.estimated_positions[-1][0], segment.estimated_positions[-1][1]
        )

    # Compute average speed if not already set
    duration_ms = segment.end_time_ms - segment.start_time_ms
    if segment.avg_speed_mps <= 0.0 and duration_ms > 0 and segment.true_distance_m > 0:
        segment.avg_speed_mps = segment.true_distance_m / (duration_ms / 1000.0)

    drift_pct = compute_drift_pct(segment.final_error_m, segment.true_distance_m)
    duration_bucket = _classify_duration(duration_ms)
    speed_regime = _classify_speed(segment.avg_speed_mps)

    return {
        "start_time_ms": segment.start_time_ms,
        "end_time_ms": segment.end_time_ms,
        "duration_ms": duration_ms,
        "duration_bucket": duration_bucket,
        "speed_regime": speed_regime,
        "avg_speed_mps": round(segment.avg_speed_mps, 2),
        "true_distance_m": round(segment.true_distance_m, 2),
        "final_error_m": round(segment.final_error_m, 2),
        "drift_pct": round(drift_pct, 4) if drift_pct != float('inf') else None,
        "pass_10pct": drift_pct < 10.0 if drift_pct != float('inf') else None,
    }


def compute_drift_report(sessions: list[SessionResult]) -> DriftReport:
    """
    Compute the full drift report, broken out by blackout duration bucket
    AND speed regime — NEVER a single pooled number (§7 guardrail #3).

    Args:
        sessions: List of SessionResult, each containing BlackoutSegments.

    Returns:
        DriftReport with by_duration, by_speed, by_duration_and_speed dicts,
        plus per-segment detail in all_segments.
    """
    report = DriftReport()
    all_errors_sq = []

    for session in sessions:
        for segment in session.segments:
            seg_result = analyze_segment(segment)
            report.all_segments.append({
                "session_id": session.session_id,
                **seg_result
            })

            drift = seg_result["drift_pct"]
            if drift is None:
                continue  # skip stationary/zero-distance segments

            # By duration
            bucket = seg_result["duration_bucket"]
            report.by_duration.setdefault(bucket, []).append(drift)

            # By speed
            regime = seg_result["speed_regime"]
            report.by_speed.setdefault(regime, []).append(drift)

            # By (duration, speed)
            combo = f"{bucket}|{regime}"
            report.by_duration_and_speed.setdefault(combo, []).append(drift)

            # For RMSE diagnostic
            all_errors_sq.append(segment.final_error_m ** 2)

    # Compute overall RMSE as diagnostic (NOT pass/fail)
    if all_errors_sq:
        report.rmse_diagnostic = math.sqrt(sum(all_errors_sq) / len(all_errors_sq))

    return report


def format_report_summary(report: DriftReport) -> str:
    """Format the drift report as a human-readable summary string."""
    lines = []
    lines.append("=" * 70)
    lines.append("DRIFT REPORT — Broken out by blackout duration & speed regime")
    lines.append("=" * 70)

    lines.append("\n--- By Blackout Duration ---")
    for bucket, drifts in sorted(report.by_duration.items()):
        avg = sum(drifts) / len(drifts)
        mx = max(drifts)
        mn = min(drifts)
        passing = sum(1 for d in drifts if d < 10.0)
        lines.append(
            f"  {bucket:25s}  n={len(drifts):3d}  "
            f"avg={avg:7.2f}%  min={mn:7.2f}%  max={mx:7.2f}%  "
            f"pass(<10%)={passing}/{len(drifts)}"
        )

    lines.append("\n--- By Speed Regime ---")
    for regime, drifts in sorted(report.by_speed.items()):
        avg = sum(drifts) / len(drifts)
        mx = max(drifts)
        mn = min(drifts)
        passing = sum(1 for d in drifts if d < 10.0)
        lines.append(
            f"  {regime:25s}  n={len(drifts):3d}  "
            f"avg={avg:7.2f}%  min={mn:7.2f}%  max={mx:7.2f}%  "
            f"pass(<10%)={passing}/{len(drifts)}"
        )

    if report.rmse_diagnostic is not None:
        lines.append(f"\n  RMSE (diagnostic only, NOT pass/fail): "
                     f"{report.rmse_diagnostic:.2f} m")

    lines.append("=" * 70)
    return "\n".join(lines)
