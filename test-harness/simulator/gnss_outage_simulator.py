"""
gnss_outage_simulator.py — Inject synthetic GNSS blackouts into drive logs

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Injects one or more synthetic GNSS blackout windows into a drive log by
NaN-filling the GPS columns during the specified intervals.  All output
from this simulator must be clearly labeled as "SIMULATED" (PRD §6.11).

References:
    PRD §6.11 (fallback when real outage index isn't found),
    PRD §6.13
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class BlackoutWindow:
    """Definition of a single GNSS blackout to inject."""
    start_time_ms: int
    duration_ms: int
    label: str = "simulated"  # always "simulated" — never unlabeled

    @property
    def end_time_ms(self) -> int:
        return self.start_time_ms + self.duration_ms


# GPS columns that get NaN-filled during a blackout
GPS_COLUMNS = ["lat", "lon", "speed_mps", "accuracy_m", "satellites", "bearing_deg"]

# Alternate column name mappings (dataset inconsistency — strip whitespace first)
GPS_COLUMN_ALIASES = {
    "latitude": "lat",
    "longitude": "lon",
    "speed": "speed_mps",
    "accuracy": "accuracy_m",
    "satellite_count": "satellites",
    "bearing": "bearing_deg",
}


def _normalize_gps_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure GPS columns use canonical names."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {}
    for alias, canonical in GPS_COLUMN_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def inject_blackout(
    df: pd.DataFrame,
    blackout: BlackoutWindow,
    timestamp_col: str = "timestamp_ms",
) -> pd.DataFrame:
    """
    Inject a single GNSS blackout into a drive log DataFrame.

    NaN-fills all GPS columns during the blackout window while preserving
    the IMU columns (accel, gyro) untouched.

    Args:
        df: Drive log DataFrame with a timestamp column and GPS columns.
        blackout: BlackoutWindow specifying start and duration.
        timestamp_col: Name of the timestamp column (epoch ms).

    Returns:
        Modified DataFrame with GPS columns NaN-filled during the blackout.
    """
    df = _normalize_gps_columns(df)

    mask = (
        (df[timestamp_col] >= blackout.start_time_ms) &
        (df[timestamp_col] < blackout.end_time_ms)
    )

    for col in GPS_COLUMNS:
        if col in df.columns:
            df.loc[mask, col] = np.nan

    return df


def inject_multiple_blackouts(
    df: pd.DataFrame,
    blackouts: list[BlackoutWindow],
    timestamp_col: str = "timestamp_ms",
) -> pd.DataFrame:
    """Inject multiple blackouts into a drive log."""
    df = _normalize_gps_columns(df)
    for bo in blackouts:
        df = inject_blackout(df, bo, timestamp_col)
    return df


def generate_periodic_blackouts(
    start_time_ms: int,
    end_time_ms: int,
    blackout_duration_ms: int = 60_000,
    gap_between_ms: int = 30_000,
) -> list[BlackoutWindow]:
    """
    Generate a series of periodic blackout windows spanning a time range.

    Useful for systematic benchmark evaluation across a full drive.

    Args:
        start_time_ms: Start of the first potential blackout.
        end_time_ms: End of the drive (no blackout starts after this).
        blackout_duration_ms: Duration of each blackout.
        gap_between_ms: Gap between end of one blackout and start of next.

    Returns:
        List of BlackoutWindow objects.
    """
    blackouts = []
    t = start_time_ms
    idx = 0
    while t + blackout_duration_ms <= end_time_ms:
        blackouts.append(BlackoutWindow(
            start_time_ms=t,
            duration_ms=blackout_duration_ms,
            label=f"simulated_periodic_{idx}",
        ))
        t += blackout_duration_ms + gap_between_ms
        idx += 1
    return blackouts


def generate_scenario_blackouts(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp_ms",
) -> list[BlackoutWindow]:
    """
    Generate blackouts matching the PS benchmark scenarios:
    1. Short: < 1 minute (50m at low speed)
    2. Long: ~1 km at 60 km/h (~60 seconds)

    Selects suitable windows from the drive based on available data length.
    """
    if df.empty:
        return []

    df = _normalize_gps_columns(df)
    t_start = int(df[timestamp_col].min())
    t_end = int(df[timestamp_col].max())
    duration = t_end - t_start

    blackouts = []

    # Scenario 1: short blackout — 30 seconds, starting 20% into the drive
    short_start = t_start + int(duration * 0.2)
    short_duration = min(30_000, int(duration * 0.3))
    if short_duration > 5000:
        blackouts.append(BlackoutWindow(
            start_time_ms=short_start,
            duration_ms=short_duration,
            label="simulated_short_<1min",
        ))

    # Scenario 2: long blackout — 60 seconds, starting 60% into the drive
    long_start = t_start + int(duration * 0.6)
    long_duration = min(60_000, int(duration * 0.3))
    if long_duration > 10_000:
        blackouts.append(BlackoutWindow(
            start_time_ms=long_start,
            duration_ms=long_duration,
            label="simulated_long_~1km",
        ))

    return blackouts
