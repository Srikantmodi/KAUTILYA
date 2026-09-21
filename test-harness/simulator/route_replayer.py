"""
route_replayer.py — Replay a drive log through the inference pipeline

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Replays a drive log CSV through the ML inference pipeline at real timestamps,
feeding samples one-by-one to the model and respecting the 100ms (10Hz)
inference interval.

This module operates in "offline replay" mode — it reads all data from a CSV
file and processes it sequentially, simulating the real-time pipeline flow
without actual wall-clock delays.

References:
    PRD §6.13, api-contracts.md §8 (inference contract)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ReplayConfig:
    """Configuration for a route replay session."""
    model_path: str
    stats_path: str
    window_size: int = 20           # 20 samples = 2 seconds at 10Hz
    inference_interval_ms: int = 100  # 100ms between inference calls
    gru_hidden_size: int = 64


@dataclass
class InferenceResult:
    """Result from a single inference step."""
    timestamp_ms: int
    speed_metric: float      # delta-v (placeholder) or absolute speed (final)
    stationary_prob: float   # sigmoid-applied probability [0, 1]


@dataclass
class ReplayResult:
    """Result of replaying an entire drive log."""
    inference_results: list[InferenceResult] = field(default_factory=list)
    estimated_positions: list[dict] = field(default_factory=list)
    session_id: str = ""


# ─── Normalization ───────────────────────────────────────────────────────────

def load_normalization_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load normalization mean/std from the npz file.

    Returns:
        (mean, std) arrays of shape (6,), dtype float32
    """
    data = np.load(stats_path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    return mean, std


def normalize_window(window: np.ndarray, mean: np.ndarray,
                     std: np.ndarray) -> np.ndarray:
    """
    Apply per-feature normalization: x_norm = (x - mean) / std

    Args:
        window: shape (20, 6), raw feature values
        mean: shape (6,), per-feature means
        std: shape (6,), per-feature standard deviations

    Returns:
        Normalized window, same shape
    """
    return (window - mean) / std


def sigmoid(x: float) -> float:
    """Apply sigmoid: 1 / (1 + exp(-x))"""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        # Numerically stable for negative x
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


# ─── Drive log loading ──────────────────────────────────────────────────────

def load_drive_log(csv_path: str) -> pd.DataFrame:
    """
    Load a drive log CSV with all the dataset-notes.md gotchas handled:
    - Strip column name whitespace
    - Try UTF-8 first, fall back to latin-1
    - Normalize timestamp to epoch ms
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')

    df.columns = df.columns.str.strip()

    # Normalize timestamp column
    if 'timestamp_ms' not in df.columns and 'time' in df.columns:
        # Detect seconds vs milliseconds
        # If max time is < 2e10, it's in seconds (epoch seconds ~1.7e9)
        if df['time'].max() < 2e10:
            df['timestamp_ms'] = (df['time'] * 1000).astype(int)
        else:
            df['timestamp_ms'] = df['time'].astype(int)

    return df


def extract_features_from_row(row: pd.Series) -> Optional[np.ndarray]:
    """
    Extract the 6-feature vector from a drive log row.

    Feature order (LOCKED):
        [leveled_ax, leveled_ay, leveled_az, gyro_yaw, gyro_pitch, gyro_roll]

    NOTE: In the full pipeline, raw accel would be leveled via
    calibration.level_accelerometer(). Since we don't have the C++ bindings
    yet, we use raw accel as a pass-through proxy. The pipeline structure
    is correct — only the leveling transform is stubbed.
    """
    try:
        # Use raw accel as proxy for leveled (stub until M3 ships)
        features = np.array([
            float(row.get('ax', 0.0)),   # leveled_ax (stubbed)
            float(row.get('ay', 0.0)),   # leveled_ay (stubbed)
            float(row.get('az', 0.0)),   # leveled_az (stubbed)
            float(row.get('gx', 0.0)),   # gyro_yaw
            float(row.get('gy', 0.0)),   # gyro_pitch
            float(row.get('gz', 0.0)),   # gyro_roll
        ], dtype=np.float32)
        return features
    except (ValueError, TypeError):
        return None


# ─── Replay engine ──────────────────────────────────────────────────────────

class RouteReplayer:
    """
    Replays a drive log through the ONNX inference model, managing the
    sliding window and GRU hidden state correctly.
    """

    def __init__(self, config: ReplayConfig):
        self.config = config
        self.mean, self.std = load_normalization_stats(config.stats_path)

        # Import onnxruntime here so module-level import doesn't fail
        # if onnxruntime isn't installed (e.g. in unit tests using mocks)
        import onnxruntime as ort
        self.session = ort.InferenceSession(config.model_path)

        # GRU hidden state — persistent across calls, reset only at session start
        self.h_state = np.zeros(
            (1, 1, config.gru_hidden_size), dtype=np.float32
        )

        # Sliding window FIFO
        self.window: list[np.ndarray] = []

    def reset_session(self):
        """Reset GRU hidden state — call ONLY at new session start."""
        self.h_state = np.zeros(
            (1, 1, self.config.gru_hidden_size), dtype=np.float32
        )
        self.window.clear()

    def push_sample(self, features: np.ndarray) -> Optional[InferenceResult]:
        """
        Push one sample into the sliding window. If the window is full (20),
        run inference and return the result.

        Args:
            features: shape (6,) raw feature vector (will be normalized)

        Returns:
            InferenceResult if window is full and inference ran, None otherwise
        """
        self.window.append(features.copy())

        # Maintain FIFO: keep only the last window_size samples
        if len(self.window) > self.config.window_size:
            self.window.pop(0)

        # Only run inference when we have a full window
        if len(self.window) < self.config.window_size:
            return None

        return self._run_inference()

    def _run_inference(self) -> InferenceResult:
        """Run model inference on the current full window."""
        # Stack window into (20, 6) array
        window_arr = np.stack(self.window, axis=0)  # (20, 6)

        # Normalize
        window_norm = normalize_window(window_arr, self.mean, self.std)

        # Reshape to (1, 20, 6) for batch dim
        input_features = window_norm.reshape(1, 20, 6).astype(np.float32)

        # Run ONNX inference
        outputs = self.session.run(
            None,
            {
                "input_features": input_features,
                "h_in": self.h_state,
            }
        )

        speed_metric_all = outputs[0]       # (1, 20, 1)
        stationary_logit_all = outputs[1]   # (1, 20, 1)
        h_out = outputs[2]                  # (1, 1, 64)

        # Take LAST timestep (causal model — only final has full context)
        speed_metric = float(speed_metric_all[0, -1, 0])
        stationary_logit = float(stationary_logit_all[0, -1, 0])

        # Apply sigmoid to stationary logit
        stationary_prob = sigmoid(stationary_logit)

        # Update persistent hidden state — NEVER reset here
        self.h_state = h_out

        return InferenceResult(
            timestamp_ms=0,  # caller sets this
            speed_metric=speed_metric,
            stationary_prob=stationary_prob,
        )

    def replay(self, df: pd.DataFrame) -> ReplayResult:
        """
        Replay an entire drive log through the inference pipeline.

        Processes samples at 10Hz intervals, feeding each through the model
        and collecting inference results.

        Args:
            df: Drive log DataFrame with timestamp_ms and IMU columns.

        Returns:
            ReplayResult with per-inference-step results.
        """
        self.reset_session()
        result = ReplayResult()

        if df.empty:
            return result

        last_inference_ms = 0
        interval = self.config.inference_interval_ms

        for _, row in df.iterrows():
            ts = int(row.get('timestamp_ms', 0))

            # Enforce 10Hz interval
            if last_inference_ms > 0 and (ts - last_inference_ms) < interval:
                continue

            features = extract_features_from_row(row)
            if features is None:
                continue

            inf_result = self.push_sample(features)
            if inf_result is not None:
                inf_result.timestamp_ms = ts
                result.inference_results.append(inf_result)
                last_inference_ms = ts

                # Store estimated position (passthrough from GNSS if available,
                # or would come from fusion in full pipeline)
                lat = row.get('lat', row.get('latitude', np.nan))
                lon = row.get('lon', row.get('longitude', np.nan))
                result.estimated_positions.append({
                    "timestamp_ms": ts,
                    "lat": float(lat) if pd.notna(lat) else None,
                    "lon": float(lon) if pd.notna(lon) else None,
                    "speed_metric": inf_result.speed_metric,
                    "stationary_prob": inf_result.stationary_prob,
                })

        return result
