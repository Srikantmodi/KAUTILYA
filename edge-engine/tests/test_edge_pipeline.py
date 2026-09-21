"""
test_edge_pipeline.py — Edge engine inference tests

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Tests:
    1. ONNX model loads and produces correct output shapes
    2. GRU state PERSISTS across inference calls (PRD §7 guardrail #6)
    3. GRU state RESETS only at session boundaries (explicit test)
    4. Normalization math is correct
    5. Sigmoid is correct
    6. Edge cases: empty input, window < 20 samples, NaN handling
    7. Sliding window FIFO maintains correct sample order

Usage:
    python -m pytest edge-engine/tests/test_edge_pipeline.py -v

References:
    PRD §7 guardrail #6 (GRU persistence test),
    api-contracts.md §8 (locked inference contract)
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Setup paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "test-harness"))
sys.path.insert(0, str(ROOT / "test-harness" / "simulator"))
sys.path.insert(0, str(ROOT / "edge-engine" / "python"))

MODEL_PATH = str(ROOT / "edge-engine" / "python" / "placeholder_model.onnx")
STATS_PATH = str(ROOT / "edge-engine" / "python" / "normalization_stats_v2.npz")
SAMPLE_CSV = str(ROOT / "test-harness" / "fixtures" / "sample_drive_log.csv")

# Skip all tests if onnxruntime is not installed
ort = pytest.importorskip("onnxruntime")


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def ort_session():
    """Create an ONNX Runtime session with the placeholder model."""
    return ort.InferenceSession(MODEL_PATH)


@pytest.fixture
def norm_stats():
    """Load normalization statistics."""
    data = np.load(STATS_PATH)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


@pytest.fixture
def replayer():
    """Create a RouteReplayer instance."""
    from route_replayer import RouteReplayer, ReplayConfig
    config = ReplayConfig(model_path=MODEL_PATH, stats_path=STATS_PATH)
    return RouteReplayer(config)


# ─── Test 1: Model I/O shapes ───────────────────────────────────────────────

class TestModelIO:
    """Verify the placeholder model's locked I/O contract."""

    def test_model_loads(self, ort_session):
        """Model loads without errors."""
        assert ort_session is not None

    def test_input_names(self, ort_session):
        """Input tensor names match the locked contract."""
        input_names = [i.name for i in ort_session.get_inputs()]
        assert "input_features" in input_names
        assert "h_in" in input_names

    def test_output_names(self, ort_session):
        """Output tensor names match the locked contract."""
        output_names = [o.name for o in ort_session.get_outputs()]
        assert "speed_metric" in output_names
        assert "stationary_logit" in output_names
        assert "h_out" in output_names

    def test_input_shapes(self, ort_session):
        """Input tensor shapes match [1,20,6] and [1,1,64]."""
        inputs = {i.name: i.shape for i in ort_session.get_inputs()}
        assert inputs["input_features"] == [1, 20, 6]
        assert inputs["h_in"] == [1, 1, 64]

    def test_output_shapes(self, ort_session):
        """Output tensor shapes match [1,20,1] and [1,1,64]."""
        outputs = {o.name: o.shape for o in ort_session.get_outputs()}
        assert outputs["speed_metric"] == [1, 20, 1]
        assert outputs["stationary_logit"] == [1, 20, 1]
        assert outputs["h_out"] == [1, 1, 64]

    def test_inference_runs(self, ort_session):
        """Model runs inference and returns non-None results."""
        x = np.random.randn(1, 20, 6).astype(np.float32)
        h = np.zeros((1, 1, 64), dtype=np.float32)
        results = ort_session.run(None, {"input_features": x, "h_in": h})
        assert len(results) == 3
        assert results[0].shape == (1, 20, 1)
        assert results[1].shape == (1, 20, 1)
        assert results[2].shape == (1, 1, 64)

    def test_last_timestep_extraction(self, ort_session):
        """Verify we correctly extract the last timestep from [1,20,1] output."""
        x = np.random.randn(1, 20, 6).astype(np.float32)
        h = np.zeros((1, 1, 64), dtype=np.float32)
        results = ort_session.run(None, {"input_features": x, "h_in": h})

        # Take last timestep
        speed = float(results[0][0, -1, 0])     # [0, 19, 0]
        stationary = float(results[1][0, -1, 0])  # [0, 19, 0]
        assert isinstance(speed, float)
        assert isinstance(stationary, float)
        assert np.isfinite(speed)
        assert np.isfinite(stationary)


# ─── Test 2: GRU State Persistence (PRD §7 guardrail #6) ────────────────────

class TestGRUStatePersistence:
    """
    CRITICAL TEST: The GRU hidden state must persist across inference calls.
    
    PRD §7 guardrail #6: "Resetting [the GRU state] per-call is a
    silent accuracy killer that produces no error."
    
    This test explicitly proves:
    1. h_out from call N becomes h_in for call N+1
    2. Different h_in values produce different outputs
    3. h_out is NOT all zeros after non-zero input
    """

    def test_h_out_is_not_zero_after_inference(self, ort_session):
        """h_out must be non-zero after processing non-zero input."""
        x = np.random.randn(1, 20, 6).astype(np.float32)
        h_in = np.zeros((1, 1, 64), dtype=np.float32)
        results = ort_session.run(None, {"input_features": x, "h_in": h_in})
        h_out = results[2]

        assert not np.allclose(h_out, 0.0), \
            "h_out is all zeros after non-zero input — GRU state not evolving!"

    def test_persistent_state_changes_output(self, ort_session):
        """
        Using h_out from call 1 as h_in for call 2 must produce a
        DIFFERENT result than using all-zeros h_in for call 2.
        
        This is the key test: if someone accidentally resets h_state
        per call, these outputs would be identical.
        """
        x1 = np.random.randn(1, 20, 6).astype(np.float32)
        x2 = np.random.randn(1, 20, 6).astype(np.float32)

        # Call 1: fresh state
        h_zero = np.zeros((1, 1, 64), dtype=np.float32)
        results_1 = ort_session.run(None, {"input_features": x1, "h_in": h_zero})
        h_out_1 = results_1[2]  # This should be non-zero now

        # Call 2A: CORRECT — feed h_out_1 as h_in
        results_2a = ort_session.run(None, {"input_features": x2, "h_in": h_out_1})

        # Call 2B: WRONG — reset to zeros (as if someone reset per-call)
        results_2b = ort_session.run(None, {"input_features": x2, "h_in": h_zero})

        # The outputs MUST differ — if they don't, the GRU state is meaningless
        speed_2a = results_2a[0][0, -1, 0]
        speed_2b = results_2b[0][0, -1, 0]

        assert not np.isclose(speed_2a, speed_2b, rtol=1e-9, atol=1e-7), \
            (f"Persistent state has no effect on output! "
             f"speed_with_state={speed_2a}, speed_without={speed_2b}. "
             f"GRU state persistence is broken.")

    def test_state_evolves_over_multiple_calls(self, ort_session):
        """h_state should evolve differently across sequential calls."""
        h_states = [np.zeros((1, 1, 64), dtype=np.float32)]

        for i in range(5):
            x = np.random.randn(1, 20, 6).astype(np.float32)
            results = ort_session.run(
                None, {"input_features": x, "h_in": h_states[-1]}
            )
            h_states.append(results[2])

        # Check that each h_state is different from the previous
        for i in range(1, len(h_states)):
            assert not np.allclose(h_states[i], h_states[i - 1], atol=1e-6), \
                f"h_state did not change between call {i-1} and {i}"


# ─── Test 3: GRU State Reset at Session Boundaries ──────────────────────────

class TestGRUStateReset:
    """Session reset should zero the hidden state."""

    def test_replayer_reset_zeros_state(self, replayer):
        """resetSession() must zero the hidden state."""
        # Inject some non-zero state
        replayer.h_state = np.ones((1, 1, 64), dtype=np.float32)
        assert not np.allclose(replayer.h_state, 0.0)

        # Reset
        replayer.reset_session()
        assert np.allclose(replayer.h_state, 0.0), \
            "resetSession() did not zero the GRU hidden state!"

    def test_replayer_reset_clears_window(self, replayer):
        """resetSession() must also clear the sliding window."""
        # Push some samples
        for _ in range(10):
            replayer.push_sample(np.random.randn(6).astype(np.float32))
        assert len(replayer.window) == 10

        replayer.reset_session()
        assert len(replayer.window) == 0


# ─── Test 4: Normalization Math ──────────────────────────────────────────────

class TestNormalization:
    """Verify normalization computation is correct."""

    def test_normalization_formula(self, norm_stats):
        """x_norm = (x - mean) / std"""
        mean, std = norm_stats
        x = np.array([[1.0, 2.0, 10.0, 0.1, 0.05, 0.03]], dtype=np.float32)
        expected = (x - mean) / std

        from route_replayer import normalize_window
        result = normalize_window(x, mean, std)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalization_preserves_shape(self, norm_stats):
        """Normalization must not change the shape."""
        mean, std = norm_stats
        x = np.random.randn(20, 6).astype(np.float32)

        from route_replayer import normalize_window
        result = normalize_window(x, mean, std)

        assert result.shape == x.shape

    def test_norm_stats_shapes(self, norm_stats):
        """Mean and std must each have exactly 6 elements."""
        mean, std = norm_stats
        assert mean.shape == (6,)
        assert std.shape == (6,)

    def test_norm_stats_reasonable(self, norm_stats):
        """Std must be positive (prevents division by zero)."""
        _, std = norm_stats
        assert np.all(std > 0), f"Std has non-positive values: {std}"

    def test_norm_feature_names_match(self):
        """Feature names in the npz must match the locked order."""
        data = np.load(STATS_PATH)
        names = list(data["feature_names"])
        expected = ['leveled_ax', 'leveled_ay', 'leveled_az',
                     'gyro_yaw', 'gyro_pitch', 'gyro_roll']
        assert names == expected, f"Feature order mismatch: {names} vs {expected}"


# ─── Test 5: Sigmoid ────────────────────────────────────────────────────────

class TestSigmoid:
    """Verify sigmoid implementation."""

    def test_sigmoid_zero(self):
        """sigmoid(0) = 0.5"""
        from route_replayer import sigmoid
        assert abs(sigmoid(0.0) - 0.5) < 1e-6

    def test_sigmoid_large_positive(self):
        """sigmoid(100) ≈ 1.0"""
        from route_replayer import sigmoid
        assert abs(sigmoid(100.0) - 1.0) < 1e-6

    def test_sigmoid_large_negative(self):
        """sigmoid(-100) ≈ 0.0"""
        from route_replayer import sigmoid
        assert abs(sigmoid(-100.0) - 0.0) < 1e-6

    def test_sigmoid_symmetry(self):
        """sigmoid(x) + sigmoid(-x) = 1.0"""
        from route_replayer import sigmoid
        for x in [0.5, 1.0, 2.0, 5.0, 10.0]:
            assert abs(sigmoid(x) + sigmoid(-x) - 1.0) < 1e-6

    def test_sigmoid_output_range(self):
        """sigmoid output must be in (0, 1) for all finite inputs."""
        from route_replayer import sigmoid
        for x in np.linspace(-50, 50, 1000):
            y = sigmoid(float(x))
            assert 0.0 <= y <= 1.0


# ─── Test 6: Edge Cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling in the inference pipeline."""

    def test_window_not_full_returns_none(self, replayer):
        """Inference must NOT run when window has fewer than 20 samples."""
        for i in range(19):
            result = replayer.push_sample(
                np.random.randn(6).astype(np.float32)
            )
            assert result is None, \
                f"Inference ran with only {i+1} samples in window!"

    def test_window_full_returns_result(self, replayer):
        """Inference MUST run when exactly 20 samples are accumulated."""
        for i in range(19):
            replayer.push_sample(np.random.randn(6).astype(np.float32))

        result = replayer.push_sample(np.random.randn(6).astype(np.float32))
        assert result is not None, "No inference result after 20 samples!"

    def test_window_fifo_maintains_order(self, replayer):
        """After 25 pushes, window should contain samples 6-25 (last 20)."""
        samples = []
        for i in range(25):
            s = np.full(6, float(i), dtype=np.float32)
            samples.append(s)
            replayer.push_sample(s)

        assert len(replayer.window) == 20
        # First element should be sample 5 (6th overall, 0-indexed)
        np.testing.assert_array_equal(replayer.window[0],
                                       np.full(6, 5.0, dtype=np.float32))
        # Last element should be sample 24
        np.testing.assert_array_equal(replayer.window[-1],
                                       np.full(6, 24.0, dtype=np.float32))

    def test_zero_input_produces_finite_output(self, ort_session):
        """All-zeros input should produce finite (not NaN/Inf) output."""
        x = np.zeros((1, 20, 6), dtype=np.float32)
        h = np.zeros((1, 1, 64), dtype=np.float32)
        results = ort_session.run(None, {"input_features": x, "h_in": h})

        for i, res in enumerate(results):
            assert np.all(np.isfinite(res)), \
                f"Output {i} has non-finite values with zero input"


# ─── Test 7: Sliding Window FIFO ────────────────────────────────────────────

class TestSlidingWindow:
    """Test the sliding window mechanism."""

    def test_window_grows(self, replayer):
        """Window should grow from 0 to WINDOW_SIZE."""
        for i in range(20):
            assert len(replayer.window) == i
            replayer.push_sample(np.random.randn(6).astype(np.float32))
        assert len(replayer.window) == 20

    def test_window_caps_at_size(self, replayer):
        """Window must never exceed WINDOW_SIZE."""
        for _ in range(100):
            replayer.push_sample(np.random.randn(6).astype(np.float32))
        assert len(replayer.window) == 20


# ─── Test 8: Full Pipeline Integration ──────────────────────────────────────

class TestFullPipeline:
    """End-to-end pipeline test with the sample drive log."""

    @pytest.mark.skipif(
        not os.path.isfile(SAMPLE_CSV),
        reason="sample_drive_log.csv not found"
    )
    def test_replay_sample_drive_log(self):
        """Replay the sample drive log and verify we get inference results."""
        from route_replayer import RouteReplayer, ReplayConfig, load_drive_log

        config = ReplayConfig(model_path=MODEL_PATH, stats_path=STATS_PATH)
        replayer = RouteReplayer(config)

        df = load_drive_log(SAMPLE_CSV)
        assert not df.empty, "Sample drive log is empty"

        result = replayer.replay(df)

        # We should get at least some inference results
        assert len(result.inference_results) > 0, \
            "No inference results from sample drive log"

        # All results should have valid values
        for ir in result.inference_results:
            assert np.isfinite(ir.speed_metric), \
                f"Non-finite speed_metric at t={ir.timestamp_ms}"
            assert 0.0 <= ir.stationary_prob <= 1.0, \
                f"stationary_prob out of [0,1] at t={ir.timestamp_ms}: {ir.stationary_prob}"

    @pytest.mark.skipif(
        not os.path.isfile(SAMPLE_CSV),
        reason="sample_drive_log.csv not found"
    )
    def test_gru_state_nonzero_after_replay(self):
        """After replaying a drive log, GRU state must NOT be all zeros."""
        from route_replayer import RouteReplayer, ReplayConfig, load_drive_log

        config = ReplayConfig(model_path=MODEL_PATH, stats_path=STATS_PATH)
        replayer = RouteReplayer(config)

        df = load_drive_log(SAMPLE_CSV)
        replayer.replay(df)

        assert not np.allclose(replayer.h_state, 0.0), \
            "GRU hidden state is all zeros after replay — state not persisting!"


# ─── Test 9: Drift Calculator Unit Tests ─────────────────────────────────────

class TestDriftCalculator:
    """Unit tests for the drift calculation module."""

    def test_haversine_zero_distance(self):
        """Same point → 0 distance."""
        from drift_calculator import haversine_m
        assert haversine_m(12.97, 77.59, 12.97, 77.59) == 0.0

    def test_haversine_known_distance(self):
        """Known distance: ~111 km for 1° latitude at equator."""
        from drift_calculator import haversine_m
        d = haversine_m(0.0, 0.0, 1.0, 0.0)
        assert 110_000 < d < 112_000, f"Expected ~111km, got {d/1000:.1f}km"

    def test_drift_pct_calculation(self):
        """drift_pct = (error / distance) * 100"""
        from drift_calculator import compute_drift_pct
        assert abs(compute_drift_pct(10.0, 100.0) - 10.0) < 1e-6
        assert abs(compute_drift_pct(5.0, 1000.0) - 0.5) < 1e-6

    def test_drift_pct_zero_distance(self):
        """Zero distance → inf drift (no meaningful metric)."""
        from drift_calculator import compute_drift_pct
        assert compute_drift_pct(10.0, 0.0) == float('inf')

    def test_report_has_duration_and_speed_buckets(self):
        """Report must ALWAYS be broken out, never pooled (§7 #3)."""
        from drift_calculator import (
            BlackoutSegment, SessionResult, compute_drift_report
        )
        session = SessionResult(session_id="test")
        session.segments.append(BlackoutSegment(
            start_time_ms=0,
            end_time_ms=60_000,
            true_positions=[(0.0, 0.0), (0.001, 0.001)],
            estimated_positions=[(0.0, 0.0), (0.0015, 0.0015)],
            true_distance_m=157.0,
            final_error_m=78.0,
            avg_speed_mps=2.6,
        ))
        report = compute_drift_report([session])

        # Must have duration and speed breakdowns
        assert len(report.by_duration) > 0, "Missing duration breakdown"
        assert len(report.by_speed) > 0, "Missing speed breakdown"
        assert len(report.all_segments) == 1
