"""
edge_runner.py — Standalone edge-engine inference pipeline (Python)

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Full Python pipeline that replicates the core inference flow outside Android:
    Load ONNX model → ingest CSV → normalize → run inference → output results

This proves the same model interface works on a second platform (edge engine).
When the C++ core-engine fusion is ready, this pipeline feeds into it.

Usage:
    python edge_runner.py --csv <drive_log.csv> \\
        [--model <placeholder_model.onnx>] \\
        [--stats <normalization_stats_v2.npz>] \\
        [--output <results.csv>]

References:
    PRD §2.1 (edge-engine is the same core with swapped I/O adapters),
    PRD §6.6 (inference contract), api-contracts.md §8
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test-harness"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test-harness" / "simulator"))

from route_replayer import (
    RouteReplayer, ReplayConfig, load_drive_log,
    extract_features_from_row, sigmoid,
)


# ─── Default paths ──────────────────────────────────────────────────────────

DEFAULT_MODEL = str(Path(__file__).parent / "placeholder_model.onnx")
DEFAULT_STATS = str(Path(__file__).parent / "normalization_stats_v2.npz")


def run_edge_inference(
    csv_path: str,
    model_path: str = DEFAULT_MODEL,
    stats_path: str = DEFAULT_STATS,
    output_path: str | None = None,
) -> list[dict]:
    """
    Run the edge inference pipeline on a CSV drive log.

    Args:
        csv_path: Path to the input drive log CSV.
        model_path: Path to the ONNX model.
        stats_path: Path to normalization stats npz.
        output_path: Optional path for output CSV. If None, prints to stdout.

    Returns:
        List of per-step result dicts.
    """
    print(f"Edge Runner — Loading model: {model_path}")
    print(f"             Stats: {stats_path}")
    print(f"             Input: {csv_path}")

    config = ReplayConfig(
        model_path=model_path,
        stats_path=stats_path,
    )
    replayer = RouteReplayer(config)

    # Load drive log
    df = load_drive_log(csv_path)
    print(f"Loaded {len(df)} rows")

    if df.empty:
        print("ERROR: Empty drive log")
        return []

    # Run replay
    t_start = time.perf_counter()
    result = replayer.replay(df)
    t_elapsed = time.perf_counter() - t_start

    print(f"Inference complete: {len(result.inference_results)} steps "
          f"in {t_elapsed:.3f}s")

    # Format results
    results = []
    for ir in result.inference_results:
        results.append({
            "timestamp_ms": ir.timestamp_ms,
            "speed_metric": round(ir.speed_metric, 6),
            "stationary_prob": round(ir.stationary_prob, 6),
            "zupt_active": 1 if ir.stationary_prob > 0.95 else 0,
        })

    # Write output
    if results:
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Results written to: {output_path}")
        else:
            # Print to stdout
            print("\n--- Inference Results ---")
            print(f"{'timestamp_ms':>15s}  {'speed_metric':>13s}  "
                  f"{'stationary_prob':>16s}  {'zupt':>4s}")
            for r in results:
                print(f"{r['timestamp_ms']:>15d}  {r['speed_metric']:>13.6f}  "
                      f"{r['stationary_prob']:>16.6f}  "
                      f"{'YES' if r['zupt_active'] else 'no':>4s}")

    # Summary statistics
    if results:
        speeds = [r['speed_metric'] for r in results]
        probs = [r['stationary_prob'] for r in results]
        zupts = sum(1 for r in results if r['zupt_active'])
        print(f"\nSummary:")
        print(f"  Speed metric range: [{min(speeds):.4f}, {max(speeds):.4f}]")
        print(f"  Stationary prob range: [{min(probs):.4f}, {max(probs):.4f}]")
        print(f"  ZUPT activations: {zupts}/{len(results)} "
              f"({100*zupts/len(results):.1f}%)")
        print(f"  Throughput: {len(results)/t_elapsed:.0f} inferences/sec")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SIH PS-26168 — Edge Engine Inference Runner",
    )
    parser.add_argument("--csv", required=True, help="Path to drive log CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to ONNX model")
    parser.add_argument("--stats", default=DEFAULT_STATS,
                        help="Path to normalization stats npz")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: stdout)")

    args = parser.parse_args()
    run_edge_inference(args.csv, args.model, args.stats, args.output)


if __name__ == "__main__":
    main()
