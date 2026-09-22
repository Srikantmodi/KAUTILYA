#!/usr/bin/env bash
# filter_osm.sh — Filter an OSM PBF extract to a smaller bounding box
#
# SIH PS-26168  Intelligent Dead Reckoning
# Owner: Member 5 (Map-Matching & Map Data Pipeline)
#
# Uses osmium-tool to clip a regional extract to a bounding box (e.g. a city)
# and optionally filter to only highway features (roads). This produces a
# much smaller PBF file for faster conversion and smaller app assets.
#
# Usage:
#   ./filter_osm.sh <input.osm.pbf> [output.osm.pbf] [bbox]
#
# Examples:
#   # Clip Karnataka to Bengaluru area (default bbox)
#   ./filter_osm.sh ../assets/karnataka-latest.osm.pbf
#
#   # Custom bbox: minlon,minlat,maxlon,maxlat
#   ./filter_osm.sh input.osm.pbf output.osm.pbf 77.5,12.8,77.7,13.1
#
# Prerequisites:
#   - osmium-tool (apt install osmium-tool / brew install osmium-tool)
#
# Bbox format: minlon,minlat,maxlon,maxlat (WGS-84 decimal degrees)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

# Default: Bengaluru city bounding box
DEFAULT_BBOX="77.4,12.8,77.8,13.15"

INPUT="${1:?Usage: $0 <input.osm.pbf> [output.osm.pbf] [bbox]}"
OUTPUT="${2:-$(dirname "$INPUT")/filtered.osm.pbf}"
BBOX="${3:-$DEFAULT_BBOX}"

# ── Prerequisite check ────────────────────────────────────────────────────

if ! command -v osmium &>/dev/null; then
    echo "ERROR: osmium-tool is required." >&2
    echo "  Ubuntu/Debian: sudo apt install osmium-tool" >&2
    echo "  macOS:         brew install osmium-tool" >&2
    exit 1
fi

# ── Ensure output directory exists ─────────────────────────────────────────

mkdir -p "$(dirname "$OUTPUT")"

# ── Step 1: Clip to bounding box ──────────────────────────────────────────

CLIPPED_TMP="$(mktemp).osm.pbf"
echo "Step 1: Clipping to bbox [$BBOX]..."
osmium extract --bbox "$BBOX" --strategy smart "$INPUT" -o "$CLIPPED_TMP" --overwrite

# ── Step 2: Filter to highway features only ───────────────────────────────

echo "Step 2: Filtering to highway=* (roads only)..."
osmium tags-filter "$CLIPPED_TMP" w/highway -o "$OUTPUT" --overwrite

# ── Cleanup ───────────────────────────────────────────────────────────────

rm -f "$CLIPPED_TMP"

echo ""
echo "Input:  $(du -h "$INPUT" | cut -f1)  $INPUT"
echo "Output: $(du -h "$OUTPUT" | cut -f1)  $OUTPUT"
echo ""
echo "Next: python convert_to_graph.py $OUTPUT road_graph.drgg"
