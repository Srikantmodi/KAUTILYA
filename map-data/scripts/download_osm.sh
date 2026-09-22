#!/usr/bin/env bash
# download_osm.sh — Download a regional OSM extract for map-matching
#
# SIH PS-26168  Intelligent Dead Reckoning
# Owner: Member 5 (Map-Matching & Map Data Pipeline)
#
# Downloads an OSM PBF extract from Geofabrik for the specified region.
# Default: Karnataka, India (covers Bengaluru — our primary test area).
#
# Usage:
#   ./download_osm.sh [region_url] [output_file]
#
# Examples:
#   ./download_osm.sh                                    # default: Karnataka
#   ./download_osm.sh https://download.geofabrik.de/asia/india/karnataka-latest.osm.pbf my_region.osm.pbf
#
# Prerequisites:
#   - wget or curl

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_URL="https://download.geofabrik.de/asia/india/karnataka-latest.osm.pbf"
DEFAULT_OUTPUT="$(dirname "$0")/../assets/karnataka-latest.osm.pbf"

URL="${1:-$DEFAULT_URL}"
OUTPUT="${2:-$DEFAULT_OUTPUT}"

# ── Ensure output directory exists ─────────────────────────────────────────

mkdir -p "$(dirname "$OUTPUT")"

# ── Download ───────────────────────────────────────────────────────────────

echo "Downloading OSM extract..."
echo "  URL:    $URL"
echo "  Output: $OUTPUT"

if command -v wget &>/dev/null; then
    wget --no-verbose --show-progress -O "$OUTPUT" "$URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$OUTPUT" "$URL"
else
    echo "ERROR: wget or curl is required" >&2
    exit 1
fi

echo "Download complete: $(du -h "$OUTPUT" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. (Optional) Filter to a smaller area:  ./filter_osm.sh $OUTPUT"
echo "  2. Convert to graph:  python convert_to_graph.py $OUTPUT output.drgg"
