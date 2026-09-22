#!/usr/bin/env python3
"""
convert_to_graph.py — OSM PBF → .drgg (Dead Reckoning Road Graph)

SIH PS-26168  Intelligent Dead Reckoning
Owner: Member 5 (Map-Matching & Map Data Pipeline)
Ref:   api-contracts.md §13, PRD §6.12

Extracts drivable-road ways from an OSM PBF file and writes a compact
binary .drgg file that core-engine/src/map_matching.cpp loads at app startup.

Only drivable roads are kept (footpaths, buildings, cycleways filtered out).
The output is a pre-converted graph — core-engine never parses raw OSM
XML/PBF at runtime.

Usage:
    python convert_to_graph.py <input.osm.pbf> <output.drgg>

Dependencies:
    pip install osmium   (pyosmium — the standard Python binding for libosmium)
"""

import struct
import sys
import math
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────────────────────
# Graph binary format constants (must match map_matching.h)
# ─────────────────────────────────────────────────────────────────────────────
DRGG_MAGIC   = b"DRGG"
DRGG_VERSION = 1

# OSM highway tags that are drivable roads (exclude pedestrian/cycle paths)
DRIVABLE_HIGHWAY_TAGS = frozenset({
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential",
    "unclassified",
    "service",
    "living_street",
    "road",                       # generic catch-all in OSM
})


# ─────────────────────────────────────────────────────────────────────────────
# OSM PBF parsing with pyosmium
# ─────────────────────────────────────────────────────────────────────────────

def parse_osm_pbf(pbf_path: str):
    """
    Parse an OSM PBF file and extract drivable road ways + referenced nodes.

    Returns:
        nodes:  dict[int, (lat, lon)]   — OSM node ID → (lat_deg, lon_deg)
        ways:   list of (way_id, [node_id, ...], oneway)
    """
    try:
        import osmium
    except ImportError:
        print("ERROR: pyosmium is required. Install with: pip install osmium",
              file=sys.stderr)
        sys.exit(1)

    # ── Pass 1: Collect drivable ways and their referenced node IDs ──

    class WayCollector(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways = []           # (way_id, [node_ids], oneway)
            self.needed_nodes = set()

        def way(self, w):
            tags = {t.k: t.v for t in w.tags}
            highway = tags.get("highway", "")
            if highway not in DRIVABLE_HIGHWAY_TAGS:
                return

            node_ids = [n.ref for n in w.nodes]
            if len(node_ids) < 2:
                return

            # Determine oneway
            ow = tags.get("oneway", "no").lower()
            junction = tags.get("junction", "").lower()

            if ow in ("yes", "1", "true") or junction == "roundabout":
                oneway = 1
            elif ow == "-1":
                # Reverse direction
                node_ids = list(reversed(node_ids))
                oneway = 1
            else:
                oneway = 0

            self.ways.append((w.id, node_ids, oneway))
            self.needed_nodes.update(node_ids)

    wc = WayCollector()
    wc.apply_file(pbf_path, locations=False)

    print(f"  Found {len(wc.ways)} drivable ways, "
          f"{len(wc.needed_nodes)} referenced nodes", file=sys.stderr)

    # ── Pass 2: Collect node coordinates ──

    class NodeCollector(osmium.SimpleHandler):
        def __init__(self, needed):
            super().__init__()
            self.needed = needed
            self.nodes = {}   # node_id → (lat, lon)

        def node(self, n):
            if n.id in self.needed:
                self.nodes[n.id] = (n.location.lat, n.location.lon)

    nc = NodeCollector(wc.needed_nodes)
    nc.apply_file(pbf_path, locations=True)

    print(f"  Resolved {len(nc.nodes)}/{len(wc.needed_nodes)} node locations",
          file=sys.stderr)

    return nc.nodes, wc.ways


def build_graph(nodes_raw, ways_raw):
    """
    Convert raw OSM data into indexed graph arrays.

    Returns:
        graph_nodes: list of (osm_id, lat, lon)   — ordered, 0-indexed
        graph_edges: list of (edge_id, from_idx, to_idx, oneway)
    """
    # Build ordered node list (only nodes that actually have coordinates)
    node_id_to_idx = {}
    graph_nodes = []

    for way_id, node_ids, oneway in ways_raw:
        for nid in node_ids:
            if nid in nodes_raw and nid not in node_id_to_idx:
                node_id_to_idx[nid] = len(graph_nodes)
                lat, lon = nodes_raw[nid]
                graph_nodes.append((nid, lat, lon))

    # Build edges: each way is split into consecutive node pairs
    graph_edges = []
    edge_id = 0

    for way_id, node_ids, oneway in ways_raw:
        for i in range(len(node_ids) - 1):
            n_from = node_ids[i]
            n_to   = node_ids[i + 1]
            if n_from not in node_id_to_idx or n_to not in node_id_to_idx:
                continue  # skip if node coordinates missing
            from_idx = node_id_to_idx[n_from]
            to_idx   = node_id_to_idx[n_to]
            graph_edges.append((edge_id, from_idx, to_idx, oneway))
            edge_id += 1

    print(f"  Graph: {len(graph_nodes)} nodes, {len(graph_edges)} edges",
          file=sys.stderr)

    return graph_nodes, graph_edges


def compute_centroid(graph_nodes):
    """Compute the centroid (mean lat, mean lon) of all nodes."""
    if not graph_nodes:
        return 0.0, 0.0
    sum_lat = sum(n[1] for n in graph_nodes)
    sum_lon = sum(n[2] for n in graph_nodes)
    n = len(graph_nodes)
    return sum_lat / n, sum_lon / n


# ─────────────────────────────────────────────────────────────────────────────
# Binary .drgg writer
# ─────────────────────────────────────────────────────────────────────────────

def write_drgg(output_path: str, graph_nodes, graph_edges,
               origin_lat: float, origin_lon: float):
    """
    Write graph to binary .drgg format matching map_matching.h spec.

    Format (little-endian):
      Header (32 bytes):
        [0-3]   char[4]   "DRGG"
        [4-7]   uint32    version = 1
        [8-11]  uint32    node_count
        [12-15] uint32    edge_count
        [16-23] float64   origin_lat_deg
        [24-31] float64   origin_lon_deg
      Nodes (N × 20 bytes):
        int32 id, float64 lat, float64 lon
      Edges (E × 20 bytes):
        int32 id, int32 from_idx, int32 to_idx, int32 oneway, int32 reserved
    """
    with open(output_path, "wb") as f:
        # Header
        f.write(DRGG_MAGIC)
        f.write(struct.pack("<I", DRGG_VERSION))
        f.write(struct.pack("<I", len(graph_nodes)))
        f.write(struct.pack("<I", len(graph_edges)))
        f.write(struct.pack("<d", origin_lat))
        f.write(struct.pack("<d", origin_lon))

        # Nodes
        for osm_id, lat, lon in graph_nodes:
            f.write(struct.pack("<i", osm_id))
            f.write(struct.pack("<d", lat))
            f.write(struct.pack("<d", lon))

        # Edges
        for edge_id, from_idx, to_idx, oneway in graph_edges:
            f.write(struct.pack("<i", edge_id))
            f.write(struct.pack("<i", from_idx))
            f.write(struct.pack("<i", to_idx))
            f.write(struct.pack("<i", oneway))
            f.write(struct.pack("<i", 0))  # reserved

    file_size = 32 + len(graph_nodes) * 20 + len(graph_edges) * 20
    print(f"  Wrote {output_path} ({file_size} bytes)", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# .drgg reader (for verification / Python-side testing)
# ─────────────────────────────────────────────────────────────────────────────

def read_drgg(path: str):
    """
    Read a .drgg file and return parsed data.

    Returns:
        dict with keys: version, origin_lat, origin_lon,
                        nodes [(id, lat, lon), ...],
                        edges [(id, from_idx, to_idx, oneway), ...]
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != DRGG_MAGIC:
            raise ValueError(f"Bad magic: {magic!r} (expected {DRGG_MAGIC!r})")

        version    = struct.unpack("<I", f.read(4))[0]
        node_count = struct.unpack("<I", f.read(4))[0]
        edge_count = struct.unpack("<I", f.read(4))[0]
        origin_lat = struct.unpack("<d", f.read(8))[0]
        origin_lon = struct.unpack("<d", f.read(8))[0]

        if version != DRGG_VERSION:
            raise ValueError(f"Unsupported version: {version}")

        nodes = []
        for _ in range(node_count):
            nid = struct.unpack("<i", f.read(4))[0]
            lat = struct.unpack("<d", f.read(8))[0]
            lon = struct.unpack("<d", f.read(8))[0]
            nodes.append((nid, lat, lon))

        edges = []
        for _ in range(edge_count):
            eid      = struct.unpack("<i", f.read(4))[0]
            from_idx = struct.unpack("<i", f.read(4))[0]
            to_idx   = struct.unpack("<i", f.read(4))[0]
            oneway   = struct.unpack("<i", f.read(4))[0]
            _reserved = struct.unpack("<i", f.read(4))[0]
            edges.append((eid, from_idx, to_idx, oneway))

    return {
        "version":    version,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "nodes":      nodes,
        "edges":      edges,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def convert_osm_to_graph(pbf_path: str, output_path: str) -> None:
    """
    Full pipeline: OSM PBF → drivable road graph → .drgg binary.

    This is the function signature from api-contracts.md §13.
    """
    print(f"Converting {pbf_path} → {output_path}", file=sys.stderr)

    nodes_raw, ways_raw = parse_osm_pbf(pbf_path)
    graph_nodes, graph_edges = build_graph(nodes_raw, ways_raw)
    origin_lat, origin_lon = compute_centroid(graph_nodes)

    write_drgg(output_path, graph_nodes, graph_edges, origin_lat, origin_lon)

    print(f"Done. Origin: ({origin_lat:.6f}, {origin_lon:.6f})",
          file=sys.stderr)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.osm.pbf> <output.drgg>",
              file=sys.stderr)
        sys.exit(1)

    pbf_path    = sys.argv[1]
    output_path = sys.argv[2]

    convert_osm_to_graph(pbf_path, output_path)


if __name__ == "__main__":
    main()
