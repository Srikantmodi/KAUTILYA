/**
 * map_matching.h — HMM-Based Map Matching Engine
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Derived from: docs/api-contracts.md §6, Master PRD §6.4, §2.4
 *
 * Owner:  Member 5 (Map-Matching & Map Data Pipeline)
 *
 * Purpose: HMM road-snapping engine that matches trajectory points to an
 * offline OSM-derived road graph using Viterbi decoding. Produces a
 * pseudo-measurement (lateral offset + heading offset) consumed directly by
 * Member 4's ukf_fusion::update_map_match() — this is CLOSED-LOOP, not
 * display-only (§2.4, §7 guardrail #5).
 *
 * Graph file format (.drgg — Dead Reckoning Road Graph):
 *   Binary, little-endian. Pre-converted from OSM PBF by convert_to_graph.py.
 *   Header (32 bytes):
 *     [0-3]   char[4]   Magic "DRGG"
 *     [4-7]   uint32    Version (1)
 *     [8-11]  uint32    Node count (N)
 *     [12-15] uint32    Edge count (E)
 *     [16-23] float64   Origin latitude (degrees) — graph centroid
 *     [24-31] float64   Origin longitude (degrees) — graph centroid
 *   Nodes (N × 20 bytes each):
 *     [0-3]   int32     OSM node ID
 *     [4-11]  float64   Latitude (degrees)
 *     [12-19] float64   Longitude (degrees)
 *   Edges (E × 20 bytes each):
 *     [0-3]   int32     Edge ID (sequential, 0-based)
 *     [4-7]   int32     From node index (0-based into node array)
 *     [8-11]  int32     To node index (0-based into node array)
 *     [12-15] int32     Oneway flag (0=bidirectional, 1=from→to only)
 *     [16-19] int32     Reserved (0)
 *
 * Conventions (api-contracts.md §0):
 *   - ENU frame, heading 0=East CCW+ (matching imu_types.h)
 *   - Angles: radians.  Distances: meters.  Speed: m/s.
 *   - Confidence/probability: float [0.0, 1.0].
 */

#ifndef CORE_ENGINE_MAP_MATCHING_H
#define CORE_ENGINE_MAP_MATCHING_H

#include "imu_types.h"
#include "ins_mechanization.h"   /* wrap_to_pi(), N_STATE constants */

#include <cstdint>
#include <vector>

/* ─────────────────────────────────────────────────────────────────────────────
 * Graph binary format constants
 * ───────────────────────────────────────────────────────────────────────────── */
static constexpr char     DRGG_MAGIC[4]   = {'D', 'R', 'G', 'G'};
static constexpr uint32_t DRGG_VERSION    = 1;
static constexpr uint32_t DRGG_HEADER_SZ  = 32;
static constexpr uint32_t DRGG_NODE_SZ    = 20;   /* int32 + float64 + float64 */
static constexpr uint32_t DRGG_EDGE_SZ    = 20;   /* 5 × int32               */

/* ─────────────────────────────────────────────────────────────────────────────
 * HMM tuning constants — all tunable, starting-point defaults
 * ───────────────────────────────────────────────────────────────────────────── */

/** Search radius for candidate road segments (meters). */
static constexpr float MM_SEARCH_RADIUS_M     = 50.0f;

/** Emission probability σ — Gaussian std dev for perpendicular distance. */
static constexpr float MM_EMISSION_SIGMA_M    = 20.0f;

/** Transition probability β — exponential decay for route-vs-observation gap. */
static constexpr float MM_TRANSITION_BETA_M   = 5.0f;

/** Max candidate segments per observation (keep closest, discard rest). */
static constexpr int   MM_MAX_CANDIDATES      = 10;

/** Large negative log-probability for impossible transitions. */
static constexpr float MM_LOG_PROB_IMPOSSIBLE  = -50.0f;

/** Earth radius (WGS-84 mean), meters — for ENU conversion. */
static constexpr double MM_EARTH_RADIUS_M     = 6371000.0;

/** Degrees ↔ radians. */
static constexpr double MM_DEG_TO_RAD = 3.14159265358979323846 / 180.0;

/* ─────────────────────────────────────────────────────────────────────────────
 * Road graph data structures
 * ───────────────────────────────────────────────────────────────────────────── */

/** A node in the road graph (OSM intersection or way endpoint). */
struct RoadNode {
    int32_t id;         /* OSM node ID (or synthetic ID for testing)         */
    double  lat_deg;    /* WGS-84 latitude, degrees  (NaN if built via ENU) */
    double  lon_deg;    /* WGS-84 longitude, degrees (NaN if built via ENU) */
    float   x_enu;      /* East position in session ENU frame, meters       */
    float   y_enu;      /* North position in session ENU frame, meters      */
};

/** A directed edge (road segment) in the road graph. */
struct RoadEdge {
    int32_t id;          /* edge/road_segment_id — returned in MapMatchResult */
    int32_t from_idx;    /* index into MapMatcher::nodes_ (0-based)           */
    int32_t to_idx;      /* index into MapMatcher::nodes_ (0-based)           */
    int32_t oneway;      /* 0 = bidirectional, 1 = oneway from→to             */
    float   length_m;    /* segment length, meters (computed at finalize)      */
    float   heading_rad; /* atan2(Δn, Δe), ENU convention (computed at finalize) */
};

/* ─────────────────────────────────────────────────────────────────────────────
 * API contract structs — api-contracts.md §6 (EXACT shapes)
 * ───────────────────────────────────────────────────────────────────────────── */

struct MapMatchCandidate {
    int   road_segment_id;
    float perpendicular_distance_m;
    float emission_probability;
};

struct MapMatchResult {
    int   matched_segment_id;
    float pseudo_measurement[2];  /* [lateral_offset_m, heading_offset_rad]  *
                                   * Same shape consumed by                  *
                                   * ukf_fusion::update_map_match()          */
    float confidence;             /* [0, 1]                                  */
};

/* ─────────────────────────────────────────────────────────────────────────────
 * MapMatcher — HMM road-snapping engine with online Viterbi decoding
 * ───────────────────────────────────────────────────────────────────────────── */

class MapMatcher {
public:
    MapMatcher();

    /* ── Graph loading from binary .drgg file ──────────────────────────── */
    /**
     * Load a pre-converted road graph from a .drgg binary file.
     * After loading, call set_session_origin() before matching.
     * @return true on success, false on I/O or format error.
     */
    bool load_graph(const char* path);

    /* ── Programmatic graph construction (testing / edge-engine) ────────  */
    /**
     * Add a node with direct ENU coordinates (no lat/lon conversion needed).
     * @param id       synthetic node ID
     * @param x_enu    east position, meters
     * @param y_enu    north position, meters
     */
    void add_node(int32_t id, float x_enu, float y_enu);

    /**
     * Add an edge between two nodes (by 0-based index into node array).
     * @param id        edge/road_segment_id
     * @param from_idx  index of the start node
     * @param to_idx    index of the end node
     * @param oneway    0 = bidirectional, 1 = oneway from→to
     */
    void add_edge(int32_t id, int32_t from_idx, int32_t to_idx,
                  int32_t oneway = 0);

    /**
     * Finalize the graph after programmatic construction — computes edge
     * geometry (lengths, headings) and builds adjacency lists.
     * Sets graph_loaded_ and origin_set_ to true.
     */
    void finalize_graph();

    /* ── Session management ────────────────────────────────────────────── */

    /**
     * Set session origin and convert all graph node lat/lon to ENU.
     * Call once when the first GNSS fix arrives (same time as UKF init).
     */
    void set_session_origin(double lat_deg, double lon_deg);

    /** Clear Viterbi history — call at session start / reset. */
    void reset();

    /* ── Core matching API (api-contracts.md §6) ───────────────────────── */

    /**
     * Match a single trajectory point to the road graph via HMM Viterbi.
     *
     * @param trajectory_point  [east, north, up] ENU meters (from NavState)
     * @param heading_rad       current heading, ENU convention (0=E, CCW+)
     * @return MapMatchResult with pseudo-measurement for the UKF, or
     *         confidence=0 / segment_id=-1 if no match found.
     */
    MapMatchResult match(const float trajectory_point[3], float heading_rad);

    /* ── Accessors ─────────────────────────────────────────────────────── */
    bool is_loaded()    const { return graph_loaded_; }
    bool has_origin()   const { return origin_set_;   }
    int  node_count()   const { return static_cast<int>(nodes_.size()); }
    int  edge_count()   const { return static_cast<int>(edges_.size()); }

private:
    /* ── Graph storage ─────────────────────────────────────────────────── */
    std::vector<RoadNode>          nodes_;
    std::vector<RoadEdge>          edges_;
    std::vector<std::vector<int>>  adjacency_;   /* node_idx → edge indices */
    bool graph_loaded_;

    /* ── Session ENU origin ────────────────────────────────────────────── */
    double origin_lat_rad_;
    double origin_lon_rad_;
    bool   origin_set_;

    /* ── Online Viterbi state (previous time-step) ─────────────────────── */
    struct ViterbiState {
        int   edge_idx;    /* index into edges_ */
        float log_prob;    /* accumulated Viterbi log-probability */
    };
    std::vector<ViterbiState> prev_viterbi_;
    bool  has_prev_state_;
    float prev_px_;        /* previous observation east  (ENU, m) */
    float prev_py_;        /* previous observation north (ENU, m) */

    /* ── Internal helpers ──────────────────────────────────────────────── */

    /** Compute length_m and heading_rad for all edges from node positions. */
    void compute_edge_geometry();

    /** Build adjacency lists (node_idx → list of edge indices). */
    void build_adjacency();

    /** Convert WGS-84 (degrees) → local ENU (meters) from session origin. */
    void latlon_to_enu(double lat_deg, double lon_deg,
                       float& east_m, float& north_m) const;

    /** Unsigned perpendicular distance from point to edge (meters). */
    float point_to_segment_distance(float px, float py, int edge_idx) const;

    /** Signed lateral offset: positive = left of road direction. */
    float signed_lateral_offset(float px, float py, int edge_idx) const;

    /**
     * Heading offset: trajectory heading minus road heading, wrapped to [-π,π].
     * For bidirectional roads, uses whichever direction minimises |offset|.
     */
    float heading_offset_to_road(float traj_heading, int edge_idx) const;

    /** Gaussian emission log-probability from perpendicular distance. */
    float emission_log_prob(float dist_m) const;

    /**
     * Transition log-probability between consecutive edge matches.
     * Connected edges get a route-vs-observation penalty; disconnected
     * edges get MM_LOG_PROB_IMPOSSIBLE.
     */
    float transition_log_prob(int from_edge, int to_edge,
                              float obs_dist) const;

    /** Check whether two edges share at least one endpoint node. */
    bool edges_share_node(int edge_a, int edge_b) const;

    /**
     * Find all edges within MM_SEARCH_RADIUS_M of point (px, py).
     * Returns edge indices, sorted by ascending distance, truncated to
     * MM_MAX_CANDIDATES.
     *
     * NOTE: uses brute-force scan — adequate for the demo-scale graphs
     * expected in this project. For city-scale graphs, add a spatial index
     * (grid hash or R-tree).
     */
    std::vector<int> find_candidate_edges(float px, float py) const;
};

/* ─────────────────────────────────────────────────────────────────────────────
 * Global singleton (accessed from JNI bridge and Python bindings)
 * ───────────────────────────────────────────────────────────────────────────── */
MapMatcher& get_map_matcher_instance();

#endif /* CORE_ENGINE_MAP_MATCHING_H */
