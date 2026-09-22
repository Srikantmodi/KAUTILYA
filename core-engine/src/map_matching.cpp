/**
 * map_matching.cpp — HMM-Based Map Matching Engine (Viterbi)
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 5 (Map-Matching & Map Data Pipeline)
 *
 * Implements:
 *   - Binary .drgg graph loading
 *   - Emission probability (Gaussian on perpendicular distance)
 *   - Transition probability (connectivity + route-vs-observation gap)
 *   - Online Viterbi decoding (sliding, no full-sequence retrace needed)
 *   - Pseudo-measurement construction for UKF closed-loop (§2.4, §7 #5)
 *   - Confidence scoring for quality-gated UKF updates
 *
 * References:
 *   api-contracts.md §6, PRD §6.4, §2.4, §6.12, §7 guardrails #5, #7
 */

#include "map_matching.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═════════════════════════════════════════════════════════════════════════════
 * Global singleton
 * ═════════════════════════════════════════════════════════════════════════════ */

static MapMatcher g_map_matcher;

MapMatcher& get_map_matcher_instance() {
    return g_map_matcher;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Constructor / Reset
 * ═════════════════════════════════════════════════════════════════════════════ */

MapMatcher::MapMatcher()
    : graph_loaded_(false),
      origin_lat_rad_(0.0),
      origin_lon_rad_(0.0),
      origin_set_(false),
      has_prev_state_(false),
      prev_px_(0.0f),
      prev_py_(0.0f)
{}

void MapMatcher::reset() {
    prev_viterbi_.clear();
    has_prev_state_ = false;
    prev_px_ = 0.0f;
    prev_py_ = 0.0f;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Graph loading from binary .drgg file
 * ═════════════════════════════════════════════════════════════════════════════ */

bool MapMatcher::load_graph(const char* path) {
    if (!path) return false;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[map_matching] failed to open graph file: %s\n", path);
        return false;
    }

    /* ── Read header (32 bytes) ──────────────────────────────────────── */
    char     magic[4];
    uint32_t version, node_count, edge_count;
    double   origin_lat, origin_lon;

    if (fread(magic, 1, 4, f) != 4 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&node_count, sizeof(uint32_t), 1, f) != 1 ||
        fread(&edge_count, sizeof(uint32_t), 1, f) != 1 ||
        fread(&origin_lat, sizeof(double), 1, f) != 1 ||
        fread(&origin_lon, sizeof(double), 1, f) != 1) {
        fprintf(stderr, "[map_matching] truncated header in %s\n", path);
        fclose(f);
        return false;
    }

    if (memcmp(magic, DRGG_MAGIC, 4) != 0) {
        fprintf(stderr, "[map_matching] bad magic in %s\n", path);
        fclose(f);
        return false;
    }
    if (version != DRGG_VERSION) {
        fprintf(stderr, "[map_matching] unsupported version %u in %s\n",
                version, path);
        fclose(f);
        return false;
    }

    /* ── Read nodes ──────────────────────────────────────────────────── */
    nodes_.clear();
    nodes_.resize(node_count);

    for (uint32_t i = 0; i < node_count; i++) {
        int32_t id;
        double  lat, lon;
        if (fread(&id, sizeof(int32_t), 1, f) != 1 ||
            fread(&lat, sizeof(double), 1, f) != 1 ||
            fread(&lon, sizeof(double), 1, f) != 1) {
            fprintf(stderr, "[map_matching] truncated node data at index %u\n", i);
            fclose(f);
            nodes_.clear();
            return false;
        }
        nodes_[i].id      = id;
        nodes_[i].lat_deg = lat;
        nodes_[i].lon_deg = lon;
        nodes_[i].x_enu   = 0.0f;
        nodes_[i].y_enu   = 0.0f;
    }

    /* ── Read edges ──────────────────────────────────────────────────── */
    edges_.clear();
    edges_.resize(edge_count);

    for (uint32_t i = 0; i < edge_count; i++) {
        int32_t eid, from_idx, to_idx, oneway, reserved;
        if (fread(&eid,      sizeof(int32_t), 1, f) != 1 ||
            fread(&from_idx, sizeof(int32_t), 1, f) != 1 ||
            fread(&to_idx,   sizeof(int32_t), 1, f) != 1 ||
            fread(&oneway,   sizeof(int32_t), 1, f) != 1 ||
            fread(&reserved, sizeof(int32_t), 1, f) != 1) {
            fprintf(stderr, "[map_matching] truncated edge data at index %u\n", i);
            fclose(f);
            nodes_.clear();
            edges_.clear();
            return false;
        }
        if (from_idx < 0 || from_idx >= (int32_t)node_count ||
            to_idx   < 0 || to_idx   >= (int32_t)node_count) {
            fprintf(stderr, "[map_matching] edge %d has out-of-range node index\n", eid);
            fclose(f);
            nodes_.clear();
            edges_.clear();
            return false;
        }
        edges_[i].id       = eid;
        edges_[i].from_idx = from_idx;
        edges_[i].to_idx   = to_idx;
        edges_[i].oneway   = oneway;
        edges_[i].length_m    = 0.0f;  /* computed after ENU conversion */
        edges_[i].heading_rad = 0.0f;
    }

    fclose(f);

    graph_loaded_ = true;
    origin_set_   = false;   /* ENU positions not yet computed — need session origin */

    fprintf(stderr, "[map_matching] loaded graph: %u nodes, %u edges from %s\n",
            node_count, edge_count, path);
    return true;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Programmatic graph construction (testing / edge-engine)
 * ═════════════════════════════════════════════════════════════════════════════ */

void MapMatcher::add_node(int32_t id, float x_enu, float y_enu) {
    RoadNode n;
    n.id      = id;
    n.lat_deg = std::numeric_limits<double>::quiet_NaN();  /* not from lat/lon */
    n.lon_deg = std::numeric_limits<double>::quiet_NaN();
    n.x_enu   = x_enu;
    n.y_enu   = y_enu;
    nodes_.push_back(n);
}

void MapMatcher::add_edge(int32_t id, int32_t from_idx, int32_t to_idx,
                          int32_t oneway) {
    RoadEdge e;
    e.id          = id;
    e.from_idx    = from_idx;
    e.to_idx      = to_idx;
    e.oneway      = oneway;
    e.length_m    = 0.0f;
    e.heading_rad = 0.0f;
    edges_.push_back(e);
}

void MapMatcher::finalize_graph() {
    compute_edge_geometry();
    build_adjacency();
    graph_loaded_ = true;
    origin_set_   = true;   /* ENU coords were provided directly */
    reset();
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Session origin & ENU conversion
 * ═════════════════════════════════════════════════════════════════════════════ */

void MapMatcher::set_session_origin(double lat_deg, double lon_deg) {
    origin_lat_rad_ = lat_deg * MM_DEG_TO_RAD;
    origin_lon_rad_ = lon_deg * MM_DEG_TO_RAD;

    /* Convert all node lat/lon to ENU relative to this origin */
    for (auto& n : nodes_) {
        latlon_to_enu(n.lat_deg, n.lon_deg, n.x_enu, n.y_enu);
    }

    compute_edge_geometry();
    build_adjacency();

    origin_set_ = true;
    reset();
}

void MapMatcher::latlon_to_enu(double lat_deg, double lon_deg,
                               float& east_m, float& north_m) const {
    double lat_rad = lat_deg * MM_DEG_TO_RAD;
    double lon_rad = lon_deg * MM_DEG_TO_RAD;

    double dlat = lat_rad - origin_lat_rad_;
    double dlon = lon_rad - origin_lon_rad_;

    north_m = static_cast<float>(dlat * MM_EARTH_RADIUS_M);
    east_m  = static_cast<float>(dlon * MM_EARTH_RADIUS_M *
                                 cos(origin_lat_rad_));
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Internal geometry helpers
 * ═════════════════════════════════════════════════════════════════════════════ */

void MapMatcher::compute_edge_geometry() {
    for (auto& e : edges_) {
        if (e.from_idx < 0 || e.from_idx >= (int32_t)nodes_.size() ||
            e.to_idx   < 0 || e.to_idx   >= (int32_t)nodes_.size()) {
            e.length_m    = 0.0f;
            e.heading_rad = 0.0f;
            continue;
        }
        const RoadNode& n1 = nodes_[e.from_idx];
        const RoadNode& n2 = nodes_[e.to_idx];

        float dx = n2.x_enu - n1.x_enu;
        float dy = n2.y_enu - n1.y_enu;

        e.length_m    = sqrtf(dx * dx + dy * dy);
        e.heading_rad = atan2f(dy, dx);   /* ENU: 0=East, CCW+ */
    }
}

void MapMatcher::build_adjacency() {
    adjacency_.clear();
    adjacency_.resize(nodes_.size());

    for (int i = 0; i < (int)edges_.size(); i++) {
        const RoadEdge& e = edges_[i];
        if (e.from_idx >= 0 && e.from_idx < (int32_t)nodes_.size())
            adjacency_[e.from_idx].push_back(i);
        if (e.to_idx >= 0 && e.to_idx < (int32_t)nodes_.size())
            adjacency_[e.to_idx].push_back(i);
    }
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Distance & offset computations
 * ═════════════════════════════════════════════════════════════════════════════ */

float MapMatcher::point_to_segment_distance(float px, float py,
                                            int edge_idx) const {
    const RoadEdge& e  = edges_[edge_idx];
    const RoadNode& n1 = nodes_[e.from_idx];
    const RoadNode& n2 = nodes_[e.to_idx];

    float x1 = n1.x_enu, y1 = n1.y_enu;
    float x2 = n2.x_enu, y2 = n2.y_enu;

    float dx = x2 - x1;
    float dy = y2 - y1;
    float len_sq = dx * dx + dy * dy;

    if (len_sq < 1e-10f) {
        /* Degenerate (zero-length) segment */
        float ddx = px - x1;
        float ddy = py - y1;
        return sqrtf(ddx * ddx + ddy * ddy);
    }

    /* Project point onto segment, clamp to [0,1] */
    float t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));

    float cx = x1 + t * dx;
    float cy = y1 + t * dy;

    float ddx = px - cx;
    float ddy = py - cy;
    return sqrtf(ddx * ddx + ddy * ddy);
}

float MapMatcher::signed_lateral_offset(float px, float py,
                                        int edge_idx) const {
    const RoadEdge& e  = edges_[edge_idx];
    const RoadNode& n1 = nodes_[e.from_idx];
    const RoadNode& n2 = nodes_[e.to_idx];

    float dx = n2.x_enu - n1.x_enu;
    float dy = n2.y_enu - n1.y_enu;
    float len = sqrtf(dx * dx + dy * dy);

    if (len < 1e-10f) return 0.0f;

    /* Cross product: positive ⇒ point is to the LEFT of segment direction.
     * cross = dx * (py − y1) − dy * (px − x1) */
    float cross = dx * (py - n1.y_enu) - dy * (px - n1.x_enu);
    return cross / len;
}

float MapMatcher::heading_offset_to_road(float traj_heading,
                                         int edge_idx) const {
    float road_heading = edges_[edge_idx].heading_rad;

    /* Forward direction offset */
    float diff_fwd = wrap_to_pi(traj_heading - road_heading);

    if (edges_[edge_idx].oneway == 1) {
        /* Oneway road — only forward direction makes sense */
        return diff_fwd;
    }

    /* Bidirectional — also consider reverse direction */
    float diff_rev = wrap_to_pi(
        traj_heading - wrap_to_pi(road_heading + static_cast<float>(M_PI)));

    return (fabsf(diff_fwd) <= fabsf(diff_rev)) ? diff_fwd : diff_rev;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * HMM probability functions
 * ═════════════════════════════════════════════════════════════════════════════ */

float MapMatcher::emission_log_prob(float dist_m) const {
    /* Gaussian: log P(d) = -d² / (2σ²)  (dropping constant normalisation) */
    return -(dist_m * dist_m) /
           (2.0f * MM_EMISSION_SIGMA_M * MM_EMISSION_SIGMA_M);
}

bool MapMatcher::edges_share_node(int edge_a, int edge_b) const {
    const RoadEdge& a = edges_[edge_a];
    const RoadEdge& b = edges_[edge_b];
    return a.from_idx == b.from_idx ||
           a.from_idx == b.to_idx   ||
           a.to_idx   == b.from_idx ||
           a.to_idx   == b.to_idx;
}

float MapMatcher::transition_log_prob(int from_edge, int to_edge,
                                      float obs_dist) const {
    if (from_edge == to_edge) {
        /* Staying on the same segment — natural and likely */
        return 0.0f;
    }

    if (edges_share_node(from_edge, to_edge)) {
        /* Connected via shared node — penalize by route-vs-observation gap.
         * Approximate route distance as half-sum of both edge lengths
         * (we're going from mid-first through shared node to mid-second). */
        float approx_route = (edges_[from_edge].length_m +
                              edges_[to_edge].length_m) * 0.5f;
        float gap = fabsf(approx_route - obs_dist);
        return -gap / MM_TRANSITION_BETA_M;
    }

    /* Not connected — heavy penalty (effectively impossible) */
    return MM_LOG_PROB_IMPOSSIBLE;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Candidate search
 * ═════════════════════════════════════════════════════════════════════════════ */

std::vector<int> MapMatcher::find_candidate_edges(float px, float py) const {
    /*
     * Brute-force scan over all edges — adequate for demo-scale graphs
     * (a few thousand edges).  For city-scale, add a spatial grid hash.
     */
    struct CandDist {
        int   edge_idx;
        float dist;
    };
    std::vector<CandDist> within_radius;
    within_radius.reserve(32);

    for (int i = 0; i < (int)edges_.size(); i++) {
        float d = point_to_segment_distance(px, py, i);
        if (d <= MM_SEARCH_RADIUS_M) {
            within_radius.push_back({i, d});
        }
    }

    /* Sort by ascending distance */
    std::sort(within_radius.begin(), within_radius.end(),
              [](const CandDist& a, const CandDist& b) {
                  return a.dist < b.dist;
              });

    /* Truncate to max candidates */
    int n = std::min((int)within_radius.size(), MM_MAX_CANDIDATES);
    std::vector<int> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = within_radius[i].edge_idx;
    }
    return result;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Core matching — online Viterbi step
 * ═════════════════════════════════════════════════════════════════════════════ */

MapMatchResult MapMatcher::match(const float trajectory_point[3],
                                 float heading_rad) {
    /* Default "no match" result */
    MapMatchResult result;
    result.matched_segment_id    = -1;
    result.pseudo_measurement[0] = 0.0f;
    result.pseudo_measurement[1] = 0.0f;
    result.confidence            = 0.0f;

    if (!graph_loaded_ || !origin_set_) {
        return result;
    }
    if (edges_.empty()) {
        return result;
    }

    float px = trajectory_point[0];   /* east  */
    float py = trajectory_point[1];   /* north */
    /* trajectory_point[2] = up, ignored for 2-D road matching */

    /* ── 1. Find candidate edges within search radius ────────────────── */
    std::vector<int> cand_edges = find_candidate_edges(px, py);
    if (cand_edges.empty()) {
        /* No roads nearby — return no-match, reset Viterbi state */
        has_prev_state_ = false;
        return result;
    }

    /* ── 2. Compute emission log-probabilities ───────────────────────── */
    struct CandInfo {
        int   edge_idx;
        float dist_m;
        float emission_lp;
    };
    std::vector<CandInfo> candidates;
    candidates.reserve(cand_edges.size());

    for (int eidx : cand_edges) {
        float d  = point_to_segment_distance(px, py, eidx);
        float lp = emission_log_prob(d);
        candidates.push_back({eidx, d, lp});
    }

    /* ── 3. Viterbi step ─────────────────────────────────────────────── */
    std::vector<ViterbiState> curr_viterbi;
    curr_viterbi.reserve(candidates.size());

    if (!has_prev_state_) {
        /* First observation — use emission probability alone */
        for (auto& c : candidates) {
            curr_viterbi.push_back({c.edge_idx, c.emission_lp});
        }
    } else {
        /* Subsequent observation — full Viterbi transition */
        float obs_dist = sqrtf((px - prev_px_) * (px - prev_px_) +
                               (py - prev_py_) * (py - prev_py_));

        for (auto& c : candidates) {
            float best_total = -std::numeric_limits<float>::max();

            for (auto& pv : prev_viterbi_) {
                float tlp   = transition_log_prob(pv.edge_idx, c.edge_idx,
                                                   obs_dist);
                float total = pv.log_prob + tlp + c.emission_lp;
                if (total > best_total) {
                    best_total = total;
                }
            }
            curr_viterbi.push_back({c.edge_idx, best_total});
        }
    }

    /* ── 4. Find best current candidate ──────────────────────────────── */
    int   best_idx  = 0;
    float best_prob = curr_viterbi[0].log_prob;
    for (int i = 1; i < (int)curr_viterbi.size(); i++) {
        if (curr_viterbi[i].log_prob > best_prob) {
            best_prob = curr_viterbi[i].log_prob;
            best_idx  = i;
        }
    }

    int best_edge_idx = curr_viterbi[best_idx].edge_idx;

    /* ── 5. Compute confidence ───────────────────────────────────────── */
    /*
     * Two factors:
     *   (a) Distance factor — how close the best match is to the road
     *       (emission probability, Gaussian).
     *   (b) Ambiguity factor — how much better the best is vs. alternatives
     *       (softmax of Viterbi log-probs, bounded by sigmoid of gap).
     *
     * confidence = distance_factor × ambiguity_factor
     */
    float best_emission_lp = candidates[best_idx].emission_lp;
    float distance_factor  = expf(best_emission_lp);

    float ambiguity_factor = 1.0f;
    if (curr_viterbi.size() > 1) {
        /* Find second-best Viterbi log-prob */
        float second_best_lp = -std::numeric_limits<float>::max();
        for (int i = 0; i < (int)curr_viterbi.size(); i++) {
            if (i != best_idx && curr_viterbi[i].log_prob > second_best_lp) {
                second_best_lp = curr_viterbi[i].log_prob;
            }
        }
        /* Sigmoid of the gap: large gap → ambiguity_factor → 1.0 */
        float gap = best_prob - second_best_lp;
        ambiguity_factor = 1.0f / (1.0f + expf(-gap * 2.0f));
    }

    float confidence = distance_factor * ambiguity_factor;
    /* Clamp to [0, 1] */
    if (confidence > 1.0f) confidence = 1.0f;
    if (confidence < 0.0f) confidence = 0.0f;

    /* ── 6. Construct pseudo-measurement ─────────────────────────────── */
    result.matched_segment_id    = edges_[best_edge_idx].id;
    result.pseudo_measurement[0] = signed_lateral_offset(px, py, best_edge_idx);
    result.pseudo_measurement[1] = heading_offset_to_road(heading_rad,
                                                          best_edge_idx);
    result.confidence = confidence;

    /* ── 7. Update Viterbi state for next call ───────────────────────── */
    prev_viterbi_   = curr_viterbi;
    has_prev_state_ = true;
    prev_px_        = px;
    prev_py_        = py;

    return result;
}
