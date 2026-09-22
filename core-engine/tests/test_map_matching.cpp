/**
 * test_map_matching.cpp — Unit tests for HMM Map Matching Engine
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 5 (Map-Matching & Map Data Pipeline)
 *
 * Test cases (per PRD §7 guardrail #7 — never validate only on happy path):
 *
 *   1. Known-geometry case with hand-computed expected match
 *   2. Deliberately ambiguous / low-confidence case proving the code
 *      correctly produces confidence < threshold (UKF would skip update)
 *   3. No candidates nearby — safe default result
 *   4. Empty graph — safe default result
 *   5. Viterbi transition consistency across sequential calls
 *   6. Heading offset correctness (bidirectional vs. oneway)
 *   7. Degenerate edge (zero length) — no crash
 *
 * Builds standalone (no GTest dependency) — uses minimal assertion macros
 * matching the project's existing test pattern.
 */

#include "map_matching.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

/* ─────────────────────────────────────────────────────────────────────────────
 * Minimal test framework
 * ───────────────────────────────────────────────────────────────────────────── */

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do {                                          \
    g_tests_run++;                                                           \
    if (!(cond)) {                                                           \
        g_tests_failed++;                                                    \
        fprintf(stderr, "  FAIL [%s:%d]: %s — %s\n",                        \
                __FILE__, __LINE__, #cond, (msg));                           \
    } else {                                                                 \
        g_tests_passed++;                                                    \
    }                                                                        \
} while (0)

#define TEST_ASSERT_NEAR(a, b, tol, msg) do {                                \
    g_tests_run++;                                                           \
    double _a = (double)(a), _b = (double)(b), _t = (double)(tol);           \
    if (fabs(_a - _b) > _t) {                                                \
        g_tests_failed++;                                                    \
        fprintf(stderr, "  FAIL [%s:%d]: %s ≈ %s  (%.6f vs %.6f, tol %.6f)" \
                " — %s\n", __FILE__, __LINE__, #a, #b, _a, _b, _t, (msg));  \
    } else {                                                                 \
        g_tests_passed++;                                                    \
    }                                                                        \
} while (0)

#define RUN_TEST(fn) do {                                                    \
    fprintf(stderr, "--- %s ---\n", #fn);                                    \
    fn();                                                                    \
} while (0)

/* ═════════════════════════════════════════════════════════════════════════════
 * Helper: build a simple two-road graph for most tests
 *
 *   Road A: (0,0)──────────(100,0)     edge 0, y = 0,   heading = 0 (East)
 *   Road B: (0,100)────────(100,100)   edge 1, y = 100, heading = 0 (East)
 *
 * Both bidirectional (oneway = 0).
 * ═════════════════════════════════════════════════════════════════════════════ */
static void build_two_road_graph(MapMatcher& mm) {
    /* Nodes */
    mm.add_node(0,   0.0f,   0.0f);   /* A-start */
    mm.add_node(1, 100.0f,   0.0f);   /* A-end   */
    mm.add_node(2,   0.0f, 100.0f);   /* B-start */
    mm.add_node(3, 100.0f, 100.0f);   /* B-end   */

    /* Edges */
    mm.add_edge(0, 0, 1, 0);   /* Road A: node 0 → node 1, bidirectional */
    mm.add_edge(1, 2, 3, 0);   /* Road B: node 2 → node 3, bidirectional */

    mm.finalize_graph();
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 1: Known-geometry — point close to Road A, far from Road B
 *
 * Hand-computed:
 *   Point (50, 3, 0):
 *     - perp distance to Road A (y=0):   3.0 m
 *     - perp distance to Road B (y=100): 97.0 m
 *   With σ = 20 m:
 *     - emission_lp(A) = -(3²)/(2×20²) = -9/800 = -0.01125
 *     - emission_lp(B) = -(97²)/(2×20²) = -9409/800 = -11.76
 *   ⇒ Road A wins overwhelmingly.
 *   lateral_offset  = +3.0 m (point is to the LEFT of eastward road)
 *   heading_offset  ≈ 0 (trajectory heading = 0 = road heading)
 *   confidence      > 0.7 (clearly unambiguous, close to road)
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_known_geometry() {
    MapMatcher mm;
    build_two_road_graph(mm);

    float pt[3] = {50.0f, 3.0f, 0.0f};
    float heading = 0.0f;   /* East, same as Road A */

    MapMatchResult r = mm.match(pt, heading);

    TEST_ASSERT(r.matched_segment_id == 0,
                "should match Road A (segment 0)");

    TEST_ASSERT_NEAR(r.pseudo_measurement[0], 3.0f, 0.1f,
                     "lateral offset should be +3.0 m (left of road)");

    TEST_ASSERT_NEAR(r.pseudo_measurement[1], 0.0f, 0.01f,
                     "heading offset should be ~0 (aligned with road)");

    TEST_ASSERT(r.confidence > 0.7f,
                "confidence should exceed UKF threshold (clear match)");

    fprintf(stderr, "  confidence = %.4f, segment_id = %d, "
            "lateral = %.4f, heading_off = %.4f\n",
            r.confidence, r.matched_segment_id,
            r.pseudo_measurement[0], r.pseudo_measurement[1]);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 2: Deliberately ambiguous — point equidistant between two roads
 *
 * Point (50, 50, 0) is exactly midway between Road A and Road B.
 * Both are 50 m away — well within search radius.
 *
 * With σ = 20 m:
 *   emission_lp = -(50²)/(2×20²) = -2500/800 = -3.125 for both
 *   → Ambiguity factor should be low (≈ 0.5)
 *   → distance_factor = exp(-3.125) ≈ 0.044
 *   → confidence should be well below 0.7
 *
 * This proves the matcher correctly signals ambiguity, causing the UKF
 * to SKIP the update rather than forcing a bad snap (§2.4, §7 #5).
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_ambiguous_low_confidence() {
    MapMatcher mm;
    build_two_road_graph(mm);

    float pt[3] = {50.0f, 50.0f, 0.0f};
    float heading = 0.0f;

    MapMatchResult r = mm.match(pt, heading);

    TEST_ASSERT(r.matched_segment_id >= 0,
                "should still return a match (even if low confidence)");

    TEST_ASSERT(r.confidence < 0.7f,
                "confidence must be below threshold for ambiguous match");

    fprintf(stderr, "  ambiguous confidence = %.4f (should be < 0.7)\n",
            r.confidence);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 3: No candidates nearby — point far from all roads
 *
 * Point (50, 500, 0) is 500 m from Road A and 400 m from Road B.
 * Both are outside the 50 m search radius.
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_no_candidates() {
    MapMatcher mm;
    build_two_road_graph(mm);

    float pt[3] = {50.0f, 500.0f, 0.0f};
    float heading = 0.0f;

    MapMatchResult r = mm.match(pt, heading);

    TEST_ASSERT(r.matched_segment_id == -1,
                "no match when point is far from all roads");

    TEST_ASSERT(r.confidence == 0.0f,
                "confidence must be 0 when no candidates found");
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 4: Empty graph — safe default without crash
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_empty_graph() {
    MapMatcher mm;
    /* No nodes, no edges, not finalized */

    float pt[3] = {10.0f, 10.0f, 0.0f};
    float heading = 0.0f;

    MapMatchResult r = mm.match(pt, heading);

    TEST_ASSERT(r.matched_segment_id == -1,
                "empty graph should return no match");

    TEST_ASSERT(r.confidence == 0.0f,
                "empty graph should return zero confidence");
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 5: Viterbi transition — sequential points along Road A
 *
 * Three consecutive points moving east along Road A at y = 2 m offset.
 * The Viterbi should consistently match Road A with increasing confidence
 * (transition history reinforces the correct road).
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_viterbi_consistency() {
    MapMatcher mm;
    build_two_road_graph(mm);

    float heading = 0.0f;

    float pt1[3] = {10.0f, 2.0f, 0.0f};
    float pt2[3] = {20.0f, 2.0f, 0.0f};
    float pt3[3] = {30.0f, 2.0f, 0.0f};

    MapMatchResult r1 = mm.match(pt1, heading);
    MapMatchResult r2 = mm.match(pt2, heading);
    MapMatchResult r3 = mm.match(pt3, heading);

    TEST_ASSERT(r1.matched_segment_id == 0, "step 1: Road A");
    TEST_ASSERT(r2.matched_segment_id == 0, "step 2: Road A");
    TEST_ASSERT(r3.matched_segment_id == 0, "step 3: Road A");

    /* After multiple consistent steps, confidence should remain high */
    TEST_ASSERT(r3.confidence > 0.7f,
                "after 3 consistent steps, confidence should be high");

    /* Lateral offset should stay consistent */
    TEST_ASSERT_NEAR(r1.pseudo_measurement[0], 2.0f, 0.1f,
                     "lateral offset step 1");
    TEST_ASSERT_NEAR(r2.pseudo_measurement[0], 2.0f, 0.1f,
                     "lateral offset step 2");
    TEST_ASSERT_NEAR(r3.pseudo_measurement[0], 2.0f, 0.1f,
                     "lateral offset step 3");

    fprintf(stderr, "  confidences: %.4f → %.4f → %.4f\n",
            r1.confidence, r2.confidence, r3.confidence);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 6: Heading offset — bidirectional and oneway
 *
 * Build a graph with:
 *   - Edge 0: bidirectional, heading = 0 (East)
 *   - Edge 1: oneway East (heading = 0, from→to only)
 *
 * Test that heading offset is computed correctly for both travel directions.
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_heading_offset() {
    MapMatcher mm;

    /* Two nodes defining a west-east road */
    mm.add_node(0,   0.0f, 0.0f);
    mm.add_node(1, 100.0f, 0.0f);

    /* Edge 0: bidirectional */
    mm.add_edge(0, 0, 1, 0);

    mm.finalize_graph();

    /* Driving east (heading=0) on a west-east road → offset ≈ 0 */
    float pt[3]  = {50.0f, 2.0f, 0.0f};
    float h_east = 0.0f;
    MapMatchResult r1 = mm.match(pt, h_east);
    TEST_ASSERT_NEAR(r1.pseudo_measurement[1], 0.0f, 0.01f,
                     "eastbound on E-W road: heading offset ≈ 0");

    mm.reset();

    /* Driving west (heading=π) on a bidirectional W-E road → offset ≈ 0 */
    float h_west = static_cast<float>(M_PI);
    MapMatchResult r2 = mm.match(pt, h_west);
    TEST_ASSERT_NEAR(fabsf(r2.pseudo_measurement[1]), 0.0f, 0.01f,
                     "westbound on bidir E-W road: heading offset ≈ 0");

    /* Now test oneway: driving west on a oneway-east road → offset ≈ ±π */
    MapMatcher mm2;
    mm2.add_node(0,   0.0f, 0.0f);
    mm2.add_node(1, 100.0f, 0.0f);
    mm2.add_edge(0, 0, 1, 1);  /* oneway East */
    mm2.finalize_graph();

    MapMatchResult r3 = mm2.match(pt, h_west);
    TEST_ASSERT_NEAR(fabsf(r3.pseudo_measurement[1]),
                     static_cast<float>(M_PI), 0.01f,
                     "westbound on oneway-east: heading offset ≈ π");
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 7: Degenerate edge — zero-length segment (no crash)
 *
 * Edge from (50,0) to (50,0) — zero length. The matcher should handle
 * this gracefully without division by zero.
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_degenerate_edge() {
    MapMatcher mm;

    mm.add_node(0, 50.0f, 0.0f);
    mm.add_node(1, 50.0f, 0.0f);  /* Same position as node 0 */
    mm.add_edge(0, 0, 1, 0);

    mm.finalize_graph();

    float pt[3] = {50.0f, 1.0f, 0.0f};
    float heading = 0.0f;

    /* Should not crash */
    MapMatchResult r = mm.match(pt, heading);

    TEST_ASSERT(r.matched_segment_id >= 0,
                "degenerate edge should still produce a match");

    fprintf(stderr, "  degenerate edge: confidence = %.4f, "
            "lateral = %.4f\n", r.confidence, r.pseudo_measurement[0]);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 8: Reset clears Viterbi state
 *
 * After reset(), the matcher should behave as if it's the first observation.
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_reset() {
    MapMatcher mm;
    build_two_road_graph(mm);

    float heading = 0.0f;

    /* Match a few points to build Viterbi history */
    float pt1[3] = {10.0f, 2.0f, 0.0f};
    float pt2[3] = {20.0f, 2.0f, 0.0f};
    mm.match(pt1, heading);
    mm.match(pt2, heading);

    /* Reset */
    mm.reset();

    /* Next match should use emission-only (no Viterbi history) */
    float pt3[3] = {50.0f, 2.0f, 0.0f};
    MapMatchResult r = mm.match(pt3, heading);

    TEST_ASSERT(r.matched_segment_id == 0,
                "after reset, should still match correctly");
    TEST_ASSERT(r.confidence > 0.5f,
                "after reset, confidence should be reasonable");
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 9: Junction graph — T-intersection transition
 *
 * Graph:
 *   Node 0 (0,0) → Node 1 (100,0): Road A (east)
 *   Node 1 (100,0) → Node 2 (100,100): Road B (north from junction)
 *
 * Points moving east then turning north should transition from A to B.
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_junction_transition() {
    MapMatcher mm;

    mm.add_node(0,   0.0f,   0.0f);
    mm.add_node(1, 100.0f,   0.0f);   /* junction */
    mm.add_node(2, 100.0f, 100.0f);

    mm.add_edge(0, 0, 1, 0);   /* Road A: east */
    mm.add_edge(1, 1, 2, 0);   /* Road B: north from junction */

    mm.finalize_graph();

    /* Drive east along Road A */
    float h_east  = 0.0f;
    float h_north = static_cast<float>(M_PI / 2.0);

    float pa[3] = {50.0f, 2.0f, 0.0f};
    float pb[3] = {80.0f, 2.0f, 0.0f};
    MapMatchResult ra = mm.match(pa, h_east);
    MapMatchResult rb = mm.match(pb, h_east);

    TEST_ASSERT(ra.matched_segment_id == 0, "pre-junction: Road A");
    TEST_ASSERT(rb.matched_segment_id == 0, "approaching junction: Road A");

    /* Turn north at junction — points well onto Road B (x≈100, y>0),
     * clearly outside Road A's 50m search radius from y=0 */
    float pc[3] = {102.0f, 30.0f, 0.0f};
    float pd[3] = {102.0f, 60.0f, 0.0f};
    MapMatchResult rc = mm.match(pc, h_north);
    MapMatchResult rd = mm.match(pd, h_north);

    TEST_ASSERT(rd.matched_segment_id == 1,
                "after junction: should match Road B");

    fprintf(stderr, "  junction transition: A=%d → A=%d → ?=%d → B=%d\n",
            ra.matched_segment_id, rb.matched_segment_id,
            rc.matched_segment_id, rd.matched_segment_id);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * TEST 10: Signed lateral offset direction
 *
 * Point to the RIGHT of an eastward road should give negative offset.
 * Point to the LEFT should give positive offset.
 * ═════════════════════════════════════════════════════════════════════════════ */
static void test_signed_offset_direction() {
    MapMatcher mm;

    mm.add_node(0,   0.0f, 0.0f);
    mm.add_node(1, 100.0f, 0.0f);
    mm.add_edge(0, 0, 1, 0);
    mm.finalize_graph();

    float heading = 0.0f;

    /* Point above road (left of eastward direction) → positive */
    float pt_left[3] = {50.0f, 5.0f, 0.0f};
    MapMatchResult rl = mm.match(pt_left, heading);
    TEST_ASSERT(rl.pseudo_measurement[0] > 0.0f,
                "point above east road → positive lateral offset (left)");

    mm.reset();

    /* Point below road (right of eastward direction) → negative */
    float pt_right[3] = {50.0f, -5.0f, 0.0f};
    MapMatchResult rr = mm.match(pt_right, heading);
    TEST_ASSERT(rr.pseudo_measurement[0] < 0.0f,
                "point below east road → negative lateral offset (right)");

    TEST_ASSERT_NEAR(rl.pseudo_measurement[0], 5.0f, 0.1f,
                     "left offset magnitude should be 5m");
    TEST_ASSERT_NEAR(rr.pseudo_measurement[0], -5.0f, 0.1f,
                     "right offset magnitude should be -5m");
}

/* ═════════════════════════════════════════════════════════════════════════════
 * main
 * ═════════════════════════════════════════════════════════════════════════════ */

int main() {
    fprintf(stderr, "\n====== test_map_matching ======\n\n");

    RUN_TEST(test_known_geometry);
    RUN_TEST(test_ambiguous_low_confidence);
    RUN_TEST(test_no_candidates);
    RUN_TEST(test_empty_graph);
    RUN_TEST(test_viterbi_consistency);
    RUN_TEST(test_heading_offset);
    RUN_TEST(test_degenerate_edge);
    RUN_TEST(test_reset);
    RUN_TEST(test_junction_transition);
    RUN_TEST(test_signed_offset_direction);

    fprintf(stderr, "\n====== Results: %d/%d passed",
            g_tests_passed, g_tests_run);
    if (g_tests_failed > 0) {
        fprintf(stderr, " (%d FAILED)", g_tests_failed);
    }
    fprintf(stderr, " ======\n\n");

    return (g_tests_failed == 0) ? 0 : 1;
}
