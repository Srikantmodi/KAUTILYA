package com.sih.deadreckoning.mapmatching

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * OsmLoader — Loads the pre-converted road graph (.drgg) from app assets.
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 5 (Map-Matching & Map Data Pipeline)
 * Ref:   PRD §6.12 — "pre-converted once and shipped as an app asset —
 *        core-engine never parses raw OSM XML/PBF at runtime"
 *
 * The .drgg file is produced offline by convert_to_graph.py and placed
 * in res/raw/. This class extracts it to the app's internal storage
 * (since native code needs a filesystem path, not a resource stream)
 * and passes the path to the C++ MapMatcher via JNI.
 *
 * Usage:
 *   val loader = OsmLoader(context)
 *   val success = loader.loadGraph()         // extracts + loads via JNI
 *   loader.setSessionOrigin(lat, lon)        // call when first GNSS fix arrives
 */
class OsmLoader(private val context: Context) {

    companion object {
        private const val TAG = "OsmLoader"

        /** Raw resource name for the pre-converted road graph. */
        private const val GRAPH_RESOURCE_NAME = "road_graph"

        /** Filename in internal storage. */
        private const val GRAPH_FILENAME = "road_graph.drgg"

        init {
            System.loadLibrary("core_engine")
        }
    }

    /** Whether the graph has been successfully loaded into C++. */
    var isLoaded: Boolean = false
        private set

    /**
     * Extract the .drgg graph from res/raw/ to internal storage and load
     * it into the C++ MapMatcher via JNI.
     *
     * @return true if the graph was loaded successfully.
     */
    fun loadGraph(): Boolean {
        try {
            val graphFile = extractGraphToInternal()
            if (graphFile == null) {
                Log.e(TAG, "Failed to extract graph to internal storage")
                return false
            }

            val success = nativeLoadGraph(graphFile.absolutePath)
            if (success) {
                isLoaded = true
                Log.i(TAG, "Graph loaded: ${graphFile.absolutePath}")
            } else {
                Log.e(TAG, "C++ failed to load graph: ${graphFile.absolutePath}")
            }
            return success
        } catch (e: Exception) {
            Log.e(TAG, "Exception loading graph", e)
            return false
        }
    }

    /**
     * Load a graph from an explicit filesystem path (for testing / edge cases).
     *
     * @param path absolute path to a .drgg file.
     * @return true on success.
     */
    fun loadGraphFromPath(path: String): Boolean {
        val success = nativeLoadGraph(path)
        isLoaded = success
        return success
    }

    /**
     * Set the session ENU origin — converts all graph node lat/lon to ENU.
     * Call once when the first GNSS fix arrives (same time as UKF init).
     *
     * @param latDeg WGS-84 latitude in degrees.
     * @param lonDeg WGS-84 longitude in degrees.
     */
    fun setSessionOrigin(latDeg: Double, lonDeg: Double) {
        if (!isLoaded) {
            Log.w(TAG, "setSessionOrigin called before graph loaded — ignored")
            return
        }
        nativeSetSessionOrigin(latDeg, lonDeg)
        Log.i(TAG, "Session origin set: ($latDeg, $lonDeg)")
    }

    /**
     * Extract the raw resource to a file in internal storage.
     * Only re-extracts if the file doesn't exist or is empty.
     */
    private fun extractGraphToInternal(): File? {
        val outFile = File(context.filesDir, GRAPH_FILENAME)

        // Skip extraction if already present and non-empty
        if (outFile.exists() && outFile.length() > 0) {
            Log.d(TAG, "Graph already extracted: ${outFile.absolutePath}")
            return outFile
        }

        // Find the raw resource ID
        val resId = context.resources.getIdentifier(
            GRAPH_RESOURCE_NAME, "raw", context.packageName
        )
        if (resId == 0) {
            Log.e(TAG, "Raw resource '$GRAPH_RESOURCE_NAME' not found. " +
                       "Run convert_to_graph.py and place the .drgg in res/raw/")
            return null
        }

        // Extract to internal storage
        try {
            context.resources.openRawResource(resId).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Extracted graph: ${outFile.length()} bytes → " +
                       outFile.absolutePath)
            return outFile
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract graph resource", e)
            outFile.delete()
            return null
        }
    }

    // ── JNI native methods (implemented in C++ jni_bridge.cpp, Member 4) ──

    /** Load graph from filesystem path into C++ MapMatcher. */
    private external fun nativeLoadGraph(path: String): Boolean

    /** Set session origin for ENU conversion. */
    private external fun nativeSetSessionOrigin(latDeg: Double, lonDeg: Double)
}
