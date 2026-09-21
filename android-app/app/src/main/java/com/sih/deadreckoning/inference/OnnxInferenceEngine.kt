/**
 * OnnxInferenceEngine.kt — On-device velocity model inference (ONNX Runtime)
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)
 *
 * Parity/fallback inference engine using ONNX Runtime Android.
 * Same InferenceResult contract as TFLiteInferenceEngine, same hidden-state
 * lifecycle, same normalization — the only difference is the runtime backend.
 *
 * This is the PRIMARY engine during the placeholder-model phase (since the
 * placeholder model is in ONNX format, not yet converted to TFLite).
 *
 * CRITICAL CONTRACT (identical to TFLiteInferenceEngine — see §8):
 *   Input:  [1, 20, 6] normalized features + [1, 1, 64] GRU h_in
 *   Output: [1, 20, 1] speed_metric (take last timestep)
 *           [1, 20, 1] stationary_logit (take last timestep, apply sigmoid)
 *           [1, 1, 64] h_out (cache and feed back)
 *
 * References:
 *   api-contracts.md §8, PRD §3.1 (onnxruntime-android as parity/fallback)
 */
package com.sih.deadreckoning.inference

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlin.math.abs
import kotlin.math.exp

/**
 * ONNX Runtime Android inference engine — parity/fallback path for
 * TFLiteInferenceEngine.
 *
 * Usage is identical to TFLiteInferenceEngine:
 * ```
 * val engine = OnnxInferenceEngine()
 * engine.initialize(context, "velocity_model.onnx")
 * val result = engine.infer(leveledAccel, rawGyro)
 * engine.resetSession()
 * engine.close()
 * ```
 */
class OnnxInferenceEngine {

    companion object {
        private const val TAG = "OnnxInferenceEngine"

        /** All constants identical to TFLiteInferenceEngine — ONE source of truth. */
        const val WINDOW_SIZE = TFLiteInferenceEngine.WINDOW_SIZE
        const val NUM_FEATURES = TFLiteInferenceEngine.NUM_FEATURES
        const val GRU_HIDDEN_SIZE = TFLiteInferenceEngine.GRU_HIDDEN_SIZE
        const val INFERENCE_TIMEOUT_MS = TFLiteInferenceEngine.INFERENCE_TIMEOUT_MS
        const val ZUPT_THRESHOLD = TFLiteInferenceEngine.ZUPT_THRESHOLD
        val NORM_MEAN = TFLiteInferenceEngine.NORM_MEAN
        val NORM_STD = TFLiteInferenceEngine.NORM_STD
    }

    /* ── Sliding window FIFO ────────────────────────────────────────────── */
    private val windowBuffer = Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) }
    private var windowHead = 0
    private var windowCount = 0

    /* ── GRU hidden state — PERSISTENT across calls ─────────────────────── */
    /**
     * Persistent GRU hidden state. Same lifecycle rules as TFLiteInferenceEngine:
     * reset only at session boundaries, never mid-session, never per-call.
     */
    private var gruHiddenState = FloatArray(GRU_HIDDEN_SIZE) { 0.0f }

    /* ── ONNX Runtime session ───────────────────────────────────────────── */
    private var isInitialized = false

    // In production: private var ortSession: OrtSession? = null
    // For now, the session is a placeholder until onnxruntime-android is
    // added to the Gradle dependencies.

    private val inferenceScope = CoroutineScope(
        Dispatchers.Default + SupervisorJob()
    )

    /* ── Public API (identical contract to TFLiteInferenceEngine) ──────── */

    /**
     * Load the ONNX model once at app startup.
     */
    fun initialize(context: Context, modelFileName: String = "velocity_model.onnx") {
        if (isInitialized) {
            Log.w(TAG, "Already initialized — skipping")
            return
        }

        Log.i(TAG, "Initializing ONNX inference engine with model: $modelFileName")

        // In production:
        //   val env = OrtEnvironment.getEnvironment()
        //   val modelPath = extractModelFromAssets(context, modelFileName)
        //   ortSession = env.createSession(modelPath)

        resetSession()
        isInitialized = true
        Log.i(TAG, "ONNX inference engine initialized")
    }

    /**
     * Push one sample and run inference if the window is full.
     * Same contract as TFLiteInferenceEngine.infer().
     */
    fun infer(leveledAccel: FloatArray, rawGyro: FloatArray): InferenceResult? {
        require(leveledAccel.size == 3) {
            "leveledAccel must have 3 elements, got ${leveledAccel.size}"
        }
        require(rawGyro.size == 3) {
            "rawGyro must have 3 elements, got ${rawGyro.size}"
        }

        val features = FloatArray(NUM_FEATURES)
        features[0] = leveledAccel[0]
        features[1] = leveledAccel[1]
        features[2] = leveledAccel[2]
        features[3] = rawGyro[0]
        features[4] = rawGyro[1]
        features[5] = rawGyro[2]

        pushToWindow(features)

        if (windowCount < WINDOW_SIZE) {
            return null
        }

        return runInferenceInternal()
    }

    /**
     * Reset GRU hidden state — ONLY at new session start.
     */
    fun resetSession() {
        gruHiddenState.fill(0.0f)
        windowHead = 0
        windowCount = 0
        Log.i(TAG, "Session reset — GRU state zeroed")
    }

    fun close() {
        inferenceScope.cancel()
        // ortSession?.close()
        isInitialized = false
        Log.i(TAG, "ONNX inference engine closed")
    }

    fun isReady(): Boolean = isInitialized

    fun getHiddenState(): FloatArray = gruHiddenState.copyOf()
    fun getWindowCount(): Int = windowCount

    /* ── Internals ─────────────────────────────────────────────────────── */

    private fun pushToWindow(features: FloatArray) {
        System.arraycopy(features, 0, windowBuffer[windowHead], 0, NUM_FEATURES)
        windowHead = (windowHead + 1) % WINDOW_SIZE
        if (windowCount < WINDOW_SIZE) windowCount++
    }

    private fun getWindowArray(): FloatArray {
        val arr = FloatArray(WINDOW_SIZE * NUM_FEATURES)
        for (i in 0 until WINDOW_SIZE) {
            val srcIdx = if (windowCount < WINDOW_SIZE) {
                i
            } else {
                (windowHead + i) % WINDOW_SIZE
            }
            System.arraycopy(
                windowBuffer[srcIdx], 0,
                arr, i * NUM_FEATURES,
                NUM_FEATURES
            )
        }
        return arr
    }

    private fun normalizeWindow(window: FloatArray): FloatArray {
        val normalized = FloatArray(window.size)
        for (i in window.indices) {
            val featureIdx = i % NUM_FEATURES
            normalized[i] = (window[i] - NORM_MEAN[featureIdx]) / NORM_STD[featureIdx]
        }
        return normalized
    }

    private fun runInferenceInternal(): InferenceResult {
        val rawWindow = getWindowArray()
        val normWindow = normalizeWindow(rawWindow)

        // In production with ONNX Runtime:
        //   val inputFeatures = OnnxTensor.createTensor(env,
        //       FloatBuffer.wrap(normWindow), longArrayOf(1, 20, 6))
        //   val hIn = OnnxTensor.createTensor(env,
        //       FloatBuffer.wrap(gruHiddenState), longArrayOf(1, 1, 64))
        //   val results = ortSession!!.run(mapOf(
        //       "input_features" to inputFeatures,
        //       "h_in" to hIn
        //   ))
        //   val speedAll = (results[0].value as Array<*>)  // [1, 20, 1]
        //   val stationaryAll = (results[1].value as Array<*>)  // [1, 20, 1]
        //   val hOut = (results[2].value as Array<*>)  // [1, 1, 64]
        //   // Take last timestep...

        // Placeholder output (same as TFLiteInferenceEngine)
        val speedMetric = computePlaceholderSpeed(normWindow)
        val stationaryLogit = computePlaceholderStationary(normWindow)
        val stationaryProb = sigmoid(stationaryLogit)
        updatePlaceholderHiddenState(normWindow)

        return InferenceResult(
            deltaV = speedMetric,
            stationaryProbability = stationaryProb
        )
    }

    private fun sigmoid(x: Float): Float {
        return if (x >= 0) {
            1.0f / (1.0f + exp(-x))
        } else {
            val expX = exp(x)
            expX / (1.0f + expX)
        }
    }

    private fun computePlaceholderSpeed(normWindow: FloatArray): Float {
        var sum = 0.0f
        val lastRow = WINDOW_SIZE - 1
        for (f in 0 until 3) {
            sum += abs(normWindow[lastRow * NUM_FEATURES + f])
        }
        return sum / 3.0f * 0.1f
    }

    private fun computePlaceholderStationary(normWindow: FloatArray): Float {
        var energy = 0.0f
        val lastRow = WINDOW_SIZE - 1
        for (f in 0 until NUM_FEATURES) {
            val v = normWindow[lastRow * NUM_FEATURES + f]
            energy += v * v
        }
        return 2.0f - energy * 0.5f
    }

    private fun updatePlaceholderHiddenState(normWindow: FloatArray) {
        val lastRow = WINDOW_SIZE - 1
        for (h in gruHiddenState.indices) {
            val featureIdx = h % NUM_FEATURES
            val input = normWindow[lastRow * NUM_FEATURES + featureIdx]
            gruHiddenState[h] = 0.9f * gruHiddenState[h] + 0.1f * input
        }
    }
}
