/**
 * TelemetryOverlay.kt — HUD overlay for live navigation telemetry
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 1 (App Shell, UI & Navigation Frontend)
 *
 * Purely presentational per §2.2 — displays state, never computes it.
 * Shows: speed, heading, mode badge, drift %, satellite/quality info.
 *
 * This is the "live drift %, mode badge" HUD from §2.1. All values
 * are read directly from NavState and NavigationMode — zero business logic.
 *
 * References:
 *   PRD §2.1  — "smooth vehicle icon, live drift %, mode badge"
 *   PRD §2.2  — "Frontend... thin, purely presentational"
 *   api-contracts.md §4   — NavState shape
 *   api-contracts.md §7   — NavigationMode enum
 */
package com.sih.deadreckoning.ui

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import com.sih.deadreckoning.fusion.NavState
import com.sih.deadreckoning.mode.NavigationMode
import kotlin.math.sqrt

/**
 * Floating HUD overlay showing live navigation telemetry.
 *
 * Layout: a vertical stack of info rows floating over the map.
 * Updated at ~10 Hz by [MapFragment] via [update].
 *
 * This view contains ZERO business logic. It formats and displays
 * values that are already computed by the C++ fusion/mode engines.
 *
 * Usage (from MapFragment or MainActivity):
 * ```
 * val overlay = TelemetryOverlay(context)
 * overlay.update(navState, mode)
 * ```
 */
class TelemetryOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : LinearLayout(context, attrs, defStyleAttr) {

    companion object {
        private const val TAG = "TelemetryOverlay"

        /** Convert m/s to km/h for display. */
        private const val MPS_TO_KMPH = 3.6f
    }

    /* ── UI elements ─────────────────────────────────────────────────────── */

    private val modeBadge: TextView
    private val speedText: TextView
    private val headingText: TextView
    private val driftText: TextView
    private val positionText: TextView

    init {
        orientation = VERTICAL
        gravity = Gravity.START
        setPadding(24, 24, 24, 24)

        // Semi-transparent dark background for readability over the map
        setBackgroundColor(Color.argb(180, 20, 20, 30))

        // Mode badge — prominent, color-coded
        modeBadge = createLabel("—", 18f, true).also { addView(it) }

        // Speed — large and readable for driver glances
        speedText = createLabel("0.0 km/h", 22f, true).also { addView(it) }

        // Heading — compass direction
        headingText = createLabel("HDG: —°", 14f, false).also { addView(it) }

        // Position (ENU) — for telemetry/debug
        positionText = createLabel("POS: — m", 12f, false).also { addView(it) }

        // Drift estimate — from covariance trace
        driftText = createLabel("DRIFT: —%", 14f, false).also { addView(it) }
    }

    /**
     * Update all telemetry displays with the latest state.
     *
     * Called at ~10 Hz by MapFragment's refresh loop. This method
     * ONLY formats and displays — no computation beyond unit conversion
     * (m/s → km/h) and covariance trace (a sum, not fusion math).
     *
     * @param state  current fused NavState from FusionBridge (Member 4)
     * @param mode   current NavigationMode from ModeManager (C++ via Member 4)
     */
    fun update(state: NavState, mode: NavigationMode) {
        // ── Mode badge ───────────────────────────────────────────────────
        modeBadge.text = mode.displayName
        modeBadge.setTextColor(mode.colorHex.toInt())

        // ── Speed (m/s → km/h) ───────────────────────────────────────────
        val speedKmh = state.speed * MPS_TO_KMPH
        speedText.text = String.format("%.1f km/h", speedKmh)

        // ── Heading (radians → degrees, 0=North, CW) ────────────────────
        // NavState heading is in radians. Convert to degrees for display.
        // Display convention: 0° = North, clockwise.
        val headingDeg = Math.toDegrees(state.headingRad.toDouble())
        val normalizedDeg = ((headingDeg % 360.0) + 360.0) % 360.0
        val compassDir = degreesToCompass(normalizedDeg)
        headingText.text = String.format("HDG: %.0f° %s", normalizedDeg, compassDir)

        // ── Position (ENU meters) ────────────────────────────────────────
        positionText.text = String.format(
            "POS: E%.1f  N%.1f  U%.1f m",
            state.positionE, state.positionN, state.positionU
        )

        // ── Drift estimate ───────────────────────────────────────────────
        // Display the position covariance trace as a rough uncertainty
        // metric. This is NOT drift computation (that's Member 6's job) —
        // it's just visualizing the diagonal of the covariance matrix
        // as a "position uncertainty radius" for the pilot's awareness.
        val positionUncertainty = computePositionUncertainty(state.covariance)
        driftText.text = String.format("UNCERT: ±%.1f m", positionUncertainty)

        // Color the drift text based on uncertainty level
        driftText.setTextColor(
            when {
                positionUncertainty < 5.0f  -> Color.parseColor("#4CAF50")  // green
                positionUncertainty < 20.0f -> Color.parseColor("#FF9800")  // amber
                else                        -> Color.parseColor("#F44336")  // red
            }
        )
    }

    /**
     * Reset all displays to default/empty state.
     * Call at session start or when data source is unavailable.
     */
    fun reset() {
        modeBadge.text = "—"
        modeBadge.setTextColor(Color.GRAY)
        speedText.text = "0.0 km/h"
        headingText.text = "HDG: —°"
        positionText.text = "POS: — m"
        driftText.text = "UNCERT: —"
        driftText.setTextColor(Color.GRAY)
    }

    /* ── Private helpers (purely presentational) ─────────────────────────── */

    /**
     * Extract position uncertainty from the covariance matrix diagonal.
     *
     * covariance is a 4×4 row-major float array (16 elements):
     *   [0]  = var(east)
     *   [5]  = var(north)
     *   [10] = var(up)
     *   [15] = var(heading)
     *
     * We display sqrt(var_east + var_north) as 2D horizontal uncertainty.
     * This is a DISPLAY metric, not a drift calculation.
     */
    private fun computePositionUncertainty(covariance: FloatArray): Float {
        if (covariance.size < 16) return 0f
        val varEast = covariance[0].coerceAtLeast(0f)
        val varNorth = covariance[5].coerceAtLeast(0f)
        return sqrt(varEast + varNorth)
    }

    /**
     * Convert compass degrees (0=N, CW) to a cardinal direction string.
     * Purely presentational — no navigation math.
     */
    private fun degreesToCompass(degrees: Double): String {
        val directions = arrayOf("N", "NE", "E", "SE", "S", "SW", "W", "NW")
        val index = ((degrees + 22.5) / 45.0).toInt() % 8
        return directions[index]
    }

    /**
     * Create a styled label for the overlay.
     */
    private fun createLabel(
        initialText: String,
        textSizeSp: Float,
        bold: Boolean
    ): TextView {
        return TextView(context).apply {
            text = initialText
            setTextColor(Color.WHITE)
            textSize = textSizeSp
            if (bold) {
                setTypeface(typeface, Typeface.BOLD)
            }
            val lp = LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
            )
            lp.bottomMargin = 8
            layoutParams = lp
        }
    }
}
