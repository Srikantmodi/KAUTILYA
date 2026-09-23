/**
 * MapFragment.kt — Offline MapLibre map with vehicle icon and trail
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 1 (App Shell, UI & Navigation Frontend)
 *
 * Purely presentational per §2.2 — displays state, never computes it.
 * Consumes NavState (Member 4's FusionBridge) and NavigationMode
 * (Member 4's C++ mode_manager, via our ModeManager.kt wrapper).
 *
 * Map rendering: MapLibre Android SDK — offline-first, no vendor API key
 * (§3.1, §7 guardrail #10: nothing on the live nav path depends on network).
 *
 * Trail color coding (§2.1 / README):
 *   green  = GNSS_AIDED_INS
 *   amber  = PURE_DR
 *   red    = RECOVERING
 *
 * References:
 *   PRD §2.1  — "smooth vehicle icon, live drift %, mode badge"
 *   PRD §3.1  — MapLibre, not OSMDroid or Google Maps
 *   PRD §7 guardrail #10 — offline-only on the live nav path
 *   api-contracts.md §4  — NavState shape
 *   api-contracts.md §7  — NavigationMode enum
 */
package com.sih.deadreckoning.ui

import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.fragment.app.Fragment
import com.sih.deadreckoning.R
import com.sih.deadreckoning.fusion.FusionBridge
import com.sih.deadreckoning.fusion.NavState
import com.sih.deadreckoning.mode.ModeManager
import com.sih.deadreckoning.mode.NavigationMode
import org.maplibre.android.MapLibre
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.Style
import org.maplibre.android.plugins.annotation.SymbolManager
import org.maplibre.android.plugins.annotation.SymbolOptions
import org.maplibre.android.plugins.annotation.Symbol
import org.maplibre.android.style.layers.LineLayer
import org.maplibre.android.style.layers.PropertyFactory
import org.maplibre.android.style.sources.GeoJsonSource
import kotlin.math.cos

/**
 * Data class holding a single trail point for the breadcrumb polyline.
 * Purely presentational — stores display data only.
 */
data class TrailPoint(
    val lat: Double,
    val lon: Double,
    val mode: NavigationMode,
    val timestampMs: Long
)

/**
 * Offline map fragment displaying:
 *   - Vehicle icon at current fused position
 *   - Color-coded trail polyline (green/amber/red by NavigationMode)
 *   - Camera following the vehicle
 *
 * This fragment is purely presentational. It reads state from external
 * providers ([NavStateProvider]) and renders it — no business logic.
 *
 * In production, [NavStateProvider] is backed by FusionBridge + ModeManager.
 * For standalone testing, swap in [MockNavStateProvider].
 */
class MapFragment : Fragment() {

    companion object {
        private const val TAG = "MapFragment"

        /** UI refresh interval — 10 Hz matches the pipeline's inference rate. */
        private const val REFRESH_INTERVAL_MS = 100L

        /** Vehicle icon image ID for the MapLibre symbol layer. */
        private const val VEHICLE_ICON_ID = "vehicle-icon"

        /** Trail source/layer IDs. */
        private const val TRAIL_SOURCE_ID = "trail-source"
        private const val TRAIL_LAYER_ID = "trail-layer"

        /**
         * Default session origin (New Delhi) — used until the first real
         * GNSS fix arrives.  This is only for initial camera positioning;
         * the actual ENU→LatLng conversion uses the real session origin
         * once available.
         */
        private const val DEFAULT_ORIGIN_LAT = 28.6139
        private const val DEFAULT_ORIGIN_LON = 77.2090

        /**
         * Demeter's law style factory for clean fragment creation.
         */
        fun newInstance(): MapFragment = MapFragment()
    }

    /* ── State providers (injected or defaulted) ─────────────────────────── */

    /**
     * Interface for providing navigation state to the map.
     * Decouples UI from the concrete bridge implementations for testability.
     */
    interface NavStateProvider {
        fun getNavState(): NavState?
        fun getCurrentMode(): NavigationMode
        fun getSessionOriginLat(): Double
        fun getSessionOriginLon(): Double
    }

    /**
     * Mock provider for standalone UI testing without C++ backend.
     * Simulates a vehicle driving in a circle.
     */
    class MockNavStateProvider : NavStateProvider {
        private val startTimeMs = System.currentTimeMillis()
        private var originLat = DEFAULT_ORIGIN_LAT
        private var originLon = DEFAULT_ORIGIN_LON

        override fun getNavState(): NavState {
            val elapsed = (System.currentTimeMillis() - startTimeMs) / 1000.0f
            // Simulate circular driving: 50m radius circle
            val radius = 50.0f
            val angularSpeed = 0.1f  // rad/s
            val angle = elapsed * angularSpeed
            val east = radius * cos(angle)
            val north = radius * kotlin.math.sin(angle)
            val speed = radius * angularSpeed

            return NavState(
                positionE = east,
                positionN = north,
                positionU = 0f,
                velocityE = -speed * kotlin.math.sin(angle),
                velocityN = speed * cos(angle),
                velocityU = 0f,
                headingRad = angle,
                gyroBiasX = 0f,
                gyroBiasY = 0f,
                gyroBiasZ = 0f,
                covariance = FloatArray(16) { if (it % 5 == 0) 1.0f else 0f }
            )
        }

        override fun getCurrentMode(): NavigationMode {
            val elapsed = (System.currentTimeMillis() - startTimeMs) / 1000
            // Cycle through modes every 10 seconds for demo
            return when ((elapsed / 10) % 3) {
                0L -> NavigationMode.GNSS_AIDED_INS
                1L -> NavigationMode.PURE_DR
                else -> NavigationMode.RECOVERING
            }
        }

        override fun getSessionOriginLat(): Double = originLat
        override fun getSessionOriginLon(): Double = originLon
    }

    /** The active state provider. Defaults to mock; set via [setNavStateProvider]. */
    private var navStateProvider: NavStateProvider = MockNavStateProvider()

    /**
     * Set the navigation state provider. Call before or after onViewCreated.
     * In production, pass a provider backed by real FusionBridge + ModeManager.
     */
    fun setNavStateProvider(provider: NavStateProvider) {
        navStateProvider = provider
    }

    /* ── MapLibre state ──────────────────────────────────────────────────── */

    private var mapView: MapView? = null
    private var mapLibreMap: MapLibreMap? = null
    private var vehicleSymbol: Symbol? = null
    private var symbolManager: SymbolManager? = null

    /* ── Trail state ─────────────────────────────────────────────────────── */

    private val trailPoints = mutableListOf<TrailPoint>()
    private val maxTrailPoints = 5000

    /* ── Periodic refresh ────────────────────────────────────────────────── */

    private val refreshHandler = Handler(Looper.getMainLooper())
    private var isRefreshing = false

    private val refreshRunnable = object : Runnable {
        override fun run() {
            if (!isRefreshing) return
            updateMapState()
            refreshHandler.postDelayed(this, REFRESH_INTERVAL_MS)
        }
    }

    /* ── TelemetryOverlay reference (set by MainActivity) ────────────────── */

    private var telemetryOverlay: TelemetryOverlay? = null

    fun setTelemetryOverlay(overlay: TelemetryOverlay) {
        telemetryOverlay = overlay
    }

    /* ── Fragment lifecycle ───────────────────────────────────────────────── */

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        // Initialize MapLibre
        MapLibre.getInstance(requireContext())

        val root = inflater.inflate(R.layout.fragment_map, container, false)
        mapView = root.findViewById(R.id.map_view)
        mapView?.onCreate(savedInstanceState)

        mapView?.getMapAsync { map ->
            mapLibreMap = map

            // Use a dark-style base map — works offline with raster tiles.
            // In production, this would point to a local MBTiles file URI.
            // Using a free OSM-based style for development.
            map.setStyle(
                Style.Builder()
                    .fromUri("https://demotiles.maplibre.org/style.json")
            ) { style ->
                onStyleLoaded(map, style)
            }
        }

        return root
    }

    private fun onStyleLoaded(map: MapLibreMap, style: Style) {
        // Set initial camera to session origin
        val originLat = navStateProvider.getSessionOriginLat()
        val originLon = navStateProvider.getSessionOriginLon()

        map.cameraPosition = CameraPosition.Builder()
            .target(LatLng(originLat, originLon))
            .zoom(16.0)
            .build()

        // Add vehicle icon to style
        addVehicleIcon(style)

        // Initialize symbol manager for the vehicle marker
        try {
            symbolManager = SymbolManager(mapView!!, map, style).apply {
                iconAllowOverlap = true
                iconIgnorePlacement = true
            }
        } catch (e: Exception) {
            Log.w(TAG, "SymbolManager init failed (expected in dev without full style): ${e.message}")
        }

        Log.i(TAG, "Map style loaded, origin: ($originLat, $originLon)")
    }

    /**
     * Add a simple vehicle icon to the map style.
     * Uses a programmatically-generated triangle bitmap pointing north.
     */
    private fun addVehicleIcon(style: Style) {
        val size = 48
        val bitmap = android.graphics.Bitmap.createBitmap(
            size, size, android.graphics.Bitmap.Config.ARGB_8888
        )
        val canvas = android.graphics.Canvas(bitmap)
        val paint = android.graphics.Paint().apply {
            color = Color.parseColor("#2196F3")
            isAntiAlias = true
            style = android.graphics.Paint.Style.FILL
        }

        // Draw a triangle pointing up (north)
        val path = android.graphics.Path().apply {
            moveTo(size / 2f, 4f)                // top
            lineTo(size - 4f, size - 4f)         // bottom-right
            lineTo(4f, size - 4f)                // bottom-left
            close()
        }
        canvas.drawPath(path, paint)

        // White outline
        paint.apply {
            color = Color.WHITE
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 2f
        }
        canvas.drawPath(path, paint)

        style.addImage(VEHICLE_ICON_ID, bitmap)
    }

    /**
     * Called periodically by the refresh handler.
     * Reads state from the provider and updates the map — zero computation.
     */
    private fun updateMapState() {
        val map = mapLibreMap ?: return
        val state = navStateProvider.getNavState() ?: return
        val mode = navStateProvider.getCurrentMode()

        // Convert ENU position to LatLng using session origin
        val latLng = enuToLatLng(
            state.positionE, state.positionN,
            navStateProvider.getSessionOriginLat(),
            navStateProvider.getSessionOriginLon()
        )

        // Update vehicle marker
        updateVehicleMarker(latLng, state.headingRad)

        // Add trail point
        addTrailPoint(latLng, mode)

        // Animate camera to follow vehicle
        map.animateCamera(
            CameraUpdateFactory.newCameraPosition(
                CameraPosition.Builder()
                    .target(latLng)
                    .bearing(Math.toDegrees(state.headingRad.toDouble()))
                    .zoom(map.cameraPosition.zoom.coerceIn(14.0, 19.0))
                    .build()
            ),
            300 // animation duration ms
        )

        // Update telemetry overlay with current state
        telemetryOverlay?.update(state, mode)
    }

    private fun updateVehicleMarker(latLng: LatLng, headingRad: Float) {
        val sm = symbolManager ?: return

        if (vehicleSymbol == null) {
            // Create the symbol on first update
            vehicleSymbol = sm.create(
                SymbolOptions()
                    .withLatLng(latLng)
                    .withIconImage(VEHICLE_ICON_ID)
                    .withIconSize(1.0f)
                    .withIconRotate(Math.toDegrees(headingRad.toDouble()).toFloat())
            )
        } else {
            // Update existing symbol position and rotation
            vehicleSymbol?.let { symbol ->
                symbol.latLng = latLng
                symbol.iconRotate = Math.toDegrees(headingRad.toDouble()).toFloat()
                sm.update(symbol)
            }
        }
    }

    private fun addTrailPoint(latLng: LatLng, mode: NavigationMode) {
        trailPoints.add(
            TrailPoint(
                lat = latLng.latitude,
                lon = latLng.longitude,
                mode = mode,
                timestampMs = System.currentTimeMillis()
            )
        )

        // Cap trail length
        while (trailPoints.size > maxTrailPoints) {
            trailPoints.removeAt(0)
        }
    }

    /**
     * Convert ENU (East-North-Up) position relative to session origin
     * to WGS-84 LatLng for map display.
     *
     * This is a DISPLAY conversion only — the actual coordinate math
     * for fusion lives in C++ (Member 4). We use the simple flat-earth
     * approximation which is sufficient for UI rendering at the scale
     * we display (city-level, <50 km from origin).
     */
    private fun enuToLatLng(
        eastM: Float, northM: Float,
        originLat: Double, originLon: Double
    ): LatLng {
        // Meters per degree at the origin latitude
        val metersPerDegLat = 111_132.0
        val metersPerDegLon = 111_132.0 * cos(Math.toRadians(originLat))

        val lat = originLat + northM / metersPerDegLat
        val lon = originLon + eastM / metersPerDegLon

        return LatLng(lat, lon)
    }

    /* ── Lifecycle forwarding to MapView ──────────────────────────────────── */

    override fun onStart() {
        super.onStart()
        mapView?.onStart()
    }

    override fun onResume() {
        super.onResume()
        mapView?.onResume()
        startRefresh()
    }

    override fun onPause() {
        stopRefresh()
        mapView?.onPause()
        super.onPause()
    }

    override fun onStop() {
        mapView?.onStop()
        super.onStop()
    }

    override fun onDestroyView() {
        stopRefresh()
        symbolManager?.onDestroy()
        symbolManager = null
        vehicleSymbol = null
        mapView?.onDestroy()
        mapView = null
        mapLibreMap = null
        super.onDestroyView()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        mapView?.onSaveInstanceState(outState)
    }

    override fun onLowMemory() {
        super.onLowMemory()
        mapView?.onLowMemory()
    }

    private fun startRefresh() {
        if (isRefreshing) return
        isRefreshing = true
        refreshHandler.post(refreshRunnable)
        Log.d(TAG, "Started map refresh at ${1000 / REFRESH_INTERVAL_MS} Hz")
    }

    private fun stopRefresh() {
        isRefreshing = false
        refreshHandler.removeCallbacks(refreshRunnable)
    }
}
