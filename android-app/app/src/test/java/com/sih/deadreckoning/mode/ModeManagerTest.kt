/**
 * ModeManagerTest.kt — Unit tests for ModeManager and NavigationMode enum
 *
 * SIH PS-26168  Intelligent Dead Reckoning
 * Owner: Member 1 (App Shell, UI & Navigation Frontend)
 *
 * Tests:
 *   1. Happy path: all 3 valid ordinals map correctly
 *   2. Edge case: invalid/unknown ordinals default to PURE_DR (safe fallback)
 *   3. Edge case: negative ordinals
 *   4. Display helpers: names and colors are non-null/non-empty
 *   5. Enum ordinal values match the C++ NavigationMode exactly
 *
 * These tests run as pure JVM unit tests — no Android framework needed.
 * The ModeManager's JNI getCurrentMode() is NOT tested here (would need
 * instrumented tests with the native library loaded). We test the Kotlin
 * enum mapping layer in isolation.
 */
package com.sih.deadreckoning.mode

import org.junit.Assert.*
import org.junit.Test

class ModeManagerTest {

    /* ── Happy path: valid ordinals ──────────────────────────────────────── */

    @Test
    fun `fromOrdinal 0 returns GNSS_AIDED_INS`() {
        val mode = NavigationMode.fromOrdinal(0)
        assertEquals(NavigationMode.GNSS_AIDED_INS, mode)
        assertEquals(0, mode.ordinalValue)
    }

    @Test
    fun `fromOrdinal 1 returns PURE_DR`() {
        val mode = NavigationMode.fromOrdinal(1)
        assertEquals(NavigationMode.PURE_DR, mode)
        assertEquals(1, mode.ordinalValue)
    }

    @Test
    fun `fromOrdinal 2 returns RECOVERING`() {
        val mode = NavigationMode.fromOrdinal(2)
        assertEquals(NavigationMode.RECOVERING, mode)
        assertEquals(2, mode.ordinalValue)
    }

    /* ── Edge case: invalid ordinals default to PURE_DR ──────────────────── */

    @Test
    fun `fromOrdinal negative returns PURE_DR`() {
        val mode = NavigationMode.fromOrdinal(-1)
        assertEquals(NavigationMode.PURE_DR, mode)
    }

    @Test
    fun `fromOrdinal out of range high returns PURE_DR`() {
        val mode = NavigationMode.fromOrdinal(99)
        assertEquals(NavigationMode.PURE_DR, mode)
    }

    @Test
    fun `fromOrdinal Int MAX_VALUE returns PURE_DR`() {
        val mode = NavigationMode.fromOrdinal(Int.MAX_VALUE)
        assertEquals(NavigationMode.PURE_DR, mode)
    }

    @Test
    fun `fromOrdinal Int MIN_VALUE returns PURE_DR`() {
        val mode = NavigationMode.fromOrdinal(Int.MIN_VALUE)
        assertEquals(NavigationMode.PURE_DR, mode)
    }

    @Test
    fun `fromOrdinal 3 returns PURE_DR (one past valid range)`() {
        // This is the most likely boundary error — C++ adds a new mode
        // but Kotlin hasn't been updated yet. Should fail safe.
        val mode = NavigationMode.fromOrdinal(3)
        assertEquals(NavigationMode.PURE_DR, mode)
    }

    /* ── Enum ordinals match C++ NavigationMode (imu_types.h) ────────────── */

    @Test
    fun `ordinal values match C++ enum`() {
        // From api-contracts.md §1:
        //   GNSS_AIDED_INS = 0
        //   PURE_DR = 1
        //   RECOVERING = 2
        assertEquals(0, NavigationMode.GNSS_AIDED_INS.ordinalValue)
        assertEquals(1, NavigationMode.PURE_DR.ordinalValue)
        assertEquals(2, NavigationMode.RECOVERING.ordinalValue)
    }

    @Test
    fun `enum has exactly 3 entries`() {
        // Guard against accidental additions without updating fromOrdinal
        assertEquals(3, NavigationMode.entries.size)
    }

    /* ── Display helpers are well-formed ─────────────────────────────────── */

    @Test
    fun `all modes have non-empty display names`() {
        for (mode in NavigationMode.entries) {
            assertTrue(
                "displayName for $mode should not be blank",
                mode.displayName.isNotBlank()
            )
        }
    }

    @Test
    fun `all modes have non-zero color values`() {
        for (mode in NavigationMode.entries) {
            assertTrue(
                "colorHex for $mode should not be 0",
                mode.colorHex != 0L
            )
        }
    }

    @Test
    fun `GNSS_AIDED_INS display name is GNSS AIDED`() {
        assertEquals("GNSS AIDED", NavigationMode.GNSS_AIDED_INS.displayName)
    }

    @Test
    fun `PURE_DR display name is DEAD RECKONING`() {
        assertEquals("DEAD RECKONING", NavigationMode.PURE_DR.displayName)
    }

    @Test
    fun `RECOVERING display name is RECOVERING`() {
        assertEquals("RECOVERING", NavigationMode.RECOVERING.displayName)
    }

    /* ── Color values match the UI spec ──────────────────────────────────── */

    @Test
    fun `GNSS_AIDED_INS color is green`() {
        // §2.1 / README: green = GNSS aided
        assertEquals(0xFF4CAF50, NavigationMode.GNSS_AIDED_INS.colorHex)
    }

    @Test
    fun `PURE_DR color is amber`() {
        // §2.1 / README: amber = pure DR
        assertEquals(0xFFFF9800, NavigationMode.PURE_DR.colorHex)
    }

    @Test
    fun `RECOVERING color is red`() {
        // §2.1 / README: red = recovering
        assertEquals(0xFFF44336, NavigationMode.RECOVERING.colorHex)
    }

    /* ── Roundtrip: ordinalValue back through fromOrdinal ────────────────── */

    @Test
    fun `all modes roundtrip through ordinalValue`() {
        for (mode in NavigationMode.entries) {
            val roundtripped = NavigationMode.fromOrdinal(mode.ordinalValue)
            assertEquals(
                "Roundtrip failed for $mode",
                mode, roundtripped
            )
        }
    }
}
