/**
 * jni_bridge.cpp — JNI bindings for core-engine
 *
 * SIH PS-26168 Intelligent Dead Reckoning
 * 
 * Provides native implementations for MapMatchingBridge and OsmLoader.
 */

#include <jni.h>
#include <string>
#include <vector>

#include "map_matching.h"

extern "C" {

/* ═════════════════════════════════════════════════════════════════════════════
 * OsmLoader.kt bindings
 * ═════════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jboolean JNICALL
Java_com_sih_deadreckoning_mapmatching_OsmLoader_nativeLoadGraph(
        JNIEnv* env, jobject /* this */, jstring path) {
    
    if (path == nullptr) return JNI_FALSE;
    
    const char* native_path = env->GetStringUTFChars(path, nullptr);
    bool success = get_map_matcher_instance().load_graph(native_path);
    env->ReleaseStringUTFChars(path, native_path);
    
    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_mapmatching_OsmLoader_nativeSetSessionOrigin(
        JNIEnv* env, jobject /* this */, jdouble latDeg, jdouble lonDeg) {
    
    get_map_matcher_instance().set_session_origin(latDeg, lonDeg);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * MapMatchingBridge.kt bindings
 * ═════════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jfloatArray JNICALL
Java_com_sih_deadreckoning_mapmatching_MapMatchingBridge_nativeMapMatch(
        JNIEnv* env, jobject /* this */, jfloatArray trajectoryPoint, jfloat headingRad) {
    
    if (trajectoryPoint == nullptr || env->GetArrayLength(trajectoryPoint) < 3) {
        return nullptr;
    }

    jfloat* point_data = env->GetFloatArrayElements(trajectoryPoint, nullptr);
    
    MapMatchResult result = get_map_matcher_instance().match(point_data, headingRad);
    
    env->ReleaseFloatArrayElements(trajectoryPoint, point_data, JNI_ABORT);

    if (result.matched_segment_id < 0) {
        return nullptr; // No match found
    }

    jfloatArray out_array = env->NewFloatArray(4);
    if (out_array == nullptr) return nullptr;

    jfloat out_data[4] = {
        static_cast<jfloat>(result.matched_segment_id),
        result.pseudo_measurement[0],
        result.pseudo_measurement[1],
        result.confidence
    };

    env->SetFloatArrayRegion(out_array, 0, 4, out_data);
    return out_array;
}

JNIEXPORT void JNICALL
Java_com_sih_deadreckoning_mapmatching_MapMatchingBridge_nativeResetMatcher(
        JNIEnv* env, jobject /* this */) {
    
    get_map_matcher_instance().reset();
}

} // extern "C"
