cmake_minimum_required(VERSION 3.22)
project(core_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find JNI headers (optional on host systems, provided automatically in Android NDK)
find_package(JNI QUIET)

# Core engine shared library (compiled into libcore_engine.so for Android app)
add_library(core_engine SHARED
    src/calibration.cpp
    src/ins_mechanization.cpp
    src/ukf_fusion.cpp
    src/mode_manager.cpp
)

target_include_directories(core_engine PUBLIC
    include/
)

if(JNI_FOUND)
    target_include_directories(core_engine PRIVATE ${JNI_INCLUDE_DIRS})
endif()

if(DEFINED ANDROID_NDK)
    target_include_directories(core_engine PUBLIC
        ${ANDROID_NDK}/sources/android/cpufeatures
    )
endif()

# Unit test executable for Member 3 — Calibration
add_executable(test_calibration
    tests/test_calibration.cpp
    src/calibration.cpp
)

target_include_directories(test_calibration PRIVATE
    include/
)

if(JNI_FOUND)
    target_include_directories(test_calibration PRIVATE ${JNI_INCLUDE_DIRS})
endif()

# Unit test executable for Member 4 — INS Mechanization
add_executable(test_ins_mechanization
    tests/test_ins_mechanization.cpp
    src/ins_mechanization.cpp
)

target_include_directories(test_ins_mechanization PRIVATE
    include/
)

# Unit test executable for Member 4 — UKF Fusion + Mode Manager
add_executable(test_ukf_fusion
    tests/test_ukf_fusion.cpp
    src/ukf_fusion.cpp
    src/ins_mechanization.cpp
    src/mode_manager.cpp
)

target_include_directories(test_ukf_fusion PRIVATE
    include/
)

enable_testing()
add_test(NAME test_calibration COMMAND test_calibration)
add_test(NAME test_ins_mechanization COMMAND test_ins_mechanization)
add_test(NAME test_ukf_fusion COMMAND test_ukf_fusion)
