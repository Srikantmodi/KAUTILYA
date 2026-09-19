cmake_minimum_required(VERSION 3.22)
project(core_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find JNI headers (optional on host systems, provided automatically in Android NDK)
find_package(JNI QUIET)

# Core engine shared library (compiled into libcore_engine.so for Android app)
add_library(core_engine SHARED
    src/calibration.cpp
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

# Unit test executable for Member 3 (and other members)
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

enable_testing()
add_test(NAME test_calibration COMMAND test_calibration)
