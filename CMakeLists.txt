# cmake minimum required version
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

project(lssg)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "-Ofast -Wall -march=native -lrt -DHAVE_CXX0X -fpic -w -fopenmp -mavx2 -mavx -msse -finline-functions")

# check if avx512f is supported
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    execute_process(COMMAND grep -c avx512f /proc/cpuinfo OUTPUT_VARIABLE AVX512F_SUPPORTED)
    if (AVX512F_SUPPORTED GREATER 0)
        message(STATUS "AVX512F supported")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512vpopcntdq")
    else()
        message(STATUS "AVX512F not supported")
    endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -g -fPIC")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0 -g -fPIC")

include_directories(src)

add_subdirectory(example)
# add_subdirectory(python)