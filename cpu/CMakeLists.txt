cmake_minimum_required(VERSION 3.26)
project(CS5234_parallel_radix_sort)

set(CMAKE_CXX_STANDARD 26)

include_directories(cpu)

add_executable(benchmark
        benchmark.cpp
        serial_radix_sort.cpp
        serial_radix_sort.h
        parallel_radix_sort.cpp
        parallel_radix_sort.h
        data_generator.cpp
        data_generator.h
)

add_executable(validate_sort
        validate_sort.cpp
        parallel_radix_sort.cpp
        parallel_radix_sort.h
        data_generator.cpp
        data_generator.h
)

add_executable(for_profiling
        for_profiling.cpp
        parallel_radix_sort.cpp
        parallel_radix_sort.h
        serial_radix_sort.cpp
        serial_radix_sort.h)

find_package(OpenMP REQUIRED)
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(validate_sort PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(for_profiling PRIVATE OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fopenmp -DNDEBUG")