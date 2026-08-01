cmake_minimum_required(VERSION 3.20)
project(up_and_out_barrier_mc LANGUAGES CXX)

option(BUILD_CUDA "Build the CUDA Monte Carlo backend when CUDA is available" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)

add_library(barrier_cpu SHARED src/barrier_cpu.cpp)
target_include_directories(barrier_cpu PRIVATE src)
if(OpenMP_CXX_FOUND)
    target_link_libraries(barrier_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(barrier_cpu PROPERTIES
    OUTPUT_NAME barrier_cpu
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75;80;86 CACHE STRING
                "CUDA architectures to build for")
        endif()
        find_package(CUDAToolkit REQUIRED)
        add_library(barrier_cuda SHARED src/barrier_cuda.cu)
        target_include_directories(barrier_cuda PRIVATE src)
        target_link_libraries(barrier_cuda PRIVATE CUDA::cudart CUDA::curand)
        set_target_properties(barrier_cuda PROPERTIES
            OUTPUT_NAME barrier_cuda
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
        message(STATUS "CUDA backend enabled for architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "CUDA compiler not found; building CPU backend only")
    endif()
endif()
