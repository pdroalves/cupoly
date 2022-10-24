PROJECT (cuPoly LANGUAGES CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

SET(LIB_NAME "cuPoly")
SET(PROJECT_VERSION_MAJOR "0")
SET(PROJECT_VERSION_MINOR "3")
SET(PROJECT_VERSION_PATCH "5")
SET(PROJECT_VERSION_TWEAK "12")
SET(CUPOLY_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${PROJECT_VERSION_TWEAK})")
SET(VERSION ${CUPOLY_VERSION})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("\n-- Configuring ${PROJECT_NAME} ${CUPOLY_VERSION} (${CMAKE_BUILD_TYPE})...\n")

SET(CMAKE_CXX_FLAGS "-Wall -Wextra")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CUDA_NVCC_FLAGS_DEBUG "-G")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/cuPolyConfig.h.in"
  "${PROJECT_SOURCE_DIR}/include/cuPoly/cuPolyConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find cuPolyConfig.h
include_directories("${PROJECT_BINARY_DIR}")

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
SET(GCC_CUDA_VERSION "gcc" CACHE STRING "GCC compiler to be used with cuda")
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

SET(CUDA_SEPARABLE_COMPILATION ON)

ENABLE_LANGUAGE(CUDA)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(NTL 11.0 REQUIRED)
FIND_PACKAGE(GMP 6.0)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(cxxopts REQUIRED)
FIND_PACKAGE(Doxygen)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread)
LIST(APPEND CUDA_NVCC_FLAGS --relocatable-device-code=true)
LIST(APPEND CUDA_NVCC_FLAGS --use_fast_math)

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#
# Lib
#
SET ( 
    libsrc 
    src/arithmetic/polynomial.cu
    src/arithmetic/context.cpp
    src/arithmetic/coprimes.cu
    src/cuda/cudaengine.cu
    src/cuda/sampler.cu
    src/cuda/dgt.cu
    src/tool/log.cu
    src/tool/version.cpp
)

INCLUDE_DIRECTORIES(
    include
    include/
 )

# BFV mode
CUDA_ADD_LIBRARY(cupolybfv STATIC ${libsrc})
SET_TARGET_PROPERTIES(cupolybfv PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES native)
TARGET_LINK_LIBRARIES(
    cupolybfv
    ${CUDA_curand_LIBRARY}
    -lntl 
    -lgmp 
)
TARGET_COMPILE_DEFINITIONS(cupolybfv PRIVATE BFV_ENGINE_MODE=1)

# CKKS mode
CUDA_ADD_LIBRARY(cupolyckks STATIC ${libsrc})
SET_TARGET_PROPERTIES(cupolyckks PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES native)
TARGET_LINK_LIBRARIES(
    cupolyckks
    ${CUDA_curand_LIBRARY}
    -lntl 
    -lgmp 
)
TARGET_COMPILE_DEFINITIONS(cupolyckks PRIVATE CKKS_ENGINE_MODE=1)

#
# Testing
#
ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)

#
# Demos
#
ADD_SUBDIRECTORY(demos)

#
# Documentation
#
# Doxygen Build
OPTION(BUILD_DOC "Build Documentation" OFF)

if(DOXYGEN_FOUND AND BUILD_DOC)
    message("Doxygen build started")
    INCLUDE(cmake/doxygen.cmake)
else(DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the documentation.")
endif(DOXYGEN_FOUND AND BUILD_DOC)

#
# Install
# 

INSTALL(TARGETS cupolybfv cupolyckks
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

INSTALL(FILES include/cuPoly/cuda/cudaengine.h include/cuPoly/cuda/sampler.h include/cuPoly/cuda/dgt.h
    DESTINATION include/cuPoly/cuda
    )

INSTALL(FILES include/cuPoly/arithmetic/polynomial.h include/cuPoly/arithmetic/context.h
    DESTINATION include/cuPoly/arithmetic
    )

INSTALL(FILES include/cuPoly/tool/log.h include/cuPoly/tool/version.h
    DESTINATION include/cuPoly/tool
    )

INSTALL(FILES include/cuPoly/settings.h include/cuPoly/cuPolyConfig.h
    DESTINATION include/cuPoly
    )
