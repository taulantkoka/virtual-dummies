cmake_minimum_required(VERSION 3.20)
project(virtual_dummies LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ==============================================================================
# Options
# ==============================================================================
option(BUILD_BENCHMARKS "Build fig7 benchmark executable (requires Armadillo)" OFF)

# ==============================================================================
# Python + pybind11
# ==============================================================================
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.5
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ==============================================================================
# BLAS vendor
# ==============================================================================
add_library(cblas_vendor INTERFACE)
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(cblas_vendor INTERFACE ${ACCELERATE_FRAMEWORK})
  target_compile_definitions(cblas_vendor INTERFACE HAS_ACCELERATE=1)
else()
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    target_link_libraries(cblas_vendor INTERFACE BLAS::BLAS)
  endif()
endif()

# ==============================================================================
# OpenMP helper
# ==============================================================================
function(enable_openmp tgt)
  if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")
    find_library(OMP_LIB omp HINTS "${LIBOMP_ROOT}/lib")
    find_path(OMP_INC omp.h HINTS "${LIBOMP_ROOT}/include")
    if(OMP_LIB AND OMP_INC)
      target_compile_options(${tgt} PRIVATE -Xpreprocessor -fopenmp)
      target_include_directories(${tgt} PRIVATE "${OMP_INC}")
      target_link_libraries(${tgt} PRIVATE "${OMP_LIB}")
    endif()
  else()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endif()
endfunction()

# ==============================================================================
# Eigen headers (vendored or fetched)
# ==============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen-5.0.0/Eigen/Dense")
  set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen-5.0.0")
else()
  include(FetchContent)
  FetchContent_Declare(eigen_fetch
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    SOURCE_SUBDIR  ""
  )
  FetchContent_Populate(eigen_fetch)
  set(EIGEN_INCLUDE_DIR "${eigen_fetch_SOURCE_DIR}")
endif()

# ==============================================================================
# vd_selectors  (always built — the main Python extension)
# ==============================================================================
pybind11_add_module(vd_selectors
  src/bindings.cpp
  src/vd_base.cpp
  src/vd_lars.cpp
  src/vd_omp.cpp
  src/vd_afs.cpp
  src/vd_afs_logistic.cpp
  src/trex.cpp
)

target_compile_features(vd_selectors PRIVATE cxx_std_17)
target_include_directories(vd_selectors PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${EIGEN_INCLUDE_DIR}
)
target_link_libraries(vd_selectors PRIVATE cblas_vendor)

if(MSVC)
  target_compile_options(vd_selectors PRIVATE /O2)
else()
  target_compile_options(vd_selectors PRIVATE -O3)
endif()

target_compile_definitions(vd_selectors PRIVATE
  EIGEN_USE_BLAS
  EIGEN_NO_DEBUG
  NDEBUG
)

if(APPLE)
  set_target_properties(vd_selectors PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup"
  )
endif()

enable_openmp(vd_selectors)

install(TARGETS vd_selectors LIBRARY DESTINATION vd_selectors)

# ==============================================================================
# fig7_benchmark  (opt-in: AD-LARS vs VD-LARS memory/runtime comparison)
#
# From tlars_core/ only two files are needed:
#   TLARS_Solver.hpp / .cpp   — the augmented-dummy LARS baseline
#   arma_cereal.hpp           — Armadillo serialization (included by TLARS_Solver.hpp)
#
# Build:
#   cmake -B build -DBUILD_BENCHMARKS=ON
#   cmake --build build --target fig7_benchmark
#   TLARS_MEM_CAP_GB=100 ./build/fig7_benchmark
# ==============================================================================
if(BUILD_BENCHMARKS)
  # --- Armadillo ---
  find_package(Armadillo QUIET)
  if(NOT Armadillo_FOUND)
    message(FATAL_ERROR
      "Armadillo not found (required for BUILD_BENCHMARKS).\n"
      "Install it or set -DCMAKE_PREFIX_PATH=/path/to/armadillo")
  endif()

  function(link_armadillo tgt)
    if(TARGET Armadillo::Armadillo)
      target_link_libraries(${tgt} PRIVATE Armadillo::Armadillo)
    elseif(TARGET armadillo)
      target_link_libraries(${tgt} PRIVATE armadillo)
    elseif(DEFINED ARMADILLO_LIBRARIES)
      target_link_libraries(${tgt} PRIVATE ${ARMADILLO_LIBRARIES})
    endif()
    if(DEFINED ARMADILLO_INCLUDE_DIRS)
      target_include_directories(${tgt} PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    endif()
  endfunction()

  # --- cereal (header-only, needed by TLARS_Solver.hpp for serialization) ---
  include(FetchContent)
  FetchContent_Declare(cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG        v1.3.2
    SOURCE_SUBDIR  ""
  )
  FetchContent_Populate(cereal)

  # --- benchmark executable ---
  find_package(Threads REQUIRED)

  add_executable(fig7_benchmark
    experiments/fig7_memory_runtime_benchmark.cpp   # the benchmark driver
    experiments/tlars_core/src/TLARS_Solver.cpp                 # AD-LARS baseline (Armadillo)
    src/vd_base.cpp                                 # VD-LARS (Eigen)
    src/vd_lars.cpp
  )

  target_compile_features(fig7_benchmark PRIVATE cxx_std_17)
  target_include_directories(fig7_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src                 # vd_lars.hpp, vd_base.hpp, ...
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/tlars_core/src      # TLARS_Solver.hpp, arma_cereal.hpp
    ${cereal_SOURCE_DIR}/include                    # <cereal/archives/...>
    ${EIGEN_INCLUDE_DIR}
  )

  if(MSVC)
    target_compile_options(fig7_benchmark PRIVATE /O2)
  else()
    target_compile_options(fig7_benchmark PRIVATE -O3 -fstrict-aliasing)
  endif()

  target_compile_definitions(fig7_benchmark PRIVATE
    EIGEN_USE_BLAS
    EIGEN_NO_DEBUG
    NDEBUG
  )

  target_link_libraries(fig7_benchmark PRIVATE cblas_vendor Threads::Threads)
  link_armadillo(fig7_benchmark)
endif()