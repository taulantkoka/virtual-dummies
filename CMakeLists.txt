cmake_minimum_required(VERSION 3.20)
project(virtual_dummies LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

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

pybind11_add_module(vd_selectors
  src/bindings.cpp
  src/vd_base.cpp
  src/vd_lars.cpp
  src/vd_omp.cpp
  src/vd_afs.cpp
  src/trex.cpp
)

target_compile_features(vd_selectors PRIVATE cxx_std_17)

target_include_directories(vd_selectors PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen-5.0.0
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

if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")
  find_library(OMP_LIB omp HINTS "${LIBOMP_ROOT}/lib")
  find_path(OMP_INC omp.h HINTS "${LIBOMP_ROOT}/include")
  if(OMP_LIB AND OMP_INC)
    target_compile_options(vd_selectors PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(vd_selectors PRIVATE "${OMP_INC}")
    target_link_libraries(vd_selectors PRIVATE "${OMP_LIB}")
  endif()
else()
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(vd_selectors PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

install(TARGETS vd_selectors LIBRARY DESTINATION vd_selectors)