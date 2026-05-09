cmake_minimum_required(VERSION 3.0.2)

find_package(CUDA)


if(CUDA_FOUND)
  project(voxelization LANGUAGES CXX CUDA)
  set(VOXELIZE_NVCC_FILES "voxelization_kernel.cu")
  set(PREPROCESS_NVCC_FILES "preprocess_kernel.cu")
else()
  project(voxelization LANGUAGES CXX)
  set(CUDA_INCLUDE_DIRS "")
  set(CUDA_LIBRARIES "")
endif()

set(CMAKE_C_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_BUILD_TYPE "Release")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_87,code=sm_87")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_89,code=sm_89")
endif()

include_directories(
  SYSTEM ${CUDA_INCLUDE_DIRS}
  include
)

add_library(voxelization STATIC
  voxelization.cpp
  ${VOXELIZE_NVCC_FILES}
  ${PREPROCESS_NVCC_FILES}
)
set_target_properties(voxelization PROPERTIES POSITION_INDEPENDENT_CODE ON )

if(CUDA_FOUND)
  target_compile_definitions(voxelization PUBLIC HAVE_CUDA_ENABLE)
endif()


target_include_directories(voxelization
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(voxelization
  ${CUDA_LIBRARIES}
)
