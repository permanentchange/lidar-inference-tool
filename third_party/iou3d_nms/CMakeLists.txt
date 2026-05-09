cmake_minimum_required(VERSION 3.0.2)

find_package(CUDA)


if(CUDA_FOUND)
  project(iou3d_nms LANGUAGES CXX CUDA)
  set(NVCC_FILES "iou3d_nms_kernel.cu")
else()
  project(iou3d_nms LANGUAGES CXX)
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

add_library(iou3d_nms STATIC
  iou3d_cpu.cpp
  iou3d_nms_api.cpp
  ${NVCC_FILES}
)
set_target_properties(iou3d_nms PROPERTIES POSITION_INDEPENDENT_CODE ON )

if(CUDA_FOUND)
  target_compile_definitions(iou3d_nms PRIVATE HAVE_CUDA_ENABLE)
endif()

target_include_directories(iou3d_nms
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(iou3d_nms
  PRIVATE
    ${CUDA_LIBRARIES}
)
