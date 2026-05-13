cmake_minimum_required(VERSION 3.0.2)

project(pillar_postprocess LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_87,code=sm_87")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_89,code=sm_89")
endif()

add_library(pillar_postprocess STATIC
  pillar_postprocess.cu
)
set_target_properties(pillar_postprocess PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pillar_postprocess
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(pillar_postprocess
  PRIVATE
    ${CUDA_LIBRARIES}
)
