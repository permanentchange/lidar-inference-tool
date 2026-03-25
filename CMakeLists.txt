cmake_minimum_required(VERSION 3.0.2)
project(inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_STUB_IF_NO_TENSORRT "Allow stub build when TensorRT is unavailable" OFF)
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
set(YAML_CPP_ROOT "" CACHE PATH "System yaml-cpp installation root")

function(require_no_conda_path name value)
  if(NOT "${value}" STREQUAL "")
    string(TOLOWER "${value}" _value_lower)
    if(_value_lower MATCHES "miniconda|anaconda|miniforge|mambaforge|/conda")
      message(FATAL_ERROR
        "${name} contains a Conda-managed path: ${value}\n"
        "Clear the CMake cache and configure with system dependencies only.")
    endif()
  endif()
endfunction()

foreach(_cache_var
    CMAKE_C_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_SHARED_LINKER_FLAGS)
  require_no_conda_path(${_cache_var} "${${_cache_var}}")
endforeach()

foreach(_env_var
    CMAKE_PREFIX_PATH
    CPATH
    C_INCLUDE_PATH
    CPLUS_INCLUDE_PATH
    LIBRARY_PATH
    CPPFLAGS
    CFLAGS
    CXXFLAGS
    LDFLAGS)
  if(DEFINED ENV{${_env_var}})
    require_no_conda_path("ENV{${_env_var}}" "$ENV{${_env_var}}")
  endif()
endforeach()

find_package(CUDA)

set(YAML_CPP_SEARCH_PREFIXES
  /usr
  /usr/local)
if(NOT "${YAML_CPP_ROOT}" STREQUAL "")
  list(INSERT YAML_CPP_SEARCH_PREFIXES 0 "${YAML_CPP_ROOT}")
endif()
if(DEFINED ENV{YAML_CPP_ROOT} AND NOT "$ENV{YAML_CPP_ROOT}" STREQUAL "")
  list(INSERT YAML_CPP_SEARCH_PREFIXES 0 "$ENV{YAML_CPP_ROOT}")
endif()
list(REMOVE_DUPLICATES YAML_CPP_SEARCH_PREFIXES)

set(YAML_CPP_LIBRARY_SUFFIXES
  lib
  lib64)
if(CMAKE_LIBRARY_ARCHITECTURE)
  list(APPEND YAML_CPP_LIBRARY_SUFFIXES "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
endif()

find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h
  PATHS ${YAML_CPP_SEARCH_PREFIXES}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_library(YAML_CPP_LIBRARY yaml-cpp
  PATHS ${YAML_CPP_SEARCH_PREFIXES}
  PATH_SUFFIXES ${YAML_CPP_LIBRARY_SUFFIXES}
  NO_DEFAULT_PATH)

set(TENSORRT_SEARCH_ROOTS
  /usr
  /usr/local
  /opt)
if(NOT "${TENSORRT_ROOT}" STREQUAL "")
  list(INSERT TENSORRT_SEARCH_ROOTS 0 "${TENSORRT_ROOT}")
endif()
if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
  list(INSERT TENSORRT_SEARCH_ROOTS 0 "$ENV{TENSORRT_ROOT}")
endif()
if(DEFINED ENV{TensorRT_ROOT} AND NOT "$ENV{TensorRT_ROOT}" STREQUAL "")
  list(INSERT TENSORRT_SEARCH_ROOTS 0 "$ENV{TensorRT_ROOT}")
endif()
if(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
  file(GLOB _tensorrt_home_candidates
    "$ENV{HOME}/TensorRT*"
    "$ENV{HOME}/workspace/TensorRT*")
  if(_tensorrt_home_candidates)
    list(APPEND TENSORRT_SEARCH_ROOTS ${_tensorrt_home_candidates})
  endif()
endif()

find_program(TRTEXEC_BIN trtexec
  HINTS ${TENSORRT_SEARCH_ROOTS}
  PATH_SUFFIXES
    bin
    targets/x86_64-linux-gnu/bin)
if(TRTEXEC_BIN)
  get_filename_component(_trtexec_dir "${TRTEXEC_BIN}" DIRECTORY)
  get_filename_component(_trtexec_parent "${_trtexec_dir}/.." ABSOLUTE)
  get_filename_component(_trtexec_grandparent "${_trtexec_dir}/../.." ABSOLUTE)
  get_filename_component(_trtexec_greatgrandparent "${_trtexec_dir}/../../.." ABSOLUTE)
  list(APPEND TENSORRT_SEARCH_ROOTS
    "${_trtexec_parent}"
    "${_trtexec_grandparent}"
    "${_trtexec_greatgrandparent}")
endif()
list(REMOVE_DUPLICATES TENSORRT_SEARCH_ROOTS)

set(TENSORRT_LIBRARY_SUFFIXES
  lib
  lib64
  targets/x86_64-linux-gnu/lib)
if(CMAKE_LIBRARY_ARCHITECTURE)
  list(APPEND TENSORRT_LIBRARY_SUFFIXES "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
endif()
if(CMAKE_SYSTEM_PROCESSOR)
  list(APPEND TENSORRT_LIBRARY_SUFFIXES "targets/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/lib")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS ${TENSORRT_SEARCH_ROOTS}
  PATH_SUFFIXES
    include
    targets/x86_64-linux-gnu/include
  NO_DEFAULT_PATH)
find_library(NVINFER_LIB nvinfer
  PATHS ${TENSORRT_SEARCH_ROOTS}
  PATH_SUFFIXES ${TENSORRT_LIBRARY_SUFFIXES}
  NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
  PATHS ${TENSORRT_SEARCH_ROOTS}
  PATH_SUFFIXES ${TENSORRT_LIBRARY_SUFFIXES}
  NO_DEFAULT_PATH)

set(INFERENCE_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INFERENCE_PRIVATE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(MISSING_GPU_DEPS)
if(NOT CUDA_FOUND)
  list(APPEND MISSING_GPU_DEPS "CUDA Toolkit / nvcc")
endif()
if(NOT TENSORRT_INCLUDE_DIR)
  list(APPEND MISSING_GPU_DEPS "TensorRT headers (NvInfer.h)")
endif()
if(NOT NVINFER_LIB)
  list(APPEND MISSING_GPU_DEPS "TensorRT library: nvinfer")
endif()
if(NOT NVINFER_PLUGIN_LIB)
  list(APPEND MISSING_GPU_DEPS "TensorRT library: nvinfer_plugin")
endif()

if(MISSING_GPU_DEPS)
  string(REPLACE ";" "\n  - " MISSING_GPU_DEPS_TEXT "${MISSING_GPU_DEPS}")
  set(MISSING_GPU_DEPS_TEXT "  - ${MISSING_GPU_DEPS_TEXT}")
  if(BUILD_STUB_IF_NO_TENSORRT)
    message(WARNING
      "Building inference stub because required GPU inference dependencies were not found:\n"
      "${MISSING_GPU_DEPS_TEXT}")
    add_library(inference SHARED
      src/inference_stub.cpp
    )
    target_include_directories(inference
      PUBLIC
        ${INFERENCE_PUBLIC_INCLUDE_DIR}
    )
  else()
    message(FATAL_ERROR
      "Full build requires CUDA and TensorRT. Missing dependencies:\n"
      "${MISSING_GPU_DEPS_TEXT}\n"
      "Set TENSORRT_ROOT to your TensorRT installation root, or configure with "
      "-DBUILD_STUB_IF_NO_TENSORRT=ON to build the stub explicitly.")
  endif()
else()
  if(NOT YAML_CPP_INCLUDE_DIR OR NOT YAML_CPP_LIBRARY)
    message(FATAL_ERROR
      "yaml-cpp was not found in system paths. Install libyaml-cpp-dev or set YAML_CPP_ROOT.")
  endif()

  add_subdirectory(third_party/iou3d_nms)
  add_subdirectory(third_party/voxelize)

  add_library(inference SHARED
    src/inference_api.cpp
    src/pillar_runtime.cpp
  )
  target_include_directories(inference
    PUBLIC
      ${INFERENCE_PUBLIC_INCLUDE_DIR}
    PRIVATE
      ${INFERENCE_PRIVATE_INCLUDE_DIR}
      ${TENSORRT_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIRS}
  )
  target_link_libraries(inference
    PRIVATE
      voxelization
      ${CUDA_LIBRARIES}
      ${NVINFER_LIB}
      ${NVINFER_PLUGIN_LIB}
  )

  add_executable(pillar_debug_tool
    tools/pillar_debug_tool.cpp
    src/pillar_runtime.cpp
    src/pillar_debug_utils.cpp
  )
  target_include_directories(pillar_debug_tool
    PRIVATE
      ${INFERENCE_PUBLIC_INCLUDE_DIR}
      ${INFERENCE_PRIVATE_INCLUDE_DIR}
      ${TENSORRT_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIRS}
      ${YAML_CPP_INCLUDE_DIR}
  )
  target_link_libraries(pillar_debug_tool
    PRIVATE
      ${CUDA_LIBRARIES}
      ${NVINFER_LIB}
      ${NVINFER_PLUGIN_LIB}
      iou3d_nms
      voxelization
      ${YAML_CPP_LIBRARY}
  )
endif()
