cmake_minimum_required(VERSION 3.0.2)

project(lidar-inference-tool
  VERSION 1.8.0
  DESCRIPTION "TensorRT lidar-inference-tool library"
  LANGUAGES CXX
)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTING "Build tests" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(ARCH x86_64)
  set(DEB_ARCH amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(ARCH aarch64)
  set(DEB_ARCH arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_package(CUDA REQUIRED)

# Respect manual compiler selection without changing the caller's environment.
# An explicit or cached CMake value takes priority over PROTOC.
if(NOT Protobuf_PROTOC_EXECUTABLE AND DEFINED ENV{PROTOC}
    AND NOT "$ENV{PROTOC}" STREQUAL "")
  set(Protobuf_PROTOC_EXECUTABLE "$ENV{PROTOC}" CACHE FILEPATH
    "The Google Protocol Buffers Compiler" FORCE)
endif()
# Otherwise prefer the compiler installed with the selected C++ headers.
find_path(Protobuf_INCLUDE_DIR google/protobuf/service.h)
if(Protobuf_INCLUDE_DIR AND NOT Protobuf_PROTOC_EXECUTABLE)
  get_filename_component(_protobuf_prefix "${Protobuf_INCLUDE_DIR}" DIRECTORY)
  find_program(Protobuf_PROTOC_EXECUTABLE NAMES protoc
    HINTS "${_protobuf_prefix}/bin"
    NO_DEFAULT_PATH
  )
endif()
find_package(Protobuf REQUIRED)

execute_process(
  COMMAND "${Protobuf_PROTOC_EXECUTABLE}" --version
  RESULT_VARIABLE _protoc_result
  OUTPUT_VARIABLE _protoc_output
  ERROR_VARIABLE _protoc_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _protoc_result EQUAL 0 OR NOT _protoc_output MATCHES "^libprotoc ([0-9.]+)$")
  message(FATAL_ERROR
    "Cannot query protoc: ${Protobuf_PROTOC_EXECUTABLE}\n${_protoc_error}\n"
    "Set -DProtobuf_PROTOC_EXECUTABLE=/path/to/matching/protoc, "
    "or run PROTOC=/path/to/matching/protoc bash rebuild.sh.")
endif()
set(_protoc_version "${CMAKE_MATCH_1}")
set(_expected_protoc_version "${Protobuf_VERSION}")
# Newer releases use e.g. C++ runtime 4.25.3 and protoc 25.3.
if(Protobuf_VERSION VERSION_GREATER_EQUAL "4.0")
  string(REGEX REPLACE "^[0-9]+\\." "" _expected_protoc_version "${Protobuf_VERSION}")
endif()
if(NOT _protoc_version VERSION_EQUAL _expected_protoc_version)
  message(FATAL_ERROR
    "Protobuf version mismatch: protoc ${_protoc_version} "
    "(${Protobuf_PROTOC_EXECUTABLE}), C++ headers ${Protobuf_VERSION} "
    "(${Protobuf_INCLUDE_DIRS}), libraries ${Protobuf_LIBRARIES}.\n"
    "Set -DProtobuf_PROTOC_EXECUTABLE=/path/to/matching/protoc "
    "or run PROTOC=/path/to/matching/protoc bash rebuild.sh.\n"
    "For an existing CMake build directory, clear its cached compiler "
    "with -UProtobuf_PROTOC_EXECUTABLE before changing PROTOC.")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(NVINFER_LIB nvinfer)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB)
  message(FATAL_ERROR
    "TensorRT not found. Set TENSORRT_INCLUDE_DIR, NVINFER_LIB and "
    "NVINFER_PLUGIN_LIB for non-standard installations.")
endif()

find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h)
find_library(YAML_CPP_LIBRARY yaml-cpp)

if(NOT YAML_CPP_INCLUDE_DIR OR NOT YAML_CPP_LIBRARY)
  message(FATAL_ERROR "yaml-cpp not found.")
endif()

message(STATUS "ARCH           : ${ARCH}")
message(STATUS "CUDA           : ${CUDA_VERSION_STRING}")
message(STATUS "TensorRT       : ${NVINFER_LIB}")
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Protobuf       : ${Protobuf_LIBRARIES}")
message(STATUS "Protobuf include: ${Protobuf_INCLUDE_DIRS} (${Protobuf_VERSION})")
message(STATUS "protoc         : ${Protobuf_PROTOC_EXECUTABLE} (${_protoc_version})")
message(STATUS "yaml-cpp       : ${YAML_CPP_LIBRARY}")

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(SPCONV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libspconv")
set(SPCONV_INCLUDE_DIR "${SPCONV_ROOT}/include")
set(SPCONV_PARSER_DIR "${SPCONV_ROOT}/parser")
set(SPCONV_LIBRARY_DIR "${SPCONV_ROOT}/lib/${ARCH}")
set(SPCONV_LIBRARY "${SPCONV_LIBRARY_DIR}/libspconv.so")

foreach(path
  "${SPCONV_INCLUDE_DIR}/spconv/engine.hpp"
  "${SPCONV_LIBRARY}"
  "${SPCONV_PARSER_DIR}/onnx-parser.cpp"
  "${SPCONV_PARSER_DIR}/onnx/onnx-ml.proto"
  "${SPCONV_PARSER_DIR}/onnx/onnx-operators-ml.proto"
)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Missing dependency: ${path}")
  endif()
endforeach()

message(STATUS "libspconv      : ${SPCONV_LIBRARY}")

set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(PROTO_SOURCES
  ${PROTO_GEN_DIR}/onnx/onnx-ml.pb.cc
  ${PROTO_GEN_DIR}/onnx/onnx-operators-ml.pb.cc
)
set(PROTO_HEADERS
  ${PROTO_GEN_DIR}/onnx/onnx-ml.pb.h
  ${PROTO_GEN_DIR}/onnx/onnx-operators-ml.pb.h
)

add_custom_command(
  OUTPUT ${PROTO_SOURCES} ${PROTO_HEADERS}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PROTO_GEN_DIR}/onnx"
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    --proto_path=${SPCONV_PARSER_DIR}
    --cpp_out=${PROTO_GEN_DIR}
    ${SPCONV_PARSER_DIR}/onnx/onnx-ml.proto
    ${SPCONV_PARSER_DIR}/onnx/onnx-operators-ml.proto
  DEPENDS
    "${Protobuf_PROTOC_EXECUTABLE}"
    ${SPCONV_PARSER_DIR}/onnx/onnx-ml.proto
    ${SPCONV_PARSER_DIR}/onnx/onnx-operators-ml.proto
  VERBATIM
)

add_library(spconv_onnx_proto STATIC ${PROTO_SOURCES})
target_include_directories(spconv_onnx_proto PUBLIC
  ${PROTO_GEN_DIR}
  ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(spconv_onnx_proto PUBLIC ${Protobuf_LIBRARIES})
set_target_properties(spconv_onnx_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src/pillar_postprocess)
add_subdirectory(third_party/iou3d_nms)
add_subdirectory(third_party/voxelize)

set(INFERENCE_SOURCES
  src/inference_api.cpp
  src/inference_cpp.cpp
  src/pillar_runtime.cpp
  src/trt_engine.cpp
  ${SPCONV_PARSER_DIR}/onnx-parser.cpp
)

set(INFERENCE_INCLUDES
  ${SRC_DIR}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${SPCONV_INCLUDE_DIR}
  ${PROTO_GEN_DIR}
  ${SPCONV_PARSER_DIR}
  ${Protobuf_INCLUDE_DIRS}
)

set(INFERENCE_LIBS
  pillar_postprocess
  iou3d_nms
  voxelization
  ${CUDA_LIBRARIES}
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  ${YAML_CPP_LIBRARY}
  ${SPCONV_LIBRARY}
  spconv_onnx_proto
)

add_library(inference SHARED ${INFERENCE_SOURCES})

target_include_directories(inference
  PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${INFERENCE_INCLUDES}
)

target_link_libraries(inference PRIVATE ${INFERENCE_LIBS})

set_target_properties(inference PROPERTIES
  BUILD_RPATH "${SPCONV_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN"
)

add_executable(perf_debug_tool
  tools/perf_debug_tool.cpp
  src/pillar_debug_utils.cpp
  ${INFERENCE_SOURCES}
)

target_include_directories(perf_debug_tool PRIVATE
  ${INCLUDE_DIR}
  ${INFERENCE_INCLUDES}
)

target_link_libraries(perf_debug_tool PRIVATE ${INFERENCE_LIBS})

set_target_properties(perf_debug_tool PROPERTIES
  BUILD_RPATH "${SPCONV_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR};$ORIGIN"
)

add_executable(infer_debug_tool
  tools/infer_debug_tool.cpp
  src/pillar_debug_utils.cpp
)

target_include_directories(infer_debug_tool PRIVATE
  ${INCLUDE_DIR}
  ${INFERENCE_INCLUDES}
)

target_link_libraries(infer_debug_tool PRIVATE inference)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_TESTING)
  enable_testing()

  add_executable(self_car_nms_test
    tests/self_car_nms_test.cpp
    third_party/iou3d_nms/iou3d_cpu.cpp
    third_party/iou3d_nms/iou3d_nms_api.cpp
  )
  target_include_directories(self_car_nms_test PRIVATE
    ${INCLUDE_DIR}
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/iou3d_nms/include
  )
  target_link_libraries(self_car_nms_test PRIVATE inference)
  add_test(NAME self_car_nms_test COMMAND self_car_nms_test)

  add_executable(config_and_input_loader_test
    tests/config_and_input_loader_test.cpp
    src/pillar_debug_utils.cpp
  )
  target_include_directories(config_and_input_loader_test PRIVATE
    ${INCLUDE_DIR}
    ${SRC_DIR}
  )
  target_link_libraries(config_and_input_loader_test PRIVATE inference)
  add_test(NAME config_and_input_loader_test COMMAND config_and_input_loader_test)

  add_test(NAME infer_debug_cli_test
    COMMAND bash
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/infer_debug_cli_test.sh
      $<TARGET_FILE:infer_debug_tool>
  )

  add_test(NAME minimal_cpp_api_cli_test
    COMMAND bash
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimal_cpp_api_cli_test.sh
      $<TARGET_FILE:minimal_cpp_api_example>
  )

  add_test(NAME centerpoint_voxel_head2engine_deps_test
    COMMAND bash
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/centerpoint_voxel_head2engine_deps_test.sh
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME package_arch_test
    COMMAND bash
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/package_arch_test.sh
      ${CMAKE_BINARY_DIR}
      ${CMAKE_SYSTEM_PROCESSOR}
  )
endif()

install(TARGETS inference
  EXPORT lidarInferenceTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS perf_debug_tool infer_debug_tool
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY "${INCLUDE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
)

get_filename_component(SPCONV_LIBRARY_REAL "${SPCONV_LIBRARY}" REALPATH)

install(FILES "${SPCONV_LIBRARY_REAL}"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RENAME libspconv.so
)

if(ARCH STREQUAL "aarch64")
  install(PROGRAMS scripts/dist/lidar_model_downloader
    DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

include(CMakePackageConfigHelpers)

set(INSTALL_CONFIGDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(EXPORT lidarInferenceTargets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${INSTALL_CONFIGDIR}
)

configure_package_config_file(
  cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_CONTACT "zkzc zkzc@todo.com")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")

set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${DEB_ARCH})
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "zkzc zkzc@todo.com")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.31)")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

include(CPack)
