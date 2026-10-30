find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME debug_tool_integration_test COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/debug_tool_integration_test.py)
  set_tests_properties(debug_tool_integration_test PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME bevfusion_engine_build_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/engine_build_test.py)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE _bevfusion_numpy_available OUTPUT_QUIET ERROR_QUIET)
  if(_bevfusion_numpy_available EQUAL 0)
    add_test(NAME bevfusion_validation_test
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/validation_test.py)
  endif()
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import yaml"
    RESULT_VARIABLE _bevfusion_yaml_available OUTPUT_QUIET ERROR_QUIET)
  if(_bevfusion_yaml_available EQUAL 0)
    add_test(NAME bevfusion_scripts_test
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts_test.py)
  endif()
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch, onnx, mmengine, mmdet3d"
    RESULT_VARIABLE _bevfusion_training_available OUTPUT_QUIET ERROR_QUIET)
  if(_bevfusion_training_available EQUAL 0)
    add_test(NAME bevfusion_centerhead_export_test
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/centerhead_export_test.py)
  endif()
endif()

add_executable(bevfusion_runtime_test runtime_test.cpp)
target_include_directories(bevfusion_runtime_test PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(bevfusion_runtime_test PRIVATE inference ${CUDA_LIBRARIES})
add_test(NAME bevfusion_runtime_test COMMAND bevfusion_runtime_test ${PROJECT_SOURCE_DIR}/demo/bevfusion_exp20261028 ${PROJECT_SOURCE_DIR}/demo/data/bevfusion/manifests/1776574418716918507.yaml)
set_tests_properties(bevfusion_runtime_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(bevfusion_io_test io_test.cpp)
target_link_libraries(bevfusion_io_test PRIVATE inference ${YAML_CPP_LIBRARY})
target_include_directories(bevfusion_io_test PRIVATE ${YAML_CPP_INCLUDE_DIR})
add_test(NAME bevfusion_io_test COMMAND bevfusion_io_test)

add_executable(bevfusion_kernels_test kernels_test.cpp)
target_include_directories(bevfusion_kernels_test PRIVATE ${INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/bevfusion ${CUDA_INCLUDE_DIRS})
target_link_libraries(bevfusion_kernels_test PRIVATE inference bevfusion_kernels ${CUDA_LIBRARIES})
add_test(NAME bevfusion_kernels_test COMMAND bevfusion_kernels_test)
set_tests_properties(bevfusion_kernels_test PROPERTIES SKIP_RETURN_CODE 77)
