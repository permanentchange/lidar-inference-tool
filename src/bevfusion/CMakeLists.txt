set(BEVFUSION_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime.cpp
  PARENT_SCOPE
)

# Follow the explicit nvcc flags used by the other CUDA modules on master.
# Do not request CUDA_STANDARD 17: older CMake cannot map it for nvcc 11.4.
string(APPEND CMAKE_CUDA_FLAGS " -std=c++17 -gencode arch=compute_87,code=sm_87")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_89,code=sm_89")
endif()

add_library(bevfusion_kernels STATIC kernels.cu)
target_include_directories(bevfusion_kernels PRIVATE ${INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(bevfusion_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
