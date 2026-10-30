set(BEVFUSION_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime.cpp
  PARENT_SCOPE
)

add_library(bevfusion_kernels STATIC kernels.cu)
target_include_directories(bevfusion_kernels PRIVATE ${INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(bevfusion_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_VERSION VERSION_LESS 3.18)
  target_compile_options(bevfusion_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_87,code=sm_87> $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_87,code=compute_87>)
endif()
