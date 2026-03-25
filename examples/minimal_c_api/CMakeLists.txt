add_executable(minimal_c_api_example
  main.cpp
)

target_include_directories(minimal_c_api_example
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(minimal_c_api_example
  PRIVATE
    inference
)

set_target_properties(minimal_c_api_example PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
