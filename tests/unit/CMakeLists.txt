include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()





add_executable(
  test_unit
  test_unit.cpp
  warp_test.cu
)
target_link_libraries(
    test_unit
  cuda cudadevrt cudart_static cublas 
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_unit)



set_target_properties(test_unit PROPERTIES
    COMPILE_FLAGS "-g"
    # LINK_FLAGS "2> error.txt"
)
# add_executable(
#   warp_test
#   warp_test.cu
# )
# target_link_libraries(
#   warp_test
#   ${CUDA_LIBRARIES} cuda cudadevrt cudart_static cublas
#   GTest::gtest_main
# )

# # add_test(NAME warp_test COMMAND warp_test)

# include(GoogleTest)
# gtest_discover_tests(warp_test)


# add_custom_target(run_tests ALL
#     COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
#     DEPENDS warp_test
# )
