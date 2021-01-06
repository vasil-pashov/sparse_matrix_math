project(sparse_matrixmath_unit_test)

enable_testing()
include(GoogleTest)

find_package(GTest REQUIRED CONFIG)

add_executable(${PROJECT_NAME} ./cpp/unit_test.cpp)
target_link_libraries(${PROJECT_NAME} GTest::GTest sparse_matrix_math)
target_compile_definitions(${PROJECT_NAME} PRIVATE ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")

gtest_discover_tests(${PROJECT_NAME})