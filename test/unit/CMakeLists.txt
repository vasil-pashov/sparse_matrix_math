project(sparse_matrixmath_unit_test)

enable_testing()
include(GoogleTest)

find_package(GTest REQUIRED CONFIG)

set(TEST_FILES
    ./cpp/unit_test_main.cpp
    ./cpp/unit_test_bicgsymmetric.cpp
    ./cpp/unit_test_bicgsquared.cpp
    ./cpp/unit_test_bicgstab.cpp
    ./cpp/unit_test_cg.cpp
)

set(INCLUDE
    include/test_common.h
    include/solver_common.h
)

add_executable(${PROJECT_NAME} ${TEST_FILES} ${INCLUDE})
target_link_libraries(${PROJECT_NAME} GTest::GTest sparse_matrix_math)
target_compile_definitions(${PROJECT_NAME} PRIVATE ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")
target_include_directories(${PROJECT_NAME} PRIVATE include)
gtest_discover_tests(${PROJECT_NAME})