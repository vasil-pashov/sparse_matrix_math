project(sparse_matrixmath_test)

enable_testing()

set(SRC
	./cpp/triplet.cpp
	./cpp/csr.cpp
    ./cpp/cg.cpp
    ./cpp/cgsquared.cpp
    ./cpp/bicgstab.cpp
    ./cpp/bicgsymmetric.cpp
)

SET(HEADERS
    include/test_common.h
)
add_executable(${PROJECT_NAME} ${SRC} ${HEADERS})
target_link_libraries(${PROJECT_NAME} doctest::doctest sparse_matrix_math)
target_compile_definitions(${PROJECT_NAME} PRIVATE ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")
target_include_directories(${PROJECT_NAME} PRIVATE include)
doctest_discover_tests(${PROJECT_NAME})

add_custom_command(TARGET 
    ${PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND_EXPAND_LISTS
)