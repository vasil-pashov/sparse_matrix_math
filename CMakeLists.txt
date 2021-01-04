project(
	sparse_matrix_math
	VERSION 0.0.1
	DESCRIPTION "Library for sparse matrices" 
)
cmake_minimum_required(VERSION 3.19.0)

set(SMM_MULTITHREADING "None" CACHE STRING "Which library to use for multuthreading.")
set_property(CACHE SMM_MULTITHREADING PROPERTY STRINGS None CPPTM)

set(SMM_UNIT_TESTS ON CACHE BOOL "Should add unit test solution to the project")

add_library(${PROJECT_NAME} INTERFACE ./include/sparse_matrix_math.h)
target_include_directories(${PROJECT_NAME} INTERFACE ./include)

if("${SMM_MULTITHREADING}" STREQUAL "CPPTM")
	target_link_libraries(${PROJECT_NAME} INTERFACE cpp_tm)
	target_compile_definitions(${PROJECT_NAME} INTERFACE SMM_MULTITHREADING_CPPTM)
endif()

if(${SMM_UNIT_TESTS})
	add_subdirectory("test" "test")
endif()