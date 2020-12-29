project(
	sparse_matrixmath
	VERSION 0.0.1
	DESCRIPTION "Library for sparse matrices" 
)
cmake_minimum_required(VERSION 3.19.0)

add_library(${PROJECT_NAME} INTERFACE ./include/sparse_matrixmath.h)
target_include_directories(${PROJECT_NAME} INTERFACE ./include)

add_subdirectory("test" "test")