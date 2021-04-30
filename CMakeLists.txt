cmake_minimum_required(VERSION 3.19.0)
project(
	sparse_matrix_math
	VERSION 0.2.0
	DESCRIPTION "Library for sparse matrices" 
)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME} INTERFACE ./include/sparse_matrix_math.h)

# Multithreading setup
set(SMM_MULTITHREADING "None" CACHE STRING "Which library to use for multuthreading.")
set_property(CACHE SMM_MULTITHREADING PROPERTY STRINGS None CPPTM)
if("${SMM_MULTITHREADING}" STREQUAL "CPPTM")
	find_package(cpp_tm 0.1.0 REQUIRED CONFIG)
	target_link_libraries(${PROJECT_NAME} INTERFACE cpp_tm::cpp_tm)
	target_compile_definitions(${PROJECT_NAME} INTERFACE SMM_MULTITHREADING_CPPTM)
endif()

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

target_include_directories(${PROJECT_NAME}
	INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
	          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS ${PROJECT_NAME}
	EXPORT ${PROJECT_NAME}_Targets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

write_basic_package_version_file(
	"${PROJECT_NAME}ConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMinorVersion)

if(UNIX)
	set(
		CONFIG_INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
		CACHE STRING "Path where config files used by find_package will be installed, gets appended to CMAKE_INSTALL_PREFIX")
else()
	set(
		CONFIG_INSTALL_DIR ${PROJECT_NAME}/cmake
		CACHE STRING "Path where config files used by find_package will be installed, gets appended to CMAKE_INSTALL_PREFIX")
endif()

configure_package_config_file(
	"${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
	"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	INSTALL_DESTINATION
	${CONFIG_INSTALL_DIR})

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CONFIG_INSTALL_DIR})

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CONFIG_INSTALL_DIR})

install(FILES ${PROJECT_SOURCE_DIR}/include/sparse_matrix_math.h DESTINATION include/sparse_matrix_math)


# Unit tests setup
set(SMM_UNIT_TESTS ON CACHE BOOL "Should add unit test solution to the project")
if(${SMM_UNIT_TESTS})
	add_subdirectory("test" "test")
endif()
