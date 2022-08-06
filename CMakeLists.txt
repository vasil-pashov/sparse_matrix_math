cmake_minimum_required(VERSION 3.24.0)
project(
	sparse_matrix_math
	VERSION 0.2.0
	DESCRIPTION "Library for sparse matrices" 
)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME} INTERFACE ./include/sparse_matrix_math.h)

option(SMM_WITH_MULTITHREADING "Should the computations be multithreaded." ON)
option(SMM_WITH_TESTS "Should add unit test solution to the project" OFF)
option(SMM_WITH_INSTALL "Should generate install target" OFF)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

# Unit tests setup
if(${SMM_WITH_TESTS})
	# Most projects don't need to call find_package and check <name>_FOUND
	# variable when working with FetchContent, because fetch content has
	# FIND_PACKAGE_ARGS which if passed will first try to execute find_package
	# with those arguments and then fallback on FetchContent. However for this
	# project we need to include a file doctest.cmake. If find_package finds
	# the package we can do it with include(doctest) because it is installed
	# into <package>/lib/cmake and cmake knows where to search for it. If
	# find_package fails and FetchContent does a fallback the file would be
	# in ${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake. It seems like there
	# is no way to tell if FetchContent_Declare with FIND_PACKAGE_ARGS did
	# find_package successfully or it had to fallback, because FetchContent_Declare
	# does not set <package>_FOUND even if the package was found.
	set(doctest_version 2.4.8)
	find_package(doctest ${doctest_version} QUIET)
	set(doctest_cmake_path "doctest")
	if(NOT ${doctest_FOUND})
		message(STATUS "Doctest was not found. Proceeding to download it.")
		FetchContent_Declare(
			doctest
			GIT_REPOSITORY https://github.com/doctest/doctest.git
			GIT_TAG "v${doctest_version}"
		)
		set(DOCTEST_NO_INSTALL ON CACHE BOOL "")
		set(DOCTEST_WITH_MAIN_IN_STATIC_LIB OFF CACHE BOOL "")
		FetchContent_MakeAvailable(doctest)
		set(doctest_cmake_path ${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
	endif()
	include(${doctest_cmake_path})
	enable_testing()

	add_subdirectory("test" "test")
endif()

# Multithreading setup
if(${SMM_WITH_MULTITHREADING})
	set(tbb_version "2021.5.0")
	FetchContent_Declare(
		tbb
		GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
		GIT_TAG "v${tbb_version}"
		FIND_PACKAGE_ARGS
			NAMES TBB
			VERSION ${tbb_version}
	)
	set(TBB_TEST OFF CACHE BOOL "")
	FetchContent_MakeAvailable(tbb)
	target_link_libraries(${PROJECT_NAME} INTERFACE TBB::tbb)
	target_compile_definitions(${PROJECT_NAME} INTERFACE -DSMM_MULTITHREADING)
endif()

if(${SMM_WITH_INSTALL})
	include(GNUInstallDirs)
	include(CMakePackageConfigHelpers)

	if(UNIX)
		include(GNUInstallDirs)
		set(include_install_dir ${CMAKE_INSTALL_INCLUDEDIR})
		set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
		set(lib_install_dir ${CMAKE_INSTALL_LIBDIR})
	else()
		set(include_install_dir "include")
		set(config_install_dir "lib/cmake/${PROJECT_NAME}")
		set(lib_install_dir "lib")
	endif()
	set(targets_export_name "${PROJECT_NAME}Targets")

	install(
		TARGETS ${PROJECT_NAME}
		EXPORT ${targets_export_name}
		ARCHIVE DESTINATION ${lib_install_dir}
		INCLUDES DESTINATION ${include_install_dir}
	)

	install(
		EXPORT ${targets_export_name}
		FILE "${targets_export_name}.cmake"
		NAMESPACE ${PROJECT_NAME}::
		DESTINATION ${config_install_dir}
	)

	write_basic_package_version_file(
		"${PROJECT_NAME}ConfigVersion.cmake"
		VERSION ${PROJECT_VERSION}
		COMPATIBILITY SameMinorVersion
	)

	configure_package_config_file(
		"${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
		"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
		INSTALL_DESTINATION ${config_install_dir}
	)

	install(
		FILES
			"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
			"${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		DESTINATION ${config_install_dir}
	)

	install(
		FILES ${PROJECT_SOURCE_DIR}/include/sparse_matrix_math.h
		DESTINATION ${include_install_dir}
	)
endif()

target_include_directories(
	${PROJECT_NAME}
	INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${include_install_dir}>
)