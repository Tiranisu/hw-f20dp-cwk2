cmake_minimum_required(VERSION 3.17.3)

project(totient)

set(SRC_FILES
    "totient.cpp")

set(SYCL_FILES
    "totient.cpp")

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake")

set(ComputeCpp_DIR CACHE STRING "NOT-FOUND")

if (NOT ComputeCpp_DIR)
    message(FATAL_ERROR
    "SYCL implementation root not provided, please specify "
    "the path to the root of the chosen SYCL implementation using "
    "ComputeCpp_DIR=<path/to/install/root>.")
endif()

find_package(ComputeCpp REQUIRED)

add_executable("${PROJECT_NAME}" "${SRC_FILES}" "${HEADER_FILES}")

target_include_directories("${PROJECT_NAME}" PRIVATE 
                           "${COMPUTECPP_INCLUDE_DIRECTORY}"
                           "${CMAKE_SOURCE_DIR}/src"
			   )

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

target_compile_definitions(totient PUBLIC -DSYCL_LANGUAGE_VERSION=2020)

add_sycl_to_target(TARGET "${PROJECT_NAME}"
                   SOURCES "${SYCL_FILES}")

# https://stackoverflow.com/a/12404666
# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

# https://stackoverflow.com/a/51448364
# find_package(OpenMP REQUIRED)
# target_link_libraries(pi PRIVATE OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
