cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(DEFAULT_BUILD_TYPE "Release")

project(cuTransDNN LANGUAGES C CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.0.0)
enable_language(C CXX CUDA)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR "cuTransDNN does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})


set(SRC_FILES_DIR ${PROJECT_SOURCE_DIR}/src)
set(HDR_FILES_DIR ${PROJECT_SOURCE_DIR}/include)
set(EIGEN_FILES_DIR ${PROJECT_SOURCE_DIR}/eigen3)
set(CUTLASS_HDR_FILES_DIR ${PROJECT_SOURCE_DIR})
set(FMHA_HDR_FILES_DIR ${PROJECT_SOURCE_DIR}/fmha)

### Find Required Libs ###
find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

# Find CUDA ARCH VERSION
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET( ${CUDA_ARCH_LIST})
list (GET CUDA_ARCH_LIST -2 ARCH_NUMBER)
SET(CUDA_ARCH sm_${ARCH_NUMBER})
SET(CUDA_COMPUTE compute_${ARCH_NUMBER})


message(--------------------------------------------)
message(CUDA_ARCH = ${CUDA_ARCH})
message(CUDA_COMPUTE = ${CUDA_COMPUTE})
message(--------------------------------------------)



set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(COMMON_HEADER_DIRS ${CUDA_PATH}/include)
set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

set(CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC -g -G -Xcompiler -Wall")
set(CUDA_FLAGS "-gencode code=${CUDA_ARCH},arch=${CUDA_COMPUTE} ${CUDA_FLAGS}")
string(REGEX REPLACE "^sm_" "" ARCH_NUMBER "${CUDA_ARCH}")
if ("${CUDA_ARCH}" STREQUAL "sm_${ARCH_NUMBER}")
	set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUMBER}")
else()
	message(FATAL_ERROR "Unknown CUDA_ARCH: ${CUDA_ARCH}")
endif()
message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS}" CACHE STRING "CUDA flags set for cuTransDNN" FORCE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -lpthread")


# Headers and sources
file(GLOB_RECURSE SOURCES "${SRC_FILES_DIR}/*")
file(GLOB_RECURSE HEADERS "${HDR_FILES_DIR}/*")

set(MAIN_FILE ${PROJECT_SOURCE_DIR}/mha.cc)

# Declaring the files
set(ALL_SRC ${HEADERS} ${MAIN_FILE} ${SOURCES})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()

add_compile_options(-Wno-unknown-pragmas)


# if (NOT $ENV{CUTLASS_LIB_DIR})
#   set(CUTLASS_LIB_DIR /usr/lib/x86_64-linux-gnu)
# else()
#   set(CUTLASS_LIB_DIR $ENV{CUTLASS_LIB_DIR})
# endif()
# string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})
# find_library(CUTLASS_LIB cutlass HINTS ${CUTLASS_LIB_DIR} ${LIBRARY_DIRS} PATH_SUFFIXES lib lib64)



add_executable(mha ${ALL_SRC})
target_include_directories(mha PRIVATE ${COMMON_HEADER_DIRS} SYSTEM ${CUDA_INCLUDE_DIRS} ${CUTLASS_HDR_FILES_DIR} ${FMHA_HDR_FILES_DIR} ${HDR_FILES_DIR} ${EIGEN_FILES_DIR})
target_link_directories(mha PRIVATE ${COMMON_LIB_DIRS})
target_link_libraries(mha PRIVATE ${CUDA_LIBRARIES} CUDA::cublas CUDA::cublasLt ${Protobuf_LIBRARIES} ${HDF5_LIBRARIES})  #  CUDA::cublas_static CUDA::cublasLt_static
# if(${CUTLASS_LIB})
# target_link_libraries(mha PUBLIC ${CUTLASS_LIB})
# endif()
set_target_properties(mha PROPERTIES POSITION_INDEPENDENT_CODE ON)