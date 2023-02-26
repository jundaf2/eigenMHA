# The variable CMAKE_CXX_STANDARD and related were introduced in CMake v3.1
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

option(DEBUG_MODE_ENC "debug enc computation result" OFF)
option(DEBUG_MODE_DEC "debug dec computation result" OFF)
option(DEBUG_RESULT_ATTENTION "debug enc attention computation result" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

# Require C++14 and disable compiler-specific extensions
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DEFAULT_BUILD_TYPE "Release")

#-------------------------------------------------------------------------------
# Project name and version
#-------------------------------------------------------------------------------
project(TransfomrerInfer LANGUAGES C CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.0.0)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR
            "TransfomrerInfer does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})


set(SRC_FILES_DIR ${PROJECT_SOURCE_DIR}/src)
set(HDR_FILES_DIR ${PROJECT_SOURCE_DIR}/include)
set(CUTLASS_HDR_FILES_DIR ${PROJECT_SOURCE_DIR})
set(FMHA_HDR_FILES_DIR ${PROJECT_SOURCE_DIR}/fmha)
# Enable languages.
enable_language(CXX C)

# TransfomrerInfer_USE_CUDA requires CMake 3.8 or newer (for DIRect CUDA support)
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# Use ${CMAKE_CXX_COMPILER} as the cuda host compiler.
if (NOT CMAKE_CUDA_HOST_COMPILER)
	set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

### Find Required Libs ###
find_package(CUDA 11.6 REQUIRED)
find_package(CUDAToolkit)
find_package(HDF5 REQUIRED)
find_package(Protobuf REQUIRED)

# Find CUDA ARCH VERSION
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET( ${CUDA_ARCH_LIST})
list (GET CUDA_ARCH_LIST 0 ARCH_NUMBER)
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


set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
set(HDF5_USE_STATIC_LIBRARIES ON)
set(Protobuf_USE_STATIC_LIBS ON)

protobuf_generate_cpp(TRANSFORMER_PROTO_SRC TRANSFORMER_PROTO_HEADER transformer.proto)


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
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS}" CACHE STRING "CUDA flags set for TransfomrerInfer" FORCE)
set(CUSPARSE_FOUND TRUE)
set(CUSPARSE_LIBRARIES "cusparse")

# Suppress warnings about MACOSX_RPATH
set(CMAKE_MACOSX_RPATH OFF CACHE BOOL "")

#-------------------------------------------------------------------------------
# Process configuration options
#-------------------------------------------------------------------------------
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -lpthread")


message(STATUS "TransfomrerInfer build type: CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "TransfomrerInfer version: v${TransfomrerInfer_VERSION_STRING}")


#-------------------------------------------------------------------------------
# Define and configure the TransfomrerInfer library
#-------------------------------------------------------------------------------

# Headers and sources
set(SOURCES 
      ${SRC_FILES_DIR}/bias_act_dropout.cpp
      ${SRC_FILES_DIR}/bias_add_transform_20314.cpp
      ${SRC_FILES_DIR}/bias_dropout_residual.cpp
      ${SRC_FILES_DIR}/context.cpp
      ${SRC_FILES_DIR}/cross_entropy.cu
      ${SRC_FILES_DIR}/cublas_wrappers.cu
      ${SRC_FILES_DIR}/cuda_util.cu
      ${SRC_FILES_DIR}/decoder.cc.cu
      ${SRC_FILES_DIR}/dropout.cpp
      ${SRC_FILES_DIR}/dropout_kernels.cu
      ${SRC_FILES_DIR}/embedding_kernels.cu
      ${SRC_FILES_DIR}/embKernels.cc.cu
      ${SRC_FILES_DIR}/encoder.cc.cu
      ${SRC_FILES_DIR}/feed_forward_layer.cpp
      ${SRC_FILES_DIR}/feed_forward.cpp
      ${SRC_FILES_DIR}/layer.cpp
      ${SRC_FILES_DIR}/layer_normalize.cpp
      ${SRC_FILES_DIR}/general_kernels.cu
      ${SRC_FILES_DIR}/manager.cpp
      ${SRC_FILES_DIR}/multihead_attention_layer.cpp
      ${SRC_FILES_DIR}/node.cpp
      ${SRC_FILES_DIR}/normalize_kernels.cu
      ${SRC_FILES_DIR}/softmax.cpp
      ${SRC_FILES_DIR}/softmax_kernels.cu
      ${SRC_FILES_DIR}/strided_batch_gemm.cpp
      ${SRC_FILES_DIR}/transform_kernels.cu
      ${SRC_FILES_DIR}/transformer_encoder_layer.cpp
      ${SRC_FILES_DIR}/transformer_weight.cc
      ${SRC_FILES_DIR}/transformer.cc
      ${SRC_FILES_DIR}/transformerKernels.cc.cu
      ${SRC_FILES_DIR}/tensor.cpp
      ${SRC_FILES_DIR}/transform_0213.cpp
      ${SRC_FILES_DIR}/util.cc.cu
      )
set(HEADERS 
      ${HDR_FILES_DIR}/bias_act_dropout.h
      ${HDR_FILES_DIR}/bias_add_transform_20314.h
      ${HDR_FILES_DIR}/bias_dropout_residual.h
      ${HDR_FILES_DIR}/block_reduce.h
      ${HDR_FILES_DIR}/common.h
      ${HDR_FILES_DIR}/context.h
      ${HDR_FILES_DIR}/cublas_wrappers.h
      ${HDR_FILES_DIR}/cuda_util.h
      ${HDR_FILES_DIR}/declaration.h
      ${HDR_FILES_DIR}/decoder.h
      ${HDR_FILES_DIR}/dropout.h
      ${HDR_FILES_DIR}/embKernels.h
      ${HDR_FILES_DIR}/encoder.h
      ${HDR_FILES_DIR}/feed_forward_layer.h
      ${HDR_FILES_DIR}/feed_forward.h
      ${HDR_FILES_DIR}/fused_adam_kernel.h
      ${HDR_FILES_DIR}/kernels.h
      ${HDR_FILES_DIR}/layer_normalize.h
      ${HDR_FILES_DIR}/layer.h
      ${HDR_FILES_DIR}/ls_cub.cuh
      ${HDR_FILES_DIR}/manager.h
      ${HDR_FILES_DIR}/model_base.h
      ${HDR_FILES_DIR}/multi_tensor_apply.cuh
      ${HDR_FILES_DIR}/multihead_attention_layer.h
      ${HDR_FILES_DIR}/node.h
      ${HDR_FILES_DIR}/softmax.h
      ${HDR_FILES_DIR}/strided_batch_gemm.h
      ${HDR_FILES_DIR}/tensor.h
      ${HDR_FILES_DIR}/transform_0213.h
      ${HDR_FILES_DIR}/transformer_encoder_layer.h
      ${HDR_FILES_DIR}/transformer_weight.h
      ${HDR_FILES_DIR}/transformer.h
      ${HDR_FILES_DIR}/transformerKernels.h
      ${HDR_FILES_DIR}/util.h
      ${TRANSFORMER_PROTO_HEADER}
      )

set(MAIN_FILE ${PROJECT_SOURCE_DIR}/transformer_example.cc)


set(TransfomrerInfer_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "The TransfomrerInfer source DIRectory" FORCE)
set(TransfomrerInfer_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} CACHE PATH "The TransfomrerInfer install DIRectory" FORCE)

# Declaring the library
set(ALL_SRC ${HEADERS} ${MAIN_FILE} ${SOURCES} ${TRANSFORMER_PROTO_SRC})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()

add_compile_options(-Wno-unknown-pragmas)

if(DEBUG_MODE_ENC)
  add_definitions(-DDEBUG_RESULT_ENC)
  message(STATUS "Debug computation result")
endif()

if(DEBUG_MODE_DEC)
  add_definitions(-DDEBUG_RESULT_DEC)
  message(STATUS "Debug computation result")
endif()

if(DEBUG_RESULT_ATTENTION)
  add_definitions(-DDEBUG_RESULT_ATTENTION)
  message(STATUS "Debug computation result")
endif()

# if (NOT $ENV{CUTLASS_LIB_DIR})
#   set(CUTLASS_LIB_DIR /usr/lib/x86_64-linux-gnu)
# else()
#   set(CUTLASS_LIB_DIR $ENV{CUTLASS_LIB_DIR})
# endif()
# string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})
# find_library(CUTLASS_LIB cutlass HINTS ${CUTLASS_LIB_DIR} ${LIBRARY_DIRS} PATH_SUFFIXES lib lib64)


add_executable(TransfomrerInfer ${ALL_SRC})
target_include_directories(TransfomrerInfer PRIVATE ${COMMON_HEADER_DIRS} SYSTEM ${CUDA_INCLUDE_DIRS} ${CUTLASS_HDR_FILES_DIR} ${FMHA_HDR_FILES_DIR} ${HDR_FILES_DIR} ${HDF5_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_link_directories(TransfomrerInfer PRIVATE ${COMMON_LIB_DIRS})
target_link_libraries (TransfomrerInfer PRIVATE ${CUDA_LIBRARIES} CUDA::cublas CUDA::cublasLt ${Protobuf_LIBRARIES} ${HDF5_LIBRARIES})  #  CUDA::cublas_static CUDA::cublasLt_static
# if(${CUTLASS_LIB})
# target_link_libraries(TransfomrerInfer PUBLIC ${CUTLASS_LIB})
# endif()
set_target_properties(TransfomrerInfer PROPERTIES POSITION_INDEPENDENT_CODE ON)