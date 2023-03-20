cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DEFAULT_BUILD_TYPE "Release")

project(eigenDNN LANGUAGES C CXX)
set(${PROJECT_NAME}_VERSION 0.0.0)
enable_language(C CXX)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR " does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

set(COMMON_FILES_DIR ${PROJECT_SOURCE_DIR}/common)
set(SRC_FILES_DIR ${PROJECT_SOURCE_DIR}/src)
set(HDR_FILES_DIR ${PROJECT_SOURCE_DIR}/include)
set(EIGEN_FILES_DIR ${PROJECT_SOURCE_DIR}/eigen)
set(EIGEN_UNSUPPORTED_FILES_DIR ${PROJECT_SOURCE_DIR}/eigen/unsupported)
set(Torch_DIR ${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch)
set(Torch_INC ${PROJECT_SOURCE_DIR}/libtorch/include)

### Find Required Libs ###
find_package(Torch REQUIRED)
include_directories(${Torch_INC})


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -no-pie -lpthread")


# Headers and sources
file(GLOB_RECURSE SOURCES "${SRC_FILES_DIR}/*")
file(GLOB_RECURSE HEADERS "${HDR_FILES_DIR}/*")

set(MAIN_FILE ${PROJECT_SOURCE_DIR}/mha.cc)

# Declaring the files
set(ALL_SRC ${HEADERS} ${MAIN_FILE} ${SOURCES} ${COMMON_FILES_DIR})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()

add_compile_options(-Wno-unknown-pragmas)

include(CTest)
add_executable(mha ${ALL_SRC})
target_include_directories(mha PRIVATE ${COMMON_HEADER_DIRS} SYSTEM ${HDR_FILES_DIR} ${EIGEN_FILES_DIR} ${EIGEN_UNSUPPORTED_FILES_DIR} ${TORCH_INCLUDES} ${COMMON_FILES_DIR})
target_link_libraries(mha PRIVATE ${TORCH_LIBRARIES})  #  CUDA::cublas_static CUDA::cublasLt_static

set_target_properties(mha PROPERTIES POSITION_INDEPENDENT_CODE ON)