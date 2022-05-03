cmake_minimum_required(VERSION 2.8)
project(E03-Memery LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# include_directories("/usr/local/cuda/include/")
# include_directories("/usr/local/cuda/lib64")

set (CMAKE_CXX_STANDARD 20)

add_executable(memory memory.cc)
target_link_libraries(memory ${CUDA_LIBRARIES})