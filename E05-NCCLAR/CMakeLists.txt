cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(E05-NCCLAR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
include(FindNCCL)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.3/)
set(Torch_DIR /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/)
set(CUDNN_INCLUDE_PATH /usr/local/cuda/include)
set(CUDNN_LIBRARY_PATH /usr/local/cuda/lib64/)
set(NCCL_TOPO_FILE /usr/local/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(ncclar ncclar.cpp)

target_link_libraries(ncclar ${MPI_LIBRARIES})
target_link_libraries(ncclar ${CUDA_LIBRARIES})
target_link_libraries(ncclar ${TORCH_LIBRARIES})