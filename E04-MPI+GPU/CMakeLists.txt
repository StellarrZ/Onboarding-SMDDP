cmake_minimum_required(VERSION 2.8)
project(E04-MPI+GPU)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.3/)
set(Torch_DIR /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/)
set(CUDNN_INCLUDE_PATH /usr/local/cuda/include)
set(CUDNN_LIBRARY_PATH /usr/local/cuda/lib64/)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(mpi+gpu mpi+gpu.cc)

target_link_libraries(mpi+gpu ${MPI_LIBRARIES})
# target_link_libraries(mpi+gpu ${CUDA_LIBRARIES})
target_link_libraries(mpi+gpu ${TORCH_LIBRARIES})