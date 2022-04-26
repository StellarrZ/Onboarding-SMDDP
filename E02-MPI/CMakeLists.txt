cmake_minimum_required(VERSION 2.8)
project(E02-MPI)

find_package(MPI REQUIRED)

set (CMAKE_CXX_COMPILER "/bin/mpicxx")
set (CMAKE_CXX_FLAGS "-g -Wall ${CMAKE_CXX_FLAGS}")
# include_directories("${MPI_CXX_INCLUDE_DIRS}")

add_executable(mpi_hello_world mpi_hello_world.cc)