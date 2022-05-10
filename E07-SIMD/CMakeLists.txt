cmake_minimum_required(VERSION 3.17)
project(E07-SIMD)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=broadwell -Wall")

# find_package(Threads)

add_executable(simd simd.cpp)

# target_link_libraries(pthread ${CMAKE_THREAD_LIBS_INIT})