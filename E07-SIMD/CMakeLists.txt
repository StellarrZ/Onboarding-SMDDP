cmake_minimum_required(VERSION 3.17)
project(E07-SIMD)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=broadwell -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=broadwell -Wall -O2")

add_executable(simd simd.cpp)
add_executable(avxloop avxloop.cpp)