cmake_minimum_required(VERSION 2.8)
project(E01-CMake)

set (CMAKE_CXX_FLAGS "-std=c++20 -Wall ${CMAKE_CXX_FLAGS}")

add_executable(Hello_World Hello_World.c)