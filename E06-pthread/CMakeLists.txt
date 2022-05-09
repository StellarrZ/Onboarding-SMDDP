cmake_minimum_required(VERSION 3.17)
project(E06-pthread)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS -pthread)

find_package(Threads)

add_executable(pthread pthread.cpp)

# target_link_libraries(pthread ${CMAKE_THREAD_LIBS_INIT})