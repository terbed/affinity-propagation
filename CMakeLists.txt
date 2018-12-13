cmake_minimum_required(VERSION 3.13)
project(affinity_propagation)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(affinity_propagation
        AffinityPropagation.cpp
        AffinityPropagation.h
        main.cpp)
