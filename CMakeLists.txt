cmake_minimum_required(VERSION 3.27)
project(CpuAvx512Test)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-mavx2 -mavx512f)
add_executable(CpuAvx512Test main.cpp)
