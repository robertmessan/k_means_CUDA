cmake_minimum_required(VERSION 3.10)
project(KMeans LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_executable(kmeans_seq kmeans_sequential.cpp)
add_executable(kmeans_par kmeans_parallel.cu)
add_executable(shared_kmeans_par shared_kmeans_par.cu)

set_target_properties(kmeans_par PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(shared_kmeans_par PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)