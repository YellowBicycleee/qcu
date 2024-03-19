cmake_minimum_required(VERSION 3.8)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_BUILD_TYPE Release)
project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

set(SOURCES src/qcd/qcu_wilson_dslash.cu)

include_directories(include)
add_library(qcu SHARED  src/dslash.cu src/qcu_clover_dslash.cu src/qcu_communicator.cu src/qcu_complex_computation.cu src/qcu_shift_storage_complex.cu src/qcu_wilson_dslash.cu src/qcu_wilson_dslash_neo.cu src/qcu_wilson_dslash_new_new.cu src/new_cg.cu)



find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)