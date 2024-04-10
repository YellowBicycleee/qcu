cmake_minimum_required(VERSION 3.8)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
set(CMAKE_CUDA_ARCHITECTURES 70)    # for V100
set(CMAKE_BUILD_TYPE Release)

project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

# set(SOURCES
#     src/qcd/qcu_wilson_dslash.cu 
#     src/qcu_storage/qcu_storage.cu
#     qcu.cu
# )

# include_directories(include)
# add_library(qcu SHARED ${SOURCES})

add_library(qcu 
    SHARED
    src/comm/qcu_communicator.cu
    src/qcd/qcu_dslash.cu
    src/qcd/qcu_wilson_dslash.cu
    src/qcu_storage/qcu_storage.cu
    src/qcu.cu
    src/algebra/qcu_algebra.cu
    src/solver/qcu_cg.cu
)

target_include_directories(qcu 
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/interface
)

add_executable(test_linear_algebra 
    ${PROJECT_SOURCE_DIR}/src/tests/linear_algebra/test_kernels.cu)
target_link_libraries(test_linear_algebra qcu)
set_target_properties(test_linear_algebra PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "tests")

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_link_libraries(qcu nccl)

target_compile_definitions(qcu PRIVATE MPI_ENABLED)