cmake_minimum_required(VERSION 3.8)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
set(CMAKE_CUDA_ARCHITECTURES 70)
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
    src/qcd/qcu_wilson_dslash.cu
    src/qcu_storage/qcu_storage.cu
    src/qcu.cu
)
target_include_directories(qcu 
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/interface
)


find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)