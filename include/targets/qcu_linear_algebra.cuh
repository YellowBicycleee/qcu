#pragma once

template <typename _Type>
__global__ void reduceSum (void* result, void* src, int vectorLength) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
}
