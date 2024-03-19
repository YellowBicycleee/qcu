#pragma once

#include "qcd/qcu_wilson_dslash.h"
#include "targets/wilson_dslash_kernel.cuh"

BEGIN_NAMESPACE(qcu)

// use this function to call kernel function, this function donnot sync inside
void WilsonDslash::calculateDslash(int daggerFlag, cudaStream_t &dslashStream) {
    void *gauge = dslashParam_->gauge;
    void *fermionIn = dslashParam_->fermionIn;
    void *fermionOut = dslashParam_->fermionOut;
    int parity = dslashParam_->parity;
    int Lx = dslashParam_->Lx;
    int Ly = dslashParam_->Ly;
    int Lz = dslashParam_->Lz;
    int Lt = dslashParam_->Lt;
    double daggerParam = daggerFlag ? -1.0 : 1.0;
    int vol = Lx * Ly * Lz * Lt;

    int gridSize = (vol / 2 + blockSize_ - 1) / blockSize_;

    dslashKernelFunc<<<gridSize, blockSize_, 0, dslashStream>>>(gauge, fermionIn, fermionOut, Lx,
                                                                Ly, Lz, Lt, parity, grid_x, grid_y,
                                                                grid_z, grid_t, daggerParam);
}

END_NAMESPACE(qcu)