#pragma once

#include <targets/shift_storage/qcu_shift_storage.cuh>

void shiftVectorStorageTwoDouble(void *dst_vec, void *src_vec, int shift_direction, int Lx, int Ly,
                                 int Lz, int Lt);

void shiftGaugeStorageTwoDouble(void *dst_vec, void *src_vec, int shift_direction, int Lx, int Ly,
                                int Lz, int Lt);

void shiftCloverStorageTwoDouble(void *dst_vec, void *src_vec, int shift_direction, int Lx, int Ly,
                                 int Lz, int Lt);