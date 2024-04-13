#include "qcd/qcu_dslash.cuh"
#include <assert.h>

BEGIN_NAMESPACE(qcu)
void DslashMV::operator()(_genvector result, _genvector src, cudaStream_t stream) {

  dslash->dslashParam_->fermionOut = result;
  dslash->dslashParam_->fermionIn = src;

  dslash->preApply();
  dslash->apply();
  dslash->postApply();

  CHECK_CUDA(cudaStreamSynchronize(dslash->dslashParam_->stream1));
  CHECK_CUDA(cudaStreamSynchronize(dslash->dslashParam_->stream2));
}

END_NAMESPACE(qcu)