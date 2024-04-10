#include "qcd/qcu_dslash.cuh"
#include <assert.h>

BEGIN_NAMESPACE(qcu)
void DslashMV::operator()(_genvector result, _genvector src, cudaStream_t stream) {

  // dslash->dslashParam_->parity = EVEN_PARITY;
  dslash->dslashParam_->fermionOut = result;
  dslash->dslashParam_->fermionIn = src;

  dslash->preApply();
  dslash->apply();
  dslash->postApply();
}
// void DslashMV::operator()(QCU_DAGGER_FLAG daggerFlag, _genvector result, _genvector src,
//                           int parity = 0, cudaStream_t stream = NULL) {

//   dslash->dslashParam_->parity = parity;
//   dslash->dslashParam_->fermionOut = result;
//   dslash->dslashParam_->fermionIn = src;

//   if (daggerFlag == QCU_DAGGER_NO) {
//     dslash->dslashParam_->daggerFlag = QCU_DAGGER_NO;
//     dslash->preApply();
//     dslash->apply();
//     dslash->postApply();
//   } else if (daggerFlag == QCU_DAGGER_YES) {
//     dslash->dslashParam_->daggerFlag = QCU_DAGGER_YES;
//     dslash->preApply();
//     dslash->apply();
//     dslash->postApply();
//   } else if (daggerFlag == QCU_DAGGER_NO_YES) {
//     // assert(0); // to modify
//     DslashParam *dslashParam = dslash->dslashParam_;
//     assert(dslashParam->dslashParam_->tempFermionIn != nullptr);
//     void *tempFermionIn = dslashParam->tempFermionIn;
//     void *realFermionIn = dslashParam->fermionIn;
//     void *realFermionOut = dslashParam->fermionOut;

//     dslashParam->daggerFlag = QCU_DAGGER_NO;
//     dslashParam->fermionOut = tempFermionIn; // D in => temp

//     dslash->preApply();
//     dslash->apply();
//     dslash->postApply();

//     // dslash->dslashParam_->daggerFlag = QCU_DAGGER_YES;
//     dslashParam->daggerFlag = QCU_DAGGER_YES;
//     dslashParam->fermionIn = tempFermionIn;
//     dslashParam->fermionOut = realFermionOut; // D^dagger temp => out
//     dslash->preApply();
//     dslash->apply();
//     dslash->postApply();
//   } else if (daggerFlag == QCU_DAGGER_YES_NO) {
//     // assert(0); // to modify
//     DslashParam *dslashParam = dslash->dslashParam_;
//     assert(dslashParam->dslashParam_->tempFermionIn != nullptr);
//     void *tempFermionIn = dslashParam->tempFermionIn;
//     void *realFermionIn = dslashParam->fermionIn;
//     void *realFermionOut = dslashParam->fermionOut;

//     // dslash->dslashParam_->daggerFlag = QCU_DAGGER_YES;
//     dslashParam->daggerFlag = QCU_DAGGER_YES;
//     dslashParam->fermionOut = tempFermionIn; // D in => temp
//     dslash->preApply();
//     dslash->apply();
//     dslash->postApply();

//     dslashParam->daggerFlag = QCU_DAGGER_NO;
//     dslashParam->fermionIn = tempFermionIn;
//     dslashParam->fermionOut = realFermionOut; // D^dagger temp => out
//     dslash->preApply();
//     dslash->apply();
//     dslash->postApply();
//   } else {
//     assert(0); // invalid dagger flag
//   }
// }
END_NAMESPACE(qcu)