#pragma once

#include "qcd/qcu_dslash.h"

BEGIN_NAMESPACE(qcu)
class WilsonDslash : public Dslash {

public:
    // use this function to call kernel function, this function donnot sync inside
    virtual void apply(int daggerFlag = 0); // to implement

    // TODO: WILSON DSLASH MatMul
    void wilsonMatMul() {}
};

END_NAMESPACE(qcu)