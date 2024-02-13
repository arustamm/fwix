#include "OneStep.h"


OneStep::OneStep(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, bool from_host) 
: cuOperator<complex4DReg, complex4DReg>(domain, range, from_host) {
  // if pointers are not provided, allocate
  
};

void OneStep::forward (bool add, std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data) {
    // 2dFFT (x-y) --> (kx-ky)
    cuFFT.cu_forward(false, )
    ps.cu_forward()
    ss.cu_forward()
  
};


void OneStep::adjoint (bool add, std::shared_ptr<complex2DReg> model, std::shared_ptr<complex2DReg> data);