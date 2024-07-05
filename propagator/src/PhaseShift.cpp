#include "PhaseShift.h"
#include <prop_kernels.cuh>
#include <cuda.h>


PhaseShift::PhaseShift(const std::shared_ptr<hypercube>& domain, float dz, float eps, 
std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data, dim3 grid, dim3 block) 
: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block), _dz_(dz), _eps_(eps) {

  _grid_ = {128, 128, 8};
  _block_ = {16, 16, 2};

  fwd_kernel = PS_kernel(&ps_forward, _grid_, _block_);
  adj_kernel = PS_kernel(&ps_adjoint, _grid_, _block_);

  d_w2 = fill_in_w(domain->getAxis(3));
  d_ky = fill_in_k(domain->getAxis(2));
  d_kx = fill_in_k(domain->getAxis(1));

  _nw_ = domain->getAxis(3).n;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&_sref_, _nw_ * sizeof(std::complex<float>)));
};

void PhaseShift::cu_forward (bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {
  if (!add) (*data)["host"]->zero();
  fwd_kernel.launch((*model)["device"], (*data)["device"], d_w2, d_kx, d_ky, _sref_, _dz_, _eps_);
};


void PhaseShift::cu_adjoint (bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {
  if (!add) (*model)["host"]->zero();
  adj_kernel.launch((*model)["device"], (*data)["device"], d_w2, d_kx, d_ky, _sref_, _dz_, _eps_);
}