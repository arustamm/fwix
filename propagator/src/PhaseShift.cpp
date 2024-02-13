#include "PhaseShift.h"
#include <prop_kernels.cuh>
#include <cuda.h>


PhaseShift::PhaseShift(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, float dz, float eps, bool from_host, dim3 grid, dim3 block) 
: CudaOperator<complex4DReg, complex4DReg>(domain, range, from_host, grid, block), _dz_(dz), _eps_(eps) {

  fwd_kernel = PS_kernel(&ps_forward, _grid_, _block_);
  adj_kernel = PS_kernel(&ps_adjoint, _grid_, _block_);

  d_w2 = fill_in_w(domain->getAxis(3));
  d_ky = fill_in_k(domain->getAxis(2));
  d_kx = fill_in_k(domain->getAxis(1));

};

void PhaseShift::cu_forward (bool add, ComplexVectorMap& model, ComplexVectorMap& data) {
  if (!add) data["host"]->zero();
  fwd_kernel.launch(model["device"], data["device"], d_w2, d_kx, d_ky, _sref_, _dz_, _eps_);
};


void PhaseShift::cu_adjoint (bool add, ComplexVectorMap& model, ComplexVectorMap& data) {
  if (!add) model["host"]->zero();
  adj_kernel.launch(model["device"], data["device"], d_w2, d_kx, d_ky, _sref_, _dz_, _eps_);
}