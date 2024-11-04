#include "PhaseShift.h"
#include <prop_kernels.cuh>
#include <cuda.h>

Injection::Injection(const std::shared_ptr<hypercube>& domain,const std::shared_ptr<hypercube>& range, 
std::map<int, std::vector<std::vector<float>> > coord_map,
complex_vector* model, complex_vector* data, dim3 grid, dim3 block) 
: CudaOperator<complex2DReg, complex4DReg>(domain, range, model, data, grid, block) {

  launcher = Injection_launcher(&injection_forward, &injection_adjoint, _grid_, _block_);
  
  int ns = domain->getAxis(2).n; // sources or receivers
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cx, ns * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cy, ns * sizeof(float)));

  set_depth(0);
};
  
void Injection::cu_forward (bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) data->zero();
  if (inject) launcher.run_fwd(model, data, d_cx, d_cy);

};
void Injection::cu_adjoint (bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data) {
  if (!add) model->zero();
  if (inject) launcher.run_adj(model, data, d_cx, d_cy);
};
