#include "Reflect.h"

Reflect::Reflect (const std::shared_ptr<hypercube>& domain, std::vector<std::shared_ptr<complex4DReg>> slow_impedance, 
  complex_vector* model, complex_vector* data, 
  dim3 grid, dim3 block, cudaStream_t stream) :
CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {

  _slow = slow_impedance[0];
  _density = slow_impedance[1];

  _grid_ = {32, 4, 4};
  _block_ = {16, 16, 4};

  nz = _slow->getHyper()->getAxis(4).n;
  nw = _slow->getHyper()->getAxis(3).n;
  ny = _slow->getHyper()->getAxis(2).n;
  nx = _slow->getHyper()->getAxis(1).n;
  
  slice_size = nx * ny * nw;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_slow_slice, 2*slice_size * sizeof(std::complex<float>)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_den_slice, 2*slice_size * sizeof(std::complex<float>)));

  launcher = Refl_launcher(&refl_forward, &refl_adjoint, _grid_, _block_, _stream_);
  launcher_in_place = Refl_launcher(&refl_forward_in, &refl_adjoint_in, _grid_, _block_, _stream_);
};

void Reflect::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();
  launcher.run_fwd(model, data, d_slow_slice, d_den_slice);

}

void Reflect::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
	if(!add) model->zero();
  launcher.run_adj(model, data, d_slow_slice, d_den_slice);
}

void Reflect::cu_forward(complex_vector* __restrict__ model) {
  launcher_in_place.run_fwd(model, model, d_slow_slice, d_den_slice);
}

void Reflect::cu_adjoint(complex_vector* __restrict__ data) {
  launcher_in_place.run_adj(data, data, d_slow_slice, d_den_slice);
}