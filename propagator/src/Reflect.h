#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <prop_kernels.cuh>

// reflecting wavefields in the volume [nz, ns, nw, ny, nx] 
class Reflect : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  Reflect (const std::shared_ptr<hypercube>& domain, std::vector<std::shared_ptr<complex4DReg>> slow_impedance, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0);

  void set_depth(int iz) {
    // copy 2 slices of the model
    int offset = iz * slice_size;
    if (iz < nz-1) {
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slow_slice, _slow->getVals() + offset, 2*slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_den_slice, _density->getVals() + offset, 2*slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
    }
    else {
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slow_slice, _slow->getVals() + offset, slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_den_slice, _density->getVals() + offset, slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
      // add the same layer to effectively have 0 reflection coeffecient
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slow_slice + slice_size, d_slow_slice, slice_size*sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_den_slice + slice_size, d_den_slice, slice_size*sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, _stream_));
    }
    
  }

  virtual ~Reflect() { 
    CHECK_CUDA_ERROR(cudaFree(d_slow_slice));
    CHECK_CUDA_ERROR(cudaFree(d_den_slice));
   };

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

  void cu_forward (complex_vector* __restrict__ model);
  void cu_adjoint (complex_vector* __restrict__ data);

private:
  std::shared_ptr<complex4DReg> _slow, _density;
  int nw, ny, nx, ns, nz;
  int slice_size;
  cuFloatComplex *d_slow_slice, *d_den_slice;

  Refl_launcher launcher, launcher_in_place;
};