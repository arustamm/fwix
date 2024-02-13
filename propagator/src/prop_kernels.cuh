#pragma once
#include <cuda_runtime.h>
#include <complex_vector.h>
#include <cuComplex.h>
#include <CudaKernel.cuh>

__global__ void ps_forward(complex_vector* model, complex_vector* data, 
  float* w2, float* kx, float* ky, cuFloatComplex slow_ref, float dz, float eps);
// void call_ps_forward(CudaKernelParams &par, complex_vector* model, complex_vector* data, float* w2, float* kx, float* ky, cuFloatComplex slow_ref, float dz, float eps);

__global__ void ps_adjoint(complex_vector* model, complex_vector* data, 
  float* w2, float* kx, float* ky, cuFloatComplex slow_ref, float dz, float eps);
  // void call_ps_adjoint(CudaKernelParams &par, complex_vector* model, complex_vector* data, float* w2, float* kx, float* ky, cuFloatComplex slow_ref, float dz, float eps);
  
typedef CudaKernel<float*, float*, float*, cuFloatComplex, float, float> PS_kernel;
