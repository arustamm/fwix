#pragma once
#include <cuda_runtime.h>
#include <complex_vector.h>
#include <cuComplex.h>
#include <KernelLauncher.cuh>

// phase shift
__global__ void ps_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps);
__global__ void ps_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps);
  __global__ void ps_inverse(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
    const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps);
typedef KernelLauncher<const float*, const float*, const float*, const cuFloatComplex*, float, float> PS_launcher;

// selector
__global__ void select_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int value, const int* __restrict__ labels);
typedef KernelLauncher<int, const int*> Selector_launcher;

  // injection
__global__ void inj_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const float* __restrict__ cx, const float* __restrict__ cy, const float* __restrict__ cz, const int* __restrict__ ids, float oz, float dz, int iz);
__global__ void inj_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const float* __restrict__ cx, const float* __restrict__ cy, const float* __restrict__ cz, const int* __restrict__ ids, float oz, float dz, int iz);
typedef KernelLauncher<const float*, const float*, const float*, const int*, float, float, int> Injection_launcher;
  
// reflection
  __global__ void refl_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* slow_slice, const complex_vector* __restrict__ den_slice);
  __global__ void refl_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice);
  __global__ void refl_forward_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice);
  __global__ void refl_adjoint_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice);
  typedef KernelLauncher<const complex_vector*, const complex_vector*> Refl_launcher;

  // taper
__global__ void taper_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int tapx, int tapy);
typedef KernelLauncher<int, int> Taper_launcher;
