#pragma once
#include <cuda_runtime.h>
#include <complex_vector.h>
#include <cuComplex.h>
#include <KernelLauncher.cuh>

// phase shift
__global__ void ps_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  float* w2, float* kx, float* ky, cuFloatComplex* slow_ref, float dz, float eps);
__global__ void ps_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  float* w2, float* kx, float* ky, cuFloatComplex* slow_ref, float dz, float eps);
  __global__ void ps_inverse(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
    float* w2, float* kx, float* ky, cuFloatComplex* slow_ref, float dz, float eps);
typedef KernelLauncher<float*, float*, float*, cuFloatComplex*, float, float> PS_launcher;
// selector
__global__ void select_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int value, int* labels);
typedef KernelLauncher<int, int*> Selector_launcher;
  // injection
__global__ void inj_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, float* cx, float* cy, float* cz, int* ids, float oz, float dz, int iz);
__global__ void inj_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, float* cx, float* cy, float* cz, int* ids, float oz, float dz, int iz);
typedef KernelLauncher<float*, float*, float*, int*, float, float, int> Injection_launcher;
  // reflection
  __global__ void refl_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, complex_vector*, complex_vector*);
  __global__ void refl_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, complex_vector*, complex_vector*);
  __global__ void refl_forward_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, complex_vector*, complex_vector*);
  __global__ void refl_adjoint_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, complex_vector*, complex_vector*);
  typedef KernelLauncher<complex_vector*, complex_vector*> Refl_launcher;
// taper
__global__ void taper_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int tapx, int tapy);
typedef KernelLauncher<int, int> Taper_launcher;
