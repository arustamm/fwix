#pragma once
#include <functional>
#include <tuple>
#include <complex_vector.h>
#include <cuda.h>
#include <iostream>

template <typename... Args>
class KernelLauncher {
public:
  KernelLauncher();
  KernelLauncher(void (*fwd_kernel)(complex_vector* __restrict__, complex_vector* __restrict__, Args...), 
  void (*adj_kernel)(complex_vector* __restrict__, complex_vector* __restrict__, Args...), 
  dim3 grid, dim3 block);
  KernelLauncher(void (*fwd_kernel)(complex_vector* __restrict__, complex_vector* __restrict__, Args...),
  dim3 grid, dim3 block);
  ~KernelLauncher();

  void run_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args);
  void run_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args);

  dim3 _grid_, _block_;
  void (*_fwd_kernel_)(complex_vector* __restrict__, complex_vector* __restrict__, Args...);
  void (*_adj_kernel_)(complex_vector* __restrict__, complex_vector* __restrict__, Args...);

};
