#pragma once
#include <functional>
#include <tuple>
#include <complex_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename... Args>
class CudaKernel {
public:
  CudaKernel();
  CudaKernel(void (*kernel)(complex_vector*, complex_vector*, Args...), dim3 grid, dim3 block);
  ~CudaKernel();

  void launch(complex_vector* model, complex_vector* data, Args... args);

  dim3 _grid_, _block_;
  void (*_kernel_)(complex_vector*, complex_vector*, Args...);
};
