#include <CudaKernel.cuh>

template <typename... Args>
CudaKernel<Args...>::CudaKernel() {};

template <typename... Args>
CudaKernel<Args...>::CudaKernel(void (*kernel)(complex_vector*, complex_vector*, Args...), dim3 grid, dim3 block) : _grid_(grid), _block_(block), _kernel_(kernel) {};

template <typename... Args>
CudaKernel<Args...>::~CudaKernel() {};

template <typename... Args>
void CudaKernel<Args...>::launch(complex_vector* model, complex_vector* data, Args... args) {;
    _kernel_<<<_grid_, _block_>>>(model, data, args...);
    CHECK_CUDA_ERROR( cudaPeekAtLastError() );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize() );
  };