#pragma once
#include <hypercube.h>
#include <complex>
#include <cuda_runtime.h>
#include <iostream>
#include <cuComplex.h>
#include <unordered_map>
// a primitive complex vector that holds up to 7-d hypercube and a pointer to a CUDA device 

// thanks to Google's AI Bard
#define ND_TO_FLAT(idx, dims) ( \
    ({ \
        size_t _flat_idx = 0; \
        size_t _dim_product = 1; \
        for (int i = sizeof(dims) / sizeof(dims[0]) - 1; i >= 0; --i) { \
            _flat_idx += (idx[i]) * _dim_product; \
            _dim_product *= dims[i]; \
        } \
        _flat_idx; \
    }) \
)

#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
};

using namespace SEP;

typedef struct complex_vector
{
    cuFloatComplex* mat;
    bool allocated = false;
    int* n;
    float* d;
    float* o;
    int nelem, ndim;
    // for kernels
    dim3 _grid_, _block_;

    // complex_vector(const std::shared_ptr<hypercube>& hyper, dim3 grid=1, dim3 block=1);

    void set_grid_block(dim3 grid, dim3 block) {
      _grid_ = grid.x * grid.y * grid.z;
      _block_ = block.x * block.y * block.z;
    }

    // complex_vector* to_device() {
    //   complex_vector* d_vec;
    //   CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&d_vec), sizeof(complex_vector)));
    //   CHECK_CUDA_ERROR(cudaMemcpyAsync(d_vec, this, sizeof(complex_vector), cudaMemcpyHostToDevice));
    //   return d_vec;
    // }

    void zero() {
      CHECK_CUDA_ERROR(cudaMemset(mat, 0, sizeof(cuFloatComplex)*nelem));
    }

    void add(complex_vector* vec);

    complex_vector* clone() {
      complex_vector* vec;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(vec->n, n, sizeof(int)*ndim, cudaMemcpyDeviceToDevice));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(vec->d, d, sizeof(float)*ndim, cudaMemcpyDeviceToDevice));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(vec->o, o, sizeof(float)*ndim, cudaMemcpyDeviceToDevice));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(vec->mat, mat, sizeof(cuFloatComplex)*ndim, cudaMemcpyDeviceToDevice));
      vec->nelem = nelem;
      vec->ndim = ndim;
      allocated = true;
      return vec;
    }

    ~complex_vector() {
      if (allocated) {
          printf("d\n");
          CHECK_CUDA_ERROR(cudaFree(mat));
          CHECK_CUDA_ERROR(cudaFree(n)); 
          CHECK_CUDA_ERROR(cudaFree(d));
          CHECK_CUDA_ERROR(cudaFree(o));
          allocated = false;
        };
    }
} complex_vector;

complex_vector* make_complex_vector(const std::shared_ptr<hypercube>& hyper, dim3 grid=1, dim3 block=1);





