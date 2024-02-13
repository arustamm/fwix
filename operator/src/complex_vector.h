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

    complex_vector(const std::shared_ptr<hypercube>& hyper) {
      nelem = hyper->getN123();
      ndim = hyper->getNdim();
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&n), sizeof(int) * ndim));
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d), sizeof(float) * ndim));
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&o), sizeof(float) * ndim));
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&mat), sizeof(cuFloatComplex) * nelem));

      int* h_n = new int[ndim];
      float* h_d = new float[ndim];
      float* h_o = new float[ndim];
      for (int i=0; i < ndim; ++i) {
        h_n[i] = hyper->getAxis(i+1).n;
        h_d[i] = hyper->getAxis(i+1).d;
        h_o[i] = hyper->getAxis(i+1).o;
      }
      CHECK_CUDA_ERROR(cudaMemcpy(n, h_n, sizeof(int)*ndim, cudaMemcpyHostToDevice));
      CHECK_CUDA_ERROR(cudaMemcpy(d, h_d, sizeof(float)*ndim, cudaMemcpyHostToDevice));
      CHECK_CUDA_ERROR(cudaMemcpy(o, h_o, sizeof(float)*ndim, cudaMemcpyHostToDevice));
      allocated = true;

      delete[] h_n;
      delete[] h_d;
      delete[] h_o;
    }

    complex_vector* to_device() {
      complex_vector* d_vec;
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_vec), sizeof(complex_vector)));
      CHECK_CUDA_ERROR(cudaMemcpy(d_vec, this, sizeof(complex_vector), cudaMemcpyHostToDevice));
      return d_vec;
    }

    void zero() {
      cudaMemset(mat, 0, sizeof(cuFloatComplex)*nelem);
    }

    ~complex_vector() {
      if (allocated) {
          CHECK_CUDA_ERROR(cudaFree(mat));
          CHECK_CUDA_ERROR(cudaFree(n)); 
          CHECK_CUDA_ERROR(cudaFree(d));
          CHECK_CUDA_ERROR(cudaFree(o));
          allocated = false;
        };
    }
} complex_vector;

typedef std::unordered_map<std::string, complex_vector*> ComplexVectorMap;






