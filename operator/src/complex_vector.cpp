#include <complex_vector.h>
#include <complex_vector.cuh>
#include <iostream>

// complex_vector::complex_vector(const std::shared_ptr<hypercube>& hyper, dim3 grid, dim3 block) {
      
//       set_grid_block(grid, block);

//       nelem = hyper->getN123();
//       ndim = hyper->getNdim();
//       CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&n), sizeof(int) * ndim));
//       CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d), sizeof(float) * ndim));
//       CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&o), sizeof(float) * ndim));
//       CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&mat), sizeof(cuFloatComplex) * nelem));

//       int* h_n = new int[ndim];
//       float* h_d = new float[ndim];
//       float* h_o = new float[ndim];
//       for (int i=0; i < ndim; ++i) {
//         h_n[i] = hyper->getAxis(i+1).n;
//         h_d[i] = hyper->getAxis(i+1).d;
//         h_o[i] = hyper->getAxis(i+1).o;
//       }
//       CHECK_CUDA_ERROR(cudaMemcpyAsync(n, h_n, sizeof(int)*ndim, cudaMemcpyHostToDevice));
//       CHECK_CUDA_ERROR(cudaMemcpyAsync(d, h_d, sizeof(float)*ndim, cudaMemcpyHostToDevice));
//       CHECK_CUDA_ERROR(cudaMemcpyAsync(o, h_o, sizeof(float)*ndim, cudaMemcpyHostToDevice));
//       allocated = true;

//       delete[] h_n;
//       delete[] h_d;
//       delete[] h_o;
//     }

complex_vector* make_complex_vector(const std::shared_ptr<hypercube>& hyper, dim3 grid, dim3 block) {

      complex_vector* vec;
      CHECK_CUDA_ERROR(cudaMallocManaged(&vec, sizeof(complex_vector)));

      vec->set_grid_block(grid, block);

      int nelem = vec->nelem = hyper->getN123();
      int ndim = vec->ndim = hyper->getNdim();
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&vec->n), sizeof(int) * ndim));
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&vec->d), sizeof(float) * ndim));
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&vec->o), sizeof(float) * ndim));
      CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&vec->mat), sizeof(cuFloatComplex) * nelem));

      int* h_n = new int[ndim];
      float* h_d = new float[ndim];
      float* h_o = new float[ndim];
      for (int i=0; i < ndim; ++i) {
        h_n[i] = hyper->getAxis(i+1).n;
        h_d[i] = hyper->getAxis(i+1).d;
        h_o[i] = hyper->getAxis(i+1).o;
      }
      CHECK_CUDA_ERROR(cudaMemcpyAsync(vec->n, h_n, sizeof(int)*ndim, cudaMemcpyHostToDevice));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(vec->d, h_d, sizeof(float)*ndim, cudaMemcpyHostToDevice));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(vec->o, h_o, sizeof(float)*ndim, cudaMemcpyHostToDevice));
      vec->allocated = true;

      delete[] h_n;
      delete[] h_d;
      delete[] h_o;

      return vec;
}


void complex_vector::add(complex_vector* vec){
  launch_add(this, vec, _grid_, _block_);
}





