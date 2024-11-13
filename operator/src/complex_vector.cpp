#include <complex_vector.h>
#include <complex_vector.cuh>
#include <iostream>

complex_vector* make_complex_vector(const std::shared_ptr<hypercube>& hyper, dim3 grid, dim3 block) {
  complex_vector* vec;
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec), sizeof(complex_vector)));

  vec->set_grid_block(grid, block);

  int nelem = vec->nelem = hyper->getN123();
  int ndim = vec->ndim = hyper->getNdim();
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->n), sizeof(int) * ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->d), sizeof(float) * ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->o), sizeof(float) * ndim));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&vec->mat), sizeof(cuFloatComplex) * nelem));

  for (int i=0; i < ndim; ++i) {
    vec->n[i] = hyper->getAxis(i+1).n;
    vec->d[i] = hyper->getAxis(i+1).d;
    vec->o[i] = hyper->getAxis(i+1).o;
  }
  vec->allocated = true;

  return vec;
};

void complex_vector::add(complex_vector* vec){
  launch_add(this, vec, _grid_, _block_);
}

complex_vector*  complex_vector::make_view() {
  complex_vector* vec;
  CHECK_CUDA_ERROR(cudaMallocManaged(&vec, sizeof(complex_vector)));

  vec->set_grid_block(_grid_, _block_);

  // Calculate the size of the new vector
  vec->ndim = this->ndim - 1;
  vec->nelem = this->nelem / this->n[this->ndim - 1]; 
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->n), sizeof(int) * vec->ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->d), sizeof(float) * vec->ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->o), sizeof(float) * vec->ndim));
  for (int i=0; i < vec->ndim; ++i) {
    vec->n[i] = this->n[i];
    vec->d[i] = this->d[i];
    vec->o[i] = this->o[i];
  }
  
  vec->allocated = false;

  return vec;
}

void complex_vector::view_at(complex_vector* view, int index) {
  if (view->allocated) throw std::runtime_error("The provided complex_vector is not a view!");
  // Calculate the offset in the original data
  int offset = index * view->nelem;
  view->mat = this->mat + offset; 
}





