#include <complex_vector.h>
#include <iostream>

complex_vector* make_complex_vector_on_device(const std::shared_ptr<hypercube>& hyper) {
  complex_vector *h_vec = new complex_vector(hyper);
  complex_vector *d_vec;

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_vec), sizeof(complex_vector)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_vec, h_vec, sizeof(complex_vector), cudaMemcpyHostToDevice));
  free(h_vec);
  return d_vec;
}






