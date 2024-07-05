#include <complex_vector.h>
#include <complex_vector.cuh>
#include <iostream>

std::shared_ptr<ComplexVectorMap> make_complex_vector_map(const std::shared_ptr<hypercube>& hyper, dim3 grid, dim3 block) {
  auto vec = std::make_shared<ComplexVectorMap>();
  (*vec)["host"] = new complex_vector(hyper, grid, block);
  (*vec)["device"] = (*vec)["host"]->to_device();
  return vec;
}


void complex_vector::add(complex_vector* vec){
  launch_add(this, vec, _grid_, _block_);
}





