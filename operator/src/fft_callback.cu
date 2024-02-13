#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>

__device__ void CB_ortho(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {
  int* n = (int*)callerInfo;
  float norm_factor = sqrtf(1.f/float((n[0] * n[1])));
  ((cufftComplex*)dataOut)[offset] = cuCmulf(element, make_cuComplex(norm_factor, 0.0f));
}
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_ortho;

cufftCallbackStoreC get_host_callback_ptr() {
  cufftCallbackStoreC h_storeCallbackPtr;
  cudaMemcpyFromSymbol(&h_storeCallbackPtr, d_storeCallbackPtr, sizeof(h_storeCallbackPtr));
  return h_storeCallbackPtr;
}