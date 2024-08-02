#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <CudaKernel.cuh>
#include <CudaKernel.cu>

template class CudaKernel<int, int*>;
__global__ void select_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int value, int* labels) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int is=0; is < NS; ++is) {
    for (int iw=iw0; iw < NW; iw += jw) {
      for (int iy=iy0; iy < NY; iy += jy) {
        for (int ix=ix0; ix < NX; ix += jx) {
          int i = ix + (iy + iw*NY)*NX;
          if (labels[i] == value) {
            int nd_ind[] = {is, iw, iy, ix};
            int ind = ND_TO_FLAT(nd_ind, dims);
            data->mat[ind] = cuCaddf(data->mat[ind], model->mat[ind]); 
          }
        }
      }
    }
  }
};
