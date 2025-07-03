#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

#define TILE_DIM 16

__global__ void ic_fwd(
    complex_vector* __restrict__ model,  
    complex_vector* __restrict__ data,
    const complex_vector* __restrict__ source_wfld
) {
    const int NX = data->n[0];
    const int NY = data->n[1];
    const int NW = data->n[2];
    const int NS = data->n[3];
    
    int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
    int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
    int iws0 = threadIdx.z + blockDim.z * blockIdx.z;

    int jx = blockDim.x * gridDim.x;
    int jy = blockDim.y * gridDim.y;
    int jws = blockDim.z * gridDim.z;

    const int dims[] = {NS, NW, NY, NX};
    const int im_dims[] = {NW, NY, NX};

    for (int iws = iws0; iws < NW*NS; iws += jws) {
        int iw = iws / NS;  // Frequency index
        int is = iws % NS; // Source index

        // Loop over spatial points in the receiver wavefield
      for (int iy = iy0; iy < NY; iy += jy) {
        for (int ix = ix0; ix < NX; ix += jx) {

          int im_ind[] = {iw, iy, ix};
          size_t flat_ind = ND_TO_FLAT(im_ind, im_dims);
          cuFloatComplex img_val = model->mat[flat_ind];

          int nd_ind[] = {is, iw, iy, ix};
          flat_ind = ND_TO_FLAT(nd_ind, dims);

          // Get values
          cuFloatComplex src_val = source_wfld->mat[flat_ind];
          data->mat[flat_ind] = cuCaddf(data->mat[flat_ind], cuCmulf(src_val, img_val));
      }
    }
  }
}

__global__ void ic_adj(
    complex_vector* __restrict__ model,  
    complex_vector* __restrict__ data,
    const complex_vector* __restrict__ source_wfld
) {
  const int NX = data->n[0];
  const int NY = data->n[1];
  const int NW = data->n[2];
  const int NS = data->n[3];
  
  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iws0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jws = blockDim.z * gridDim.z;

  const int dims[] = {NS, NW, NY, NX};
  const int im_dims[] = {NW, NY, NX};

  for (int iws = iws0; iws < NW*NS; iws += jws) {
      int iw = iws / NS;  // Frequency index
      int is = iws % NS; // Source index

      // Loop over spatial points in the receiver wavefield
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {

        int nd_ind[] = {is, iw, iy, ix};
        size_t flat_ind = ND_TO_FLAT(nd_ind, dims);

        // Get values
        cuFloatComplex src_val = source_wfld->mat[flat_ind];
        cuFloatComplex rec_val = data->mat[flat_ind];
        
        // Adjoint: conj(source) * receiver
        cuFloatComplex contribution = cuCmulf(cuConjf(src_val), rec_val);

        // Atomic add to model (multiple threads will write to same (iw,iy,ix) location)
        int im_ind[] = {iw, iy, ix};
        size_t img_flat_ind = ND_TO_FLAT(im_ind, im_dims);
        
        atomicAdd(&(model->mat[img_flat_ind].x), cuCrealf(contribution));
        atomicAdd(&(model->mat[img_flat_ind].y), cuCimagf(contribution));
      }
    }
  }
}



