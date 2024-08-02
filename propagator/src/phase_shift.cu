#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <CudaKernel.cuh>
#include <CudaKernel.cu>

template class CudaKernel<float*, float*, float*, cuFloatComplex*, float, float>;

__global__ void ps_forward(const complex_vector* __restrict__ model, complex_vector* __restrict__ data, float* w2, float* kx, float* ky, cuFloatComplex* slow_ref, float dz, float eps) {

  float a, b, c, re, im;
  int flat_ind;
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
      float sre = cuCrealf(slow_ref[iw]);
      float sim = cuCimagf(slow_ref[iw]);
      for (int iy=iy0; iy < NY; iy += jy) {
        for (int ix=ix0; ix < NX; ix += jx) {
          a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
          b = w2[iw]*(sim-eps*sre);
          c = sqrtf(a*a + b*b);
          re = sqrtf((c+a)/2);
          im = -sqrtf((c-a)/2);
          // convert 4d index to flat index
          int nd_ind[] = {is, iw, iy, ix};
          flat_ind = ND_TO_FLAT(nd_ind, dims);

          float att = exp(im*dz);
          float coss = cos(re*dz);
          float sinn = sin(re*dz);

          float mre = cuCrealf(model->mat[flat_ind]);
          float mim = cuCimagf(model->mat[flat_ind]);

          re = att * (mre * coss + mim * sinn);
          im = att * (-mre * sinn + mim * coss);

          data->mat[flat_ind] = cuCaddf(data->mat[flat_ind], make_cuFloatComplex(re, im)); 
        }
      }
    }
  }
};

__global__ void ps_adjoint(complex_vector* __restrict__ model, const complex_vector* __restrict__ data, float* w2, float* kx, float* ky, cuFloatComplex* slow_ref, float dz, float eps) {
  
  float a, b, c, re, im;
  int flat_ind;

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
      float sre = cuCrealf(slow_ref[iw]);
      float sim = cuCimagf(slow_ref[iw]);
      for (int iy=iy0; iy < NY; iy += jy) {
        for (int ix=ix0; ix < NX; ix += jx) {
          a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
          b = w2[iw]*(sim-eps*sre);
          c = sqrtf(a*a + b*b);
          re = sqrtf((c+a)/2);
          im = -sqrtf((c-a)/2);

          // convert 4d index to flat index
          int nd_ind[] = {is, iw, iy, ix};
          flat_ind = ND_TO_FLAT(nd_ind, dims);

          float att = exp(im*dz);
          float coss = cos(re*dz);
          float sinn = sin(re*dz);

          float dre = cuCrealf(data->mat[flat_ind]);
          float dim = cuCimagf(data->mat[flat_ind]);

          re = att * (dre * coss - dim * sinn);
          im = att * (dre * sinn + dim * coss);

          model->mat[flat_ind] = cuCaddf(model->mat[flat_ind], make_cuFloatComplex(re, im));
        }
      }
    }
  }
};