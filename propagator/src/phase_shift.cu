#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <CudaKernel.cuh>
#include <CudaKernel.cu>

template class CudaKernel<float*, float*, float*, cuFloatComplex, float, float>;

__global__ void ps_forward(complex_vector* model, complex_vector* data, float* w2, float* kx, float* ky, cuFloatComplex slow_ref, float dz, float eps) {

  float a, b, c, re, im;
  int flat_ind;
  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix = threadIdx.x + blockDim.x*blockIdx.x;
  int iy = threadIdx.y + blockDim.y*blockIdx.y;
  int iw = threadIdx.z + blockDim.z*blockIdx.z;
  int nd_ind[] = {0, iw, iy, ix};

  float sre = cuCrealf(slow_ref);
  float sim = cuCimagf(slow_ref);

  if (iw < NW && iy < NY && ix < NX) {
    for (int is=0; is < NS; ++is) {
      
      a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
      b = w2[iw]*(sim-eps*sre);
      c = sqrtf(a*a + b*b);
      re = sqrtf((c+a)/2);
      im = -sqrtf((c-a)/2);
      
      nd_ind[0] = is;
      flat_ind = ND_TO_FLAT(nd_ind, dims);
      
      float att = exp(im*dz);
      float coss = cos(re*dz);
      float sinn = sin(re*dz);

      data->mat[flat_ind].x += att * (model->mat[flat_ind].x * coss + model->mat[flat_ind].y * sinn);
      data->mat[flat_ind].y += att * (-model->mat[flat_ind].x * sinn + model->mat[flat_ind].y * coss);
    }
  }
};

__global__ void ps_adjoint(complex_vector* model, complex_vector* data, float* w2, float* kx, float* ky, cuFloatComplex slow_ref, float dz, float eps) {
  
  float a, b, c, re, im;
  int flat_ind;

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix = threadIdx.x + blockDim.x*blockIdx.x;
  int iy = threadIdx.y + blockDim.y*blockIdx.y;
  int iw = threadIdx.z + blockDim.z*blockIdx.z;
  int nd_ind[] = {0, iw, iy, ix};

  float sre = cuCrealf(slow_ref);
  float sim = cuCimagf(slow_ref);

  if (iw < NW && iy < NY && ix < NX) {
    for (int is=0; is < NS; ++is) {
      a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
      b = w2[iw]*(sim-eps*sre);
      c = sqrtf(a*a + b*b);
      re = sqrtf((c+a)/2);
      im = -sqrtf((c-a)/2);

      nd_ind[0] = is;
      flat_ind = ND_TO_FLAT(nd_ind, dims);
      float att = exp(im*dz);
      float coss = cos(re*dz);
      float sinn = sin(re*dz);

      model->mat[flat_ind].x += att * (data->mat[flat_ind].x * coss - data->mat[flat_ind].y * sinn);
      model->mat[flat_ind].y += att * (data->mat[flat_ind].x * sinn + data->mat[flat_ind].y * coss);
    }
  }
};