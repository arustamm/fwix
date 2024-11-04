#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

template class KernelLauncher<float*, float*>;

__global__ void inj_forward(const complex_vector* __restrict__ model, complex_vector* __restrict__ data, float* cx, float* cy) {

  float a, b, c, re, im;
  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int NS = data->n[3];
  
  int mNW = model->n[0];
  int NTRACE = model->n[1];

  float OX = data->o[0];
  float OY = data->o[1];
  float DX = data->d[0];
  float DY = data->d[1];
  int dims[] = {NS, NW, NY, NX};

  int iw0 = threadIdx.x + blockDim.x*blockIdx.x;
  int itrace0 = threadIdx.y + blockDim.y*blockIdx.y;

  int jw = blockDim.x * gridDim.x;
  int js = blockDim.y * gridDim.y;

  for (int itrace=itrace0; itrace < NTRACE; itrace += jtrace) {
    int iy = (cy[itrace]-OY)/DY;
    float y = OY + iy*DY;
    float ly = (cy[itrace] - y) / cy[itrace];

    int ix = (cx[itrace]-OX)/DX;
    float x = OX + ix*DX;
    float lx = (cx[itrace] - x) / cx[itrace];
    
    for (int iw=iw0; iw < mNW; iw += jw) {
      // convert 4d index to flat index
      int ind = ND_TO_FLAT({ids[itrace], iw}, {mNS,mNW});
      int cx0cy0 = ND_TO_FLAT({ids[itrace], iw, iy, ix}, dims);
      int cx1cy0 = ND_TO_FLAT({ids[itrace], iw, iy, ix+1}, dims);
      int cx0cy1 = ND_TO_FLAT({ids[itrace], iw, iy+1, ix}, dims);
      int cx1cy1 = ND_TO_FLAT({ids[itrace], iw, iy+1, ix+1}, dims);

      cuFloatComplex val = model->mat[ind];

      data->mat[cx0cy0] = cuCaddf(data->mat[cx0cy0], lx*ly*val); 
      data->mat[cx1cy0] = cuCaddf(data->mat[cx1cy0], (1-lx)*ly*val); 
      data->mat[cx0cy1] = cuCaddf(data->mat[cx0cy1], lx*(1-ly)*val); 
      data->mat[cx1cy1] = cuCaddf(data->mat[cx1cy1], (1-lx)*(1-ly)*val); 

    }
  }
};

__global__ void inj_adjoint(complex_vector* __restrict__ model, const complex_vector* __restrict__ data, float* cx, float* cy) {

  float a, b, c, re, im;
  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int NS = data->n[3];

  int mNW = model->n[0];
  int mNS = model->n[1];

  float OX = data->o[0];
  float OY = data->o[1];
  float DX = data->d[0];
  float DY = data->d[1];
  int dims[] = {NS, NW, NY, NX};

  int iw0 = threadIdx.x + blockDim.x*blockIdx.x;
  int is0 = threadIdx.y + blockDim.y*blockIdx.y;

  int jw = blockDim.x * gridDim.x;
  int js = blockDim.y * gridDim.y;

  for (int is=is0; is < mNS; is += js) {
    int iy = (cy[is]-OY)/DY;
    float y = OY + iy*DY;
    float ly = (cy[is] - y) / cy[is];

    int ix = (cx[is]-OX)/DX;
    float x = OX + ix*DX;
    float lx = (cx[is] - x) / cx[is];
    
    for (int iw=iw0; iw < mNW; iw += jw) {
      // convert 4d index to flat index
      int ind = ND_TO_FLAT({is, iw}, {mNS,mNW});
      int cx0cy0 = ND_TO_FLAT({is, iw, iy, ix}, dims);
      int cx1cy0 = ND_TO_FLAT({is, iw, iy, ix+1}, dims);
      int cx0cy1 = ND_TO_FLAT({is, iw, iy+1, ix}, dims);
      int cx1cy1 = ND_TO_FLAT({is, iw, iy+1, ix+1}, dims);

      cuFloatComplex val = data->mat[cx0cy0]*lx*ly + data->mat[cx1cy0]*(1-lx)*ly + 
                          data->mat[cx0cy1]*lx*(1-ly) + data->mat[cx1cy1]*(1-lx)*(1-ly); 

      model->mat[ind] = cuCaddf(model->mat[ind], val);

    }
  }
};