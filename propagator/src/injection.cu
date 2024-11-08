#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

template class KernelLauncher<float*, float*, float*, int*>;

__global__ void inj_forward(const complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  float* cx, float* cy, float* cz, int* ids) {

  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int NS = data->n[3];
  int NZ = data->n[4];
  
  int mNW = model->n[0];
  int NTRACE = model->n[1];
  int mdims[] = {NTRACE,mNW};

  float OX = data->o[0];
  float OY = data->o[1];
  float OZ = data->o[4];
  float DX = data->d[0];
  float DY = data->d[1];
  float DZ = data->d[4];
  int dims[] = {NZ, NS, NW, NY, NX};

  int iw0 = threadIdx.x + blockDim.x*blockIdx.x;
  int itrace0 = threadIdx.y + blockDim.y*blockIdx.y;

  int jw = blockDim.x * gridDim.x;
  int jtrace = blockDim.y * gridDim.y;

  cuFloatComplex w[8]; 

  for (int itrace=itrace0; itrace < NTRACE; itrace += jtrace) {
    int iy = (cy[itrace]-OY)/DY;
    float y = OY + iy*DY;
    float ly = 1.f - (cy[itrace] - y) / DY;

    int ix = (cx[itrace]-OX)/DX;
    float x = OX + ix*DX;
    float lx = 1.f - (cx[itrace] - x) / DX;

    int iz = (cz[itrace]-OZ)/DZ;
    float z = OZ + iz*DZ;
    float lz = 1.f - (cz[itrace] - z) / DZ;

    int id = ids[itrace];

    w[0] = make_cuFloatComplex(lz * lx * ly, 0.0f);
    w[1] = make_cuFloatComplex(lz * (1 - lx) * ly, 0.0f);
    w[2] = make_cuFloatComplex(lz * lx * (1 - ly), 0.0f);
    w[3] = make_cuFloatComplex(lz * (1 - lx) * (1 - ly), 0.0f);
    w[4] = make_cuFloatComplex((1 - lz) * lx * ly, 0.0f);
    w[5] = make_cuFloatComplex((1 - lz) * (1 - lx) * ly, 0.0f);
    w[6] = make_cuFloatComplex((1 - lz) * lx * (1 - ly), 0.0f);
    w[7] = make_cuFloatComplex((1 - lz) * (1 - lx) * (1 - ly), 0.0f);
    
    for (int iw=iw0; iw < mNW; iw += jw) {
      int idx_ind[] = {itrace, iw};
      int idx_cx0cy0cz0[] = {iz, id, iw, iy, ix};
      int idx_cx1cy0cz0[] = {iz, id, iw, iy, ix + 1};
      int idx_cx0cy1cz0[] = {iz, id, iw, iy + 1, ix};
      int idx_cx1cy1cz0[] = {iz, id, iw, iy + 1, ix + 1};
      int idx_cx0cy0cz1[] = {iz + 1, id, iw, iy, ix};
      int idx_cx1cy0cz1[] = {iz + 1, id, iw, iy, ix + 1};
      int idx_cx0cy1cz1[] = {iz + 1, id, iw, iy + 1, ix};
      int idx_cx1cy1cz1[] = {iz + 1, id, iw, iy + 1, ix + 1};

      // convert 4d index to flat index
      int ind = ND_TO_FLAT(idx_ind, mdims);
      int cx0cy0cz0 = ND_TO_FLAT(idx_cx0cy0cz0, dims);
      int cx1cy0cz0 = ND_TO_FLAT(idx_cx1cy0cz0, dims);
      int cx0cy1cz0 = ND_TO_FLAT(idx_cx0cy1cz0, dims);
      int cx1cy1cz0 = ND_TO_FLAT(idx_cx1cy1cz0, dims);

      int cx0cy0cz1 = ND_TO_FLAT(idx_cx0cy0cz1, dims);
      int cx1cy0cz1 = ND_TO_FLAT(idx_cx1cy0cz1, dims);
      int cx0cy1cz1 = ND_TO_FLAT(idx_cx0cy1cz1, dims);
      int cx1cy1cz1 = ND_TO_FLAT(idx_cx1cy1cz1, dims);

      cuFloatComplex val = model->mat[ind];

      data->mat[cx0cy0cz0] = cuCaddf(data->mat[cx0cy0cz0],cuCmulf(w[0],val)); 
      data->mat[cx1cy0cz0] = cuCaddf(data->mat[cx1cy0cz0],cuCmulf(w[1],val)); 
      data->mat[cx0cy1cz0] = cuCaddf(data->mat[cx0cy1cz0],cuCmulf(w[2],val)); 
      data->mat[cx1cy1cz0] = cuCaddf(data->mat[cx1cy1cz0],cuCmulf(w[3],val)); 

      data->mat[cx0cy0cz1] = cuCaddf(data->mat[cx0cy0cz1],cuCmulf(w[4],val)); 
      data->mat[cx1cy0cz1] = cuCaddf(data->mat[cx1cy0cz1],cuCmulf(w[5],val)); 
      data->mat[cx0cy1cz1] = cuCaddf(data->mat[cx0cy1cz1],cuCmulf(w[6],val)); 
      data->mat[cx1cy1cz1] = cuCaddf(data->mat[cx1cy1cz1],cuCmulf(w[7],val));

    }
  }
};

__global__ void inj_adjoint(complex_vector* __restrict__ model, const complex_vector* __restrict__ data, 
  float* cx, float* cy, float* cz, int* ids) {

  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int NS = data->n[3];
  int NZ = data->n[4];
  
  int mNW = model->n[0];
  int NTRACE = model->n[1];
  int mdims[] = {NTRACE,mNW};

  float OX = data->o[0];
  float OY = data->o[1];
  float OZ = data->o[4];
  float DX = data->d[0];
  float DY = data->d[1];
  float DZ = data->d[4];
  int dims[] = {NZ, NS, NW, NY, NX};

  int iw0 = threadIdx.x + blockDim.x*blockIdx.x;
  int itrace0 = threadIdx.y + blockDim.y*blockIdx.y;

  int jw = blockDim.x * gridDim.x;
  int jtrace = blockDim.y * gridDim.y;
  
  cuFloatComplex w[8];

  for (int itrace=itrace0; itrace < NTRACE; itrace += jtrace) {
    int iy = (cy[itrace]-OY)/DY;
    float y = OY + iy*DY;
    float ly = 1.f - (cy[itrace] - y) / DY;

    int ix = (cx[itrace]-OX)/DX;
    float x = OX + ix*DX;
    float lx = 1.f - (cx[itrace] - x) / DX;

    int iz = (cz[itrace]-OZ)/DZ;
    float z = OZ + iz*DZ;
    float lz = 1.f - (cz[itrace] - z) / DZ;

    int id = ids[itrace];

    w[0] = make_cuFloatComplex(lz * lx * ly, 0.0f);
    w[1] = make_cuFloatComplex(lz * (1 - lx) * ly, 0.0f);
    w[2] = make_cuFloatComplex(lz * lx * (1 - ly), 0.0f);
    w[3] = make_cuFloatComplex(lz * (1 - lx) * (1 - ly), 0.0f);
    w[4] = make_cuFloatComplex((1 - lz) * lx * ly, 0.0f);
    w[5] = make_cuFloatComplex((1 - lz) * (1 - lx) * ly, 0.0f);
    w[6] = make_cuFloatComplex((1 - lz) * lx * (1 - ly), 0.0f);
    w[7] = make_cuFloatComplex((1 - lz) * (1 - lx) * (1 - ly), 0.0f);
    
    for (int iw=iw0; iw < mNW; iw += jw) {

      int idx_ind[] = {itrace, iw};
      int idx_cx0cy0cz0[] = {iz, id, iw, iy, ix};
      int idx_cx1cy0cz0[] = {iz, id, iw, iy, ix + 1};
      int idx_cx0cy1cz0[] = {iz, id, iw, iy + 1, ix};
      int idx_cx1cy1cz0[] = {iz, id, iw, iy + 1, ix + 1};
      int idx_cx0cy0cz1[] = {iz + 1, id, iw, iy, ix};
      int idx_cx1cy0cz1[] = {iz + 1, id, iw, iy, ix + 1};
      int idx_cx0cy1cz1[] = {iz + 1, id, iw, iy + 1, ix};
      int idx_cx1cy1cz1[] = {iz + 1, id, iw, iy + 1, ix + 1};
      
      // convert 4d index to flat index
      int ind = ND_TO_FLAT(idx_ind, mdims);
      int cx0cy0cz0 = ND_TO_FLAT(idx_cx0cy0cz0, dims);
      int cx1cy0cz0 = ND_TO_FLAT(idx_cx1cy0cz0, dims);
      int cx0cy1cz0 = ND_TO_FLAT(idx_cx0cy1cz0, dims);
      int cx1cy1cz0 = ND_TO_FLAT(idx_cx1cy1cz0, dims);

      int cx0cy0cz1 = ND_TO_FLAT(idx_cx0cy0cz1, dims);
      int cx1cy0cz1 = ND_TO_FLAT(idx_cx1cy0cz1, dims);
      int cx0cy1cz1 = ND_TO_FLAT(idx_cx0cy1cz1, dims);
      int cx1cy1cz1 = ND_TO_FLAT(idx_cx1cy1cz1, dims);

      cuFloatComplex val = cuCmulf(data->mat[cx0cy0cz0],w[0]);
      val = cuCaddf(val, cuCmulf(data->mat[cx1cy0cz0],w[1])); 
      val = cuCaddf(val, cuCmulf(data->mat[cx0cy1cz0],w[2])); 
      val = cuCaddf(val, cuCmulf(data->mat[cx1cy1cz0],w[3])); 
      val = cuCaddf(val, cuCmulf(data->mat[cx0cy0cz1],w[4]));
      val = cuCaddf(val, cuCmulf(data->mat[cx1cy0cz1],w[5])); 
      val = cuCaddf(val, cuCmulf(data->mat[cx0cy1cz1],w[6]));
      val = cuCaddf(val, cuCmulf(data->mat[cx1cy1cz1],w[7]));  

      model->mat[ind] = cuCaddf(model->mat[ind], val);

    }
  }
};