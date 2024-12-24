#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <cuda_runtime.h>
#include <KernelLauncher.cuh>
#include <cuComplex.h>
#include <prop_kernels.cuh>

using namespace SEP;

class PhaseShift :  public CudaOperator<complex4DReg, complex4DReg> {
public:
    PhaseShift(const std::shared_ptr<hypercube>& domain, float dz, float eps, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid=1, dim3 block=1);

    void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
    void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
    void cu_inverse (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

    void set_slow(std::complex<float>* sref) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(_sref_, sref, _nw_*sizeof(std::complex<float>), cudaMemcpyHostToDevice));
    }

    virtual void set_grid_block(dim3 grid, dim3 block);

    ~PhaseShift() {
        CHECK_CUDA_ERROR(cudaFree(d_w2));
        CHECK_CUDA_ERROR(cudaFree(d_kx));
        CHECK_CUDA_ERROR(cudaFree(d_ky));
        CHECK_CUDA_ERROR(cudaFree(_sref_));
    }

protected:
    PS_launcher launcher;
    PS_launcher launcher_inv;
    cuFloatComplex* _sref_;
    float *d_w2, *d_kx, *d_ky;
    float _dz_;
    float _eps_;
    int _nw_;

    float* fill_in_k(const axis& ax) {
        float *k;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&k, sizeof(float)*ax.n));
        auto h_k = std::vector<float>(ax.n);
	    float dk = 2*M_PI/(ax.d*(ax.n-1));
        for (int ik=0; ik<h_k.size()/2; ik++) 
            h_k[ik] = ik*dk;
        for (int ik=1; ik<h_k.size()/2; ik++) 
            h_k[ax.n-ik] = -h_k[ik];
        CHECK_CUDA_ERROR(cudaMemcpy(k, h_k.data(), sizeof(float)*ax.n, cudaMemcpyHostToDevice));
        return k;
    }

    float* fill_in_w(const axis& ax) {
        float *w;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&w, sizeof(float)*ax.n));
        auto h_w = std::vector<float>(ax.n);
        for (int i=0; i < h_w.size(); ++i) {
            float f = ax.o + i*ax.d;
            f = 2*M_PI*f;
            h_w[i] = f*f;
        }
        CHECK_CUDA_ERROR(cudaMemcpy(w, h_w.data(), sizeof(float)*ax.n, cudaMemcpyHostToDevice));
        return w;
    }

};
