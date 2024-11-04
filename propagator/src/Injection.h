

// operator injecting a wavelet or data into the set of wavefields: [Ns, Nw, Nx, Ny]
class Injection : public cuOperator<complex2DReg, complex4DReg>  {
public:
  Injection(const std::shared_ptr<hypercube>& domain,const std::shared_ptr<hypercube>& range, 
  std::map<int, std::vector<std::vector<float>> > coord_map,
  complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid=1, dim3 block=1);
  
  ~Injection() {
    CHECK_CUDA_ERROR(cudaFree(d_cx));
    CHECK_CUDA_ERROR(cudaFree(d_cy));
  };

  void set_depth (int& iz) {
    if (map.count() > 0) {
      inject = true;
      cx = map[iz][0];
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_cx, cx.data(), sizeof(float)*_size_, cudaMemcpyHostToDevice));
      cy = map[iz][1];
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_cy, cy.data(), sizeof(float)*_size_, cudaMemcpyHostToDevice));
    } 
    else {
      inject = false;
    }
  };

  void cu_forward (bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data);

private:
  Injection_launcher launcher;
  float *d_cx, *d_cy;
  bool inject;
  std::vector<float> cx, cy;
}
