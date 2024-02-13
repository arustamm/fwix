

// operator injecting a wavelet or data into the set of wavefields: [Ns, Nw, Nx, Ny]
class Injection : public cuOperator<complex2DReg, complex4DReg>  {
  public:
    Injection() {};
    ~Injection() {};

    void forward(bool add, const std::unique_ptr<complex2DReg> model, std::unique_ptr<complex4DReg> data) {

    }

    void cu_forward(bool add, const std::complex<float>* d_model, std::complex<float>* d_data) {

    }
}
