
  // operator to propagate 2D wavefield in (x-y) from top down to the bottom of the model for multiple sources and freqs (ns-nw)
  class OneWay {
    OneWay () {};
    ~OneWay() {};

    void cu_forward (bool add, ComplexVectorMap& model, ComplexVectorMap& data) {
      for (int iz = 0; iz < NZ; ++iz) {
        inject->set_depth(iz);
        inject->cu_forward(false, ..., _wfld_prev);
        // 2dFFT (x-y) --> (kx-ky)
        onestep->set_depth(iz);
        onestep->cu_forward(true, _wfld_prev, _wfld_next);
      } 
      
    };


    void adjoint (bool add, std::unique_ptr<complex2DReg> model, std::unique_ptr<complex2DReg> data);
  }