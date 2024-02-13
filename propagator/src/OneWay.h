
  // operator to propagate 2D wavefield in (x-y) from top down to the bottom of the model for multiple sources and freqs (ns-nw)
  class OneWay {
    OneWay () {};
    ~OneWay() {};

    void forward (bool add, std::unique_ptr<complex4DReg> model, std::unique_ptr<complex4DReg> data) {
      for (int iz = 0; iz < NZ; ++iz) {
        inject->set_depth(iz);
        inject->cu_forward(false, )
        // 2dFFT (x-y) --> (kx-ky)
        onestep->cu_forward()
      } 
      
    };


    void adjoint (bool add, std::unique_ptr<complex2DReg> model, std::unique_ptr<complex2DReg> data);
  }