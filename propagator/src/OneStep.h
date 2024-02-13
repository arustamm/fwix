
  // operator to propagate 2D wavefield ONCE in (x-y) for multiple sources and freqs (ns-nw) 
  class OneStep :  {
    OneStep () {};
    ~OneStep() {};

    void forward (bool add, std::unique_ptr<complex4DReg> model, std::unique_ptr<complex4DReg> data) {
        // 2dFFT (x-y) --> (kx-ky)
        cuFFT.cu_forward(false, )
      
    };


    void adjoint (bool add, std::unique_ptr<complex2DReg> model, std::unique_ptr<complex2DReg> data);
  }