#pragma once
#include <float1DReg.h>
#include <float2DReg.h>
#include "complex2DReg.h"
#include "complex3DReg.h"
#include "complex1DReg.h"
#include "paramObj.h"
#include "boost/multi_array.hpp"
#include  "opencv2/core.hpp"
#include <future>

namespace SEP {


class RefSampler

	{
	public:

		RefSampler(std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par);
		RefSampler(const std::shared_ptr<complex4DReg>& slow, std::shared_ptr<paramObj> par);

		inline std::complex<float>* get_ref_slow(int iz, int iref) {return slow_ref.data() + (iref + iz*_nref_)*_nw_;}
		inline int* get_ref_labels(int iz) { return ref_labels.data() + iz*_nw_*(_ny_+pady)*(_nx_+padx);}

		void sample_at_depth(std::shared_ptr<complex4DReg> slow, int iz);
		std::future<void> sample_at_depth_async(std::shared_ptr<complex4DReg> slow, int iz);

		int _nx_, _ny_, _nref_, _nz_, _nw_, padx, pady;
		
	private:
		
		void kmeans_sample(const std::shared_ptr<complex4DReg>& slow);
		
		boost::multi_array<int, 4> ref_labels;
		boost::multi_array<std::complex<float>, 3> slow_ref;

		

	};
}
