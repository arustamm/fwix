#pragma once
#include <float1DReg.h>
#include <float2DReg.h>
#include "complex2DReg.h"
#include "complex3DReg.h"
#include "complex1DReg.h"
#include "boost/multi_array.hpp"
#include  "opencv2/core.hpp"

namespace SEP {


class RefSampler

	{
	public:

		RefSampler(const std::shared_ptr<complex4DReg>& slow, int nref);

		inline std::complex<float>* get_ref_slow(int iz, int iref) {return slow_ref.data() + (iref + iz*_nref_)*_nw_;}
		inline int* get_ref_labels(int iz) { return ref_labels.data() + iz*_nw_*_ny_*_nx_;}

		int _nx_, _ny_, _nref_, _nz_, _nw_;

	private:

		void kmeans_sample();

		const std::shared_ptr<complex4DReg>& _slow_;
		boost::multi_array<int, 4> ref_labels;
		boost::multi_array<std::complex<float>, 3> slow_ref;

		

	};
}
