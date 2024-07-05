#include <RefSampler.h>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <utility>
#include "opencv2/core.hpp"
#include <tbb/tbb.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

using namespace SEP;
using namespace std::placeholders;

RefSampler::RefSampler(const std::shared_ptr<complex4DReg>& slow, int nref) : _slow_(slow) {
			_nref_ = nref;
			_nx_ = _slow_->getHyper()->getAxis(1).n;
			_ny_ = _slow_->getHyper()->getAxis(2).n;
			_nw_ = _slow_->getHyper()->getAxis(3).n;
			_nz_ = _slow_->getHyper()->getAxis(4).n;

			ref_labels.resize(boost::extents[_nz_][_nw_][_ny_][_nx_]);
			slow_ref.resize(boost::extents[_nz_][_nref_][_nw_]);
			kmeans_sample();
		};


void RefSampler::kmeans_sample() {
	tbb::parallel_for(tbb::blocked_range2d<int>(0,_nw_,0,_nz_),
		[=](const tbb::blocked_range2d<int> &r) {
		for (int iz=r.cols().begin(); iz < r.cols().end(); iz++) {
			for (int iw=r.rows().begin(); iw < r.rows().end(); iw++) {
				int offset = (iw + iz*_nw_)*_nx_*_ny_;
				std::complex<float>* ptr_slow_ref = _slow_->getVals() + offset;
				int* ptr_labels = ref_labels.data() + offset; 
				// prepare opencv matrices for processing
				cv::Mat_<std::complex<float>> slow_slice(_nx_*_ny_, 1, ptr_slow_ref);
				cv::Mat_<int> labels(_nx_*_ny_, 1, ptr_labels);
				cv::Mat_<std::complex<float>> centers(_nref_, 1);
				// stopping criteria
				cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 1e-12);
				// compute centers & labels
				double obj = cv::kmeans(slow_slice, _nref_, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
				// copy to slow_ref array
				for (int iref=0; iref < _nref_; ++iref) {
					std::complex<float> sref = centers.at<std::complex<float>>(iref, 0);
					slow_ref[iz][iref][iw] = sref;
				}
			}
		}
	});
}

