
#include <OneWay.h>

using namespace SEP;

void Downward::cu_forward(bool add, const complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	for (int iz=0; iz < NZ-1; ++iz) {
		// slice through 5d wfld to get 4d wfld
		data->view_at(curr, iz);
		data->view_at(next, iz+1);
		// propagate one step in-place
		prop->cu_forward(true, curr, next);
	}

}

void Downward::cu_adjoint(bool add, complex_vector* __restrict__ model, const complex_vector* __restrict__ data) {

		if(!add)  model->zero();
		model_k->zero();

		for (int iref=0; iref < _nref_; ++iref) {

			select->set_value(iref);
			select->cu_adjoint(0, _wfld_ref,data);

			fft2d->cu_forward(_wfld_ref);

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_adjoint(1, model_k, _wfld_ref);
		}

		fft2d->cu_adjoint(1, model, model_k);

}
