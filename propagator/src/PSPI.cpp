
#include <OneStep.h>

using namespace SEP;

void PSPI::cu_forward(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {

	  if(!add) (*data)["host"]->zero();

		// pad->forward(model,model_pad,0);
	  fft2d->cu_forward(0,model,model_k);

		for (int iref=0; iref < _nref_; ++iref) {

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_forward(0, model_k, _wfld_ref);

			fft2d->cu_adjoint(_wfld_ref);
			// // taper->forward(_wfld_ref,_wfld_ref,1);
			
			select->set_value(iref);
			select->cu_forward(1, _wfld_ref,data);
		}

}

void PSPI::cu_adjoint(bool add, std::shared_ptr<ComplexVectorMap> model, std::shared_ptr<ComplexVectorMap> data) {

		if(!add)  (*model)["host"]->zero();
		(*model_k)["host"]->zero();

		for (int iref=0; iref < _nref_; ++iref) {

			select->set_value(iref);
			select->cu_adjoint(0, _wfld_ref,data);

			fft2d->cu_forward(_wfld_ref);

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_adjoint(1, model_k, _wfld_ref);
		}

		fft2d->cu_adjoint(1, model, model_k);

}
