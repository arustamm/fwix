import pyCudaWEM
import pyOperator as Op
import genericIO
import numpy as np
from pyVector import superVector


class Propagator(Op.Operator):
	def __init__(self,model, data, slowness, wavelet, par):
		self.setDomainRange(model,data)
		geometry = par.pars["geometry"]
		self.cppMode = pyCudaWEM.Propagator(
			model.getHyper().cppMode, data.getHyper().cppMode, 
			slowness.getHyper().cppMode, wavelet.cppMode,
			geometry["sx"], geometry["sy"], geometry["sz"], geometry["s_ids"],
			geometry["rx"], geometry["ry"], geometry["rz"], geometry["r_ids"],
			par.cppMode
		)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)


class PhaseShift(Op.Operator):
	def __init__(self,model,data, dz, eps=0):
		self.setDomainRange(model,data)
		self.cppMode = pyCudaWEM.PhaseShift(model.getHyper().cppMode, dz, eps)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)

	def set_slow(self,slow):
		self.cppMode.set_slow(slow)


class RefSampler:
	def __init__(self, slow, nref):
		self.cppMode = pyCudaWEM.RefSampler(slow.cppMode, nref)

	def get_ref_slow(self, iz, iref):
		return self.cppMode.get_ref_slow(iz,iref)
	
	def get_ref_labels(self, iz):
		return self.cppMode.get_ref_labels(iz)
	

class PSPI(Op.Operator):
	def __init__(self, model, data, slow, par):
		self.cppMode = pyCudaWEM.PSPI(model.getHyper().cppMode, slow.cppMode, par.cppMode)
		self.setDomainRange(model, data)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)

	def set_depth(self, iz):
		self.cppMode.set_depth(iz)


class NSPS(Op.Operator):
	def __init__(self, model, data, slow, par):
		self.cppMode = pyCudaWEM.NSPS(model.getHyper().cppMode, slow.cppMode, par.cppMode)
		self.setDomainRange(model, data)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)

	def set_depth(self, iz):
		self.cppMode.set_depth(iz)

class Injection(Op.Operator):
	def __init__(self, model, data, oz, dz, cx, cy, cz, ids):
		self.cppMode = pyCudaWEM.Injection(model.getHyper().cppMode, data.getHyper().cppMode, oz, dz, cx, cy, cz, ids)
		self.setDomainRange(model, data)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)

	def set_coords(self, cx, cy, cz, ids):
		self.cppMode.set_coords(cx, cy, cz, ids)

	def set_depth(self, iz):
		self.cppMode.set_depth(iz)


class Downward(Op.Operator):
	def __init__(self, model, data, slow, par):
		self.cppMode = pyCudaWEM.Downward(model.getHyper().cppMode, slow.cppMode, par.cppMode)
		self.setDomainRange(model, data)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)

	def forward(self,data):
		self.cppMode.forward(data.cppMode)

	def adjoint(self,model):
		self.cppMode.adjoint(model.cppMode)

	def set_depth(self, iz):
		self.cppMode.set_depth(iz)

class Upward(Op.Operator):
	def __init__(self, model, data, slow, par):
		self.cppMode = pyCudaWEM.Upward(model.getHyper().cppMode, slow.cppMode, par.cppMode)
		self.setDomainRange(model, data)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)

	def set_depth(self, iz):
		self.cppMode.set_depth(iz)

