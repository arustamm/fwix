import pyCudaWEM
import pyOperator as Op
import genericIO
import numpy as np
from pyVector import superVector

class StreamingPropagator(Op.Operator):
	# model here refers to the slowness model
	# data refers to the recorded traces
	def __init__(self, model, data, wavelet, par, geometry, nbatches):
		self.setDomainRange(model,data)
		# cpp code needs the hypercube corresponding to the injected source traces
		self.cppMode = pyCudaWEM.StreamingPropagator(
			wavelet.getHyper().cppMode, data.getHyper().cppMode, 
			model.vecs[0].getHyper().cppMode, wavelet.cppMode,
			geometry["sx"], geometry["sy"], geometry["sz"], geometry["s_ids"],
			geometry["rx"], geometry["ry"], geometry["rz"], geometry["r_ids"],
			par.cppMode, nbatches
		)

	def forward(self,add,model,data):
		mod = [m.cppMode for m in model]
		self.cppMode.forward(add, mod, data.cppMode)

class Propagator(Op.Operator):
	# model here refers to the slowness model
	# data refers to the recorded traces
	def __init__(self, model, data, wavelet, par, geometry):
		self.setDomainRange(model,data)
		# cpp code needs the hypercube corresponding to the injected source traces
		self.cppMode = pyCudaWEM.Propagator(
			wavelet.getHyper().cppMode, data.getHyper().cppMode, 
			model.vecs[0].getHyper().cppMode, wavelet.cppMode,
			geometry["sx"], geometry["sy"], geometry["sz"], geometry["s_ids"],
			geometry["rx"], geometry["ry"], geometry["rz"], geometry["r_ids"],
			par.cppMode
		)

	def forward(self,add,model,data):
		mod = [m.cppMode for m in model]
		self.cppMode.forward(add, mod, data.cppMode)

	def get_compression_ratio(self):
		"""
		Returns the compression ratio of the propagator.
		This is the ratio of the number of input samples to the number of output samples.
		"""
		return self.cppMode.get_compression_ratio()
	
class ExtendedBorn(Op.Operator):
	# model here refers to the slowness model
	# data refers to the recorded traces
	def __init__(self, model, data, slow_den, propagator):
		self.setDomainRange(model,data)
		mod = [m.cppMode for m in slow_den]
		# cpp code needs the hypercube corresponding to the injected source traces
		self.cppMode = pyCudaWEM.ExtendedBorn(
			model[0].getHyper().cppMode, data.getHyper().cppMode, 
			mod, propagator.cppMode
		)

	def forward(self,add,model,data):
		mod = [m.cppMode for m in model]
		self.cppMode.forward(add, mod, data.cppMode)


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
	def __init__(self, slow, par):
		self.cppMode = pyCudaWEM.RefSampler(slow.cppMode, par.cppMode)

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

