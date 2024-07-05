import pyCudaOperator
import pyOperator as Op
import genericIO
import numpy as np
from pyVector import superVector


class cuFFT2d(Op.Operator):
	def __init__(self,model,data):
		self.setDomainRange(model,data)
		self.cppMode = pyCudaOperator.cuFFT2d(model.getHyper().cppMode)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)


	

