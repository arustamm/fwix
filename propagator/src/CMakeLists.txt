

# compiling CUDA files into objects
set(CU_SRC phase_shift.cu)
set(CU_INC prop_kernels.cuh)
# add_library(cuda_objects OBJECT ${CU_SRC} ${CU_INC})
# set_property(TARGET cuda_objects PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# compiling cpp into objects
set(CPP_SRC PhaseShift.cpp)
set(CPP_INC PhaseShift.h)
# add_library(cpp_objects OBJECT ${CPP_SRC} ${CPP_INC})
# set_property(TARGET cpp_objects PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_library(CudaWEM STATIC ${CU_SRC} ${CU_INC} ${CPP_SRC} ${CPP_INC})
set_property(TARGET CudaWEM PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CudaWEM
						CudaOperator
						genericCpp sepVector hypercube
						jsonCpp sep3d sep
					  CUDA::cudart_static CUDA::cufft_static
						)

install(TARGETS CudaWEM DESTINATION lib)
