

# compiling cpp into objects
set(CPP_SRC prop_unit_test.cpp)
set(CPP_INC)
add_executable(prop_unit_test ${CPP_SRC} ${CPP_INC})
set_property(TARGET prop_unit_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(prop_unit_test
		CudaOperator CudaWEM
		genericCpp sepVector hypercube
		CUDA::cudart_static CUDA::cufft_static
		GTest::gtest_main GTest::gtest)

add_test(cuWEM_gtests prop_unit_test)
