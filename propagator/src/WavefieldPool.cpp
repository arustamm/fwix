#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <WavefieldPool.h>
#include <ioModes.h>
#include "zfp.h"
#include <queue>

void WavefieldPool::initialize(
  std::shared_ptr<hypercube> wfld_hyper, 
  std::shared_ptr<paramObj> par)  {
  
  _slice_size_bytes = wfld_hyper->getN123() * sizeof(std::complex<float>);

  int nwflds_to_store = par->getInt("wflds_to_store", 3);
  error_bound = par->getFloat("compress_error", 1E-6);

  // Create shared pools
  wfld_pool.resize(nwflds_to_store);
  events_pool.resize(nwflds_to_store);
  zfp_stream_pool.resize(nwflds_to_store);
  zfp_field_pool.resize(nwflds_to_store);
  auto ax = wfld_hyper->getAxes();
  
  // Initialize shared resources...
  for (int i = 0; i < nwflds_to_store; ++i) {
    wfld_pool[i] = std::make_shared<complex4DReg>(wfld_hyper);
    CHECK_CUDA_ERROR(cudaHostRegister(wfld_pool[i]->getVals(), 
                  wfld_hyper->getN123()*sizeof(std::complex<float>), 
                  cudaHostRegisterDefault));
    CHECK_CUDA_ERROR(cudaEventCreate(&events_pool[i]));
    // Initialize ZFP resources...
    zfp_stream_pool[i] = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp_stream_pool[i], error_bound);
    // Note: The data pointer is null, it will be set just-in-time
    zfp_field_pool[i] = zfp_field_4d(nullptr, zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);
  }
	// Initialize dedicated decompression resources
	decompress_wfld = std::make_shared<complex4DReg>(wfld_hyper);
	CHECK_CUDA_ERROR(cudaHostRegister(decompress_wfld->getVals(), 
                  wfld_hyper->getN123()*sizeof(std::complex<float>), 
                  cudaHostRegisterDefault));
	decompress_stream = zfp_stream_open(NULL);
	zfp_stream_set_accuracy(decompress_stream, error_bound);
	// Associate the decompression field with the dedicated wavefield
	decompress_field = zfp_field_4d(decompress_wfld->getVals(), zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);
}

void WavefieldPool::compress_slice_async(int iz, complex_vector* __restrict__ wfld, cudaStream_t stream, std::vector<std::vector<char>>& storage) {
    int pool_idx = iz % wfld_pool.size();
    auto event = events_pool[pool_idx];
    auto wfld_buffer = wfld_pool[pool_idx];
    
    // Copy GPU data to host buffer
    CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld_buffer->getVals(), wfld->mat, 
                     _slice_size_bytes, cudaMemcpyDeviceToHost, stream));
    
    // Record event after copy
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream));
    
    // Launch async compression
    auto future = compress_slice_impl(iz, pool_idx, event, storage);
    

        // std::lock_guard<std::mutex> lock(compression_mutex);
        compression_futures.push(std::move(future));
    // }
    
}

std::future<void> WavefieldPool::compress_slice_impl(int iz, int pool_idx, cudaEvent_t event, std::vector<std::vector<char>>& storage) {
    return std::async(std::launch::async, [this, iz, pool_idx, event, &storage]() {
        auto wfld_buffer = wfld_pool[pool_idx];
        auto zfp_s = zfp_stream_pool[pool_idx];
        auto zfp_f = zfp_field_pool[pool_idx];

        // Wait for GPU->CPU copy to complete
        CHECK_CUDA_ERROR(cudaEventSynchronize(event));

        // Set up ZFP field
        zfp_field_set_pointer(zfp_f, reinterpret_cast<float*>(wfld_buffer->getVals()));

        // Compress
        size_t max_size = zfp_stream_maximum_size(zfp_s, zfp_f);
        std::vector<char> compressed_data(max_size);
        
        bitstream* stream = stream_open(compressed_data.data(), compressed_data.size());
        zfp_stream_set_bit_stream(zfp_s, stream);
        zfp_stream_rewind(zfp_s);

        size_t actual_size = zfp_compress(zfp_s, zfp_f);
        stream_close(stream);

        if (actual_size == 0) {
            throw std::runtime_error("Compression failed for slice " + std::to_string(iz));
        }

        compressed_data.resize(actual_size);
        
        // Thread-safe storage
        // {
            // std::lock_guard<std::mutex> lock(compression_mutex);
        storage[iz] = std::move(compressed_data);
        // }
    });
}

void WavefieldPool::decompress_slice(int iz, std::vector<char>& compressed_data) {
    
    if (compressed_data.empty()) {
        throw std::runtime_error("Error: Compressed data for slice " + std::to_string(iz) + " is not available.");
    }

    // Use a dedicated pool index for decompression or make it thread-safe
    // static std::mutex decompress_mutex;
    // std::lock_guard<std::mutex> lock(decompress_mutex);

    bitstream* stream = stream_open((void*)compressed_data.data(), compressed_data.size());
    zfp_stream_set_bit_stream(decompress_stream, stream);
    zfp_stream_rewind(decompress_stream);

    if (!zfp_decompress(decompress_stream, decompress_field)) {
        stream_close(stream);
        throw std::runtime_error("ZFP decompression failed for slice " + std::to_string(iz));
    }

    stream_close(stream);
}

void WavefieldPool::check_ready() {
    // std::lock_guard<std::mutex> lock(compression_mutex);
    if (compression_futures.size() >= wfld_pool.size()) {
        compression_futures.front().wait();
        compression_futures.pop();
    }
}

void WavefieldPool::wait_to_finish() {
    // std::lock_guard<std::mutex> lock(compression_mutex);
    while (!compression_futures.empty()) {
        compression_futures.front().wait();
        compression_futures.pop();
    }
}

void WavefieldPool::cleanup() {
  while (!compression_futures.empty()) {
        compression_futures.front().wait();
        compression_futures.pop();
    }

    for (const auto& wfld_ptr : wfld_pool) {
        if (wfld_ptr && wfld_ptr->getVals()) {
            // This is the counterpart to cudaHostRegister
            CHECK_CUDA_ERROR(cudaHostUnregister(wfld_ptr->getVals()));
        }
    }

    for (auto event : events_pool) {
        if (event) {
            CHECK_CUDA_ERROR(cudaEventDestroy(event));
        }
    }

    for (size_t i = 0; i < zfp_stream_pool.size(); ++i) {
        if (zfp_stream_pool[i]) {
            zfp_stream_close(zfp_stream_pool[i]);
        }
        if (zfp_field_pool[i]) {
            zfp_field_free(zfp_field_pool[i]);
        }
    }

    if (decompress_stream) {
			zfp_stream_close(decompress_stream);
    }
    if (decompress_field) {
			zfp_field_free(decompress_field);
    }
		if (decompress_wfld && decompress_wfld->getVals())
			CHECK_CUDA_ERROR(cudaHostUnregister(decompress_wfld->getVals()));
}



