#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <Reflect.h>

#include <sep_reg_file.h>
#include <utils.h>
#include <ioModes.h>
#include "zfp.h"
#include <queue>

class WavefieldPool {
public:
    WavefieldPool(std::shared_ptr<hypercube> wfld_hyper, 
                         std::shared_ptr<paramObj> par) {
        initialize(wfld_hyper, par);
    }

    ~WavefieldPool() {
        cleanup();
    }

    // Compression/decompression interface
    void compress_slice_async(int iz, complex_vector* wfld, cudaStream_t stream, 
                              std::vector<std::vector<char>>& storage);
    void decompress_slice(int iz, std::vector<char>& compressed_data);
    
    // Resource management
    std::shared_ptr<complex4DReg> get_wfld_buffer(int pool_idx) { return wfld_pool[pool_idx]; }
    cudaEvent_t get_event(int pool_idx) { return events_pool[pool_idx]; }
    
    // Pipeline management
    void check_ready();
    void wait_to_finish();
    void clear_pipeline();

private:
    void initialize(std::shared_ptr<hypercube> wfld_hyper, std::shared_ptr<paramObj> par);
    void cleanup();
    
    std::future<void> compress_slice_impl(int iz, int pool_idx, cudaEvent_t event, std::vector<std::vector<char>>& storage);

    // Resources
    size_t _slice_size_bytes;
    std::vector<std::shared_ptr<complex4DReg>> wfld_pool;
    std::vector<cudaEvent_t> events_pool;
    std::vector<zfp_stream*> zfp_stream_pool;
    std::vector<zfp_field*> zfp_field_pool;
    double error_bound;

    // Add dedicated decompression resources
    zfp_stream* decompress_stream;
    zfp_field* decompress_field;
    std::shared_ptr<complex4DReg> decompress_wfld;  // Dedicated wavefield for decompression
    
    // Pipeline management
    std::queue<std::future<void>> compression_futures;
    std::mutex compression_mutex;  // For thread safety
    
    // Configuration
    int nwflds_to_store;
    double rel_error_bound;
    size_t slice_size_bytes;
};
