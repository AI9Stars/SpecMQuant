#pragma once
#include "../../utils.cuh"

void gptq_marlin_repack(uint32_t const* b_q_weight_ptr, 
                        uint32_t const* perm_ptr,
                        int64_t size_k, 
                        int64_t size_n,
                        int64_t num_bits, 
                        bool has_perm,
                        uint32_t* out_ptr,
                        cudaStream_t stream);