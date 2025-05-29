#pragma once
#include <cuda_fp16.h>

void marlin_qqq_gemm(int8_t* a,
                              int32_t* b_q_weight,
                              float* s_tok,
                              float* s_ch,
                              half* s_group,
                              int32_t* workspace, int64_t size_m,
                              int64_t size_n, int64_t size_k,
                              int groupsize,
                              int32_t* c,
                              half* d,
                              cudaStream_t stream
                            );