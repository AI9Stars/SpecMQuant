// Implemented by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }

#pragma once
#include <cuda_fp16.h>
#include "../../utils.cuh"

void w4a8_qoq_group_gemm_forward_cuda(const Stream& stream,
                        int8_t* in_feats,
                        int8_t* kernel,
                        int8_t* zeros,
                        int8_t* scales_i8,
                        half2* wscales,
                        half* ascales,
                        half* out_feats,
                        int num_in_feats,
                        int num_in_channels,
                        int num_out_feats,
                        int num_out_channels);
