#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <math.h>

// HIP kernel for Split-KV reduce_segments
// output: [num_tokens, num_query_heads, head_size] (half)
// segm_output: [num_tokens, num_query_heads, max_num_segments, head_size_padded] (half)
// segm_max: [num_tokens, num_query_heads, max_num_segments] (float)
// segm_expsum: [num_tokens, num_query_heads, max_num_segments] (float)
// seq_lens: [num_seqs] (int)
// query_start_len: [num_seqs + 1] (int)

__device__ __forceinline__ int find_seq_idx_hip(
    const int* __restrict__ query_start_len_ptr,
    int target_idx,
    int num_seqs
) {
    int left = 0;
    int right = num_seqs;
    while (left < right) {
        int mid = (left + right) / 2;
        int val = query_start_len_ptr[mid];
        if (val <= target_idx) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left - 1;
}

__global__ void reduce_segments_hip_kernel(
    __half* __restrict__ output_ptr,
    const __half* __restrict__ segm_output_ptr,
    const float* __restrict__ segm_max_ptr,
    const float* __restrict__ segm_expsum_ptr,
    const int* __restrict__ seq_lens_ptr,
    const int* __restrict__ query_start_len_ptr,
    int num_seqs,
    int num_query_heads,
    int head_size,
    int head_size_padded,
    int max_num_segments,
    int tile_size,
    int64_t output_stride_0,
    int64_t output_stride_1
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_size) return;

    int seq_idx = find_seq_idx_hip(query_start_len_ptr, token_idx, num_seqs);
    if (seq_idx < 0) seq_idx = 0;
    if (seq_idx >= num_seqs) seq_idx = num_seqs - 1;

    int seq_len = seq_lens_ptr[seq_idx];
    int tiles_per_segment = (seq_len + max_num_segments * tile_size - 1) / (max_num_segments * tile_size);
    if (tiles_per_segment < 1) tiles_per_segment = 1;
    int act_num_segments = (seq_len + tiles_per_segment * tile_size - 1) / (tiles_per_segment * tile_size);
    if (act_num_segments > max_num_segments) act_num_segments = max_num_segments;

    int64_t meta_offset_base = ((int64_t)token_idx * num_query_heads + head_idx) * max_num_segments;

    // Pass 1: find overall max
    float overall_max = -1e30f;
    for (int seg = 0; seg < act_num_segments; ++seg) {
        float m = segm_max_ptr[meta_offset_base + seg];
        if (m > overall_max) overall_max = m;
    }

    float overall_max_safe = (overall_max <= -1e20f) ? 0.0f : overall_max;

    // Pass 2: compute rescaled expsum & accumulator
    float overall_expsum = 0.0f;
    float acc_sum = 0.0f;

    int64_t out_offset_base = (((int64_t)token_idx * num_query_heads + head_idx) * max_num_segments) * head_size_padded;

    for (int seg = 0; seg < act_num_segments; ++seg) {
        float m = segm_max_ptr[meta_offset_base + seg];
        float expsum = segm_expsum_ptr[meta_offset_base + seg];
        float diff = (m <= -1e20f) ? -10000.0f : (m - overall_max_safe);
        float weight = (m <= -1e20f) ? 0.0f : expf(diff);

        overall_expsum += expsum * weight;

        float val = __half2float(segm_output_ptr[out_offset_base + seg * head_size_padded + d]);
        acc_sum += val * weight;
    }

    float final_val = 0.0f;
    if (overall_expsum > 0.0f) {
        final_val = acc_sum / overall_expsum;
    }

    int64_t out_idx = (int64_t)token_idx * output_stride_0 + (int64_t)head_idx * output_stride_1 + d;
    output_ptr[out_idx] = __float2half(final_val);
}

void reduce_segments_hip(
    torch::Tensor output,
    torch::Tensor segm_output,
    torch::Tensor segm_max,
    torch::Tensor segm_expsum,
    torch::Tensor seq_lens,
    torch::Tensor query_start_len,
    int64_t num_seqs,
    int64_t num_query_heads,
    int64_t head_size,
    int64_t head_size_padded,
    int64_t max_num_segments,
    int64_t tile_size
) {
    int num_tokens = output.size(0);
    dim3 grid(num_tokens, num_query_heads);
    dim3 block(head_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    reduce_segments_hip_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output.data_ptr()),
        reinterpret_cast<const __half*>(segm_output.data_ptr()),
        reinterpret_cast<const float*>(segm_max.data_ptr()),
        reinterpret_cast<const float*>(segm_expsum.data_ptr()),
        reinterpret_cast<const int*>(seq_lens.data_ptr()),
        reinterpret_cast<const int*>(query_start_len.data_ptr()),
        (int)num_seqs,
        (int)num_query_heads,
        (int)head_size,
        (int)head_size_padded,
        (int)max_num_segments,
        (int)tile_size,
        output.stride(0),
        output.stride(1)
    );
}
