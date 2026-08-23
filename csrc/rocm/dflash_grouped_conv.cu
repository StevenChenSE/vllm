#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

// HIP kernel for 1D Grouped Convolution in DFlash2
// hidden_states: [N, hidden_size] (half)
// delta: [N, taps, num_groups] (half)
// base: [taps, hidden_size] (half)
// output: [N, hidden_size] (half)

__global__ void grouped_conv_fused_hip_kernel(
    const __half* __restrict__ hidden_ptr,
    const __half* __restrict__ delta_ptr,
    const __half* __restrict__ base_ptr,
    __half* __restrict__ output_ptr,
    int N,
    int64_t stride_delta_n,
    int64_t stride_delta_t,
    int64_t stride_delta_g,
    int num_groups,
    int group_size,
    int hidden_size,
    int block_size
) {
    // Grid: blockIdx.x -> token batch (N), blockIdx.y -> group (num_groups)
    // Block: threadIdx.x -> element within group (group_size, e.g. 256)
    int n = blockIdx.x;
    int g = blockIdx.y;
    int e = threadIdx.x;

    if (n >= N || e >= group_size) return;

    int pos = n % block_size;
    int prev_n = (n > 0) ? (n - 1) : 0;
    bool valid_tap1 = (pos >= 1);

    // Base weights
    float base_0 = __half2float(base_ptr[0 * hidden_size + g * group_size + e]);
    float base_1 = __half2float(base_ptr[1 * hidden_size + g * group_size + e]);

    // Delta weights
    float delta_0 = __half2float(delta_ptr[n * stride_delta_n + 0 * stride_delta_t + g * stride_delta_g]);
    float delta_1 = __half2float(delta_ptr[n * stride_delta_n + 1 * stride_delta_t + g * stride_delta_g]);

    float coeff_0 = base_0 + delta_0;
    float coeff_1 = base_1 + delta_1;

    // Current hidden
    float h_0 = __half2float(hidden_ptr[n * hidden_size + g * group_size + e]);
    float out = coeff_0 * h_0;

    // Tap 1
    if (valid_tap1) {
        float h_1 = __half2float(hidden_ptr[prev_n * hidden_size + g * group_size + e]);
        out += coeff_1 * h_1;
    }

    output_ptr[n * hidden_size + g * group_size + e] = __float2half(out);
}

torch::Tensor grouped_conv_fused_hip(
    torch::Tensor hidden_states,
    torch::Tensor delta,
    torch::Tensor base,
    int64_t block_size,
    int64_t num_groups,
    int64_t group_size
) {
    auto output = torch::empty_like(hidden_states);
    int N = hidden_states.size(0);
    int hidden_size = num_groups * group_size;

    dim3 grid(N, num_groups);
    dim3 block(group_size);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    grouped_conv_fused_hip_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(hidden_states.data_ptr()),
        reinterpret_cast<const __half*>(delta.data_ptr()),
        reinterpret_cast<const __half*>(base.data_ptr()),
        reinterpret_cast<__half*>(output.data_ptr()),
        N,
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        num_groups,
        group_size,
        hidden_size,
        block_size
    );

    return output;
}
