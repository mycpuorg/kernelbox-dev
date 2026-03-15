// Numerically stable 1-D softmax in two passes.
// Pass 1: find max, subtract, exponentiate.
// Pass 2: normalize by sum (via atomicAdd to shared).
//
// NOTE: This is a simple single-block implementation for demonstration.
// For production use, you'd want a multi-block reduction.
extern "C" __global__ void softmax_1d(
    const float *input,
    float *output,
    unsigned int n)
{
    extern __shared__ float sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared for reduction
    float val = (idx < n) ? input[idx] : -1e30f;
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Tree reduction to find max within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < n)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Exponentiate
    float exp_val = (idx < n) ? expf(val - max_val) : 0.0f;
    sdata[threadIdx.x] = exp_val;
    __syncthreads();

    // Sum reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum_exp = sdata[0];

    if (idx < n)
        output[idx] = exp_val / sum_exp;
}
