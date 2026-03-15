// Blend two signals with per-element weights.
// 3 inputs (signal_a, signal_b, weights), 2 outputs (blended, residual)
extern "C" __global__ void weighted_blend(
    const float *signal_a,
    const float *signal_b,
    const float *weights,
    float *blended,
    float *residual,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float w = weights[idx];
        float b = w * signal_a[idx] + (1.0f - w) * signal_b[idx];
        blended[idx] = b;
        residual[idx] = signal_a[idx] - b;
    }
}
