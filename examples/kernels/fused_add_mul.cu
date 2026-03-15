// Two inputs, two outputs: sum = x + y, prod = x * y
extern "C" __global__ void fused_add_mul(
    const float *x, const float *y,
    float *sum_out, float *prod_out,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        sum_out[idx] = x[idx] + y[idx];
        prod_out[idx] = x[idx] * y[idx];
    }
}
