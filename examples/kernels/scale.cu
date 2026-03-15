// Single input, single output: output = input * 2.5
extern "C" __global__ void scale(const float *input,
                                  float *output,
                                  unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = input[idx] * 2.5f;
}
