extern "C" __global__ void saxpy(const float *input,
                                  float *output,
                                  unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = 2.0f * input[idx] + 3.0f;
}
