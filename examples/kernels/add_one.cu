extern "C" __global__ void add_one(const unsigned int *input,
                                    unsigned int *output,
                                    unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = input[idx] + 1;
}
