"""Tutorial 06: Shared memory and explicit grid/block.

A block-level reduction kernel that computes the sum of each 256-element chunk.
Uses shared memory for the reduction tree.

Run: kbox iterate examples/dev/test_tutorial_06_shared_memory.py --once
"""
import torch

BLOCK_REDUCE_KERNEL = r"""
extern "C" __global__ void block_sum(
    const float *in0, float *out0, unsigned int n
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int idx = bid * block_size + tid;

    // Load into shared memory
    sdata[tid] = (idx < n) ? in0[idx] : 0.0f;
    __syncthreads();

    // Tree reduction
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        out0[bid] = sdata[0];
    }
}
"""

BLOCK = 256
N_BLOCKS = 16
N = BLOCK * N_BLOCKS

def init_once():
    x = torch.randn(N, device="cuda")
    expected = x.view(N_BLOCKS, BLOCK).sum(dim=1)
    return {
        "kernel_source": BLOCK_REDUCE_KERNEL,
        "inputs": [x],
        "expected": [expected],
        "outputs": "float32;n=%d" % N_BLOCKS,
        "grid": N_BLOCKS,
        "block": BLOCK,
        "smem": BLOCK * 4,  # sizeof(float) * BLOCK
        "atol": 1e-3,       # reductions accumulate error
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
