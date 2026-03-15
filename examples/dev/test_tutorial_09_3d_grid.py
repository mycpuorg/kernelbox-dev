"""Tutorial 09: 3D grid and block dimensions.

Demonstrates a 2D matrix operation using 2D grid and block dimensions.
Grid and block can be tuples of (x, y, z).

Run: kbox iterate examples/dev/test_tutorial_09_3d_grid.py --once
"""
import torch

TRANSPOSE_KERNEL = r"""
extern "C" __global__ void transpose(
    const float *in0, float *out0, unsigned int rows, unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        out0[col * rows + row] = in0[row * cols + col];
    }
}
"""

ROWS = 64
COLS = 128
BLOCK_X = 16
BLOCK_Y = 16

def init_once():
    import numpy as np
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": TRANSPOSE_KERNEL,
        "inputs": [x],
        "expected": [x.T.contiguous()],
        "outputs": "float32;n=%d" % (ROWS * COLS),
        "grid": ((COLS + BLOCK_X - 1) // BLOCK_X,
                 (ROWS + BLOCK_Y - 1) // BLOCK_Y, 1),
        "block": (BLOCK_X, BLOCK_Y, 1),
    }

def run(inputs, kernel):
    import numpy as np
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0),
        kernel.out_ptr(0),
        np.uint32(ROWS),
        np.uint32(COLS),
    ])]
