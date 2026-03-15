"""Reference kernel_mode() for the public CUDA pairwise task."""
import numpy as np


def kernel_mode(kernels, input_ptrs, output_ptrs, n, scratch_ptr=None,
                inputs_meta=None, outputs_meta=None):
    _ = scratch_ptr, inputs_meta, outputs_meta
    block = 256
    grid = (n + block - 1) // block
    return [
        {
            "kernel": kernels[0],
            "grid": grid,
            "block": block,
            "params": [input_ptrs[0], output_ptrs[0], np.uint32(n)],
        },
        {
            "kernel": kernels[1],
            "grid": grid,
            "block": block,
            "params": [input_ptrs[0], output_ptrs[1], np.uint32(n)],
            "clear_outputs": False,
        },
    ]
