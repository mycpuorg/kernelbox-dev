"""Triton softmax kernel test. Run with: kbox iterate examples/dev/test_triton_softmax.py --once
"""
import torch
import numpy as np
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def init_once():
    rows, cols = 128, 256
    x = torch.randn(rows, cols, device="cuda")
    BLOCK = triton.next_power_of_2(cols)
    return {
        "triton_kernel": softmax_kernel,
        "triton_constexprs": {"BLOCK_SIZE": BLOCK},
        "inputs": [x],
        "expected": [torch.softmax(x, dim=-1)],
        "outputs": 1,
        "grid": rows,
        "atol": 1e-4,
    }

def run(inputs, kernel):
    x = inputs[0]
    rows, cols = x.shape
    out = kernel(x, params=[
        kernel.out_ptr(0),        # output_ptr
        kernel.in_ptr(0),         # input_ptr
        np.int32(x.stride(0)),    # input_row_stride
        np.int32(x.stride(0)),    # output_row_stride (same shape)
        np.int32(cols),           # n_cols
    ])
    return [out.view(rows, cols)]
