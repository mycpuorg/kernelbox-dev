import torch
import operator

# ── Inline CUDA: aten.gelu.default ──
from torch.utils.cpp_extension import load_inline as _load_inline
_GELU_KERNEL_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void gelu_kernel(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}

torch::Tensor gelu_kernel_forward(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""
_gelu_kernel_ext = _load_inline("gelu_kernel_ext", "", cuda_sources=[_GELU_KERNEL_CUDA_SRC], functions=["gelu_kernel_forward"])

# ── Inline CUDA: aten.relu.default ──
from torch.utils.cpp_extension import load_inline as _load_inline
_RELU_KERNEL_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void relu_kernel(
    const float *in0, float *out0, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = x > 0 ? x : 0;
    }
}

torch::Tensor relu_kernel_forward(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""
_relu_kernel_ext = _load_inline("relu_kernel_ext", "", cuda_sources=[_RELU_KERNEL_CUDA_SRC], functions=["relu_kernel_forward"])

def run(input):
    out = {}
    out["gelu_1"] = _gelu_kernel_ext.gelu_kernel_forward(input["x"])
    out["add_1"] = torch.ops.aten.add.Tensor(out["gelu_1"], input["bias"])
    out["mm_1"] = torch.ops.aten.mm.default(out["add_1"], input["weight"])
    out["relu_1"] = _relu_kernel_ext.relu_kernel_forward(out["mm_1"])
    out["view_1"] = torch.ops.aten.view.default(out["relu_1"], [2, 16, 32])
    out["mul_1"] = torch.ops.aten.mul.Tensor(out["view_1"], input["scale"])
    return [out["mul_1"]]
