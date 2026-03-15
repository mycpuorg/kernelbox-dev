"""Hidden CUDA task variant for secure kernel_mode evaluation."""
import torch

from kernelbox.h5 import TensorDict, TestSuite
from kernelbox.task_defs.cuda_pairwise_public import ADD_ONE, DOUBLE


def _make_case(name, n, shift, scale):
    x = torch.linspace(-2.0, 2.0, n, device="cuda", dtype=torch.float32)
    x = x * scale + shift
    inputs = TensorDict(x=x)
    expected = [x + 1.0, x * 2.0]
    return name, inputs, expected


def init_once():
    suite = TestSuite(
        cases=[
            _make_case("small", 1024, 0.25, 1.0),
            _make_case("medium", 2048, -0.5, 1.5),
            _make_case("large", 4096, 1.0, 0.75),
        ],
        input_keys=["x"],
        n_expected=2,
        directory="hidden_cuda_pairwise",
    )
    return {
        "kernel_source": [ADD_ONE, DOUBLE],
        "suite": suite,
        "outputs": 2,
        "atol": 1e-5,
        "bench_suite_per_case": True,
        "warmup": 1,
        "iters": 3,
    }
