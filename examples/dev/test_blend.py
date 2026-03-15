"""Weighted blend: 3 inputs -> 2 outputs (blended, residual).

    kbox iterate examples/dev/test_blend.py

Multi-output demo: the h5 file has expected_blend and expected_residual,
which load_test auto-splits into expected[0] and expected[1] (sorted).
"""


def init_once():
    return {"h5": "examples/data/blend.h5"}


def run(inputs):
    w = inputs.weights
    blended  = w * inputs.signal_a + (1.0 - w) * inputs.signal_b
    residual = inputs.signal_a - blended
    return [blended, residual]
