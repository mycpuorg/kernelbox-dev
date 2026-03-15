"""Deliberately broken MLP to demo --dump for debugging.

    kbox iterate examples/dev/test_mlp_debug.py --dump debug_mlp.h5

Then inspect the dump:

    python3 -c "
    import h5py
    with h5py.File('debug_mlp.h5') as f:
        for name in f:
            g = f[name]
            print(f'{name}:')
            print(f'  max_abserr = {g.attrs[\"max_abserr\"]:.6e}')
            print(f'  max_relerr = {g.attrs[\"max_relerr\"]:.6e}')
            print(f'  shape      = {list(g.attrs[\"shape\"])}')
    "
"""
import torch


def init_once():
    return {"h5": "examples/data/mlp.h5"}


def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2 + 0.01]  # intentional error
