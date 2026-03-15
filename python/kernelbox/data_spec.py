"""Data spec resolution: spec string -> torch.Tensor."""

import os

import numpy as np


def from_spec(spec, dtype=None, n=1024, device='cpu', seed=None):
    """Resolve a data spec string to a torch.Tensor.

    Args:
        spec: Data spec string (zeros, ones, const:V, seq, seq:K,
              rand, rand:A:B, randn, randn:M:S, or file path)
        dtype: torch dtype (default: torch.float32)
        n: Number of elements
        device: Target device ('cpu' or 'cuda')
        seed: Random seed (None for non-deterministic)
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    if seed is not None:
        torch.manual_seed(seed)

    if spec == 'zeros':
        return torch.zeros(n, dtype=dtype, device=device)
    elif spec == 'ones':
        return torch.ones(n, dtype=dtype, device=device)
    elif spec.startswith('const:'):
        val = float(spec[6:])
        return torch.full((n,), val, dtype=dtype, device=device)
    elif spec == 'seq':
        return torch.arange(n, dtype=dtype, device=device)
    elif spec.startswith('seq:'):
        start = float(spec[4:])
        return torch.arange(start, start + n, dtype=dtype, device=device)
    elif spec == 'rand':
        return torch.rand(n, dtype=dtype, device=device)
    elif spec.startswith('rand:'):
        parts = spec[5:].split(':')
        lo, hi = float(parts[0]), float(parts[1])
        return lo + (hi - lo) * torch.rand(n, dtype=dtype, device=device)
    elif spec == 'randn':
        return torch.randn(n, dtype=dtype, device=device)
    elif spec.startswith('randn:'):
        parts = spec[6:].split(':')
        mean, std = float(parts[0]), float(parts[1])
        return mean + std * torch.randn(n, dtype=dtype, device=device)
    else:
        return _load_file_spec(spec, dtype, n, device)


# Torch dtype -> numpy dtype mapping (used for file loading).
_TORCH_TO_NP = None

def _torch_to_np(dtype):
    """Map a torch dtype to the corresponding numpy dtype."""
    import torch
    global _TORCH_TO_NP
    if _TORCH_TO_NP is None:
        _TORCH_TO_NP = {
            torch.float32: np.float32, torch.int32: np.int32,
            torch.float16: np.float16, torch.float64: np.float64,
            torch.uint8: np.uint8,
        }
    return _TORCH_TO_NP.get(dtype, np.float32)


def _load_file_spec(spec, dtype, n, device):
    """Load data from a file spec."""
    import torch

    path = spec
    if path.startswith('file:'):
        path = path[5:]

    name = None
    if ':' in path:
        path, name = path.rsplit(':', 1)

    ext = os.path.splitext(path)[1].lower()

    if ext in ('.pt', '.pth'):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(data, dict):
            tensor = data[name] if name else next(iter(data.values()))
        elif isinstance(data, (list, tuple)):
            tensor = data[0] if not name else data[int(name)]
        else:
            tensor = data
        tensor = tensor.to(dtype).flatten()[:n]
        if len(tensor) < n:
            tensor = torch.cat([tensor, torch.zeros(n - len(tensor), dtype=dtype)])
        return tensor.to(device)

    elif ext in ('.h5', '.hdf5'):
        import h5py
        if not name:
            raise ValueError("HDF5 files require dataset name: path.h5:dataset")
        with h5py.File(path, 'r') as f:
            arr = f[name][:]
        np_dtype = _torch_to_np(dtype)
        arr = arr.astype(np_dtype).flatten()[:n]
        if len(arr) < n:
            arr = np.concatenate([arr, np.zeros(n - len(arr), dtype=np_dtype)])
        return torch.from_numpy(arr).to(device)

    else:
        np_dtype = _torch_to_np(dtype)
        arr = np.fromfile(path, dtype=np_dtype, count=n)
        if len(arr) < n:
            arr = np.concatenate([arr, np.zeros(n - len(arr), dtype=np_dtype)])
        return torch.from_numpy(arr[:n].copy()).to(device)
