"""KernelBox iterate-mode package."""

import importlib as _importlib

from .data_spec import from_spec

__all__ = ['from_spec', 'dev', 'KernelSession', 'Session',
           'iterate', 'watch_file', 'h5']


def __getattr__(name):
    """Lazy-import torch-dependent modules on first access."""
    if name in ('dev', 'KernelSession', 'Session'):
        mod = _importlib.import_module('.dev', __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    if name in ('iterate', 'watch_file'):
        mod = _importlib.import_module('.iterate', __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    if name == 'h5':
        mod = _importlib.import_module('.h5', __name__)
        globals()['h5'] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
