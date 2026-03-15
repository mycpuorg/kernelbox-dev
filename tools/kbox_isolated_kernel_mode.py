#!/usr/bin/env python3
"""One-shot isolated kernel_mode planner runner."""
import os
import sys
from pathlib import Path


def _ensure_kernelbox_importable():
    try:
        import kernelbox  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "python",
        here.parent.parent.parent / "python",
    ]
    for candidate in candidates:
        if (candidate / "kernelbox").is_dir():
            sys.path.insert(0, str(candidate))
            return


_ensure_kernelbox_importable()

from kernelbox.isolated_kernel_mode import main


if __name__ == "__main__":
    main()
