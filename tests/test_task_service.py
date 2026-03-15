#!/usr/bin/env python3
"""CPU-only regression tests for task service helpers."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

from kernelbox.task_service import _candidate_file, _iter_input_tensors
from kernelbox.tasks import list_tasks


@dataclass
class _InputBox:
    x: torch.Tensor
    note: str


class TaskServiceTests(unittest.TestCase):
    def test_list_tasks_are_stably_sorted(self):
        names = [task.name for task in list_tasks()]
        self.assertEqual(names, sorted(names))

    def test_iter_input_tensors_accepts_single_tensor(self):
        x = torch.randn(4)
        self.assertEqual(_iter_input_tensors(x), [x])

    def test_iter_input_tensors_filters_supported_containers(self):
        x = torch.randn(4)
        y = torch.randn(2)

        self.assertEqual(_iter_input_tensors({"x": x, "label": "ignored"}), [x])
        self.assertEqual(_iter_input_tensors((x, "ignored", y)), [x, y])
        self.assertEqual(_iter_input_tensors(_InputBox(x=x, note="ignored")), [x])
        self.assertEqual(_iter_input_tensors(v for v in (x, "ignored", y)), [x, y])

    def test_candidate_file_requires_exactly_one_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _candidate_file()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _candidate_file(kernel_mode_code="print('x')", kernel_mode_path="foo.py")

    def test_candidate_file_validates_existing_file(self):
        missing = Path(tempfile.gettempdir()) / "kbox_missing_kernel_mode.py"
        with self.assertRaises(FileNotFoundError):
            _candidate_file(kernel_mode_path=missing)

    def test_candidate_file_materializes_temporary_code(self):
        path, cleanup = _candidate_file(kernel_mode_code="def kernel_mode():\n    return []\n")
        try:
            self.assertEqual(path, cleanup)
            self.assertTrue(Path(path).is_file())
            self.assertIn("def kernel_mode", Path(path).read_text(encoding="utf-8"))
        finally:
            if cleanup is not None:
                Path(cleanup).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
