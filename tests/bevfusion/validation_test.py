#!/usr/bin/env python3
"""Check selective sample loading and FP16 validation guards without model imports."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/bevfusion'))
from _lib.validation import check_finite_tensors, selected_batches


class ValidationTest(unittest.TestCase):
    """Protect validation from hidden nonfinite outputs and unnecessary image decoding."""

    def test_only_requested_samples_are_loaded_and_collated(self):
        """Exercise sparse indices in a dataset that rejects every unrequested access."""
        class Dataset:
            """Provide two permitted samples in a much larger logical dataset."""

            def __getitem__(self, index):
                """Fail immediately if selection tries to decode an unrequested frame."""
                if index not in (7, 999):
                    raise AssertionError('Unselected frame decoded')
                return 'frame-' + str(index)

        loader = SimpleNamespace(dataset=Dataset(), collate_fn=tuple)
        self.assertEqual(list(selected_batches(loader, [7, 999])),
                         [(7, ('frame-7',)), (999, ('frame-999',))])
        self.assertEqual(list(selected_batches(loader, [])), [])

    def test_nonfinite_head_tensor_is_rejected(self):
        """Scan every tensor, including head outputs absent from original feature hooks."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(AssertionError, 'dump is empty'):
                check_finite_tensors(root)
            np.save(root / 'images.npy', np.zeros((2, 3), np.float32))
            np.save(root / 'score.npy', np.ones((2, 3), np.float32))
            self.assertEqual(check_finite_tensors(root), 2)
            for invalid in (np.nan, np.inf, -np.inf):
                np.save(root / 'score.npy', np.array([invalid], np.float32))
                with self.assertRaisesRegex(AssertionError, 'score.npy'):
                    check_finite_tensors(root)


if __name__ == '__main__':
    unittest.main()
