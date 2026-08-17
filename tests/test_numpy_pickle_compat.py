import io
import pickle
import unittest

import numpy as np

from grail.core.io import _load_numpy_compatible_pickle


class NumpyPickleCompatTest(unittest.TestCase):
    def test_loads_numpy_2_core_path_in_numpy_1_environment(self):
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        payload = pickle.dumps(expected, protocol=0)
        payload = payload.replace(b"cnumpy.core.", b"cnumpy._core.")

        actual = _load_numpy_compatible_pickle(io.BytesIO(payload))

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
