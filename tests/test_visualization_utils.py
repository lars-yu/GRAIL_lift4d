import unittest

import numpy as np
import torch

from grail.visualization.utils.vis_utils import motion_seq_to_scenepic


class MotionSeqToScenepicTest(unittest.TestCase):
    def test_detaches_optimization_tensors_before_numpy_conversion(self):
        motion_seq = {
            "human_seq": {
                "vertices": torch.randn(2, 3, 3, requires_grad=True),
                "joints_pos": torch.randn(2, 2, 3, requires_grad=True),
                "triangles": torch.tensor([[0, 1, 2]], dtype=torch.int64),
                "rigid": False,
            },
            "obj_seq": {
                "vertices": torch.randn(4, 3, requires_grad=True),
                "triangles": torch.tensor([[0, 1, 2]], dtype=torch.int64),
                "transforms": torch.eye(4).repeat(2, 1, 1).requires_grad_(),
                "rigid": True,
            },
            "static_table_seq": {
                "vertices": torch.randn(4, 3, requires_grad=True),
                "triangles": torch.tensor([[0, 1, 2]], dtype=torch.int64),
                "transforms": torch.eye(4).repeat(2, 1, 1).requires_grad_(),
                "rigid": True,
            },
        }
        hoi_data = {"human_data": {"trans": torch.zeros(2, 3)}}

        result = motion_seq_to_scenepic(motion_seq, hoi_data)

        self.assertIsInstance(result["human_seq"]["vertices"], np.ndarray)
        self.assertIsInstance(result["obj_seq"]["vertices"], np.ndarray)
        self.assertIsInstance(result["obj_seq"]["transforms"], np.ndarray)
        self.assertIsInstance(result["static_table_seq"]["vertices"], np.ndarray)
        self.assertTrue(np.isfinite(result["human_seq"]["vertices"]).all())


if __name__ == "__main__":
    unittest.main()
