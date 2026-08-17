import unittest

import torch

from grail.optimization.approach import (
    approach_offsets,
    hand_to_mesh_surface_distance,
    smoothstep_approach_ramp,
)
from grail.optimization.loss_terms import contact_anchor_distance_loss


class HumanApproachTests(unittest.TestCase):
    def test_exact_hand_to_triangle_surface_distance(self):
        hand = torch.tensor([[0.25, 0.25, 1.0]])
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]])
        distance = hand_to_mesh_surface_distance(
            hand, vertices, faces, top_k=1, candidate_faces=1
        )
        self.assertAlmostEqual(float(distance), 1.0, places=5)

    def test_ramp_before_window_monotonic_and_held_after_contact(self):
        ramp = smoothstep_approach_ramp(121, contact_frame=80, approach_window=30)
        self.assertTrue(torch.all(ramp[:50] == 0))
        self.assertTrue(torch.all(ramp[1:] >= ramp[:-1]))
        self.assertTrue(torch.all(ramp[80:] == 1))

    def test_approach_distance_is_bounded(self):
        ramp = smoothstep_approach_ramp(121, contact_frame=80, approach_window=30)
        offsets, distance = approach_offsets(
            ramp, torch.tensor(2.0), torch.tensor([1.0, 0.0, 0.0]), max_distance=0.35
        )
        self.assertAlmostEqual(float(distance), 0.35, places=6)
        self.assertTrue(torch.all(offsets[:50] == 0))
        self.assertTrue(torch.allclose(offsets[80:], torch.tensor([0.35, 0.0, 0.0])))

    def test_scalar_approach_optimization_reaches_three_cm(self):
        ramp = smoothstep_approach_ramp(121, contact_frame=80, approach_window=30)
        distance = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([distance], lr=0.02)
        for _ in range(120):
            optimizer.zero_grad()
            offsets, _ = approach_offsets(
                ramp, distance, torch.tensor([1.0, 0.0, 0.0]), max_distance=0.35
            )
            surface_distance = (0.20 - offsets[80, 0]).abs()
            loss = contact_anchor_distance_loss(surface_distance, target=0.02, delta=0.02)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                distance.clamp_(0.0, 0.35)
        final_surface_distance = abs(0.20 - float(distance))
        self.assertLess(final_surface_distance, 0.03)


if __name__ == "__main__":
    unittest.main()
