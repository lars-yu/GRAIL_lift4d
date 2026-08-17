import unittest

import numpy as np
import torch

from grail.optimization.loss_terms import temporal_soft_contact_loss
from grail.optimization.motion_state import (
    build_static_relative_depth_target,
    detect_object_motion,
    infer_contact_hand,
    resolve_contact_hint,
)


def _masks(frame_num, move_start=None):
    masks = np.zeros((frame_num, 48, 64), dtype=bool)
    for frame in range(frame_num):
        shift = 0 if move_start is None else max(0, frame - move_start) // 2
        masks[frame, 18:30, 20 + shift : 32 + shift] = True
    return masks


def _centers(frame_num=60, move_start=None, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.tile(np.array([0.1, -0.1, 2.0]), (frame_num, 1))
    centers += rng.normal(scale=0.0005, size=centers.shape)
    if move_start is not None:
        steps = np.maximum(0, np.arange(frame_num) - move_start)
        centers[:, 0] += 0.008 * steps
        centers[:, 2] -= 0.004 * steps
    return centers


class ObjectMotionStateTests(unittest.TestCase):
    def test_static_jitter_fails_instead_of_inventing_motion(self):
        with self.assertRaisesRegex(ValueError, "motion onset"):
            detect_object_motion(_centers(), _masks(60))

    def test_single_frame_spike_does_not_trigger(self):
        centers = _centers()
        centers[30] += np.array([0.15, -0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "motion onset"):
            detect_object_motion(centers, _masks(60))

    def test_static_then_motion_detects_onset_without_sg31(self):
        expected = 25
        state = detect_object_motion(
            _centers(move_start=expected), _masks(60, expected), contact_hint=48
        )
        self.assertLessEqual(abs(state.move_start_frame - expected), 3)
        self.assertGreaterEqual(state.confidence, 0.55)
        expected_detection = np.asarray(
            __import__("scipy.ndimage").ndimage.median_filter(
                _centers(move_start=expected), size=(5, 1), mode="nearest"
            )
        )
        np.testing.assert_allclose(state.detection_center_cam, expected_detection, atol=1e-6)

    def test_contact_hint_expands_stationary_noise_baseline_without_cropping_search(self):
        rng = np.random.default_rng(9)
        frame_num = 121
        expected = 80
        centers = np.tile(np.array([0.1, -0.1, 2.0]), (frame_num, 1))
        centers[:15] += rng.normal(scale=0.0005, size=(15, 3))
        centers[15:expected] += rng.normal(scale=0.004, size=(expected - 15, 3))
        steps = np.maximum(0, np.arange(frame_num) - expected)
        centers[:, 0] += 0.01 * steps
        centers[:, 2] -= 0.005 * steps

        state = detect_object_motion(
            centers,
            _masks(frame_num, expected),
            contact_hint=49,
        )

        self.assertLessEqual(abs(state.move_start_frame - expected), 3)
        self.assertGreaterEqual(state.confidence, 0.55)

    def test_static_relative_depth_target(self):
        z = np.r_[np.full(20, 2.0), np.linspace(2.0, 1.7, 20)]
        static_z, target = build_static_relative_depth_target(z, 20, transition_frames=4)
        np.testing.assert_allclose(target[:20], static_z)
        self.assertAlmostEqual(float(target[-1] - target[23]), float(z[-1] - z[23]), places=6)

    def test_target_uses_smooth_z_not_detection_z(self):
        centers = _centers(move_start=25)
        smooth_z = np.full(60, 2.0)
        smooth_z[25:] -= 0.003 * np.arange(35)
        state = detect_object_motion(
            centers,
            _masks(60, 25),
            smoothed_z=smooth_z,
            config={"transition_frames": 0},
        )
        self.assertAlmostEqual(float(state.z_target[0]), 2.0, places=6)
        self.assertAlmostEqual(
            float(state.z_target[-1] - state.z_target[state.move_start_frame]),
            float(smooth_z[-1] - smooth_z[state.move_start_frame]),
            places=6,
        )

    def test_soft_contact_hint_cannot_crop_physical_window(self):
        frames = torch.arange(40, 51)
        distances = torch.tensor([0.15, 0.12, 0.09, 0.05, 0.021, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
        _, weights = temporal_soft_contact_loss(
            distances,
            frames,
            contact_hint=20,
            hint_sigma=5,
            hint_floor=0.2,
            softmin_temperature=0.01,
        )
        self.assertTrue(torch.all(weights > 0))
        self.assertEqual(int(frames[weights.argmax()]), 44)

    def test_cache_contact_start_precedes_interaction_fallback(self):
        self.assertEqual(resolve_contact_hint(None, 49, 80, 121), (49, "cache"))
        self.assertEqual(resolve_contact_hint(None, None, 80, 121), (80, "inter_start"))
        self.assertEqual(resolve_contact_hint(55, 49, 80, 121), (55, "cli"))

    def test_auto_hand_uses_first_valid_cache_label(self):
        self.assertEqual(infer_contact_hand("auto", [None, ["R_Hand"]]), "right")
        self.assertEqual(infer_contact_hand("auto", [["L_Hand", "R_Hand"]]), "both")


if __name__ == "__main__":
    unittest.main()
