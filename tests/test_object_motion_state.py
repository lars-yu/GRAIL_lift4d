import unittest

import numpy as np

from grail.optimization.motion_state import (
    build_static_relative_depth_target,
    detect_object_motion,
)


def _centers(frame_num=60, move_start=None, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.tile(np.array([0.1, -0.1, 2.0]), (frame_num, 1))
    centers += rng.normal(scale=0.00025, size=centers.shape)
    if move_start is not None:
        steps = np.maximum(0, np.arange(frame_num) - move_start + 1)
        centers[:, 0] += 0.008 * steps
        centers[:, 2] -= 0.004 * steps
    return centers


def _masks(frame_num=60, move_start=None, *, jitter=False):
    masks = np.zeros((frame_num, 64, 80), dtype=bool)
    for frame in range(frame_num):
        baseline_jitter = frame % 2 if jitter else 0
        shift = 0
        if move_start is not None and frame >= move_start:
            shift = frame - move_start + 1
        masks[frame, 22:38, 20 + baseline_jitter + shift : 36 + baseline_jitter + shift] = True
    return masks


class ObjectMotionStateTests(unittest.TestCase):
    def test_static_mask_and_lift4d_jitter_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "motion onset"):
            detect_object_motion(_centers(), _masks(jitter=True))

    def test_single_frame_mask_anomaly_does_not_unlock_object(self):
        masks = _masks()
        masks[30] = np.roll(masks[30], 10, axis=1)
        with self.assertRaisesRegex(ValueError, "motion onset"):
            detect_object_motion(_centers(), masks)

    def test_three_of_five_sustained_motion_backtracks_to_first_evidence(self):
        expected = 25
        state = detect_object_motion(
            _centers(move_start=expected),
            _masks(move_start=expected),
            config={"strong_iou_drop_floor": 0.20},
        )
        self.assertEqual(state.move_start_frame, expected)
        np.testing.assert_array_equal(state.static[:expected], True)
        np.testing.assert_array_equal(state.moving[expected:], True)

    def test_gpt_contact_hint_cannot_change_motion_state(self):
        centers = _centers(move_start=28)
        masks = _masks(move_start=28)
        early = detect_object_motion(centers, masks, contact_hint=2)
        late = detect_object_motion(centers, masks, contact_hint=57)
        self.assertEqual(early.move_start_frame, late.move_start_frame)
        np.testing.assert_array_equal(early.moving, late.moving)
        np.testing.assert_allclose(early.z_target, late.z_target)

    def test_in_place_rotation_is_detected_from_iou(self):
        masks = _masks()
        for frame in range(24, 60):
            masks[frame] = False
            if (frame - 24) % 2:
                masks[frame, 18:42, 25:31] = True
            else:
                masks[frame, 27:33, 16:40] = True
        state = detect_object_motion(_centers(), masks)
        self.assertEqual(state.move_start_frame, 24)
        self.assertGreater(state.mask_iou_drop[24], state.thresholds["strong_iou_drop"])

    def test_static_relative_depth_target_has_exact_hard_freeze(self):
        z = np.r_[np.linspace(1.99, 2.01, 20), np.linspace(2.0, 1.7, 20)]
        static_z, target = build_static_relative_depth_target(z, 20)
        np.testing.assert_array_equal(target[:20], np.float32(static_z))
        self.assertAlmostEqual(float(target[-1] - target[20]), float(z[-1] - z[20]), places=6)

    def test_target_uses_smoothed_lift4d_z(self):
        smooth_z = np.full(60, 2.0)
        smooth_z[25:] -= 0.003 * np.arange(35)
        state = detect_object_motion(
            _centers(move_start=25),
            _masks(move_start=25),
            smoothed_z=smooth_z,
        )
        self.assertAlmostEqual(
            float(state.z_target[-1] - state.z_target[state.move_start_frame]),
            float(smooth_z[-1] - smooth_z[state.move_start_frame]),
            places=6,
        )

    def test_mask_motion_detected_when_lift4d_center_is_static(self):
        centers = _centers(move_start=None)
        masks = _masks(move_start=24)
        state = detect_object_motion(
            centers,
            masks,
            config={"lift4d_speed_floor_m": 1.0, "required_consecutive_mask_frames": 3},
        )
        self.assertEqual(state.move_start_frame, 24)
        self.assertTrue(np.all(state.moving[24:]))


if __name__ == "__main__":
    unittest.main()
