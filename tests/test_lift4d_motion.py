import os
import tempfile
import unittest

import numpy as np
import torch

from grail.adapters.lift4d import (
    align_lift4d_motion_to_foundationpose,
    fit_translation_scale,
    load_motion_npz,
    save_motion_npz,
    weighted_kabsch_umeyama,
)
from grail.optimization.loss_computer import LossComputer
from grail.optimization.loss_terms import lift4d_motion_loss


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _poses(translations, rotations=None):
    translations = np.asarray(translations, dtype=np.float64)
    n = translations.shape[0]
    out = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
    out[:, :3, 3] = translations
    if rotations is not None:
        out[:, :3, :3] = np.asarray(rotations, dtype=np.float64)
    return out


class Lift4DMotionAdapterTest(unittest.TestCase):
    def test_known_se3_point_cloud_recovers_rotation_and_translation(self):
        rng = np.random.default_rng(4)
        src = rng.normal(size=(128, 3))
        R = _rotz(0.6)
        t = np.array([0.3, -0.2, 1.1])
        dst = src @ R.T + t

        fit = weighted_kabsch_umeyama(src, dst, min_points=8)

        self.assertTrue(fit.valid)
        np.testing.assert_allclose(fit.R, R, atol=1e-6)
        np.testing.assert_allclose(fit.t, t, atol=1e-6)
        self.assertLess(fit.rmse, 1e-8)

    def test_outlier_robust_fit_stays_stable(self):
        rng = np.random.default_rng(5)
        src = rng.normal(size=(200, 3))
        R = _rotz(-0.4)
        t = np.array([-0.2, 0.5, 0.7])
        dst = src @ R.T + t
        dst[:20] += rng.normal(scale=10.0, size=(20, 3))

        fit = weighted_kabsch_umeyama(src, dst, min_points=8, robust_iters=4)

        self.assertTrue(fit.valid)
        np.testing.assert_allclose(fit.R, R, atol=1e-2)
        np.testing.assert_allclose(fit.t, t, atol=1e-2)
        self.assertLess(fit.rmse, 0.05)

    def test_pure_rotation_about_non_origin_center_has_no_spurious_translation(self):
        center = np.array([1.0, 0.2, -0.3])
        R0 = np.eye(3)
        R1 = _rotz(np.pi / 2)
        # T maps canonical points to camera space while rotating about `center`.
        lift = _poses(
            [center - R0 @ center, center - R1 @ center],
            rotations=[R0, R1],
        )
        motion_path = None
        with tempfile.TemporaryDirectory() as td:
            motion_path = os.path.join(td, "motion.npz")
            save_motion_npz(
                motion_path,
                frame_indices=np.array([0, 1]),
                object_poses_cam=lift,
                motion_confidence=np.array([1.0, 1.0]),
                rigid_fit_rmse=np.array([0.0, 0.0]),
                object_scales=np.array([1.0, 1.0]),
                image_size=(64, 64),
                canonical_object_center=center,
            )
            motion = load_motion_npz(motion_path)
            aligned = align_lift4d_motion_to_foundationpose(
                motion,
                fp_poses_world=_poses([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]),
                camera_c2w=np.eye(4),
                translation_scale=1.0,
            )
        np.testing.assert_allclose(aligned.object_poses[:, :3, 3], [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], atol=1e-6)

    def test_camera_to_world_rotation_maps_lift_translation(self):
        lift = _poses([[0, 0, 0], [1, 0, 0]])
        motion = type(
            "Motion",
            (),
            {
                "frame_indices": np.array([0, 1]),
                "object_poses_cam": lift,
                "motion_confidence": np.array([1.0, 1.0]),
                "rigid_fit_rmse": np.array([0.0, 0.0]),
                "object_scales": np.array([1.0, 1.0]),
                "camera_convention": "opencv_camera",
                "canonical_object_center": np.zeros(3),
                "source_path": "memory",
            },
        )()
        cam = np.eye(4)
        cam[:3, :3] = _rotz(np.pi / 2)
        aligned = align_lift4d_motion_to_foundationpose(
            motion,
            fp_poses_world=_poses([[0, 0, 0], [0, 0, 0]]),
            camera_c2w=cam,
            translation_scale=2.0,
        )
        np.testing.assert_allclose(aligned.object_poses[1, :3, 3], [0.0, 2.0, 0.0], atol=1e-6)

    def test_motion_npz_without_depth_or_alpha_loads(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "motion_only.npz")
            save_motion_npz(
                path,
                frame_indices=np.array([0]),
                object_poses_cam=_poses([[0, 0, 0]]),
                motion_confidence=np.array([1.0]),
                rigid_fit_rmse=np.array([0.0]),
                object_scales=np.array([1.0]),
                image_size=(16, 16),
            )
            motion = load_motion_npz(path)
        self.assertEqual(motion.object_poses_cam.shape, (1, 4, 4))

    def test_translation_scale_fit_uses_one_global_scalar(self):
        lift = _poses([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        fp = _poses([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
        scale = fit_translation_scale(
            fp,
            lift,
            valid=np.array([True, True, True]),
            confidence=np.ones(3),
            anchor_pos=0,
            camera_c2w=np.eye(4),
            canonical_center=np.zeros(3),
        )
        self.assertAlmostEqual(scale, 2.0, places=6)

    def test_dynamic_camera_config_raises(self):
        motion = type(
            "Motion",
            (),
            {
                "frame_indices": np.array([0]),
                "object_poses_cam": _poses([[0, 0, 0]]),
                "motion_confidence": np.array([1.0]),
                "rigid_fit_rmse": np.array([0.0]),
                "object_scales": np.array([1.0]),
                "camera_convention": "opencv_camera",
                "canonical_object_center": np.zeros(3),
                "source_path": "memory",
            },
        )()
        with self.assertRaisesRegex(ValueError, "fixed camera"):
            align_lift4d_motion_to_foundationpose(
                motion,
                fp_poses_world=_poses([[0, 0, 0]]),
                camera_c2w=np.eye(4),
                camera_mode="dynamic",
                translation_scale=1.0,
            )


class Lift4DMotionLossTest(unittest.TestCase):
    def test_legacy_full_se3_loss_is_prohibited(self):
        prior_t = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        prior_R = torch.eye(3).repeat(3, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "prohibited"):
            lift4d_motion_loss(
                prior_t,
                prior_R,
                prior_t,
                prior_R,
                torch.tensor([True, True, True]),
                torch.ones(3),
                anchor_frame=0,
            )

    def test_disabled_loss_config_does_not_require_prior(self):
        computer = LossComputer(None, None, "cpu", lambda _: None, 0, None)
        total, losses = computer.compute_loss(
            data=object(),
            pred=object(),
            loss_cfg={"lift4d_motion": {"enabled": False, "weight": 1.0}},
        )
        self.assertEqual(total, 0.0)
        self.assertEqual(losses, {})


if __name__ == "__main__":
    unittest.main()
