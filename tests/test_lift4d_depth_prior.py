import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from grail.adapters.lift4d_depth import load_lift4d_depth_prior, project_opencv_translation
from grail.optimization.data_types import OptParams
from grail.optimization.loss_terms import (
    full_frame_indices,
    foundationpose_camera_rays,
    lift4d_depth_trend_loss,
    positive_depth_scale,
    ray_depth_translation,
)


def _write_prior(path, frame_num=25, point_num=16, **extra):
    rng = np.random.default_rng(7)
    base = rng.normal(scale=0.02, size=(point_num, 3)).astype(np.float32)
    base[:, 2] += 3.0
    motion = np.zeros((frame_num, 1, 3), dtype=np.float32)
    motion[:, 0, 2] = np.linspace(0.0, -0.25, frame_num)
    trajectories = base[None] + motion
    K = np.repeat(
        np.asarray([[900.0, 0.0, 320.0], [0.0, 910.0, 240.0], [0.0, 0.0, 1.0]])[
            None
        ],
        frame_num,
        axis=0,
    ).astype(np.float32)
    payload = {
        "frame_indices": np.arange(frame_num, dtype=np.int64),
        "point_trajectories_cam": trajectories,
        "canonical_points": base,
        "point_visibility": np.ones((frame_num, point_num), dtype=bool),
        "point_fit_inliers": np.ones((frame_num, point_num), dtype=bool),
        "point_opacity": np.ones(point_num, dtype=np.float32),
        "valid_point_count": np.full(frame_num, point_num, dtype=np.int64),
        "camera_intrinsics": K,
        "camera_convention": np.asarray("opencv_camera"),
    }
    payload.update(extra)
    np.savez_compressed(path, **payload)


class Lift4DDepthPriorTests(unittest.TestCase):
    def test_ray_depth_projection_invariance(self):
        fp = torch.tensor([[0.3, -0.2, 3.0], [-0.1, 0.4, 2.0]], dtype=torch.float64)
        ray = foundationpose_camera_rays(fp)
        moved = ray_depth_translation(ray, torch.tensor([1.2, 4.5], dtype=torch.float64))
        K = np.repeat(np.eye(3)[None], 2, axis=0)
        K[:, 0, 0] = 800.0
        K[:, 1, 1] = 810.0
        K[:, 0, 2] = 320.0
        K[:, 1, 2] = 240.0
        original_px = project_opencv_translation(fp.numpy(), K)
        moved_px = project_opencv_translation(moved.numpy(), K)
        np.testing.assert_allclose(original_px, moved_px, atol=1e-10)

    def test_only_depth_is_optimized(self):
        frame_num = 5
        params = OptParams(
            human_trans_global=torch.zeros(3),
            human_trans_res=torch.zeros(frame_num, 3),
            human_pose_res=torch.zeros(frame_num, 1, 6),
            hand_pose_res=torch.zeros(frame_num, 1, 6),
            obj_R_res=torch.zeros(frame_num, 6),
            obj_t_res=None,
            obj_depth_res=torch.zeros(frame_num, requires_grad=True),
            human_approach_distance=torch.zeros((), requires_grad=True),
            obj_z_opt=None,
            log_lift4d_depth_scale=torch.zeros((), requires_grad=True),
        )
        self.assertIsNone(params.obj_t_res)
        self.assertEqual(tuple(params.obj_depth_res.shape), (frame_num,))
        self.assertIsNone(params.obj_z_opt)

    def test_121_frames_are_all_supervised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            _write_prior(path, frame_num=121, point_num=64)
            prior = load_lift4d_depth_prior(
                path, frame_num=121, stable_point_count=64, min_stable_points=8
            )
        np.testing.assert_array_equal(prior.frame_indices, np.arange(121))
        self.assertEqual(int(prior.prior_used.sum()), 121)
        self.assertEqual(len(full_frame_indices(121)), 121)

    def test_interval_greater_than_one_fails(self):
        with self.assertRaisesRegex(ValueError, "interval=1"):
            full_frame_indices(121, interval=4)

    def test_noncontinuous_frame_indices_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            bad = np.arange(25, dtype=np.int64)
            bad[12] = 11
            _write_prior(path, frame_indices=bad)
            with self.assertRaisesRegex(ValueError, "np.arange"):
                load_lift4d_depth_prior(path, frame_num=25, min_stable_points=8)

    def test_strong_smoothing_reduces_high_frequency_jitter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            frame_num = 121
            rng = np.random.default_rng(11)
            jitter = 0.08 * ((-1.0) ** np.arange(frame_num))
            base = rng.normal(scale=0.01, size=(64, 3)).astype(np.float32)
            base[:, 2] += 3.0
            trajectories = np.repeat(base[None], frame_num, axis=0)
            trajectories[:, :, 2] += jitter[:, None]
            _write_prior(path, frame_num=frame_num, point_num=64, point_trajectories_cam=trajectories)
            prior = load_lift4d_depth_prior(
                path, frame_num=frame_num, stable_point_count=64, min_stable_points=8
            )
        raw_hf = np.mean(np.abs(np.diff(prior.z_raw, n=2)))
        smooth_hf = np.mean(np.abs(np.diff(prior.z, n=2)))
        self.assertLess(smooth_hf, 0.2 * raw_hf)

    def test_smoothing_removes_every_four_frame_spikes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            frame_num = 121
            base = np.zeros((64, 3), dtype=np.float32)
            base[:, 2] = 3.0
            trajectories = np.repeat(base[None], frame_num, axis=0)
            trajectories[50::4, :, 2] += 0.15
            _write_prior(
                path,
                frame_num=frame_num,
                point_num=64,
                point_trajectories_cam=trajectories,
            )
            prior = load_lift4d_depth_prior(
                path, frame_num=frame_num, stable_point_count=64, min_stable_points=8
            )
        raw_step = np.abs(np.diff(prior.z_raw))
        smooth_step = np.abs(np.diff(prior.z))
        self.assertGreater(raw_step[49:].max(), 0.1)
        self.assertLess(smooth_step[49:].max(), 0.02)

    def test_negative_depth_scale_impossible(self):
        for raw in (-100.0, -1.0, 0.0, 2.0, 100.0):
            scale = positive_depth_scale(torch.tensor(raw))
            self.assertGreater(float(scale), 0.0)
            self.assertGreaterEqual(float(scale), 0.25)
            self.assertLessEqual(float(scale), 4.0)

    def test_lift4d_missing_npz_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_lift4d_depth_prior("/definitely/missing/real_prior.npz", frame_num=25)

    def test_lift4d_frame_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            _write_prior(path, frame_num=25)
            with self.assertRaisesRegex(ValueError, "frame count mismatch"):
                load_lift4d_depth_prior(path, frame_num=24, min_stable_points=8)

    def test_object_poses_cam_not_used_for_depth_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            poison = np.full((25, 4, 4), np.nan, dtype=np.float32)
            _write_prior(
                path,
                object_poses_cam=poison,
                motion_confidence=np.full(25, -999.0, dtype=np.float32),
            )
            prior = load_lift4d_depth_prior(path, frame_num=25, min_stable_points=8)
            self.assertTrue(np.isfinite(prior.z).all())
            self.assertTrue((prior.z > 0).all())

    def test_lift4d_depth_loss_gradient(self):
        z_opt = torch.linspace(3.0, 2.8, 25, requires_grad=True)
        lift_z = torch.linspace(1.5, 1.2, 25)
        weights = torch.ones(25)
        fp_ray = torch.ones(25, 3)
        rotation = torch.eye(3).repeat(25, 1, 1).requires_grad_(True)
        log_scale = torch.zeros((), requires_grad=True)
        loss = lift4d_depth_trend_loss(
            z_opt, lift_z, weights, positive_depth_scale(log_scale), delta=0.03
        )
        loss.backward()
        self.assertIsNotNone(z_opt.grad)
        self.assertGreater(float(z_opt.grad.abs().sum()), 0.0)
        self.assertIsNone(fp_ray.grad)
        self.assertIsNone(rotation.grad)
        self.assertIsNotNone(log_scale.grad)

    def test_camera_z_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            _write_prior(path)
            prior = load_lift4d_depth_prior(path, frame_num=25, min_stable_points=8)
        fp = torch.tensor([[0.2, 0.1, 3.0]] * 25)
        optimized = ray_depth_translation(foundationpose_camera_rays(fp), torch.from_numpy(prior.z))
        self.assertTrue(torch.all(fp[:, 2] > 0))
        self.assertTrue(np.all(prior.center_cam[:, 2] > 0))
        self.assertTrue(torch.all(optimized[:, 2] > 0))

    def test_projection_pixel_drift(self):
        rng = np.random.default_rng(4)
        fp = np.column_stack(
            [rng.uniform(-0.4, 0.4, 20), rng.uniform(-0.3, 0.3, 20), rng.uniform(2.0, 4.0, 20)]
        )
        ray = fp / fp[:, 2:3]
        optimized = ray * rng.uniform(1.0, 6.0, 20)[:, None]
        K = np.repeat(np.eye(3)[None], 20, axis=0)
        K[:, 0, 0] = 1000.0
        K[:, 1, 1] = 995.0
        K[:, :2, 2] = [640.0, 360.0]
        error = np.linalg.norm(
            project_opencv_translation(optimized, K) - project_opencv_translation(fp, K), axis=1
        )
        self.assertLess(float(error.max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
