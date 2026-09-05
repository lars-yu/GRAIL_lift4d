import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


GEM_SMPL_ROOT = Path(__file__).resolve().parents[1] / "imports" / "GEM-SMPL"
if str(GEM_SMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(GEM_SMPL_ROOT))

from hmr4d.model.genmo.contact_guidance import (  # noqa: E402
    ContactGuidance,
    ContactGuidanceConfig,
    ROOT_VELOCITY_SLICE,
    allowed_channel_mask,
    arm_channel_slices,
    generate_sampling_noise,
    make_contact_guidance,
    root_candidate_improves_hold,
    should_use_root_fallback,
)
from grail.adapters.gem_smpl import (  # noqa: E402
    _calibrate_human_points,
    robust_human_depth_scale,
)
from grail.pipelines.genmo_contact_guidance import (  # noqa: E402
    OBJECT_TRAJECTORY_METHOD,
    _build_frozen_object_motion,
    _camera_aligned_ground_basis,
    _compose_anchored_object_trajectory,
)
from grail.optimization.motion_state import detect_object_motion  # noqa: E402


class GenmoContactGuidanceTests(unittest.TestCase):
    def test_top_view_keeps_camera_right_on_screen_right(self):
        angle = np.deg2rad(135.0)
        camera_to_global = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float32,
        )
        right, forward = _camera_aligned_ground_basis(camera_to_global)
        camera_right_global = camera_to_global[0, :]
        camera_forward_global = camera_to_global[2, :]
        self.assertGreater(float(np.dot(camera_right_global, right)), 0.999)
        self.assertGreater(float(np.dot(camera_forward_global, forward)), 0.999)
        self.assertAlmostEqual(float(np.dot(right, forward)), 0.0, places=6)

    def test_object_motion_is_first_pose_frozen_then_anchor_relative(self):
        raw = np.repeat(np.eye(4, dtype=np.float32)[None], 6, axis=0)
        raw[:, :3, 3] = np.array(
            [[1, 2, 4], [2, 2, 4], [3, 2, 4], [2, 0, 4], [0, 2, 4], [1, 1, 4]],
            dtype=np.float32,
        )
        motion_depth = np.array([9.8, 9.9, 10.0, 10.0, 10.2, 9.8], dtype=np.float32)
        poses, translations = _compose_anchored_object_trajectory(
            raw, motion_depth, 3
        )

        np.testing.assert_array_equal(poses[:3], np.repeat(raw[:1], 3, axis=0))
        target_depth = raw[0, 2, 3] + motion_depth - motion_depth[3]
        expected = raw[:, :3, 3] / raw[:, 2:3, 3] * target_depth[:, None]
        expected[:3] = raw[0, :3, 3]
        np.testing.assert_allclose(
            poses[:, :3, 3], expected
        )
        np.testing.assert_allclose(translations, expected)
        np.testing.assert_allclose(poses[3, :3, :3], raw[0, :3, :3])
        self.assertIn("relative_depth", OBJECT_TRAJECTORY_METHOD)

    def test_object_rotation_is_frozen_before_contact_then_uses_foundationpose(self):
        raw = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
        angles = np.linspace(0.0, 0.4, 5)
        raw[:, 0, 0] = np.cos(angles)
        raw[:, 0, 1] = -np.sin(angles)
        raw[:, 1, 0] = np.sin(angles)
        raw[:, 1, 1] = np.cos(angles)
        raw[:, 2, 3] = 3.0
        depth = np.repeat(2.0, 5)
        poses, _ = _compose_anchored_object_trajectory(raw, depth, 3)
        np.testing.assert_allclose(poses[2, :3, :3], raw[0, :3, :3], atol=1e-6)
        np.testing.assert_allclose(poses[3, :3, :3], raw[3, :3, :3], atol=1e-6)
        self.assertFalse(np.allclose(poses[3, :3, :3], poses[4, :3, :3]))

    def test_motion_onset_defines_contact_and_freeze_boundary(self):
        frame_num = 20
        centers = np.repeat(np.array([[0.0, 0.0, 2.0]]), frame_num, axis=0)
        masks = np.zeros((frame_num, 20, 30), dtype=bool)
        for frame in range(frame_num):
            x = 2 if frame < 10 else 2 + frame - 9
            masks[frame, 4:8, x : x + 4] = True
        state = detect_object_motion(
            centers,
            masks,
            smoothed_z=centers[:, 2],
            config={"baseline_frames": 5, "required_consecutive_mask_frames": 3},
        )
        self.assertEqual(state.move_start_frame, 10)
        self.assertTrue(state.static[:10].all())
        self.assertTrue(state.moving[10:].all())

    def test_new_trajectory_ignores_stale_optimizer_cache_and_is_read_only(self):
        raw = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
        raw[:, 2, 3] = 3.0
        prior = SimpleNamespace(stable_point_ids=np.arange(7))
        state = SimpleNamespace(
            move_start_frame=3,
            confidence=1.0,
            thresholds={},
        )
        prior.z = np.array([2, 2, 2, 2, 2.1], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            np.savez(output_dir / "optimized_object_motion.npz", poses_in_cam=np.nan)
            poses, diagnostics = _build_frozen_object_motion(
                raw, prior, state, "/tmp/prior.npz", output_dir
            )
            self.assertFalse(poses.flags.writeable)
            self.assertFalse(diagnostics["optimizer_run"])
            self.assertFalse(diagnostics["contact_losses_enabled"])
            with np.load(output_dir / "anchored_lift4d_object_motion.npz") as saved:
                self.assertEqual(str(saved["trajectory_method"]), OBJECT_TRAJECTORY_METHOD)
                self.assertTrue(np.isfinite(saved["poses_in_cam"]).all())

    def test_frame0_gt_depth_scale_is_robust_and_preserves_projection(self):
        predicted = np.array([4.0, 4.1, 3.9, 4.0, 4.2, 4.0])
        gt = predicted * 0.75
        gt[-1] = 20.0
        valid = np.ones_like(predicted, dtype=bool)
        scale, diagnostics = robust_human_depth_scale(
            predicted, gt, valid, min_samples=4
        )
        self.assertAlmostEqual(scale, 0.75, places=6)
        self.assertEqual(diagnostics["valid_surface_samples"], 6)

        points = torch.tensor([[1.0, -2.0, 4.0], [0.5, 1.0, 2.0]])
        calibrated = _calibrate_human_points(points, torch.zeros(3), scale)
        self.assertTrue(
            torch.allclose(points[:, :2] / points[:, 2:], calibrated[:, :2] / calibrated[:, 2:])
        )

    def test_verified_left_and_right_arm_channel_masks(self):
        expected = {
            "left": {
                "left_collar": (72, 78),
                "left_shoulder": (90, 96),
                "left_elbow": (102, 108),
                "left_wrist": (114, 120),
            },
            "right": {
                "right_collar": (78, 84),
                "right_shoulder": (96, 102),
                "right_elbow": (108, 114),
                "right_wrist": (120, 126),
            },
        }
        for hand, expected_ranges in expected.items():
            actual = {
                name: (channel_slice.start, channel_slice.stop)
                for name, channel_slice in arm_channel_slices(hand).items()
            }
            self.assertEqual(actual, expected_ranges)
            mask = allowed_channel_mask(hand, include_root=False)
            self.assertEqual(int(mask.sum()), 24)
            self.assertEqual(int(mask[ROOT_VELOCITY_SLICE].sum()), 0)
        overlap = allowed_channel_mask("left", include_root=False) * allowed_channel_mask(
            "right", include_root=False
        )
        self.assertEqual(int(overlap.sum()), 0)

    def test_arm_only_guidance_cannot_change_root_or_other_channels(self):
        reference = torch.zeros(1, 7, 151)
        x0 = reference.clone()
        x0[..., 72:78] = 0.2
        x0[..., ROOT_VELOCITY_SLICE] = 0.3

        def palm_position(motion, _hand):
            arm = motion[..., 72:78].sum(dim=-1, keepdim=True)
            root = motion[..., ROOT_VELOCITY_SLICE].sum(dim=-1, keepdim=True)
            return torch.cat((arm + root, arm * 0, root * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=3,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                reference_motion=reference,
                include_root=False,
                guidance_strength=0.7,
            ),
            palm_position,
        )
        guided = callback(x0, torch.tensor([25]))
        allowed = allowed_channel_mask("left", include_root=False).bool()
        self.assertTrue(torch.equal(guided[..., ~allowed], x0[..., ~allowed]))
        self.assertTrue(torch.equal(guided[..., ROOT_VELOCITY_SLICE], x0[..., ROOT_VELOCITY_SLICE]))
        self.assertFalse(torch.equal(guided[..., 72:78], x0[..., 72:78]))
        self.assertTrue(torch.isfinite(guided).all())

    def test_trajectory_contact_guidance_persists_after_contact(self):
        reference = torch.ones(1, 6, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=2,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                object_surface_targets=torch.zeros(6, 3),
                reference_motion=reference,
                contact_transition_frames=0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=0.1,
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([25]))
        self.assertTrue(torch.equal(guided[:, :2], reference[:, :2]))
        self.assertTrue(torch.all(guided[:, 2:, 72] < reference[:, 2:, 72]))

    def test_two_segment_targets_use_static_pre_contact_and_post_contact_sequence(self):
        reference = torch.ones(1, 6, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=3,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                pre_contact_surface_target=torch.zeros(3),
                post_contact_surface_targets=torch.zeros(3, 3),
                reference_motion=reference,
                contact_transition_frames=0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=0.1,
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([25]))
        self.assertTrue(torch.equal(guided[:, :3], reference[:, :3]))
        self.assertTrue(torch.all(guided[:, 3:, 72] < reference[:, 3:, 72]))

    def test_contact_frame_weight_is_reported_and_finite(self):
        reference = torch.ones(1, 5, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=2,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                reference_motion=reference,
                contact_frame_position_weight=1.25,
                w_reference=0.0,
                w_temporal=0.0,
            ),
            palm_position,
        )
        callback(reference, torch.tensor([25]))
        self.assertEqual(callback.step_diagnostics[-1]["contact_frame_position_weight"], 1.25)

    def test_root_fallback_only_when_arm_result_fails_a_gate(self):
        kwargs = {
            "reach_error_threshold": 0.03,
            "max_arm_rotation_change_limit_deg": 30.0,
        }
        self.assertFalse(should_use_root_fallback(0.03, 30.0, **kwargs))
        self.assertTrue(should_use_root_fallback(0.031, 10.0, **kwargs))
        self.assertTrue(should_use_root_fallback(0.01, 30.1, **kwargs))

    def test_worse_root_fallback_candidate_is_not_selected(self):
        self.assertFalse(
            root_candidate_improves_hold(0.033, 0.025, 0.082, 0.030, 0.028, 0.088)
        )
        self.assertTrue(
            root_candidate_improves_hold(0.033, 0.025, 0.082, 0.031, 0.024, 0.070)
        )

    def test_relative_velocity_weight_changes_post_contact_gradient(self):
        reference = torch.zeros(1, 6, 151)
        reference[:, :, 72] = torch.tensor([0.0, 0.0, 0.2, 0.4, 0.3, 0.7])

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        def guide(weight):
            callback = ContactGuidance(
                ContactGuidanceConfig(
                    contact_frame=2,
                    selected_hand="left",
                    object_surface_target=torch.zeros(3),
                    object_surface_targets=torch.zeros(6, 3),
                    reference_motion=reference,
                    contact_transition_frames=0,
                    w_reference=0.0,
                    w_temporal=0.0,
                    post_contact_relative_velocity_weight=weight,
                    post_contact_worst_frame_weight=0.0,
                    guidance_strength=0.1,
                    grad_clip_norm=100.0,
                ),
                palm_position,
            )
            return callback(reference, torch.tensor([25]))

        without_velocity = guide(0.0)
        with_velocity = guide(8.0)
        self.assertFalse(torch.allclose(without_velocity, with_velocity))

    def test_root_multiplier_only_scales_root_gradient(self):
        reference = torch.zeros(1, 5, 151)
        x0 = reference.clone()
        x0[..., 78:84] = 0.1
        x0[..., ROOT_VELOCITY_SLICE] = 0.1

        def palm_position(motion, _hand):
            arm = motion[..., 78:84].sum(dim=-1, keepdim=True)
            root = motion[..., ROOT_VELOCITY_SLICE].sum(dim=-1, keepdim=True)
            return torch.cat((arm, root, arm * 0), dim=-1)

        def guide(multiplier):
            callback = ContactGuidance(
                ContactGuidanceConfig(
                    contact_frame=2,
                    selected_hand="right",
                    object_surface_target=torch.zeros(3),
                    reference_motion=reference,
                    include_root=True,
                    guidance_strength=0.1,
                    grad_clip_norm=100.0,
                    root_guidance_multiplier=multiplier,
                ),
                palm_position,
            )
            return callback(x0, torch.tensor([49]))

        base = guide(1.0)
        boosted = guide(4.0)
        self.assertTrue(torch.allclose(base[..., 78:84], boosted[..., 78:84]))
        base_root_delta = base[..., ROOT_VELOCITY_SLICE] - x0[..., ROOT_VELOCITY_SLICE]
        boosted_root_delta = boosted[..., ROOT_VELOCITY_SLICE] - x0[..., ROOT_VELOCITY_SLICE]
        self.assertTrue(torch.allclose(boosted_root_delta, base_root_delta * 4.0))

    def test_sampling_noise_is_seed_reproducible(self):
        first = generate_sampling_noise((1, 8, 151), device="cpu", seed=42)
        second = generate_sampling_noise((1, 8, 151), device="cpu", seed=42)
        different = generate_sampling_noise((1, 8, 151), device="cpu", seed=43)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))

    def test_disabled_guidance_returns_original_callback_path(self):
        self.assertIsNone(make_contact_guidance(None, lambda motion, hand: motion[..., :3]))

        from hmr4d.network.genmo.genmo_diffusion import GENMODiffusion

        class AttrDict(dict):
            __getattr__ = dict.__getitem__

        class FakeDiffusion:
            def __init__(self):
                self.denoised_fn = "unset"

            def ddim_sample_loop_with_aux(self, model, shape, **kwargs):
                self.denoised_fn = kwargs["denoised_fn"]
                return {"sample": kwargs["noise"], "pred_x": kwargs["noise"]}

        module = GENMODiffusion.__new__(GENMODiffusion)
        torch.nn.Module.__init__(module)
        module.test_gen_only_diffusion = FakeDiffusion()
        module.denoiser = torch.nn.Identity()
        module.args = AttrDict(out_attr=["pred_x"])
        module.model_cfg = SimpleNamespace(
            diffusion=AttrDict(sampler="ddim", ddim_eta=0.0)
        )
        module.eval()
        motion = torch.zeros(1, 4, 151)
        inputs = {
            "length": torch.tensor([4]),
            "B": 1,
            "L": 4,
            "motion": motion,
            "f_cond": torch.zeros(1, 4, 1),
            "f_uncond": torch.zeros(1, 4, 1),
            "sample_indices_dict": {},
        }
        torch.manual_seed(123)
        expected_noise = torch.randn_like(motion)
        torch.manual_seed(123)
        actual = module.forward_test(inputs)
        self.assertIsNone(module.test_gen_only_diffusion.denoised_fn)
        self.assertTrue(torch.equal(actual["pred_x"], expected_noise))

    def test_nonfinite_guidance_fails_before_sampling_continues(self):
        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=0,
                selected_hand="right",
                object_surface_target=torch.zeros(3),
                reference_motion=torch.zeros(1, 1, 151),
            ),
            lambda motion, hand: motion[..., :3] * torch.tensor(float("nan")),
        )
        with self.assertRaisesRegex(FloatingPointError, "NaN/Inf"):
            callback(torch.zeros(1, 1, 151), torch.tensor([10]))

    def test_stop_after_stage35_never_enters_stage4(self):
        from grail.pipelines.recon_4dhoi import run_pipeline_steps

        called = []
        steps = [
            (3.5, "skip_step35", lambda videos, args: called.append("3.5")),
            (4, "skip_step4", lambda videos, args: called.append("4")),
        ]
        args = SimpleNamespace(
            skip_step35=False,
            skip_step4=False,
            genmo_contact_guidance_poc=True,
            stop_after_genmo_guidance=True,
        )
        run_pipeline_steps(["rand00033"], args, steps=steps)
        self.assertEqual(called, ["3.5"])


if __name__ == "__main__":
    unittest.main()
