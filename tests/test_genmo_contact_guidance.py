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
    group_channel_slices,
    leg_channel_slices,
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
from grail.optimization.finger_grasp import (  # noqa: E402
    signed_point_mesh_distance,
)


def _unit_cube_mesh():
    """Axis-aligned unit cube [-0.5,0.5]^3 as (verts[8,3], faces[12,3])."""
    v = torch.tensor([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=torch.float32)
    f = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=torch.long)
    return v, f


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
        candidate = raw[:, :3, 3] / raw[:, 2:3, 3] * target_depth[:, None]
        expected = raw[0, :3, 3] + candidate - candidate[3]
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
        np.testing.assert_allclose(poses[3], raw[0], atol=1e-6)
        expected_rotation = raw[4, :3, :3] @ raw[3, :3, :3].T @ raw[0, :3, :3]
        np.testing.assert_allclose(poses[4, :3, :3], expected_rotation, atol=1e-6)
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
                arm_gradient_smooth_kernel=1,  # test raw per-frame behavior
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([25]))
        self.assertTrue(torch.equal(guided[:, :2], reference[:, :2]))
        self.assertTrue(torch.all(guided[:, 2:, 72] < reference[:, 2:, 72]))

    def test_approach_smoothness_weight_accepted_and_finite(self):
        reference = torch.ones(1, 6, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=4,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                pre_contact_surface_target=torch.zeros(3),
                post_contact_surface_targets=torch.zeros(2, 3),
                reference_motion=reference,
                contact_transition_frames=4,
                approach_smoothness_weight=5.0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=0.1,
                arm_gradient_smooth_kernel=1,
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([25]))
        self.assertTrue(torch.isfinite(guided).all())

    def test_negative_approach_smoothness_weight_raises(self):
        reference = torch.ones(1, 6, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=4,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                pre_contact_surface_target=torch.zeros(3),
                post_contact_surface_targets=torch.zeros(2, 3),
                reference_motion=reference,
                contact_transition_frames=4,
                approach_smoothness_weight=-1.0,
                guidance_strength=0.1,
            ),
            palm_position,
        )
        with self.assertRaises(ValueError):
            callback(reference, torch.tensor([25]))

    def test_post_contact_hold_radius_has_zero_gradient_inside_two_cm(self):
        reference = torch.zeros(1, 6, 151)
        reference[:, :, 72] = torch.tensor([0.0, 0.0, 0.010, 0.015, 0.019, 0.018])

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
                post_contact_hold_radius=0.02,
                post_contact_relative_velocity_weight=0.0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=0.1,
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([25]))
        self.assertTrue(torch.allclose(guided, reference))

    def test_post_contact_hold_radius_pulls_back_only_outside_two_cm(self):
        reference = torch.zeros(1, 6, 151)
        reference[:, :, 72] = torch.tensor([0.0, 0.0, 0.010, 0.030, 0.040, 0.050])

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
                post_contact_hold_radius=0.02,
                post_contact_relative_velocity_weight=0.0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=0.1,
                grad_clip_norm=100.0,
                arm_gradient_smooth_kernel=1,  # test raw per-frame behavior
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([25]))
        self.assertTrue(torch.allclose(guided[:, :3], reference[:, :3]))
        self.assertTrue(torch.all(guided[:, 3:, 72] < reference[:, 3:, 72]))

    def test_arm_guidance_remains_active_with_small_cap_at_final_ddim_step(self):
        # v22: at t=0 the arm cap drops to arm_final_update_cap, but guidance
        # stays active and the inner loop accumulates several small steps.
        reference = torch.ones(1, 4, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=1,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                object_surface_targets=torch.zeros(4, 3),
                reference_motion=reference,
                post_contact_hold_radius=0.02,
                diffusion_steps=50,
                arm_max_update_cap=0.20,
                arm_final_update_cap=0.04,
                final_inner_steps=5,
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([0]))
        frame_update = torch.linalg.vector_norm(guided - reference, dim=-1)
        diag = callback.step_diagnostics[-1]
        self.assertGreater(float(frame_update.max()), 0.0)
        # Each inner step is capped at arm_final_update_cap ...
        self.assertLessEqual(diag["arm_applied_update_norm"], 0.04 + 1e-6)
        self.assertAlmostEqual(diag["scheduled_update_cap"], 0.04, places=5)
        # ... but the inner loop (up to t0_max_inner_steps=40) accumulates more
        # displacement, bounded by that many capped steps (v25 #3).
        self.assertLessEqual(float(frame_update.max()), 40 * 0.04 + 1e-4)
        self.assertGreaterEqual(diag["inner_iterations"], 1)

    def test_guidance_update_cap_is_applied_independently_per_frame(self):
        reference = torch.zeros(1, 3, 151)
        reference[..., 72] = 1.0

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=0,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                object_surface_targets=torch.zeros(3, 3),
                reference_motion=reference,
                contact_transition_frames=0,
                post_contact_hold_radius=0.0,
                post_contact_relative_velocity_weight=0.0,
                post_contact_worst_frame_weight=0.0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=100.0,
                grad_clip_norm=100.0,
                arm_max_update_cap=0.10,
                arm_final_update_cap=0.02,
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([49]))
        frame_update = torch.linalg.vector_norm(guided - reference, dim=-1)
        self.assertTrue(
            torch.allclose(frame_update, torch.full_like(frame_update, 0.10))
        )

    def test_root_guidance_keeps_nonzero_final_scale(self):
        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=0,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                reference_motion=torch.zeros(1, 1, 151),
                root_guidance_final_scale=0.10,
            ),
            lambda motion, _hand: motion[..., :3],
        )
        self.assertAlmostEqual(
            float(callback._time_scale(torch.tensor([0]), root=False)), 1.0
        )
        self.assertAlmostEqual(
            float(callback._time_scale(torch.tensor([0]), root=True)), 0.10
        )

    def test_inner_step_schedule_matches_ddim_phase(self):
        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=0,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                reference_motion=torch.zeros(1, 1, 151),
                late_inner_steps=2,
                final_inner_steps=5,
                inner_step_late_threshold=4,
                diffusion_steps=50,
            ),
            lambda motion, _hand: motion[..., :3],
        )
        # v25 (#3): t=0 runs up to t0_max_inner_steps (default 40) to reach
        # convergence, not the fixed final_inner_steps; earlier phases unchanged.
        self.assertEqual(callback._inner_step_schedule(torch.tensor([0])), 40)
        self.assertEqual(callback._inner_step_schedule(torch.tensor([3])), 2)
        self.assertEqual(callback._inner_step_schedule(torch.tensor([4])), 2)
        self.assertEqual(callback._inner_step_schedule(torch.tensor([10])), 1)
        self.assertEqual(callback._inner_step_schedule(torch.tensor([49])), 1)

    def test_final_step_inner_loop_moves_more_than_single_step(self):
        reference = torch.ones(1, 4, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        def displacement(timestep):
            callback = ContactGuidance(
                ContactGuidanceConfig(
                    contact_frame=1,
                    selected_hand="left",
                    object_surface_target=torch.zeros(3),
                    object_surface_targets=torch.zeros(4, 3),
                    reference_motion=reference,
                    post_contact_hold_radius=0.0,
                    diffusion_steps=50,
                    arm_max_update_cap=0.20,
                    arm_final_update_cap=0.04,
                ),
                palm_position,
            )
            guided = callback(reference, torch.tensor([timestep]))
            return float(torch.linalg.vector_norm(guided - reference, dim=-1).max())

        # v25 (#3): t=0 runs the full inner loop (up to 40, converging), an early
        # DDIM step runs a single inner iteration, so t=0 moves much more.
        single = displacement(10)
        multi = displacement(0)
        self.assertGreater(single, 0.0)
        self.assertGreater(multi, single * 1.5)

    def test_arm_and_root_caps_are_independent(self):
        reference = torch.zeros(1, 3, 151)
        x0 = reference.clone()
        x0[..., 72] = 1.0
        x0[..., ROOT_VELOCITY_SLICE] = 1.0

        def palm_position(motion, _hand):
            arm = motion[..., 72:73]
            root = motion[..., ROOT_VELOCITY_SLICE].sum(dim=-1, keepdim=True)
            return torch.cat((arm + root, arm * 0, root * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=0,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                object_surface_targets=torch.zeros(3, 3),
                reference_motion=reference,
                include_root=True,
                contact_transition_frames=0,
                post_contact_hold_radius=0.0,
                post_contact_relative_velocity_weight=0.0,
                post_contact_worst_frame_weight=0.0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=100.0,
                grad_clip_norm=100.0,
                root_guidance_final_scale=1.0,
                arm_max_update_cap=0.20,
                arm_final_update_cap=0.04,
                root_max_update_cap=0.05,
                root_final_update_cap=0.02,
            ),
            palm_position,
        )
        callback(x0, torch.tensor([30]))
        diag = callback.step_diagnostics[-1]
        self.assertLessEqual(diag["arm_applied_update_norm"], 0.20 + 1e-6)
        self.assertLessEqual(diag["root_applied_update_norm"], 0.05 + 1e-6)
        # The arm cap is larger, so the arm update exceeds the root cap.
        self.assertGreater(diag["arm_applied_update_norm"], 0.05 + 1e-6)

    def test_root_gradient_is_temporally_smoothed(self):
        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=0,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                reference_motion=torch.zeros(1, 20, 151),
                include_root=True,
                contact_transition_frames=0,
                root_gradient_smooth_kernel=9,
                root_guidance_final_scale=1.0,
                root_guidance_multiplier=1.0,
            ),
            lambda motion, _hand: motion[..., :3],
        )
        spike = torch.zeros(1, 20, 3)
        spike[0, 10, :] = 1.0
        smoothed = callback._smooth_root_gradient(spike, torch.tensor([0]))
        input_tv = float((spike[:, 1:] - spike[:, :-1]).abs().sum())
        output_tv = float((smoothed[:, 1:] - smoothed[:, :-1]).abs().sum())
        self.assertGreater(float(smoothed.abs().sum()), 0.0)
        self.assertLess(output_tv, input_tv)

    def test_inner_line_search_rejects_nonimproving_update(self):
        # Palm already at the target -> zero gradient -> nothing should be
        # accepted, so the motion is returned unchanged.
        reference = torch.zeros(1, 3, 151)

        def palm_position(motion, _hand):
            value = motion[..., 72:73]
            return torch.cat((value, value * 0, value * 0), dim=-1)

        callback = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=0,
                selected_hand="left",
                object_surface_target=torch.zeros(3),
                object_surface_targets=torch.zeros(3, 3),
                reference_motion=reference,
                contact_transition_frames=0,
                post_contact_hold_radius=0.0,
                w_reference=0.0,
                w_temporal=0.0,
                guidance_strength=100.0,
                inner_line_search=True,
                final_inner_steps=5,
            ),
            palm_position,
        )
        guided = callback(reference, torch.tensor([0]))
        self.assertTrue(torch.allclose(guided, reference))
        self.assertEqual(callback.step_diagnostics[-1]["inner_candidate_accepted"], 0)

    # ------------------------------------------------------------------
    # v23 whole-body guidance
    # ------------------------------------------------------------------
    @staticmethod
    def _whole_body_kin(motion, _hand):
        # palm from left_collar x-channel (72); root position from the root
        # velocity channels (148:151); the four feet Y from the ankle/foot
        # channels so leg/foot gradients are exercised.
        palm = torch.cat((motion[..., 72:73], motion[..., 73:74], motion[..., 74:75]), dim=-1)
        root = motion[..., 148:151]

        def on_y(v):
            return torch.cat((v * 0, v, v * 0), dim=-1)

        feet = torch.stack(
            (
                on_y(motion[..., 36:37]),  # left_ankle
                on_y(motion[..., 54:55]),  # left_foot
                on_y(motion[..., 42:43]),  # right_ankle
                on_y(motion[..., 60:61]),  # right_foot
            ),
            dim=-2,
        )
        return {"palm_position": palm, "root_position": root, "foot_positions": feet}

    def _whole_body_config(self, **overrides):
        base = dict(
            contact_frame=3,
            selected_hand="left",
            object_surface_target=torch.zeros(3),
            reference_motion=torch.zeros(1, 6, 151),
            include_root=True,
            include_torso=True,
            include_legs=True,
            diffusion_steps=50,
        )
        base.update(overrides)
        return ContactGuidanceConfig(**base)

    def test_signed_point_mesh_distance_sign_and_magnitude(self):
        v, f = _unit_cube_mesh()
        pts = torch.tensor([
            [0.0, 0.0, 0.0],    # center: inside, ~0.5 from each face
            [0.9, 0.0, 0.0],    # outside +x by 0.4
            [0.0, 0.0, -0.7],   # outside -z by 0.2
        ], dtype=torch.float32)
        sd = signed_point_mesh_distance(pts, v, f, candidate_faces=12)
        self.assertLess(float(sd[0]), 0.0)                     # inside -> negative
        self.assertAlmostEqual(float(sd[0]), -0.5, delta=0.05)  # ~0.5 deep
        self.assertGreater(float(sd[1]), 0.0)                  # outside -> positive
        self.assertAlmostEqual(float(sd[1]), 0.4, delta=0.05)
        self.assertGreater(float(sd[2]), 0.0)
        self.assertAlmostEqual(float(sd[2]), 0.2, delta=0.05)

    def test_signed_point_mesh_distance_is_differentiable(self):
        v, f = _unit_cube_mesh()
        p = torch.tensor([[0.9, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        sd = signed_point_mesh_distance(p, v, f, candidate_faces=12)
        sd.sum().backward()
        self.assertIsNotNone(p.grad)
        self.assertGreater(float(p.grad.abs().sum()), 0.0)

    def test_gaussian_time_smooth_reduces_frame_to_frame_jitter(self):
        # A spiky per-frame signal must have its temporal variation reduced,
        # which is what tames post-contact hand jitter under a large cap.
        g = torch.zeros(1, 20, 6)
        g[0, ::2] = 1.0  # alternate frames -> maximal jitter
        smoothed = ContactGuidance._gaussian_time_smooth(g, 7)
        raw_tv = float((g[:, 1:] - g[:, :-1]).abs().sum())
        smooth_tv = float((smoothed[:, 1:] - smoothed[:, :-1]).abs().sum())
        self.assertLess(smooth_tv, raw_tv)
        self.assertGreater(float(smoothed.abs().sum()), 0.0)
        # kernel<=1 is a no-op
        self.assertTrue(torch.equal(ContactGuidance._gaussian_time_smooth(g, 1), g))

    def test_rigid_depth_placement_translates_palm_without_scaling(self):
        # A pure translation offset must shift the decoded palm by exactly the
        # same vector at every frame (no scaling): relative geometry preserved.
        offset = torch.tensor([0.1, 0.0, -1.0])
        reference = torch.zeros(1, 4, 151)
        reference[..., 72] = torch.tensor([0.0, 0.2, 0.4, 0.6])

        def palm_position(motion, _hand):
            v = motion[..., 72:73]
            return torch.cat((v, v * 0, v * 0), dim=-1)

        def build(translation):
            cb = ContactGuidance(
                ContactGuidanceConfig(
                    contact_frame=1,
                    selected_hand="left",
                    object_surface_target=torch.zeros(3),
                    reference_motion=reference,
                    human_translation_global=translation,
                ),
                palm_position,
            )
            _, comps = cb._compute_guidance_loss(reference, torch.tensor([10]))
            return comps["palm"].detach()

        base_palm = build(None)
        shifted_palm = build(offset)
        self.assertTrue(
            torch.allclose(shifted_palm, base_palm + offset.reshape(1, 1, 3), atol=1e-6)
        )

    def test_effective_human_depth_scale_default_is_one(self):
        cfg = ContactGuidanceConfig(
            contact_frame=0,
            selected_hand="left",
            object_surface_target=torch.zeros(3),
            reference_motion=torch.zeros(1, 1, 151),
        )
        self.assertEqual(cfg.human_depth_scale, 1.0)

    def test_group_channel_slices_cover_arm_torso_legs_by_name(self):
        slices = group_channel_slices("left")
        # arm: left_collar(k12)=72:78, shoulder(k15)=90:96, elbow(k17)=102:108, wrist(k19)=114:120
        self.assertEqual(slices["arm"], [slice(72, 78), slice(90, 96), slice(102, 108), slice(114, 120)])
        # torso: spine1(k2)=12:18, spine2(k5)=30:36, spine3(k8)=48:54
        self.assertEqual(slices["torso"], [slice(12, 18), slice(30, 36), slice(48, 54)])
        # root velocity slice
        self.assertEqual(slices["root"], [ROOT_VELOCITY_SLICE])
        # legs include both left and right hip/knee/ankle/foot
        self.assertEqual(len(slices["legs"]), 8)

    def test_leg_channel_slices_left_and_right(self):
        legs = leg_channel_slices("left")
        # left_hip(k0)=0:6, left_knee(k3)=18:24, left_ankle(k6)=36:42, left_foot(k9)=54:60
        self.assertEqual(legs["left_hip"], slice(0, 6))
        self.assertEqual(legs["left_knee"], slice(18, 24))
        self.assertEqual(legs["left_ankle"], slice(36, 42))
        self.assertEqual(legs["left_foot"], slice(54, 60))
        # right_hip(k1)=6:12, right_ankle(k7)=42:48, right_foot(k10)=60:66
        self.assertEqual(legs["right_hip"], slice(6, 12))
        self.assertEqual(legs["right_ankle"], slice(42, 48))
        self.assertEqual(legs["right_foot"], slice(60, 66))

    def test_arm_on_root_legs_off_at_final_ddim_step(self):
        cb = ContactGuidance(
            self._whole_body_config(root_leg_fade_fraction=0.30),
            self._whole_body_kin,
        )
        self.assertAlmostEqual(float(cb._time_scale(torch.tensor([0]), root=False)), 1.0)
        self.assertAlmostEqual(float(cb._root_leg_time_scale(torch.tensor([0]))), 0.0)
        self.assertGreater(float(cb._root_leg_time_scale(torch.tensor([25]))), 0.0)

    def test_root_offset_ramp_starts_zero_reaches_target_at_contact(self):
        cb = ContactGuidance(
            self._whole_body_config(contact_frame=5, contact_transition_frames=4,
                                    reference_motion=torch.zeros(1, 10, 151)),
            self._whole_body_kin,
        )
        ramp = cb._root_offset_ramp(10, torch.device("cpu"), torch.float32)
        self.assertAlmostEqual(float(ramp[0]), 0.0)
        self.assertAlmostEqual(float(ramp[5]), 1.0)
        self.assertTrue(torch.allclose(ramp[5:], torch.ones_like(ramp[5:])))

    def test_root_vertical_lock_penalizes_height_change(self):
        cb = ContactGuidance(
            self._whole_body_config(w_root_vertical_lock=1.0),
            self._whole_body_kin,
        )
        flat = torch.zeros(1, 6, 151)  # root Y (channel 149) constant -> no height change
        _, comps_flat = cb._compute_guidance_loss(flat, torch.tensor([10]))
        raised = torch.zeros(1, 6, 151)
        raised[..., 149] = torch.linspace(0.0, 0.3, 6)  # root Y rises over time
        _, comps_raised = cb._compute_guidance_loss(raised, torch.tensor([10]))
        self.assertAlmostEqual(float(comps_flat["root_vertical_loss"]), 0.0)
        self.assertGreater(float(comps_raised["root_vertical_loss"]), 0.0)

    def test_support_foot_loss_zero_when_static_and_ignores_swing_foot(self):
        probs = torch.zeros(6, 4)
        probs[:, 0] = 1.0  # left ankle is the support foot
        probs[:, 1] = 1.0  # left foot is the support foot
        cb = ContactGuidance(
            self._whole_body_config(foot_contact_probs=probs, ground_height=0.0),
            self._whole_body_kin,
        )
        static = torch.zeros(1, 6, 151)  # all feet channels constant -> no sliding
        _, comps_static = cb._compute_guidance_loss(static, torch.tensor([10]))
        self.assertAlmostEqual(float(comps_static["support_foot_loss"]), 0.0)

        swing = torch.zeros(1, 6, 151)
        swing[..., 42] = torch.linspace(0.0, 0.5, 6)  # move RIGHT ankle (swing, prob 0)
        _, comps_swing = cb._compute_guidance_loss(swing, torch.tensor([10]))
        self.assertAlmostEqual(float(comps_swing["support_foot_loss"]), 0.0)

        support = torch.zeros(1, 6, 151)
        support[..., 36] = torch.linspace(0.0, 0.5, 6)  # move LEFT ankle (support, prob 1)
        _, comps_support = cb._compute_guidance_loss(support, torch.tensor([10]))
        self.assertGreater(float(comps_support["support_foot_loss"]), 0.0)

    def test_group_update_caps_are_independent(self):
        cb = ContactGuidance(
            self._whole_body_config(
                contact_frame=0,
                reference_motion=torch.zeros(1, 3, 151),
                contact_transition_frames=0,
                post_contact_hold_radius=0.0,
                post_contact_relative_velocity_weight=0.0,
                post_contact_worst_frame_weight=0.0,
                w_reference=0.0,
                w_temporal=0.0,
                w_root_vertical_lock=0.0,
                leg_reference_weight=0.0,
                arm_reference_weight=0.0,
                arm_smoothness_weight=0.0,
                guidance_strength=100.0,
                grad_clip_norm=100.0,
                arm_max_update_cap=0.20,
                arm_final_update_cap=0.04,
                torso_max_update_cap=0.05,
                leg_max_update_cap=0.05,
                root_max_update_cap=0.05,
                object_surface_targets=torch.zeros(3, 3),
            ),
            self._whole_body_kin,
        )
        x0 = torch.zeros(1, 3, 151)
        x0[..., 72] = 1.0   # arm channel -> palm
        x0[..., 148:151] = 1.0  # root channels
        x0[..., 36] = 1.0   # left ankle (leg)
        cb(x0, torch.tensor([30]))
        diag = cb.step_diagnostics[-1]
        self.assertLessEqual(diag["arm_applied_update_norm"], 0.20 + 1e-6)
        self.assertLessEqual(diag["leg_applied_update_norm"], 0.05 + 1e-6)
        self.assertLessEqual(diag["root_applied_update_norm"], 0.05 + 1e-6)
        self.assertLessEqual(diag["torso_applied_update_norm"], 0.05 + 1e-6)

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
                arm_gradient_smooth_kernel=1,  # test raw per-frame behavior
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
                    # Keep the per-frame caps out of the way so the velocity
                    # term's effect on the gradient magnitude is visible (the
                    # gradient direction is identical, so a saturated cap would
                    # normalize both weights to the same update).
                    arm_max_update_cap=100.0,
                    arm_final_update_cap=100.0,
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
                    # Disable the caps and the shared line-search scale so this
                    # test isolates pure root-gradient multiplier linearity.
                    arm_max_update_cap=100.0,
                    arm_final_update_cap=100.0,
                    root_max_update_cap=100.0,
                    root_final_update_cap=100.0,
                    root_gradient_smooth_kernel=0,
                    inner_line_search=False,
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

    # ------------------------------------------------------------------
    # v24 action/contact fix
    # ------------------------------------------------------------------
    def _wb(self, reference, **ov):
        return ContactGuidance(
            self._whole_body_config(reference_motion=reference, **ov), self._whole_body_kin
        )

    def test_v24_default_contact_target_is_true_surface_point(self):
        # standoff 0 => the palm-point guidance target IS the true surface point.
        import inspect
        from grail.adapters.gem_smpl import run_contact_guided_genmo
        sig = inspect.signature(run_contact_guided_genmo)
        self.assertEqual(sig.parameters["contact_standoff_m"].default, 0.0)

    def test_v24_default_clearance_not_greater_than_hold_radius(self):
        cfg = ContactGuidanceConfig(
            contact_frame=1, selected_hand="left",
            object_surface_target=torch.zeros(3), reference_motion=torch.zeros(1, 3, 151),
        )
        self.assertLessEqual(cfg.contact_min_clearance, cfg.post_contact_hold_radius)
        with self.assertRaises(ValueError):
            ContactGuidanceConfig(
                contact_frame=1, selected_hand="left",
                object_surface_target=torch.zeros(3), reference_motion=torch.zeros(1, 3, 151),
                contact_min_clearance=0.05, post_contact_hold_radius=0.02,
            )

    def test_v24_initial_and_final_error_share_one_target(self):
        ref = torch.zeros(1, 6, 151)
        P = torch.tensor([0.1, 0.2, 0.3])
        cb = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=3, selected_hand="left",
                object_surface_target=P, reference_motion=ref,
                pre_contact_surface_target=P,
                post_contact_surface_targets=P.reshape(1, 3).repeat(3, 1),
                contact_transition_frames=2, guidance_strength=0.1,
            ),
            lambda m, _h: torch.cat((m[..., 72:73], m[..., 73:74], m[..., 74:75]), dim=-1),
        )
        _, comps = cb._compute_guidance_loss(ref, torch.tensor([10]))
        ts = comps["target_seq"]
        self.assertTrue(torch.allclose(ts[:, 2], P.reshape(1, 3), atol=1e-6))  # pre-contact
        self.assertTrue(torch.allclose(ts[:, 3], P.reshape(1, 3), atol=1e-6))  # post-contact[0]

    def test_v24_root_target_is_reference_root_plus_residual(self):
        T, frame = 8, 5
        ref = torch.zeros(1, T, 151)
        ref[..., 148] = torch.linspace(0.0, 1.0, T)  # reference root X walks forward
        delta = torch.tensor([0.3, 0.0, 0.0])
        cb = self._wb(ref, contact_frame=frame, contact_transition_frames=3,
                      root_target_delta_global=delta, w_root_target=1.0)
        ramp = cb._root_offset_ramp(T, torch.device("cpu"), torch.float32)
        cand = ref.clone()
        cand[..., 148] = ref[..., 148] + ramp * delta[0]  # ref_root + residual ramp
        _, comps = cb._compute_guidance_loss(cand, torch.tensor([10]))
        self.assertLess(float(comps["root_target_loss"]), 1e-6)
        _, comps0 = cb._compute_guidance_loss(ref, torch.tensor([10]))  # no residual
        self.assertGreater(float(comps0["root_target_loss"]), 1e-4)

    def test_v24_root_residual_zero_before_approach_start(self):
        T, frame, trans = 10, 8, 3  # approach_start = 5
        ref = torch.zeros(1, T, 151)
        cb = self._wb(ref, contact_frame=frame, contact_transition_frames=trans,
                      root_target_delta_global=torch.tensor([0.5, 0.0, 0.0]))
        ramp = cb._root_offset_ramp(T, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.allclose(ramp[: frame - trans], torch.zeros(frame - trans)))
        self.assertAlmostEqual(float(ramp[frame]), 1.0)

    def test_v24_root_residual_constant_and_velocity_zero_after_contact(self):
        T, frame, trans = 10, 5, 3
        ref = torch.zeros(1, T, 151)
        ref[..., 148] = torch.linspace(0.0, 2.0, T)
        delta = torch.tensor([0.4, 0.0, 0.0])
        cb = self._wb(ref, contact_frame=frame, contact_transition_frames=trans,
                      root_target_delta_global=delta, w_root_velocity=1.0)
        ramp = cb._root_offset_ramp(T, torch.device("cpu"), torch.float32)
        desired_x = ref[..., 148] + ramp * delta[0]
        dv = desired_x[:, 1:] - desired_x[:, :-1]
        rv = ref[..., 148][:, 1:] - ref[..., 148][:, :-1]
        # After contact the residual is constant, so desired velocity == reference.
        self.assertTrue(torch.allclose(dv[:, frame:], rv[:, frame:], atol=1e-6))
        cand = ref.clone(); cand[..., 148] = desired_x  # follow desired exactly
        _, comps = cb._compute_guidance_loss(cand, torch.tensor([10]))
        self.assertLess(float(comps["root_velocity_loss"]), 1e-6)

    def test_v24_reference_vertical_motion_preserved(self):
        T = 6
        ref = torch.zeros(1, T, 151)
        ref[..., 149] = torch.linspace(0.0, 0.3, T)  # reference root Y bends down/up
        cb = self._wb(ref, w_root_vertical_lock=1.0)
        _, comps = cb._compute_guidance_loss(ref, torch.tensor([10]))  # candidate == reference
        self.assertLess(float(comps["root_vertical_loss"]), 1e-6)
        cand = ref.clone(); cand[..., 149] = cand[..., 149] + torch.linspace(0.0, 0.2, T)
        _, comps2 = cb._compute_guidance_loss(cand, torch.tensor([10]))
        self.assertGreater(float(comps2["root_vertical_loss"]), 1e-4)

    def test_v24_arm_active_at_final_ddim_step(self):
        cb = self._wb(torch.zeros(1, 6, 151))
        self.assertAlmostEqual(float(cb._group_time_scale("arm", torch.tensor([0]))), 1.0)
        self.assertAlmostEqual(float(cb._group_time_scale("torso", torch.tensor([0]))), 1.0)

    def test_v24_root_and_legs_off_at_final_ddim_step(self):
        cb = self._wb(torch.zeros(1, 6, 151))
        self.assertAlmostEqual(float(cb._group_time_scale("root", torch.tensor([0]))), 0.0)
        self.assertAlmostEqual(float(cb._group_time_scale("legs", torch.tensor([0]))), 0.0)

    def test_v24_torso_reference_and_temporal_losses_present(self):
        ref = torch.zeros(1, 6, 151)
        cb = self._wb(ref, include_torso=True, torso_reference_weight=1.0)
        torso_sl = group_channel_slices("left")["torso"][0]
        cand = ref.clone()
        cand[..., torso_sl.start:torso_sl.stop] = 0.5  # torso deviates from reference
        _, comps = cb._compute_guidance_loss(cand, torch.tensor([10]))
        self.assertGreater(float(comps["torso_reference_loss"]), 0.0)
        self.assertIn("torso_smoothness_loss", comps)

    def test_v24_ground_contact_uses_reference_height_not_global_min(self):
        T = 6
        probs = torch.zeros(T, 4); probs[:, 1] = 1.0  # left foot planted
        ref = torch.zeros(1, T, 151); ref[..., 54] = 0.2  # ref left-foot Y elevated, constant
        cb = self._wb(ref, foot_contact_probs=probs, ground_height=0.0)
        _, comps = cb._compute_guidance_loss(ref, torch.tensor([10]))  # candidate == reference
        self.assertLess(float(comps["ground_contact_loss"]), 1e-6)  # 0.2 != global-min 0, yet fine
        cand = ref.clone(); cand[..., 54] = 0.0
        _, comps2 = cb._compute_guidance_loss(cand, torch.tensor([10]))
        self.assertGreater(float(comps2["ground_contact_loss"]), 1e-4)

    def test_v24_swing_foot_not_locked_by_support_loss(self):
        T = 6
        probs = torch.ones(T, 4)  # all feet reported high-contact
        ref = torch.zeros(1, T, 151); ref[..., 60] = torch.linspace(0.0, 0.5, T)  # right foot swings
        cb = self._wb(ref, foot_contact_probs=probs, ground_height=0.0)
        _, comps = cb._compute_guidance_loss(ref, torch.tensor([10]))  # follows reference swing
        self.assertLess(float(comps["support_foot_loss"]), 1e-6)

    def test_v24_candidate_metrics_report_total_and_safety(self):
        cb = self._wb(torch.zeros(1, 6, 151), foot_contact_probs=torch.zeros(6, 4))
        m = cb._candidate_metrics(
            torch.zeros(1, 6, 151), torch.zeros(1, 6, 151), torch.tensor([10])
        )
        for key in ("total_loss", "contact_error_m", "max_palm_step_m", "foot_slide_m"):
            self.assertIn(key, m)

    def test_v24_translation_is_not_default_follow_mode(self):
        import inspect
        from grail.adapters.gem_smpl import run_contact_guided_genmo
        d = inspect.signature(run_contact_guided_genmo).parameters[
            "post_contact_follow_mode"
        ].default
        self.assertEqual(d, "pose")
        self.assertNotEqual(d, "translation")

    def test_v24_finger_grasp_default_disabled(self):
        cli = (Path(__file__).resolve().parents[1] / "grail" / "pipelines" / "recon_4dhoi.py").read_text()
        block = cli[cli.index('"--genmo-finger-grasp"'):]
        block = block[: block.index(")")]
        self.assertIn("default=False", block)
        self.assertFalse(bool(getattr(SimpleNamespace(), "genmo_finger_grasp", False)))

    def test_v24_renderer_does_not_rotate_hand_pose_channels(self):
        src = (Path(__file__).resolve().parents[1] / "tools" / "render_genmo_grail_hybrid.py").read_text()
        self.assertNotIn("148:151] @", src)  # no rotation of the 165-dim right-hand pose

    def test_v24_camera_to_world_preserves_hand_object_distance(self):
        rng = np.random.RandomState(0)
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        c2w_R, c2w_t = Q, rng.randn(3)
        hand_cam, obj_cam = rng.randn(3), rng.randn(3)
        # human and object are carried to world by the SAME camera->world SE(3).
        hand_w = hand_cam @ c2w_R.T + c2w_t
        obj_w = obj_cam @ c2w_R.T + c2w_t
        d_cam = float(np.linalg.norm(hand_cam - obj_cam))
        d_w = float(np.linalg.norm(hand_w - obj_w))
        self.assertLess(abs(d_cam - d_w), 1e-5)
        src = (Path(__file__).resolve().parents[1] / "tools" / "render_genmo_grail_hybrid.py").read_text()
        self.assertIn("t_cam @ c2w_R.T + c2w_t", src)  # object path
        self.assertIn("trans @ R.T + t", src)          # human path, same (R, t)

    def test_v24_sampling_noise_reproducible_same_seed(self):
        a = generate_sampling_noise((2, 3, 151), device=torch.device("cpu"), seed=7)
        b = generate_sampling_noise((2, 3, 151), device=torch.device("cpu"), seed=7)
        self.assertTrue(torch.equal(a, b))

    def test_v24_preserves_v22_v23_public_symbols(self):
        from hmr4d.model.genmo.contact_guidance import (
            allowed_channel_mask, group_channel_slices, ROOT_VELOCITY_SLICE,
        )
        from grail.optimization.finger_grasp import signed_point_mesh_distance
        self.assertTrue(callable(signed_point_mesh_distance))
        self.assertEqual(ROOT_VELOCITY_SLICE, slice(148, 151))

    # ------------------------------------------------------------------
    # v25 contact-closure fix
    # ------------------------------------------------------------------
    def _closure_cb(self, **ov):
        reference = torch.zeros(1, 6, 151)
        reference[:, :, 72] = torch.tensor([0.0, 0.0, 0.0, 0.05, 0.05, 0.05])

        def palm(m, _h):
            v = m[..., 72:73]
            return torch.cat((v, v * 0, v * 0), dim=-1)

        base = dict(
            contact_frame=3, selected_hand="left",
            object_surface_target=torch.zeros(3), object_surface_targets=torch.zeros(6, 3),
            reference_motion=reference, contact_transition_frames=2,
            post_contact_hold_radius=0.02, w_reference=0.0, w_temporal=0.0,
            arm_gradient_smooth_kernel=1,
        )
        base.update(ov)
        return ContactGuidance(ContactGuidanceConfig(**base), palm), reference

    def test_v25_contact_frame_loss_is_independent_term(self):
        cb, reference = self._closure_cb()
        _, comps = cb._compute_guidance_loss(reference, torch.tensor([10]))
        self.assertIn("contact_frame_position_loss", comps)
        # contact frame (3) sits 5 cm from the surface (> 2 cm hold) -> nonzero.
        self.assertGreater(float(comps["contact_frame_position_loss"]), 0.0)

    def test_v25_contact_frame_term_respects_hold_dead_zone(self):
        # When the contact frame is already within the 2 cm hold, the independent
        # term is zero (it must not fight GENMO inside the dead zone).
        reference = torch.zeros(1, 6, 151)
        reference[:, :, 72] = torch.tensor([0.0, 0.0, 0.0, 0.010, 0.010, 0.010])

        def palm(m, _h):
            v = m[..., 72:73]
            return torch.cat((v, v * 0, v * 0), dim=-1)

        cb = ContactGuidance(
            ContactGuidanceConfig(
                contact_frame=3, selected_hand="left",
                object_surface_target=torch.zeros(3), object_surface_targets=torch.zeros(6, 3),
                reference_motion=reference, contact_transition_frames=2,
                post_contact_hold_radius=0.02,
            ),
            palm,
        )
        _, comps = cb._compute_guidance_loss(reference, torch.tensor([10]))
        self.assertAlmostEqual(float(comps["contact_frame_position_loss"]), 0.0)

    def test_v25_t0_inner_loop_runs_up_to_40(self):
        cb, _ = self._closure_cb(final_inner_steps=5, t0_max_inner_steps=40)
        self.assertEqual(int(cb._inner_step_schedule(torch.tensor([0]))), 40)

    def test_v25_inner_after_is_best_not_last(self):
        cb, reference = self._closure_cb(final_inner_steps=8, guidance_strength=0.2)
        cb(reference, torch.tensor([0]))
        diag = cb.step_diagnostics[-1]
        # The returned candidate is the minimum-contact-error one, so the reported
        # "after" error never exceeds the last iteration's error.
        self.assertLessEqual(
            diag["inner_contact_error_after_m"], diag["inner_contact_error_last_m"] + 1e-9
        )

    def test_v25_inner_loop_converges_below_cap(self):
        # With a reachable target the loop early-stops (2 cm or patience) well
        # before the 40-step cap.
        cb, reference = self._closure_cb(
            final_inner_steps=40, t0_max_inner_steps=40, guidance_strength=0.3,
            arm_max_update_cap=0.5, arm_final_update_cap=0.5,
        )
        cb(reference, torch.tensor([0]))
        diag = cb.step_diagnostics[-1]
        self.assertLess(int(diag["inner_iterations"]), 40)

    def test_v25_no_swing_phase_does_not_zero_root_target(self):
        # The v24 "root_target * 0.3 when no swing" scaling was removed (#7).
        src = (Path(__file__).resolve().parents[1] / "grail" / "adapters" / "gem_smpl.py").read_text()
        self.assertNotIn("root_target_delta_global * 0.3", src)


if __name__ == "__main__":
    unittest.main()
