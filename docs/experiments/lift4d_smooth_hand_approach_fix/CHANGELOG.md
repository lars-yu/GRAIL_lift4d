# Lift4D smooth hand approach fix

## Scope

This branch fixes the real `rand00033` hand-approach discontinuity without
reintroducing object xyz/Kabsch/rotation supervision. Formal object motion is
still camera-Z-only from the real Lift4D point trajectory; FoundationPose xy
and rotation remain fixed, and contact/grasp geometry is detached from
`obj_depth_res`.

## Per-file changes

- `grail/optimization/hoi_optimizer.py`
  - `init_data`: stores initial hand camera points/pixels and computes a
    distance/speed-derived approach window, but no longer builds a final ray
    target from the pre-Stage-A mesh.
  - `refresh_hand_ray_targets_after_object_stage`: calls `forward`, converts
    the Stage-A detached mesh to OpenCV camera coordinates, performs a real
    surface lookup, and refreshes every ray/depth/fallback field.
  - `capture_stage_boundary_state`: freezes hand position, velocity, pose
    residual, and relative anchor at the end of Stage B.
  - `_apply_stage_gradient_masks`: adds a five-frame overlap before both the
    approach and moving intervals. No object variables are opened in B/C.

- `grail/optimization/motion_state.py`
  - `detect_object_motion`: uses three consecutive mask-IoU frames as the
    primary onset evidence. Centroid, area, and Lift4D speed affect confidence
    only; Lift4D speed is no longer an AND gate. Static sequences still fail
    fast and moving remains latched.

- `grail/optimization/hand_object_ray_ik.py`
  - `mesh_surface_depth_at_pixels`: adds OpenCV ray-triangle intersection,
    nearest-current-hand-depth hit selection, and a projected-vertex fallback
    flag.
  - `camera_ray_hand_targets`: selects the hand-side offset instead of always
    subtracting depth, and uses one `t_move` endpoint for the pre-contact
    minimum-jerk path to prevent surface-selection jumps.
  - `approach_window_from_fps`: derives a bounded 20-60 frame window from
    required displacement, maximum hand speed, and FPS.

- `grail/optimization/approach.py`
  - Adds the shared minimum-jerk ramp `10s^3-15s^4+6s^5`; the compatibility
    smoothstep API now returns the same ramp.

- `grail/optimization/loss_computer.py`
  - Uses projected sampled SMPL-X vertices for human silhouette loss, sampling
    before projection to avoid the previous GPU peak.
  - Uses the shared ramp for contact distance targets, fixed Stage-B relative
    anchors for Stage C, and adds local hand path/velocity/acceleration/jerk,
    pose-residual acceleration, boundary position/velocity, and continuity
    losses. Stage C uses a joint overlap phase whose contact target and weight
    transition continuously from approach to grasp. Every contact geometry
    path remains object-detached.

- `grail/optimization/data_types.py`, `evaluator.py`
  - Carries fixed hand rays/pixels, surface fallback flags, approach distance,
    and Stage-B boundary state with frame-safe truncation.

- `scripts/run_lift4d_vggt_optimization.py`
  - Refreshes targets after Stage A, captures the Stage-B endpoint, enables
    local trajectory losses and overlap, limits root approach to 3 cm, and
    writes `hand_trajectory_diagnostics.csv/png` plus boundary gates.

- `configs/recon_4dhoi/pickup_smplx.yaml`
  - Adds mask-IoU consecutive-frame, hand-speed/window, root-limit, and
    boundary-tail configuration.

- `tests/test_object_motion_state.py`, `tests/test_hand_object_ray_ik.py`,
  `tests/test_lift4d_formal_runner.py`
  - Add static-Lift4D mask onset, ray surface/side/refresh, minimum-jerk
    endpoint, dynamic-window, Stage-B overlap, and local-loss coverage tests.

## Formal rand00033 result

Output:
`pickup_table/generation/lift4d_mask_motion_ray_ik/rand00033_smooth_hand_approach_20260819_retry4`

- `t_move`: 105 -> 89
- Lift4D supervision: 121/121
- contact distance: 2.4708 cm
- maximum hand-object distance step in the approach/boundary diagnostic
  window: 1.6234 cm
- 103->104 hand step: 0.8276 cm; 104->105: 0.9369 cm
- approach maximum hand step: 2.3751 cm
- maximum hand speed: 1.1807 m/s; acceleration: 38.1615 m/s^2; jerk finite
- body keypoint RMSE: 22.2915 -> 20.2292 px
- hand keypoint RMSE: 17.2414 -> 13.1788 px
- human mask IoU delta: +0.004624
- moving frames under 5 cm: 100%
- all acceptance gates: true

## Final retry5 result (2026-08-19)

The latest joint-phase implementation was rerun from the real rand00033
inputs with `--stage-a-niter 400 --stage-b-niter 600 --stage-c-niter 600`.

- Output: `pickup_table/generation/lift4d_mask_motion_ray_ik/rand00033_smooth_hand_approach_20260819_retry5`
- `t_move=89`, confidence `1.0`, Lift4D supervision `121/121`
- Contact-frame hand-object distance: `1.9154 cm`
- `103->104=0.3878 cm`; `104->105=0.1387 cm`
- Maximum diagnostic-window hand-object change: `1.2973 cm`
- Boundary hand step: `1.5336 cm`; approach maximum hand step: `2.3541 cm`
- Maximum speed/acceleration/jerk: `1.1815 m/s`, `36.6456 m/s^2`, `1872.2815`
- Body keypoint RMSE: `22.2915 -> 20.3341 px`; hand RMSE: `17.2414 -> 13.1034 px`
- Human mask IoU delta: `+0.004572`; moving frames under 5 cm: `100%`
- Contact/grasp gradient w.r.t. `obj_depth_res`: `0`; periodic four-frame jumps: `0`
- All acceptance gates: `true`

Stage C now applies contact/ray/path losses in the five-frame joint overlap.
The saved-result top-view renderer sets `skip_contact_label_loading=true`; its
validation log reports `CONTACT_CACHE_READ_NOT_DETECTED`, so GPT/contact-cache
labels are not read by either formal optimization or formal rendering.

Formal videos regenerated from retry5:

- `rendered/lift4d_rigid_motion.mp4` (1280x720, 121 frames, 30 fps)
- `rendered/foundationpose_vs_lift4d_vs_optimized.mp4` (3840x720, 121 frames, 30 fps)
- `grail_optimized_top_view.mp4` (1280x720, 121 frames, 30 fps)
