# Test results

Verified on 2026-08-18 in the remote `Grail` conda environment.

```bash
python -m py_compile \
  grail/adapters/lift4d.py \
  grail/adapters/lift4d_depth.py \
  grail/optimization/motion_state.py \
  grail/optimization/data_types.py \
  grail/optimization/hoi_optimizer.py \
  grail/optimization/loss_computer.py \
  grail/optimization/loss_terms.py \
  scripts/run_lift4d_vggt_optimization.py \
  scripts/visualize_lift4d_depth_prior.py \
  scripts/render_saved_hoi_top_view.py \
  tests/test_lift4d_formal_runner.py \
  tests/test_object_motion_state.py \
  tests/test_numpy_pickle_compat.py
PYTHONPATH=$PWD python -m unittest discover -s tests -p 'test*.py'
git diff --check
git diff --cached --check
```

Results:

- `py_compile`: passed.
- `unittest`: 51 tests passed.
- worktree and staged whitespace checks: passed.
- 121-frame test: 121/121 prior frames supervised.
- `interval=4`: rejected.
- poisoned `object_poses_cam` and rotation gradients: do not affect depth loss.
- VGGT stage config: `include_human=True`, `include_object=False`.
- disabled motion state: legacy 30/5/10 weights, hard contact radius, and no static lock.
- static object hard-freeze: translation and rotation deviation are both exactly zero.
- formal videos: all three contain 121 frames at 30 fps and are non-empty.

## Real rand00033 execution

The real RGB, HMR, masks, FoundationPose poses, mesh, and strict contact cache are
present. The strict cache contains `contact_start_idx=49` and `R_Hand`.

The real Lift4D motion-only NPZ is:

```text
/home/jiaoyufei_insta360.com/PRE/GRAIL_4d/imports/Lift4D/lift4d_scgs/output/custom_kid_001_indoor2_pickup_table_rand00033_lift4d_object_1_node/lift4d_motion_prior.npz
```

It contains 121 frames and 4059 points, with
`frame_indices == np.arange(121)` and finite values. Formal GRAIL supervision
uses only `point_trajectories_cam`; `object_poses_cam` is ignored.

Runtime results:

```text
contact hint/source: 49 / cache
contact hand: right
t_move/confidence: 80 / 0.8857142857
contact window/selected frame: [72, 82] / 80
Lift4D supervised frames: 121 / 121
static raw z std: 0.0180305652 m
static target z std: 2.38418579e-7 m
optimized static z std: 2.38418579e-7 m
static translation/rotation deviation: 0 m / 0
maximum optimized depth step: 0.0189487934 m
maximum optimized depth acceleration: 0.0189986229 m
pre/post hand-object distance: 0.3703799844 m / 0.0296992380 m
human approach distance: 0.0686557740 m
foot sliding: 0.0043949303 m/frame
```

Formal output directory:

```text
/home/jiaoyufei_insta360.com/PRE/GRAIL_4d/pickup_table/generation/lift4d_motion_state_soft_contact/rand00033_formal_static_hardfreeze_20260818
```

It contains the real-data CSV/PNG/PKL/JSON diagnostics plus:

```text
lift4d_rigid_motion.mp4
foundationpose_vs_lift4d_vs_optimized.mp4
grail_optimized_top_view.mp4
```

No synthetic data or VGGT object-depth loss was used.
