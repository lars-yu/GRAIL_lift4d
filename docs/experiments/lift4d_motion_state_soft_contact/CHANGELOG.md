# Lift4D motion-state and soft-contact changelog

## Runtime behavior

| File | Changed functions | Previous behavior | New behavior | Switch / disable |
| --- | --- | --- | --- | --- |
| `grail/adapters/lift4d_depth.py` | `load_lift4d_depth_prior` | Only the median7 + SG31 center was exposed. | Also exposes a median5 detection center while preserving fixed stable Gaussian IDs and the full-frame smoothed Z prior. | Motion-state use is disabled with `object_motion_state.enabled: false`. |
| `grail/optimization/motion_state.py` | `detect_object_motion`, `build_static_relative_depth_target`, contact helpers | No physical object-motion onset existed. | Detects persistent static-to-moving onset from Lift4D and masks, fails on low confidence, and creates a static-relative `z_target`. | `object_motion_state.enabled: false`. |
| `grail/optimization/data_types.py` | `Lift4DDepth`, `HOIData` fields | No onset/contact provenance or soft weights were carried. | Carries detection centers, Z target, motion state, contact hint source/window, and selected contact. | Fields remain inert when motion state is disabled. |
| `grail/optimization/hoi_optimizer.py` | `init_data`, `initialize_obj_depth_from_lift4d`, `forward`, approach initialization/masking, export | Contact frame was a hard timing anchor, FoundationPose jitter entered the pre-motion pose, and approach direction used one contact-frame object position. | Resolves CLI/cache/interaction hint provenance, infers hand labels, detects `t_move`, directly initializes anchor-relative Lift4D depth, hard-freezes the complete pre-motion object pose, ends approach at `t_move`, uses the median static object world position, and exports fail-fast pose diagnostics. | `object_motion_state.enabled: false` restores hard-contact timing and raw smoothed Z targets. |
| `grail/optimization/loss_terms.py` | temporal contact helpers | Contact used a hard frame radius. | Adds positive temporal priors and soft-min contact selection over the full physical window. | Disabled motion state uses the legacy hard-frame loss path. |
| `grail/optimization/loss_computer.py` | Lift4D depth/velocity, static, contact, approach, relative losses | Used `prior.z` directly and hard contact timing. | Uses full-frame `z_target`, locks pre-motion depth, soft-selects contact, and starts relative consistency at `t_move`. | `object_motion_state.enabled: false`; static loss is omitted by the runner. |
| `scripts/run_lift4d_vggt_optimization.py` | parser, stage configs, diagnostics | Defaulted contact to frame 80, required VGGT object depth, and ran human global alignment before object depth. | Contact override defaults to `None`; VGGT is optional and human-only; stages run object depth, fixed-object human approach, then bounded joint refinement; writes motion/contact diagnostics and fail-fast quality metrics. | YAML motion-state switch restores 30/5/10 legacy Lift4D/velocity/FP weights; omit `--use-vggt-human-depth` to disable VGGT. |
| `scripts/visualize_lift4d_depth_prior.py` | formal comparison and rigid rendering | Compared against raw smoothed Z, reused jittering per-frame FoundationPose during static frames, and only showed the right-hand distance. | Uses the same hard-frozen static pose convention as optimization, validates it, renders both the three-panel comparison and `lift4d_rigid_motion.mp4`, and annotates real confidence/RMSE/valid-point data. | N/A. |
| `scripts/render_saved_hoi_top_view.py` | top-view annotations | Showed only right-hand distance. | Shows both hands, static/moving state, hint, window, and selected contact. | N/A. |
| `configs/recon_4dhoi/pickup_smplx.yaml` | optimization defaults | Had no motion-state/soft-contact defaults. | Adds the requested fail-fast motion state and soft-contact defaults. | Set `object_motion_state.enabled: false`. |

## Compatibility and tests

`grail/adapters/lift4d.py` is included because the baseline already tracks
`tests/test_lift4d_motion.py` but omitted the module it imports. It remains a
legacy/prohibited compatibility adapter; the formal object-depth path imports
only `lift4d_depth.py`, and `lift4d_motion_loss` still raises `RuntimeError`.

`tests/test_object_motion_state.py` covers onset robustness, median5 isolation,
static-relative Z, hint-window behavior, cache precedence, and hand inference.
`tests/test_lift4d_formal_runner.py` covers optional VGGT/contact overrides,
human-only VGGT supervision, disabled-switch legacy loss configuration,
anchor-relative initialization, static rotation/ray/Z hard-freezing, and bounded
Stage 3C depth refinement. `tests/test_numpy_pickle_compat.py` covers loading
NumPy 2.x pickles in the production environment.
