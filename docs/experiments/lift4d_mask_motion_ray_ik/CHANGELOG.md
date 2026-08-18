# Lift4D mask motion and ray IK

## Per-file changes

- `grail/optimization/motion_state.py`: `detect_object_motion` now consumes
  adjacent mask pairs with median/MAD thresholds and 3/5 latch voting. The old
  hint-driven/future-window onset is gone. Disable with
  `optimization.object_motion_state.enabled: false` to use the legacy path.
- `grail/optimization/hand_object_ray_ik.py`: added mask-distance hand
  selection, FPS-derived smoothstep windows, detached mesh-surface depth
  lookup, camera-ray targets, and continuous grasp helpers. The module is only
  used when motion-state mode is enabled.
- `grail/optimization/hoi_optimizer.py`: formal `init_data` bypasses GPT/cache
  contacts; `forward` optimizes camera-Z only with static hard freeze; added
  `initialize_postcontact_pose_residuals` and explicit upper-body/arms frame
  masks. Set motion-state mode off to retain legacy contact-label behavior.
- `grail/optimization/loss_computer.py`: formal contact, ray-IK, and
  translation-follow losses detach object geometry; contact is continuous over
  the precontact/moving intervals. Legacy loss dispatch remains available for
  non-formal configurations.
- `grail/optimization/loss_terms.py`: retained legacy APIs but explicitly
  prohibits the old full-SE(3)/Kabsch Lift4D supervision.
- `grail/optimization/data_types.py`, `evaluator.py`: carry real Lift4D depth,
  motion-state, ray-target, and diagnostic provenance through the formal path.
- `scripts/run_lift4d_vggt_optimization.py`: implements Stage A/B/C separation,
  best-state restoration, postcontact residual propagation, raw/weighted loss
  reporting, full-frame diagnostics, and real-camera intrinsics export.
- `scripts/visualize_lift4d_depth_prior.py`: renders real RGB/mesh comparison
  and rigid-motion videos; missing legacy contact CSV fields are derived from
  physical `t_move` only.
- `scripts/render_saved_hoi_top_view.py`: renders the saved real optimized
  result from the top view and supports formal CSVs without GPT fields.
- `configs/recon_4dhoi/pickup_smplx.yaml`: adds configurable adjacent-mask
  thresholds, `pre_contact_seconds`, automatic hand selection, and removes
  formal softmin contact settings.
- `grail/models/**`, `.gitignore`: commits the runtime model source (no weights)
  and unignores only `grail/models/**` so clean clones can import it.
- `tests/test_object_motion_state.py`, `tests/test_hand_object_ray_ik.py`,
  `tests/test_lift4d_formal_runner.py`: cover motion detection, ray invariance,
  detached object gradients, static priority, Stage C all-frame gradients,
  continuity initialization, and real 121-frame supervision invariants.

To roll back the complete change, use `git revert <final-commit-sha>`; no
generated outputs or external runtime symlinks are part of the commit.

- Replaced future-window onset evidence with adjacent mask IoU, centroid, area,
  and auxiliary Lift4D speed signals.
- Added median+MAD thresholds, 3/5 voting, explicit static-frame hard freeze,
  and fail-fast onset detection.
- Added raw-keypoint distance-transform hand selection and camera-ray target
  utilities independent of GPT/contact cache labels.
- Detached object geometry from formal human contact and grasp losses.
- Stage B/C no longer optimize object depth; Stage A restores its best state.
- Tracked runtime `grail/models` source in the repository.
- Added post-contact pose-residual propagation from `t_move` to every moving
  frame before Stage 3C. This removes the one-frame reversion to raw HMR and
  makes continuous hand-object supervision effective.
- Added explicit before/after body and hand keypoint RMSE, human-mask IoU
  delta, `motion_state_diagnostics.csv/png`, and exact camera-intrinsics export.
- Formal real-data run `rand00033_formal_20260818_retry6` completed with all
  acceptance gates passing: `t_move=105`, right hand selected from raw 2D mask
  distance, Lift4D `121/121` frames, moving-frame contact under 5 cm `100%`,
  contact distance `0.02628 m`, maximum adjacent contact change `0.00222 m`,
  static optimized Z std `2.38e-7 m`, object-contact gradient `0`, and no
  four-frame periodic jumps.
- Real comparison rendering completed for
  `foundationpose_vs_lift4d_vs_optimized.mp4` and
  `lift4d_rigid_motion.mp4`. The first CPU top-view attempt exposed a
  CPU/CUDA texture-device mismatch; the formal rerun used `--device cuda` and
  completed `121/121` frames as `grail_optimized_top_view.mp4`.
