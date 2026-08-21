# Test Results

## Code and unit tests

- Updated tests/test_palm_contact.py: terminal palm fixture now declares frame_num=4; Stage C assertions freeze frames 0:5 and allow contact-hand gradients only after frame 4.
- Updated grail/optimization/loss_computer.py: palm target and observed-pixel sequences are checked against data.frame_num; mismatches raise ValueError. Stage B precontact contact-anchor window includes t_move.
- Updated grail/optimization/hoi_optimizer.py: Stage B gradient mask includes t_move; Stage C uses start=min(frame_num, motion_frame+1), end=frame_num; Stage C does not modify t_move; non-contact hands stay frozen.
- Targeted tests: python -m unittest tests.test_palm_contact tests.test_lift4d_formal_runner -v -> 36 tests passed.
- Full suite: python -m unittest discover -s tests -v -> 83 tests passed.

## Real rand00033 retries

All retries used the real 121-frame RGB video, HMR NPZ, object mesh, FoundationPose poses_in_cam.pkl, render config, GRAIL masks/depth caches, and Lift4D motion-only prior. No synthetic fallback, Kabsch re-enable, object xy/rotation optimization, t_move override, or acceptance-gate removal was used.

### retry1
- Failure: output parent directory did not exist, so the log could not be created.
- Root cause: run-environment output path.
- Follow-up: created the intended output directory before retrying.

### retry2
- Failure: foreground SSH session termination stopped the process and left no durable log.
- Root cause: process lifetime tied to SSH session.
- Follow-up: switched to nohup and persisted run.log.

### retry3
- Failure: ValueError: not enough values to unpack (expected 3, got 1).
- Root cause: one-segment video-id.
- Follow-up: used canonical three-segment id pickup_table/dl300_delta/kid_001_indoor2-pickup-table_rand00033.

### retry4
- Failure: missing imports/GEM-SMPL/....
- Root cause: relocated worktree had a self-referential imports symlink.
- Follow-up: repaired the symlink to the preserved dependency tree.

### retry5
- Failure: missing data/g1_smplx/g1_smplx_param.npz.
- Root cause: relocated worktree had a self-referential data symlink.
- Follow-up: repaired the symlink to the preserved data tree.

### retry6
- Failure: TypeError: float() argument must be a string or a real number, not dict.
- Root cause: first_frame_pose.pickle (render dictionary) was passed as the per-frame object pose file.
- Follow-up: selected the real pose_estimation_output/poses_in_cam.pkl.

### retry7
- Failure: FileNotFoundError for masks at the canonical three-segment cache path.
- Root cause: real masks existed under masks/dl300_delta/dl300, while code resolved masks/pickup_table/dl300_delta.
- Follow-up: added a directory mapping to the existing real cache; no data was synthesized.

### retry8
- Failure: FileNotFoundError for depth at the canonical three-segment cache path.
- Root cause: real depth existed under depth/dl300_delta/dl300, while code resolved depth/pickup_table/dl300_delta.
- Follow-up: added a directory mapping to the existing real cache.

### retry9
- Result: full optimization completed and wrote diagnostics, then formal gates failed.
- Observed: 121 frames loaded; t_move=89; contact-frame hand-object distance 0.09452 m; moving-frame fraction under 5 cm 0; palm reprojection p95 25.48 px; palm depth error at t_move 0.19471 m; palm 3D error 0.19480 m; moving palm surface median 0.38661 m; contact/object-depth gradient 0.
- Root cause analysis: the complete run exposed a real data/optimization mismatch rather than an initialization failure. The precontact contact-anchor loss ended at move_start-1, so it did not supervise the Stage-B endpoint.

### retry10
- Modification before retry: changed the precontact contact-anchor loss window to include move_start (t_move), then reran targeted and full tests.
- Tests after modification: targeted 36 passed; full suite 83 passed.
- Result: full optimization completed and formal gates still failed.
- Observed: t_move=89; contact-frame hand-object distance 0.09490 m; moving-frame fraction under 5 cm 0; adjacent hand-object change 0.30450 m; approach hand step 0.05081 m; palm reprojection p95 25.48 px; palm depth error 0.19500 m; palm 3D error 0.19510 m; moving palm surface median 0.38659 m; moving palm surface under 1.5 cm 0%; palm patch coverage under 1 cm 0%; contact/object-depth gradient 0.
- Conclusion: including the Stage-B endpoint did not materially change the real result. The remaining failure is retained as a genuine debug result; no prohibited workaround was applied.

## Artifacts

- retry9 output: .../lift4d_palm_ray_contact_render_fix/rand00033_palm_ray_contact_20260821_retry9
- retry10 output: .../lift4d_palm_ray_contact_render_fix/rand00033_palm_ray_contact_20260821_retry10
- Both outputs retain run.log, optimization_metrics.json, palm/contact diagnostics, motion diagnostics, and traceback.
- The formal acceptance result is not promoted because the required gates failed.

### retry11
- Command: the same real `rand00033` command as retry10, with Stage C `human_trans_res` enabled at `lr=0.001` and a post-contact mask starting at `t_move+1`.
- Failure stage: formal acceptance after optimization.
- Error summary: `t_move=89`; post-contact depth improved but remained about 0.18--0.38 m late; formal gates failed for moving contact, surface, reprojection p95, and coverage.
- Root cause: Stage C had no trainable human translation residual, so the raw HMR jump after frame 89 could not be corrected.
- Modified files: `grail/optimization/hoi_optimizer.py`, `scripts/run_lift4d_vggt_optimization.py`.
- Change and tests: added masked Stage C `human_trans_res`; targeted 36 passed, full 83 passed, and `py_compile` passed.

### retry12
- Command: real `rand00033`, Stage B and Stage C `human_trans_res` both enabled at `lr=0.001`.
- Failure stage: formal acceptance after optimization.
- Error summary: contact-frame distance `0.00062 m`, t_move depth error `0.00567 m`, moving fraction under 5 cm `3.1%`, moving surface median `0.214 m`, reprojection p95 `25.48 px`; eight acceptance gates failed.
- Root cause: Stage B translation corrected the endpoint, but Stage C step size was insufficient for the post-contact HMR jump; the global reprojection p95 was dominated by frames before the existing loss window.
- Modified files: `scripts/run_lift4d_vggt_optimization.py` (Stage B translation residual).
- Change and tests: targeted 36 passed, full 83 passed, and `py_compile` passed; complete traceback and diagnostics were retained in the retry12 output directory.

### retry13
- Command: real `rand00033`, Stage C `human_trans_res lr=0.005`.
- Failure stage: formal acceptance after optimization.
- Error summary: moving fraction under 5 cm `90.6%`, adjacent change `0.04165 m`, moving surface median `0.01914 m`, p95 reprojection remained `25.48 px`; seven gates failed.
- Root cause: the higher Stage C translation rate fixed most post-contact motion, but frames 0--23 still had no palm reprojection supervision and residual surface/depth error remained.
- Modified files: `scripts/run_lift4d_vggt_optimization.py`.
- Change and tests: targeted 36 passed, full 83 passed, and `py_compile` passed; retry13 log and formal traceback were retained.

### retry14
- Command: real `rand00033`, added `window_start=0` to Stage B palm reprojection, allowed Stage B translation from frame 0, and used Stage C translation `lr=0.01`.
- Failure stage: formal acceptance after optimization.
- Error summary: p95 reprojection improved to `1.64 px`; moving fraction under 5 cm reached `100%`; remaining failures were approach step `0.0405 m`, t_move depth `0.0144 m`, moving surface coverage `65.6%`, penetration `0.0190 m`, and adjacent palm change.
- Root cause: full-sequence reprojection solved the early p95, but an abrupt frame-24 translation response violated approach continuity and stronger post-contact translation still traded against body alignment.
- Modified files: `grail/optimization/loss_computer.py`, `grail/optimization/hoi_optimizer.py`, `scripts/run_lift4d_vggt_optimization.py`.
- Change and tests: strict frame-length checks remained intact; targeted 36 passed, full 83 passed, and `py_compile` passed. Full retry14 artifacts and traceback were retained.

### retry15
- Command: real `rand00033`, Stage B translation residual from frame 0 with a 29-frame ramp; Stage C contact weights strengthened and contact-hand pose residual learning rate set to `0.0005`.
- Failure stage: formal acceptance after optimization.
- Result: moving frames under 5 cm `100%`; moving palm surface median `0.002734 m`; surface under 1.5 cm `100%`; patch coverage `0.4520`; palm reprojection p95 `1.4336 px`.
- Failed gates: approach hand step `0.04091 m`; body keypoint RMSE increase `7.090 px`; hand keypoint RMSE increase `7.901 px`; t_move palm depth `0.012416 m`; maximum penetration `0.01999 m`; adjacent palm-object change `0.03964 m`.
- Root cause: stronger post-contact fitting preserved contact metrics but still traded against body/hand reprojection and boundary continuity. Complete `run.log`, traceback, metrics, and diagnostics are retained in `rand00033_palm_ray_contact_20260821_retry15`.

### retry16
- Modification before retry: precontact palm depth, target-3D, and surface losses now ramp with the hand-ray approach weight; penetration weight increased to `5000`; contact-hand Stage C pose residual learning rate reduced to `0.00015`.
- Tests after modification: targeted 36 passed; full suite 83 passed; `py_compile` passed.
- Result: real 121-frame optimization completed. Moving frames under 5 cm `100%`; moving palm surface median `0.002927 m`; surface under 1.5 cm `100%`; patch coverage `0.4547`; palm reprojection median/p95 `0.5469/1.4132 px`; contact-frame hand-object distance `0.000698 m`; contact/object-depth gradient `0`.
- Passed gates include all-frame Lift4D supervision, t_move 3D error, contact distance, moving contact/surface, coverage, reprojection, static depth, and periodic-jump checks.
- Failed gates: approach hand step `0.040984 m`; body keypoint RMSE increase `7.150 px`; hand keypoint RMSE increase `7.980 px`; t_move palm depth `0.012332 m`; maximum penetration `0.020078 m`; maximum adjacent palm-object change `0.039526 m`.
- Root cause: the ramp fixed precontact palm depth and surface alignment, but the remaining endpoint depth, penetration, and human reprojection tradeoff persists under the formal constraints. Complete `run.log`, traceback, metrics, and diagnostics are retained in `rand00033_palm_ray_contact_20260821_retry16`.

### retry17
- Modification before retry: replaced the first penetration proxy with a signed plane proxy; strengthened Stage-B terminal palm depth/surface constraints; used Stage-C body/hand reprojection weights `3/1`, terminal depth weight `5000`, surface weight `600`, signed penetration weight `5000`, and translation `lr=0.006`.
- Tests after modification: targeted 36 passed; full suite 83 passed; `py_compile` passed.
- Result: body/hand keypoint RMSE increases `4.683/2.100 px` and t_move palm depth error `0.007156 m` passed. Moving palm surface median was `0.011663 m`, fraction under 1.5 cm `65.625%`, patch coverage `11.51%`, maximum penetration `0.01450 m`, approach step `0.04196 m`, and maximum adjacent palm-object change `0.04279 m`.
- Formal status: failed approach, moving surface median/fraction/coverage, penetration, and adjacent-palm gates. Debug artifacts are retained in `rand00033_palm_ray_contact_20260821_retry17`.

### retry18
- Modification before retry: restored Stage-B translation `lr=0.0005` with the 29-frame ramp; reduced Stage-B terminal surface weight to `10`; used Stage-C translation `lr=0.008`, contact-hand pose `lr=0.0001`, surface weight `300`, and penetration weight `20000` with 3 mm clearance.
- Tests after modification: targeted 36 passed; full suite 83 passed; `py_compile` passed.
- Result: body/hand keypoint RMSE increases `4.510/2.747 px`; t_move depth error `0.009761 m`; moving surface median `0.010585 m`; surface fraction under 1.5 cm `71.875%`; coverage `13.775%`; penetration `0.01147 m`; approach step `0.04097 m`; adjacent change `0.03982 m`.
- Formal status: failed approach, moving surface median/fraction/coverage, penetration, and adjacent-palm gates. Debug artifacts are retained in `rand00033_palm_ray_contact_20260821_retry18`.

### retry19
- Modification before retry: changed signed penetration to candidate-triangle exact point-to-surface distance using `pytorch3d.ops.knn_points`; added a 2.5 cm maximum-step penalty to Stage-B/Stage-C hand velocity; increased Stage-C palm surface weight to `1000`.
- Tests after modification: targeted 36 passed; full suite 83 passed; `py_compile` passed.
- Failure stage: Stage 3B terminated before formal acceptance with `RuntimeError: The size of tensor a (64) must match the size of tensor b (3) at non-singleton dimension 2`.
- Root cause: the outward-normal mask was expanded as `[P,1,K]` using `outward[:, None]` instead of `[P,K,1]`. Full `run.log` is retained in `rand00033_palm_ray_contact_20260821_retry19`.

### retry20
- Fix before retry: corrected candidate-face normal orientation to `outward[..., None]`; no loss weights, input paths, frame count, t_move, or acceptance gates were changed from retry19.
- Verification before the real run: targeted tests `36/36`; full suite `83/83`; `py_compile` passed for `loss_computer.py`, `hoi_optimizer.py`, the formal runner, and both updated test modules.
- Command: real `rand00033`, 121 frames, automatically detected `t_move=89`, formal iterations `400/600/600`, right contact hand. No synthetic fallback, Kabsch, object xy/rotation optimization, contact gradient to `obj_depth_res`, or gate removal was used.
- Passed 17/22 gates: all-frame Lift4D supervision; positive OpenCV Z; static depth; contact/boundary distance; moving-frame contact; moving surface median/fraction; palm depth/3D at t_move; palm reprojection median/p95; hand keypoint drift; mask IoU; adjacent hand-object change; periodic-jump check. Contact/object-depth gradient remained exactly `0`.
- Key metrics: contact distance `0.000632 m`; moving frames under 5 cm `100%`; moving surface median `0.005819 m`; surface fraction under 1.5 cm `93.75%`; palm reprojection median/p95 `0.5925/1.5094 px`; t_move palm depth/3D errors `0.009824/0.010295 m`; optimized static-Z std `2.38e-7 m`.
- Failed 5/22 gates: approach hand step `0.039054 m` (limit 0.03); body keypoint RMSE increase `5.3575 px` (limit 5); palm patch coverage `23.635%` (minimum 30%); maximum penetration `0.012522 m` (limit 0.003); maximum adjacent palm-object change `0.035508 m` (limit 0.013).
- Formal status: failed and retained strictly as debug output in `rand00033_palm_ray_contact_20260821_retry20`; `optimization_metrics.json`, `hoi_data.pkl`, full `run.log`, traceback, CSV diagnostics, and plots are present.

### retry21
- Modification before retry: switched palm/object distance and coverage to batched full-palm/full-object `knn_points`; added worst-fraction signed penetration and maximum-step reduction. Stage-B/Stage-C hand maximum step was `0.012 m`, weight `100`, reduction `max`; Stage-C body reprojection weight was `5`.
- Verification before the real run: targeted tests `36/36`; full suite `83/83`; `py_compile` passed.
- Result: approach step `0.015521 m`, body/hand keypoint RMSE changes `+2.274/+0.853 px`, maximum penetration `0`, but contact became unreachable: contact distance `0.131620 m`, moving frames under 5 cm `28.125%`, moving surface median `0.056467 m`, surface under 1.5 cm `0%`, coverage `0%`, t_move depth/3D `0.208315/0.208482 m`, and palm reprojection p95 `6.499 px`.
- Root cause: the `1.2 cm` maximum-step constraint over-constrained Stage C and prevented the hand from reaching the contact target. Formal status failed 9 gates; artifacts are retained in `rand00033_palm_ray_contact_20260821_retry21`.

### retry22
- Modification before retry: relaxed Stage-B/Stage-C hand maximum step from `0.012 m` to `0.025 m` while retaining `max_step_weight=100` and `max_step_reduction=max`; reduced Stage-C coverage weight from `1000` to `300`. Full KNN coverage, worst-fraction penetration, Stage-C body weight `5`, and all acceptance gates were retained.
- Verification before the real run: targeted tests `36/36`; retry22 code synchronization and `py_compile` passed. The full suite had already passed `83/83` on the unchanged loss/optimizer code after retry21.
- Command: real `rand00033`, 121 frames, automatically detected `t_move=89`, formal iterations `400/600/600`, right contact hand; no synthetic fallback, Kabsch, object xy/rotation optimization, contact gradient to `obj_depth_res`, or gate removal.
- Result: body/hand keypoint RMSE changes `+0.763/-1.120 px`, palm reprojection median/p95 `0.573/2.834 px`, maximum penetration `0`, optimized static-Z std `2.38e-7 m`, and contact/object-depth gradient `0`. However contact distance was `0.035389 m`, moving frames under 5 cm `6.25%`, t_move depth/3D `0.099191/0.099224 m`, moving surface median `0.102467 m`, surface under 1.5 cm `0%`, coverage `0%`, boundary palm step `0.018006 m`, and maximum adjacent palm-object change `0.035318 m`.
- Root cause: relaxing the step limit restored Stage-B approach but Stage-C joint refinement converged to a non-contact local solution; the unchanged formal gates correctly rejected it. Formal status failed 9 gates and remains debug-only in `rand00033_palm_ray_contact_20260821_retry22`; complete metrics, logs, traceback, CSV diagnostics, and plots are retained.

### retry23
- Modification before retry: retained `max_step=0.025 m`, full KNN coverage, worst-fraction penetration, and all gates, but reduced Stage-B/Stage-C `max_step_weight` from `100` to `25` after retry22's Stage-C initial hand-velocity term reached `12768`.
- Verification before the real run: targeted tests `36/36`; full suite `83/83`; five-file `py_compile` passed.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, formal iterations `400/600/600`, right contact hand.
- Result: moving frames under 5 cm `81.25%`, approach step `0.026748 m`, body/hand keypoint RMSE changes `+2.745/+1.867 px`, palm reprojection median/p95 `0.578/2.980 px`, and maximum penetration `0` passed. Contact distance was `0.035272 m`, t_move depth/3D `0.093083/0.093105 m`, moving surface median `0.032019 m`, surface under 1.5 cm `0%`, coverage `1.311%`, boundary palm step `0.020376 m`, and maximum adjacent palm-object change `0.030061 m`.
- Root cause: lowering the maximum-step penalty avoided retry22's fully non-contact Stage-C solution, but Stage B still failed to place the frozen `t_move` palm at the depth/3D target. Stage C cannot correct frame `t_move` by design. Formal status failed 8 unchanged gates; complete debug artifacts are retained in `rand00033_palm_ray_contact_20260821_retry23`.

### retry24
- Modification before retry: increased Stage-B terminal palm depth/3D/surface weights from `5000/500/10` to `20000/2000/100`; Stage-C configuration and the `max_step=0.025`, `max_step_weight=25` continuity constraint were unchanged.
- Verification before the real run: targeted tests `36/36`; full suite `83/83`; five-file `py_compile` passed.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, formal iterations `400/600/600`, right contact hand.
- Result: contact-frame hand-object distance improved to `0.007256 m`; moving frames under 5 cm `81.25%`; body/hand keypoint RMSE changes `+3.098/+2.046 px`; maximum penetration `0.010318 m`; t_move palm depth/3D errors `0.048156/0.049300 m`; moving surface median `0.028947 m`; surface fraction under 1.5 cm `6.25%`; coverage `1.724%`; boundary/approach step `0.030756 m`; adjacent change `0.029891 m`.
- Root cause: the stronger terminal losses improved the surface contact distance but left the palm-center ray target misaligned; Huber terminal gradients remained saturated and Stage C could not update frozen frame `t_move`. Formal status failed 10 unchanged gates; complete debug artifacts are retained in `rand00033_palm_ray_contact_20260821_retry24`.

### retry25
- Modification before retry: increased Stage-B terminal palm depth/3D weights to `100000/10000` after retry24 showed Huber-saturated endpoint errors; Stage-C weights, `max_step=0.025 m`, and `max_step_weight=25` remained unchanged.
- Verification before the real run: targeted tests `36/36`; full suite `83/83`; five-file `py_compile` passed.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, formal iterations `400/600/600`, right contact hand.
- Result: contact distance `0.001960 m`, t_move depth/3D `0.010100/0.011682 m`, moving frames under 5 cm `81.25%`, body/hand keypoint RMSE changes `+3.355/+2.149 px`, and approach/surface metrics otherwise remained stable. Failed boundary/approach hand step `0.037254 m`, moving surface median `0.027098 m`, surface fraction under 1.5 cm `9.375%`, coverage `5.334%`, penetration `0.012462 m`, and adjacent change `0.033305 m`.
- Root cause: endpoint targeting reached the frozen t_move target, but the terminal correction was too abrupt; Stage B's final hand step exceeded the continuity gate and propagated into Stage-C surface/penetration errors. Formal status failed 9 gates; complete debug artifacts are retained in `rand00033_palm_ray_contact_20260821_retry25`.

### retry26
- Modification before retry: retained retry25's Stage-B terminal palm depth/3D weights (`100000/10000`) and reduced the Stage-B/Stage-C maximum hand-step threshold from `0.025 m` to `0.018 m`; the penalty weight remained `25` and all formal gates remained enabled.
- Verification before the real run: targeted tests `36/36`; full suite `83/83`; five-file `py_compile` passed.
- Command: the same real `rand00033` command, 121 frames, automatic `t_move=89`, formal iterations `400/600/600`, right contact hand.
- Failure stage and error: optimization completed, then formal acceptance raised `RuntimeError`; nine unchanged gates failed for moving contact, boundary/approach steps, moving surface fraction/median, patch coverage, penetration, and adjacent palm-object continuity.
- Result: contact distance `0.001809 m`; t_move depth/3D `0.009742/0.011848 m`; moving frames under 5 cm `71.875%`; boundary/approach steps `0.035374/0.035626 m`; moving surface median `0.029437 m`; surface under 1.5 cm `9.375%`; patch coverage `4.310%`; maximum penetration `0.014735 m`; maximum adjacent palm-object change `0.031267 m`; body/hand keypoint RMSE changes `+3.461/+2.399 px`; palm reprojection median/p95 `0.546/4.158 px`; contact/object-depth gradient remained exactly `0`.
- Root cause: the tighter all-window step limit competed with the endpoint target without reducing the t_move boundary jump; it reduced moving-contact coverage from retry25's `81.25%` to `71.875%` and increased penetration. The final runner therefore restores the previously validated `0.025 m` threshold; complete retry26 logs, traceback, metrics, CSV diagnostics, and plots remain debug-only in `rand00033_palm_ray_contact_20260821_retry26`.
