# Test Results

## Follow-up code change (2026-08-22)

- Branch: `codex/lift4d-smooth-contact-retry20-fix`, based on `b0d7afd4de227f5c2c395f5037d7e2fb6068530e`.
- Single algorithm change: smooth cubic approach scheduling, a strict Stage-B window ending at `t_move`, no Stage-B human translation residual or terminal palm pull, and a detached post-contact hand/object anchor with object-depth gradients blocked.
- Modified files: `grail/optimization/hand_object_ray_ik.py`, `grail/optimization/hoi_optimizer.py`, `grail/optimization/loss_computer.py`, `scripts/run_lift4d_vggt_optimization.py`, `tests/test_hand_object_ray_ik.py`, `tests/test_lift4d_formal_runner.py`, `tests/test_palm_contact.py`.
- Targeted verification: `tests.test_hand_object_ray_ik`, `tests.test_palm_contact`, and `tests.test_lift4d_formal_runner` passed (`48/48`). Full suite passed (`85/85`). All four edited runtime modules compiled with `py_compile`.
- Real run status: no new 121-frame optimization was executed in this follow-up because the remote run command/input recovery was not completed. Existing retry20 remains the latest real result and is not promoted or relabeled. No new metrics, videos, NPZ, PKL, MP4, or training data were added.
- Existing retry20 artifacts remain at `/home/jiaoyufei_insta360.com/PRE/GRAIL_4d_stage3_depth_approach_backup/pickup_table/generation/lift4d_palm_ray_contact_render_fix/rand00033_palm_ray_contact_20260821_retry20` and retain its documented 17/22 formal-gate result.

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
### retry27 (invalid smoothstep provenance)
- Real-data run: 121 frames, automatically detected `t_move=89`, right contact hand, formal iterations `400/600/600`.
- The remote worktree was accidentally on commit `b0d7afd4de227f5c2c395f5037d7e2fb6068530e` (before the smoothstep/post-contact-anchor commit `55e68aadeaab38f899eb4818797d208c4bec7648`). This run is therefore not evidence for the final smoothstep code and remains a debug-only diagnostic.
- Result: contact distance `0.144653 m`; moving frames under 5 cm `40.625%`; moving palm-surface median `0.055325 m`; surface under 1.5 cm `0%`; palm patch coverage `0%`; palm reprojection median/p95 `0.3553/25.4786 px`; t_move palm depth/3D errors `0.217596/0.217797 m`; approach step `0.029094 m`; body keypoint RMSE increase `4.2968 px`; penetration `0`; object-depth gradient `0`.
- Complete `run.log`, `optimization_metrics.json`, CSV diagnostics, plots, and the formal-gate traceback are retained in `rand00033_palm_ray_contact_20260822_retry27`.

### retry28 (55e68aa)
- Verification before the real run: targeted tests `37/37`, full suite `85/85`, and required `py_compile` checks passed. The run used commit `55e68aadeaab38f899eb4818797d208c4bec7648`, the smoothstep and detached post-contact-anchor implementation.
- Command: real `rand00033`, 121 frames, automatically detected `t_move=89`, right contact hand, `--stage-a-niter 400 --stage-b-niter 600 --stage-c-niter 600`, CUDA execution. No synthetic fallback, Kabsch alignment, object XY/rotation optimization, contact gradient to `obj_depth_res`, or acceptance-gate bypass was used.
- Result: contact distance `0.144653 m`; moving frames under 5 cm `40.625%`; moving palm-surface median `0.055325 m`; surface under 1.5 cm `0%`; palm patch coverage `0%`; palm reprojection median/p95 `0.3553/25.4786 px`; t_move palm depth/3D errors `0.217596/0.217797 m`; approach max step `0.029094 m`; boundary step `0.002494 m`; body/hand keypoint RMSE increases `4.2968/1.3065 px`; maximum penetration `0`; maximum adjacent palm-object change `0.030672 m`; object-depth gradient `0`.
- Passed gates include all-frame Lift4D supervision, positive camera-Z, static optimized depth, approach/boundary step, body/hand drift, penetration, hand-object change under 5 cm, mask IoU, median palm reprojection, and periodic-jump checks. The formal result is `false`; failed gates are contact distance, moving contact/surface, t_move palm depth/3D, palm reprojection p95, coverage, and maximum adjacent palm-object change.
- Root cause: smoothstep fixed the approach-step gate, but Stage-B intentionally leaves terminal palm depth/3D/surface weights at zero and Stage-C freezes frames `<=t_move`. The frozen t_move palm therefore remains about 21.8 cm from the contact target, and post-contact refinement cannot repair that frame. This is a single unresolved Stage-B endpoint/contact-loss issue, not a reason to weaken the formal gates.
- Complete `run.log`, `optimization_metrics.json`, `hoi_data.pkl`, CSV diagnostics, and plots are retained in `rand00033_palm_ray_contact_20260822_retry28`. The run ended with the formal-gate `RuntimeError`; no formal video render was published because the renderer correctly rejects debug-only output.

### retry29
- Modification before retry: restored moderate single-frame Stage-B terminal palm weights to `20000/2000/100` for depth/3D/surface after retry28 showed the frozen endpoint was unreachable. All other losses, automatic mask-motion `t_move`, Stage-B/C windows, detached targets, zero object-depth contact gradient, and formal gates were unchanged.
- Verification before the real run: targeted `37/37`, full `85/85`, and `py_compile` passed.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, right contact hand, `400/600/600` iterations, CUDA.
- Result: contact distance `0.068974 m`; moving under 5 cm `96.875%`; moving surface median `0.015133 m`; surface under 1.5 cm `46.875%`; t_move palm depth/3D `0.099375/0.109316 m`; approach/boundary steps `0.033451/0.032744 m`; body/hand RMSE increases `7.1285/4.2003 px`; palm reprojection median/p95 `0.2661/25.6535 px`; coverage `0%`; penetration `0`; adjacent palm-object change `0.037811 m`; object-depth gradient `0`.
- Root cause: a single terminal frame remained too abrupt and still did not solve the 3D ray target. Formal status failed; complete logs, metrics, CSV diagnostics, and plots are retained in `rand00033_palm_ray_contact_20260822_retry29`.

### retry30
- Modification before retry: changed the terminal palm losses to an 8-frame contiguous window ending at automatically detected `t_move`; terminal coefficients were `100000/10000/100` for depth/3D/surface. Targets and object surface vertices remained detached, and no acceptance gate or object-depth gradient path changed.
- Verification before the real run: targeted `38/38`, full `86/86`, `py_compile`, and `git diff --check` passed.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, right contact hand, `400/600/600` iterations, CUDA.
- Result: contact distance `0.047370 m`; moving under 5 cm `100%`; moving surface median `0.013360 m`; surface under 1.5 cm `84.375%`; t_move palm depth/3D `0.016620/0.096074 m`; approach/boundary steps `0.036221/0.001706 m`; body/hand RMSE increases `8.5077/5.5152 px`; palm reprojection median/p95 `0.2235/31.6172 px`; coverage `1.4547%`; penetration `0.004194 m`; adjacent palm-object change `0.032506 m`; object-depth gradient `0`.
- Failure evidence: frames 80--86 show palm speeds `0.39--1.09 m/s` and reprojection error up to `52.67 px`; frame 89 remains `0.096 m` from the 3D target, and the Stage-C transition at frame 90 reaches `1.35 m/s`. The uniform window plus high coefficients therefore spread a large endpoint pull without a smooth normalized temporal weighting.
- Formal status: failed and debug-only. The complete `run.log` ends with the formal-gate traceback; `optimization_metrics.json`, `palm_contact_diagnostics.csv`, `hand_trajectory_diagnostics.csv`, `motion_state_diagnostics.csv`, and plots are retained in `rand00033_palm_ray_contact_20260822_retry30`. No formal video was published.

### retry31
- Modification before retry: replaced the uniform 8-frame terminal aggregation with a monotone cubic-smoothstep ramp normalized to unit mean, and reduced terminal coefficients to `2000/500/10` for depth/3D/surface. The window still ends at automatically detected `t_move`; detached targets/object vertices, Stage-B/C boundaries, camera-Z-only Lift4D supervision, and all formal gates were unchanged.
- Verification before the real run: targeted tests `39/39`, full suite `87/87`, and `git diff --check` passed. The first launch had an incorrect prior path; its complete `FileNotFoundError` is retained as `retry31_input_path_traceback.log`, then the corrected real run completed with the exact 121-frame inputs.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, right contact hand, `400/600/600` iterations, CUDA. No synthetic fallback, manual contact frame, Kabsch, object XY/rotation optimization, contact gradient to `obj_depth_res`, or gate bypass was used.
- Result: approach max step `0.029386 m`, boundary step `0.002097 m`, boundary velocity change `0.049537 m`, maximum penetration `0`, contact/object-depth gradient `0`, hand keypoint RMSE increase `2.009 px`, and median palm reprojection `0.351 px`. The smooth ramp removed the retry30 boundary jump, but the frozen endpoint remained unreachable: contact distance `0.100464 m`, t_move depth/3D errors `0.172378/0.172499 m`, moving frames under 5 cm `75%`, moving surface median `0.042966 m`, surface under 1.5 cm `0%`, patch coverage `0%`, palm reprojection p95 `25.479 px`, body RMSE increase `5.065 px`, and maximum adjacent palm-object change `0.030604 m`.
- Root cause: temporal smoothing and moderate terminal coefficients fix endpoint impulse/continuity but do not correct the approximately 17 cm palm target mismatch across the terminal window. The existing per-frame ray/surface target trajectory is therefore unreachable under the unchanged approach and Stage-C freeze constraints. Formal status is `false`; all artifacts, including the final formal-gate traceback, remain debug-only in `rand00033_palm_ray_contact_20260822_retry31`.

### retry32
- Modification before retry: kept the normalized monotone cubic-smoothstep 8-frame terminal window and detached target/object-surface paths from retry31, and changed only Stage-B `palm_depth` terminal aggregation to an explicit squared error (`terminal_loss="squared"`). The squared term preserves gradient at the observed large endpoint error without restoring the unstable historical terminal coefficients. Stage-B/C boundaries, automatic mask-motion `t_move`, camera-Z-only Lift4D supervision, zero contact gradient to `obj_depth_res`, and all formal gates were unchanged.
- Verification before the real run: targeted palm/formal tests `40/40`, full suite `88/88`, remote `py_compile`, and `git diff --check` passed.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, right contact hand, `--stage-a-niter 400 --stage-b-niter 600 --stage-c-niter 600`, CUDA. No synthetic fallback, manual contact frame, Kabsch, object XY/rotation optimization, contact gradient to object depth, or acceptance-gate bypass was used.
- Result: contact-frame hand/object distance `0.067931 m`; moving contact under 5 cm `96.875%`; approach/boundary steps `0.029661/0.002737 m`; boundary velocity change `0.342912 m`; body/hand keypoint RMSE increases `7.452/4.557 px`; palm reprojection median/p95 `0.272/25.658 px`; t_move palm depth/3D errors `0.088865/0.101623 m`; moving palm-surface median `0.014739 m`; surface under 1.5 cm `56.25%`; patch coverage `0%`; maximum penetration `0.000041 m`; maximum adjacent palm-object change `0.031482 m`; contact/object-depth gradient `0`.
- Formal status: `false`. Failed gates were contact distance, body keypoint drift, palm reprojection p95, t_move depth/3D, moving surface median/fraction, patch coverage, and adjacent palm-object continuity. The squared depth terminal reduced the depth-loss raw value during Stage B, but the remaining 3D ray target and visual contact mismatch were not resolved under the fixed Stage-B endpoint and Stage-C freeze contract.
- Complete `run.log` (including the final formal-gate traceback), `optimization_metrics.json`, `palm_contact_diagnostics.csv`, trajectory/motion diagnostics, and plots are retained in `rand00033_palm_ray_contact_20260822_retry32`. No formal video was published because the renderer correctly rejects `formal_result=false`.

### retry33
- Modification before retry: changed only Stage-B `palm_target_3d` terminal aggregation from Huber to squared error. Stage-B `palm_depth` already used squared terminal error; all smoothstep window weights, detached targets/object vertices, automatic mask-motion contact detection, Stage-B/Stage-C boundaries, Lift4D camera-Z-only supervision, and formal gates were unchanged.
- Initial launch failure: the first command used a one-segment `--video-id` and stopped with `ValueError: not enough values to unpack (expected 3, got 1)`; the complete traceback is retained at the beginning of the retry33 `run.log`. Root cause was the command argument format, not the implementation. The corrected retry used `pickup_table/dl300_delta/kid_001_indoor2-pickup-table_rand00033`.
- Verification before the corrected real run: targeted tests `41/41`, full suite `89/89`, and `git diff --check` passed.
- Command: real `rand00033`, 121 frames, automatic `t_move=89`, right contact hand, `--stage-a-niter 400 --stage-b-niter 600 --stage-c-niter 600`, CUDA. No synthetic fallback, manual contact frame, Kabsch, object XY/rotation optimization, contact gradient to `obj_depth_res`, or acceptance-gate bypass was used.
- Result: contact distance `0.065720 m`; moving frames under 5 cm `78.125%`; approach/boundary steps `0.025475/0.009135 m`; boundary velocity change `0.229690 m`; body/hand keypoint RMSE increases `5.5199/4.0995 px`; palm reprojection median/p95 `4.2099/26.3340 px`; t_move palm depth/3D errors `0.117098/0.130039 m`; moving palm surface median `0.029795 m`; surface under 1.5 cm `0%`; patch coverage `1.5086%`; maximum penetration `0`; maximum adjacent palm-object change `0.031597 m`; contact/object-depth gradient `0`.
- Formal status: failed. The unchanged gates rejected contact distance, moving contact fraction, body drift, palm reprojection p95, t_move depth/3D, moving surface median/fraction, patch coverage, and adjacent continuity. The endpoint squared loss did not recover the frozen t_move target; no gate was weakened or removed.
- Complete artifacts are retained at `/home/jiaoyufei_insta360.com/PRE/GRAIL_4d_stage3_depth_approach_backup/pickup_table/generation/lift4d_palm_ray_contact_render_fix/rand00033_palm_ray_contact_20260822_retry33`, including `run.log`, final formal traceback, `optimization_metrics.json`, motion/contact CSVs, and diagnostic plots. No formal video was published because the result was debug-only.

### retry34
- Modification before retry: changed only Stage-B `palm_target_3d` terminal weight from `500` to `100`; squared terminal loss, normalized 8-frame smooth window, detached targets/object vertices, automatic `t_move`, Stage-B/C boundaries, zero object-depth contact gradient, and formal gates were unchanged.
- Verification before the real run: targeted tests `41/41`, full suite `89/89`, and `git diff --check` passed.
- Result: automatic `t_move=89`, right contact hand, contact distance `0.065011 m`; moving frames under 5 cm `78.125%`; approach/boundary steps `0.025589/0.009136 m`; body/hand RMSE increases `5.5155/4.1200 px`; palm reprojection median/p95 `4.4085/26.3340 px`; t_move depth/3D errors `0.116976/0.129803 m`; moving surface median `0.030135 m`; surface under 1.5 cm `0%`; patch coverage `1.4907%`; maximum penetration `0`; adjacent palm-object change `0.030729 m`; contact/object-depth gradient `0`.
- Formal status: failed unchanged gates for contact distance, moving contact, body drift, reprojection p95, t_move depth/3D, surface, coverage, and adjacent continuity. Complete `run.log`, traceback, metrics, CSV diagnostics, and plots are retained in `rand00033_palm_ray_contact_20260822_retry34`; no formal video was published.

### retry35
- Modification before retry: changed only Stage-B `palm_depth` terminal weight from `2000` to `5000` to test stronger smooth-window endpoint supervision. All other code and gates were unchanged.
- Verification before the real run: targeted tests `41/41`, full suite `89/89`, and `git diff --check` passed.
- Result: automatic `t_move=89`, right contact hand, contact distance `0.094964 m`; moving frames under 5 cm `71.875%`; approach step `0.028226 m`; body/hand RMSE increases `6.0323/5.4268 px`; palm reprojection p95 `27.3403 px`; t_move depth/3D errors `0.082583/0.147659 m`; moving surface median `0.028603 m`; surface under 1.5 cm `0%`; patch coverage `1.6703%`; maximum adjacent palm-object change `0.042122 m`.
- Formal status: failed and worse than retry34 on contact, 3D endpoint, drift, reprojection, and continuity. The final source therefore restores the retry34 value `2000`; complete retry35 artifacts remain debug-only in `rand00033_palm_ray_contact_20260822_retry35`.

### t_move=89 single-frame reachability pause (2026-08-22)
- Ran the real `pickup_table/dl300_delta/kid_001_indoor2-pickup-table_rand00033` input with automatic `t_move=89` and the detected right contact hand. The diagnostic initialized the real object mesh, masks, cached depth, FoundationPose poses, and Lift4D prior, then optimized only frame 89 for 80 iterations per mode. It did not run Stage A/B/C, formal gates, retry36, or a complete 121-frame optimization.
- Baseline actual palm world: `[5.287203, 14.166409, 0.745542] m`; physical target palm world: `[5.509652, 13.827654, 0.986320] m`; delta: `[0.222449, -0.338756, 0.240778] m`, norm `0.471395 m`. Baseline palm reprojection was `1.2939 px`.
- Delta decomposition used the normalized camera ray and ground direction: camera-ray unit `[-0.394055, 0.775901, -0.492645]`, projection `-0.469116 m`, component `[0.184857, -0.363987, 0.231107] m`; ground unit `[-0.218848, -0.975759, 0]`, projection `0.281861 m`, component `[-0.061685, -0.275029, 0] m`.
- `arm_only` (joints `13,14,16,17,18,19,20,21`): palm distance `0.205630 m`, reprojection `7.0427 px`; failed both thresholds.
- `arm_shoulder` (upper-body/shoulder plus arm joints): palm distance `0.018505 m`, reprojection `0.9111 px`; reprojection passed, but 5 mm distance did not.
- `arm_root_residual` (arm joints plus camera-ray root residual clamped to `0.05 m`): palm distance `0.144706 m`, reprojection `4.5175 px`; reprojection passed, but 5 mm distance did not.
- None of the three modes reached both `palm distance <= 5 mm` and `reprojection < 5 px`. The correct next step is to pause and investigate single-frame IK reachability before any full 121-frame run.
- Verification after this follow-up: remote full unittest discovery `90/90`, required `py_compile`, and `git diff --check` all passed.
