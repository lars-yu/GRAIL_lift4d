# Test Results

## Validation so far

- Local syntax check on the current unsynced copies: `PYTHONPYCACHEPREFIX=/tmp/grail-pyc
  python3 -m py_compile remote_work/*.py` passed. This does not replace the
  required remote-environment test run.

- Remote `py_compile` passed for all modified optimizer/model/runner files.
- Remote `git diff --check` passed (the existing `smplx_model.py` CRLF warning is
  line-ending metadata, not a whitespace error).
- Existing `tests.test_hand_object_ray_ik`: 10 tests passed.
- Existing `tests.test_lift4d_formal_runner`: 18 tests passed.
- New local source checks include explicit rejection of Stage-B
  `hand_pose_res`; the remote `tests.test_palm_contact` run must be repeated
  after the latest source sync.

## Real rand00033 attempt

Command used the real 121-frame video, HMR NPZ, object mesh, FoundationPose
poses, GRAIL mask cache, and Lift4D motion-only NPZ. Stage 3A completed without
OOM and logged `contact_or_grasp_grad_obj_depth_res: 0`. The attempt stopped
before Stage 3B because frame 0 was outside the approach window but was still
subject to the 15 px ray fallback limit (`123.87 px`). The code now records
such pre-window fallback frames and keeps the <=15 px fail-fast rule for the
approach/contact window. The output remains debug-only and was not promoted.

The later debug candidate proved that opening contact-hand pose residuals in
Stage B can reduce the palm reprojection term, but that path is intentionally
non-compliant with the specification and must not be promoted. The current
local source instead rejects that configuration and keeps hand residuals in
Stage C only. Full real rerun, four formal videos, remote unittest coverage,
and syncing these latest local changes remain pending while the remote SSH
command approval is unavailable. The debug directory must be removed only
after reconnecting to the remote host; `retry5` remains the sole retained
formal result directory.
