# Test results

Remote environment: `~/miniconda3/envs/Grail/bin/python` on the two-5090
server, branch `codex/lift4d-smooth-hand-approach-fix`.

- `py_compile`: all modified Python files passed.
- `tests.test_object_motion_state`: 8 tests, OK.
- `tests.test_hand_object_ray_ik`: 10 tests, OK.
- `tests.test_lift4d_formal_runner`: 18 tests, OK.
- Full `unittest discover -s tests -p "test_*.py" -v`: 65 tests, OK.
- Formal `rand00033` run: 400/600/600 iterations, 121/121 Lift4D frames,
  all acceptance gates true.
- Render validation: rigid and top-view MP4s are 1280x720, comparison MP4 is
  3840x720, all have 121 frames at 30 FPS and non-empty video streams.
- `git diff --cached --check`: passed before each commit.

The first tuning run exposed an OOM caused by projecting the complete human
mesh before sampling. That path was changed to sample first; the final run
completed without OOM.

Final retry5 audit:

- `python -m py_compile` passed after the Stage-C and renderer changes.
- `unittest discover -s tests -p "test_*.py" -v`: 65 tests, OK; the targeted
  formal-runner suite is 18 tests, OK.
- The final retry5 directory contains all required CSV/PNG/JSON/PKL outputs and
  three real-data videos. `ffprobe` reports 121 frames at 30 fps for each.
- `t_move=89`; contact distance `0.0191540 m`; maximum diagnostic distance
  change `0.0129729 m`; boundary step `0.0153360 m`.
- The top-view rerender log contains no `Loading contact labels from cache` or
  `Contact labels (` line.
