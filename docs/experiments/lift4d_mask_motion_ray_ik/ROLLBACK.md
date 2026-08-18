The change is isolated to branch `codex/lift4d-mask-motion-ray-ik`. After the
single commit is pushed, rollback is recoverable with:

```bash
git revert <commit-sha>
```

Do not reset or overwrite the protected baseline worktree.

The formal run used only real inputs from rand00033 and was written under a
new output directory:

`pickup_table/generation/lift4d_mask_motion_ray_ik/rand00033_formal_20260818_retry6`

The completed comparison artifacts are under its `rendered/` subdirectory.
The attempted CPU top-view output is not certified because rendering failed
before producing a valid MP4.
