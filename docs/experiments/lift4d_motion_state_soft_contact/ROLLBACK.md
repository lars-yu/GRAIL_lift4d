# Rollback

To retain the implementation but restore the prior optimization behavior, set:

```yaml
object_motion_state:
  enabled: false
```

This restores the pre-change raw smoothed Z targets, hard contact timing, legacy
30/5/10 loss weights, and disables the pre-motion static lock. VGGT remains
optional; omit `--use-vggt-human-depth` to keep it out of all losses.

To completely undo the experiment after it is committed, use the recoverable
revert operation:

```bash
git revert <experiment-commit-hash>
```

Do not use `git reset --hard`; the protected source worktree contains unrelated
user changes.
