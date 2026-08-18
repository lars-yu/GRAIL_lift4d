# Rollback

The two logical commits on this branch are:

1. `d1b246f33ebb115582ac307938e86bab0e720cb2` — mask-first onset and Stage-A
   ray target refresh.
2. The final branch `HEAD` — smooth endpoint, Stage-B/C overlap, local hand
   trajectory losses, diagnostics, tests, and formal-output documentation.

To remove this experiment while retaining the parent branch:

```bash
git switch codex/lift4d-smooth-hand-approach-fix
git revert HEAD
git revert d1b246f33ebb115582ac307938e86bab0e720cb2
```

Generated outputs, input data, checkpoints, and runtime `imports/data`
symlinks are outside the commits and are not removed by a Git revert.
