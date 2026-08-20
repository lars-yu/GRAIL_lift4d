# Rollback

The complete pre-change source state is commit
`166043ea78d425a78068c4e575e9e61f3da9155a` on
`codex/lift4d-smooth-hand-approach-fix`.

To return without rewriting shared history:

```bash
git fetch origin
git switch codex/lift4d-smooth-hand-approach-fix
git reset --keep 166043ea78d425a78068c4e575e9e61f3da9155a
```

To undo this change after it is committed on its feature branch, create a new
revert commit with `git revert <palm-fix-commit>`. Runtime `imports/` and `data/`
symlinks must not be staged, deleted, or restored as part of rollback.
