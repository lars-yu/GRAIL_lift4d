# Fixed-object adaptive human IK

This mode keeps the complete recovered object trajectory immutable and chooses
the smallest human solve that can satisfy the physical palm target.

## Outer cascade

The cascade is an outer state machine, not a differentiable mode switch inside
one optimizer iteration:

1. **M0 upper body** optimizes the contact-side arm, shoulders, and torso.  The
   root and both legs are frozen.  Adaptive mode gives this probe 250 iterations.
2. **M1 planted stance** inherits M0, opens the root and lower body, but limits
   the ground root correction to 8 cm.  HMR support episodes remain anchored in
   world space.  Contact, anatomy, and both 1 cm support-anchor gates must pass.
3. **M2 step aware** is used only when M1 fails.  M1 is rolled back, preserving
   the useful M0 upper-body seed.  The solver then creates alternating stance
   and swing phases and optimizes the full body against them.

Only the selected mode is continued into Stage C.  The failed probe records and
their rejection reasons are written to `optimization_metrics.json`.

## Step plan

For a required ground root displacement `d` and maximum individual step length
`L`, M2 uses `rounds = ceil(||d|| / L)` and two steps per round.  Each round
advances both feet by the same fraction of `d`.  For example, a 34 cm correction
with the default 22 cm limit creates four swings: right, left, right, left; every
individual foot displacement is about 17 cm.

During a swing, horizontal interpolation uses smoothstep and vertical motion
uses a sinusoidal 5 cm arc.  The other foot remains a fixed world-space stance
anchor.  After touchdown, the new location becomes the next stance anchor; the
foot is never pulled back to its original HMR coordinate.  The root reference
is the mean ground displacement of both planned feet, so the pelvis advances as
the footsteps advance instead of sliding independently.

## Command-line controls

The existing fixed-object command enables the cascade by default:

```bash
--fixed-object-human-ik \
--fixed-object-ik-mode adaptive \
--ik-probe-niter 250 \
--ik-stance-niter 350 \
--stance-root-max 0.08 \
--max-step-length 0.22 \
--swing-foot-clearance 0.05 \
--max-approach-steps 4 \
--min-swing-frames 8 \
--step-settle-frames 3 \
--first-step auto
```

`--fixed-object-ik-mode step` forces M2 for debugging.  `upper-body` and
`stance` force the corresponding single mode and fail fast if its gates do not
pass.

## Acceptance

M2 evaluates foot sliding only across adjacent stance frames.  Swing frames and
lift-off/touchdown transitions are not misclassified as sliding.  In addition
to the existing fixed-grasp, palm, penetration, and elbow gates, M2 requires:

- stance-anchor mean and maximum error at most 1 cm;
- swing target maximum error at most 3 cm;
- touchdown target maximum error at most 2 cm;
- achieved swing clearance at least 2 cm;
- final touchdown step at most 3 cm per frame.

The diagnostics include the selected mode, every attempted mode, step count,
maximum planned step length, swing/touchdown errors, swing clearance, and stance
foot sliding.
