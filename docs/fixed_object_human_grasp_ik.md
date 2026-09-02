# Fixed-object human grasp IK

This mode treats the recovered object trajectory as immutable and solves a new
SMPL-X human trajectory around it. It is intended for clips where the object
pose is reliable but monocular human depth leaves the hand too far from the
object.

## Objective

At the detected contact frame `t_move`, the ray-cast object-shell point is
converted from world coordinates into the object's local frame. The palm
centre is not placed directly on that shell: its centre-to-mesh thickness is
estimated from the initial posed SMPL-X palm and clamped to 1.2--4 cm. This
removes the contradictory objective that previously pulled an internal palm
joint through the object while penetration loss pushed the hand mesh out.

```text
p_grasp_obj = (p_target_world - t_obj[t_move]) R_obj[t_move]
```

The post-contact target then follows the complete fixed object SE(3) motion:

```text
p_target_world[t] = p_grasp_obj R_obj[t]^T + t_obj[t]
```

Using an object-local anchor is important: a constant world-space translation
offset slides when the object rotates.

The human solve is whole-body and hierarchical:

1. Seed a bounded root-translation ramp only over the approach window; earlier
   recovered HMR frames remain unchanged.
2. Optimize human root rotation/translation, hips, knees, ankles, torso, and
   the contact-side arm. The opposite arm and head stay locked.
3. Build an immutable world-space ankle anchor for every contiguous support
   episode. Root translation remains optimizable on supporting frames, while
   leg IK keeps the supporting foot on its anchor.
4. Project every body-joint residual into a hard anatomical trust region and
   penalize contact-elbow angles outside 5--165 degrees.
5. Enforce palm-centre position, signed palm normal, signed palm tangent,
   partial palm-shell coverage, and penetration losses. Finger residuals are
   frozen by default.

The runner rejects a result unless contact-frame and all post-contact palm
errors are at most 5 mm by default. It also rejects either support-foot world
anchor error or adjacent support-foot displacement above 1 cm, and rejects an
elbow angle above 170 degrees. The object pose is compared against its frozen
reference at the end of the solve with a `1e-8` numerical tolerance.

## Usage

Use the normal formal runner arguments and add:

```bash
python scripts/run_lift4d_vggt_optimization.py \
  --config-file <config.yaml> \
  --video-id <video_id> \
  --video-file <video.mp4> \
  --hmr-file <hmr.npz> \
  --mesh-file <object.obj> \
  --foundationpose-poses <poses.pkl> \
  --render-config <render.pkl> \
  --cache-dir <cache_dir> \
  --results-dir <results_dir> \
  --lift4d-prior <lift4d_prior.npz> \
  --output-dir <output_dir> \
  --fixed-object-human-ik
```

Useful controls:

- `--max-human-global-alignment 0.35`: maximum per-frame root correction.
- `--fixed-grasp-threshold 0.005`: hard palm-to-grasp acceptance threshold in metres.
- `--refine-contact-fingers`: optional Stage-C finger refinement; off by default.
- Fixed-object whole-body IK solves one contact arm. An automatic `both`
  candidate is resolved to the nearer hand; an explicit `--contact-hand both`
  fails fast so two independent palms are never collapsed into one target.
- `--stage-b-niter` and `--stage-c-niter`: pre-contact and post-contact solve iterations.

The serialized output records the object-local grasp anchor, its transported
world trajectory, and the final object-pose-lock diagnostics under `meta`.
