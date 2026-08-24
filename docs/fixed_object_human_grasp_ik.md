# Fixed-object human grasp IK

This mode treats the recovered object trajectory as immutable and solves a new
SMPL-X human trajectory around it. It is intended for clips where the object
pose is reliable but monocular human depth leaves the hand too far from the
object.

## Objective

At the detected contact frame `t_move`, the surface target is converted from
world coordinates into the object's local frame:

```text
p_grasp_obj = (p_target_world - t_obj[t_move]) R_obj[t_move]
```

The post-contact target then follows the complete fixed object SE(3) motion:

```text
p_target_world[t] = p_grasp_obj R_obj[t]^T + t_obj[t]
```

Using an object-local anchor is important: a constant world-space translation
offset slides when the object rotates.

The human solve is hierarchical:

1. Apply one bounded, constant ground-plane translation to the complete human
   track. Because it is constant over time, it adds no foot velocity.
2. Optimize only torso, shoulders, arms, and the contact hand. Pelvis/root
   rotation and all leg joints stay locked.
3. Permit per-frame root translation only where neither foot is classified as
   supporting. Supporting frames receive zero root-translation gradient.
4. Enforce palm position, palm normal, surface coverage, and penetration losses.

The runner rejects a result unless contact-frame and all post-contact palm
errors are at most 5 mm by default. It also rejects support-foot displacement
above 1 cm in any adjacent frame. The object pose is compared against its
frozen reference at the end of the solve with a `1e-8` numerical tolerance.

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

- `--max-human-global-alignment 0.35`: maximum constant ground-plane correction.
- `--fixed-grasp-threshold 0.005`: hard palm-to-grasp acceptance threshold in metres.
- `--stage-b-niter` and `--stage-c-niter`: pre-contact and post-contact solve iterations.

The serialized output records the object-local grasp anchor, its transported
world trajectory, and the final object-pose-lock diagnostics under `meta`.
