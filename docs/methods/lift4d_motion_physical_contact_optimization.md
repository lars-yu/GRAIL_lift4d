# Lift4D Motion and Physical Palm Contact Optimization

## Scope

This method combines GRAIL human reconstruction with a real Lift4D motion-only
prior for object camera-Z motion and a semantic palm/finger contact objective.
The implementation is fixed-camera and uses only real RGB, HMR, masks, depth
cache, FoundationPose poses, the real object mesh, and the real Lift4D NPZ.
Synthetic contact frames, Kabsch object alignment, object XY/rotation
optimization, and free XYZ human translation are prohibited.

## Coordinates and Inputs

Human and object optimization states are stored in world coordinates. Rendering
uses the fixed OpenCV camera. FoundationPose supplies the object image-plane
ray, translation anchor, and rotation; GRAIL's renderer intrinsics are the only
intrinsics used for palm projection and contact rays.

For an observed palm pixel `(u, v)` and GRAIL intrinsics `K`, the camera ray is

```python
ray_cam = torch.stack([(u - cx) / fx, (v - cy) / fy,
                       torch.ones_like(u)], dim=-1)
ray_cam = ray_cam / torch.linalg.norm(ray_cam, dim=-1, keepdim=True)
ray_world = torch.matmul(camera_to_world_R, ray_cam[..., None]).squeeze(-1)
```

The ray is transformed with the fixed camera-to-world rotation and never with
Lift4D's camera matrix.

## Lift4D Prior and Object Constraints

Stable Lift4D point trajectories are converted to a relative camera-Z target
anchored to the first FoundationPose depth. Lift4D supervises all frames' Z
motion, while FoundationPose remains the source of object XY and rotation.
The static interval is hard-frozen. Object depth residuals are the only object
optimization variables; contact/object losses detach object vertices and
translations, so `d(contact_loss)/d(obj_depth_res) == 0`.

## Motion State and Contact Frames

`t_move` is detected from sustained mask/centroid/area evidence and Lift4D
motion. It is not assumed to be the first physical hand contact. The local
scanner evaluates candidate frames `[t_move - 8, t_move]` from identical
initialized parameters. It selects the earliest row satisfying all physical
patch gates, preferring modes B, C, D, E, F. If none passes, it records
`local_contact_feasible=false` and selects the lowest normalized Pareto score.

## Semantic Contact Patches

The palm and finger patches are the existing semantic G1-SMPL-X mappings. The
scanner reports nearest-mesh min/median distance, the fraction of semantic
vertices within 1 cm, signed penetration and penetrating fraction. Bone palm
center error remains a diagnostic only; it is not a physical 5 mm acceptance
gate.

## Local Reachability Modes

The scanner tests: A arms; B arms plus shoulders/clavicles; C and D add a
one-dimensional camera-ray root residual bounded at 2 cm and 3 cm; E and F add
upper torso/spine with 3 cm and 5 cm bounds. The residual is represented by the
bounded scalar `max_distance * tanh(raw_distance)` multiplied by the normalized
world ray and the minimum-jerk approach ramp. No free XYZ root residual exists.

## Four Optimization Stages

### Stage A: Object depth

Optimize only Lift4D camera-Z residuals with static hard-freeze and FoundationPose
image-plane/rotation constraints.

### Stage B1: Smooth approach

Optimize the selected human joint scope over `approach_start -> t_contact`.
The minimum-jerk ramp is `10u^3 - 15u^4 + 6u^5`; it has zero endpoint velocity.
Palm surface, coverage, penetration, reprojection, silhouette, keypoint, pose,
velocity, acceleration, and jerk terms are active.

### Stage B2: Static contact hold

Optimize only `t_contact -> t_move` while the object is static. The closest
semantic palm vertices at `t_contact` are recorded as a detached fixed patch
anchor. This stage prevents a contact solution from immediately leaving the
object before motion begins.

### Stage C: Common motion

Freeze all human pose frames `<= t_move` and optimize only post-motion frames.
The contact patch center at `t_contact` defines the translation-only relative
anchor:

```python
contact_offset = patch_center[t_contact].detach() - obj.trans[t_contact].detach()
target = obj.trans[t_move + 1:].detach() + contact_offset[None]
```

No object rotation is introduced. Contact hand pose residuals are Stage-C-only;
the non-contact hand remains frozen.

## Losses and Gradient Flow

Palm ray/depth/3D losses use detached physical targets. Surface and coverage
losses use semantic patch vertices against a detached object mesh. Penetration
uses the same detached object geometry. Reprojection and keypoint losses retain
human gradients. Temporal path, velocity, acceleration, jerk, boundary, and
post-contact relative losses enforce continuity. The post-contact relative loss
uses the fixed semantic patch center, never an uncontacted bone palm center.

## Acceptance and Debug Results

The physical contact gates are palm patch median distance `<= 15 mm`, coverage
within 1 cm `>= 30%`, maximum penetration `< 3 mm`, palm reprojection `<= 5 px`,
body/hand RMSE increase `<= 5 px`, and mask IoU decrease `<= 0.03`. Full-sequence
gates additionally require approach/boundary step `< 30 mm`, adjacent contact
change `< 13 mm`, moving contact under 5 cm `>= 95%`, p95 palm reprojection
`< 5 px`, and zero object-depth contact gradient. A failed run is retained as
`formal_result=false` debug output and is never presented as formal success.

## Rendering

The saved real `hoi_data.pkl` is rendered with the fixed GRAIL camera and real
object mesh. Front overlays use the original RGB video; top views use a fixed
scene-centered camera. Debug outputs must be labelled `DEBUG — formal_result=false`.
The contact-window view covers `[t_contact - 10, t_move + 15]` and overlays frame,
phase, patch median/coverage, penetration, reprojection, drift, and ray residual.

## Reproducibility and Limits

The formal command records all real input paths, automatic `t_move`, selected
`t_contact`, optimization mode, stage iterations, and scan JSON in result meta.
Retry36 changed physical patch selection, automatic contact selection, and the
B1/B2/C windows. Retry37 made the single follow-up continuity correction found
by retry36: Stage-C post-contact anchoring now uses the detached semantic patch
center rather than an uncontacted bone palm center. Retry37 was the final
121-frame experiment. It remained `formal_result=false` because the physical
patch, moving-contact, continuity, approach-step, p95 reprojection, and surface
coverage gates still failed. No retry38 or retry39 was started. The retained
debug renders are evidence for diagnosis only and must not be presented as a
formal success.
