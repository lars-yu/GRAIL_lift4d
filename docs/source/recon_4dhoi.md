# 4D HOI Reconstruction

Recovers the full 4D human-object interaction (SMPL-X body pose + MANO hand
pose + 6-DoF object trajectory) from a generated or captured RGB video.

## Quickstart

```bash
# Manipulation/pickup — SMPL-X (default, expects object to move)
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results

# Manipulation/pickup — SOMA body model
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --config configs/recon_4dhoi/manip_soma.yaml

# Terrain / sitting (static object — bypasses FoundationPose)
python -m grail.pipelines.recon_4dhoi --dataset syn_stairs --results_dir results \
    --config configs/recon_4dhoi/loco_smplx.yaml

# Dynamic/around-camera video — VGGT-Omega geometry backend
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --config configs/recon_4dhoi/pickup_smplx_vggt_dynamic.yaml
```

VGGT-Omega inference requires a CUDA-visible process. If the local shell cannot
see `/dev/nvidia*`, run the VGGT stages on the GPU host rather than relying on
CPU fallback. Cache-only steps such as VGGT-Blender alignment, observation
extraction, and validation can still run without CUDA when the required VGGT
outputs already exist and the expensive GPU stages are skipped or `--skip_done`
can reuse complete caches.

Validated outputs land under
`results/generation/4dhoi_recon_smplx_valid/{dataset}/{category}/{video_id}/`:

- `hoi_data/hoi_data.pkl` — body params + object 6-DoF poses per frame
- `result_vis/recon_result.mp4` — overlaid reconstruction on the input
- `result_vis/recon_comparison.mp4` — side-by-side input vs. recon
- `result_vis/recon_result_top_view.mp4` — top-down view
- `result_vis/recon_result.html` — interactive ScenePic viewer
- `mesh_data/` — the canonical object mesh used in optimization

## Pipeline steps

```{list-table}
:widths: 5 25 70
:header-rows: 1

* - #
  - Stage
  - Notes
* - 1
  - Human pose
  - GEM-SMPL body + WiLoR hands, fused per-frame. ~45 s/video on an L40S.
* - 2
  - Preprocess
  - SAM2 mask tracking + MoGe monocular depth for fixed-camera runs. VGGT runs keep SAM2 masks here and estimate geometry in the VGGT stages.
* - 2.1
  - VGGT reconstruction
  - Dynamic-camera only. Runs VGGT-Omega on all video frames and writes `vggt/depth.npy`, `confidence.npy`, `intrinsics.npy`, `c2w.npy`, `raw_scene.ply`, and `metadata.json`.
* - 2.2
  - VGGT-Blender alignment
  - Dynamic-camera only. Uses first-frame static pixels to estimate one Sim(3) from VGGT world to Blender world, then writes `vggt_aligned/c2w_blender.npy`, metric depth, metric point clouds, `aligned_scene.ply`, and `alignment/sim3_vggt_to_blender.json`.
* - 2.3
  - Dynamic human world motion
  - Dynamic-camera only. Converts GENMO/WiLoR camera-space root translation/orientation with per-frame `T_B<-C_t` and caches `vggt_aligned/<video_id>/human_motion/motion_world.npz` as a preview/debug artifact; local body pose, hand pose, shape, and auxiliary observations are preserved.
* - 2.4
  - VGGT observations
  - Dynamic-camera only. Extracts eroded SAM2 human/object/static point observations in Blender metric world space.
* - 3
  - Object pose
  - FoundationPose 6-DoF tracking from cached masks + RGB. Dynamic-camera runs pass per-frame VGGT intrinsics and initialize frame 0 with `T_C0<-O = inv(T_B<-C0) T_B<-O0`; VGGT depth is not fed into FoundationPose. Dynamic runs track the original frame sequence only, because ffmpeg-interpolated frames would need interpolated camera calibration. Dynamic runs also save `pose_estimation_output/poses_in_world.pkl` / `.npy` as `T_B<-O,t` for downstream validation.
* - 3.1
  - Dynamic object world pose
  - Dynamic-camera only. Backfills/validates `pose_estimation_output/poses_in_world.pkl` from FoundationPose `poses_in_cam.pkl` and `vggt_aligned/<video_id>/c2w_blender.npy` without re-running tracking.
* - 4
  - HOI optimization
  - Multi-stage; dynamic-camera runs keep VGGT camera fixed (`optimize_camera: false`) and optimize human/object residuals in Blender world. Dynamic-camera outputs do not apply the legacy final Savitzky-Golay trajectory smoothing; continuity is handled by the staged temporal losses and then validated in world space. If `pose_estimation_output/poses_in_world.pkl` exists, it is consumed as the object initialization while `poses_in_cam.pkl` remains available for tracking/interactions. The optimizer writes and reuses `vggt_aligned/<video_id>/human_motion/motion_world_optimizer.npz`, which includes the interaction-frame alignment and character height/shape normalization used by the actual optimization. Uses OpenAI vision calls inside `grail/core/contact_label.py`
    to detect contact joints per interval. Defaults use `gpt-4o`.
    Heaviest stage (~9-10 min/video on L40S).
* - 5
  - Filter
  - Quality thresholds: human-position error, mask alignment, keypoint
    tracking, contact penalty, penetration, motion magnitude. Dynamic-camera
    runs use the aligned VGGT `c2w_blender.npy` + per-frame intrinsics for
    object mask consistency, and skip the legacy SLAM camera-translation filter
    because camera motion is expected.
* - 6
  - Visualize
  - PyTorch3D top-down + side-by-side renders, ScenePic HTML.
```

## Dynamic-camera VGGT mode

The fixed-camera/MoGe path remains the default via:

```yaml
geometry_backend: moge
camera_mode: fixed
```

For generated orbit or moving-camera videos, use:

```yaml
geometry_backend: vggt
camera_mode: dynamic
vggt:
  checkpoint_path: imports/vggt-omega/checkpoints/vggt_omega_1b_512.pt
  scene_reference_dir: generation/scene_reference/{origin}
  alignment_mode: first_frame_correspondence  # first_frame_correspondence | static_scene_icp | hybrid
  static_scene_ply: null  # optional override; defaults to scene_reference_dir/static_scene.ply
  object_init_pose_path: null  # optional override; defaults to scene_reference_dir/object_init_pose.npy
```

Passing `--geometry_backend vggt` on the command line implies
`camera_mode=dynamic` unless `--camera_mode` is explicitly supplied; invalid
mixed pairs such as `geometry_backend=vggt` with `camera_mode=fixed` are rejected
before any expensive stage starts.

The dynamic cache layout is rooted under `results/generation/4dhoi_recon_cache/`:

- `vggt/{video_id}/` — raw VGGT depth/confidence/intrinsics/cameras and raw point cloud.
- `vggt_aligned/{video_id}/` — one global Sim(3), Blender-world dynamic cameras, metric depth, and aligned point clouds.
- `vggt_aligned/{video_id}/human_motion/motion_world.npz` — cached GENMO/WiLoR human motion with only the root pose transformed into Blender metric world.
- `vggt_aligned/{video_id}/human_motion/motion_world_optimizer.npz` — optimizer-compatible human world motion cache with interaction alignment and character height/shape normalization metadata.
- `vggt_observations/{video_id}/` — masked human/object/static VGGT points in Blender metric world.
- `foundation_pose_output/{video_id}/pose_estimation_output/poses_in_world.pkl` — cached object trajectory `T_B<-O,t`; validation uses this directly for raw object world trajectory diagnostics when optimization output is not available yet.
- `{output_dir}/{video_id}/validation/metrics.json` — camera trajectory and alignment diagnostics.

Before launching long GPU-heavy stages, a read-only cache audit can be run with:

```bash
python -m grail.pipelines.recon_4dhoi \
    --config configs/recon_4dhoi/pickup_smplx_vggt_dynamic.yaml \
    --check_vggt_readiness
```

To stop at the milestone recommended for moving-camera bring-up, run only the
geometry closed loop before launching GENMO/FoundationPose/optimization:

```bash
python -m grail.pipelines.recon_4dhoi \
    --config configs/recon_4dhoi/pickup_smplx_vggt_dynamic.yaml \
    --video_id dataset/category/video_id \
    --vggt_geometry_only --skip_done
```

This executes only VGGT reconstruction, VGGT-Blender Sim(3) alignment, and the
metric-camera/depth validation stage. It is the preferred first GPU run for a
new orbit video; once `vggt_aligned/c2w_blender.npy`, metric depth, alignment
PLYs, and validation metrics look correct, rerun the full dynamic pipeline.
The matching audit is:

```bash
python -m grail.pipelines.recon_4dhoi \
    --config configs/recon_4dhoi/pickup_smplx_vggt_dynamic.yaml \
    --video_id dataset/category/video_id \
    --check_vggt_readiness --vggt_geometry_only
```

This geometry-scoped readiness report checks only raw VGGT, scene reference,
aligned Sim(3)/metric depth, and geometry validation artifacts. The default
readiness scope remains `full`, which also requires observations,
FoundationPose world poses, dynamic human motion, optimization output, and
post-processed `_valid` output.
Validation caches record `validation_scope` in `metrics.json`: geometry-stage
validation writes `geometry`, while the post-optimization validation writes
`full`. A `full` cache satisfies the geometry audit, but a geometry-only cache
is not treated as complete for the full dynamic pipeline when `--skip_done` is
enabled.

It prints a JSON report for the VGGT raw cache, Blender scene reference, aligned
metric geometry, observations, dynamic human motion, and FoundationPose world
object pose. The command exits non-zero when required files are missing or frame
counts/shapes are inconsistent. The aligned-cache check also requires
`vggt_aligned/metadata.json` provenance, including `alignment_inputs`, so stale
Sim(3) outputs from a different VGGT cache or Blender reference are rejected
before long downstream stages reuse them.
When a validation directory is configured, the readiness report also checks the
camera plots, PLY alignment artifacts, `camera_motion.json`, and `metrics.json`
provenance/flag fields so a stale or partial validation cache cannot be treated
as a completed geometry closed-loop check.
For full-chain audits, the pipeline also passes the dynamic output
`hoi_data.pkl`; readiness verifies dynamic/VGGT metadata, fixed-camera
optimization (`optimize_camera: false`), no final trajectory smoothing,
world-coordinate human/object outputs, dynamic depth/static-scene eval terms,
and lightweight validation mesh samples.
It also checks the filtered/post-processed `_valid/.../hoi_data/hoi_data.pkl`
when present in the full pipeline, requiring `postprocess_preserve_world_coordinates`
and zero human/object offsets so retarget-ready outputs remain in Blender metric
world coordinates.

Static scene reference export is available for `.blend` files:

```bash
blender -b scene.blend --python scripts/export_scene_reference.py -- \
    --output_dir scene_reference --render_depth --render_masks --strict
```

The alignment step can consume this export directly: pass
`scene_reference/depth_gt_00000.exr`, `scene_reference/camera_init_K.npy`, and
`scene_reference/camera_init_c2w.npy` when running the alignment module outside
the full pipeline. The exporter writes EXR depth through Blender's compositor
when available and falls back to ray-cast camera-Z depth with the same OpenCV
intrinsics convention used by the alignment code.
For production references, keep `--strict` enabled so missing human/object/camera
artifacts fail early. If a scene uses non-standard mesh names, pass explicit
`--human_regex` and `--object_regex`; for static-only alignment debugging, add
`--allow_missing_human --allow_missing_object` intentionally.
Dynamic FoundationPose initialization uses `scene_reference/object_init_pose.npy`
when that file exists and computes `T_C0<-O = inv(T_B<-C0) @ T_B<-O0`; if a
static-only scene reference omits the manipulated object, the pipeline falls
back to the legacy `first_frame_pose.pickle` object pose. If dynamic
initialization is provided without a legacy `first_frame_pose.pickle`, the
adapter assumes the object mesh has already been exported at metric scale from
the scene reference. Human/object masks behave the same way: scene-reference
masks are used when present, otherwise the first-frame FoundationPose masks are
used as the foreground exclusions for alignment.
For datasets where the `.blend` contains only the static room but the legacy
FoundationPose prep directory has the target object pose/masks, enrich the
static-only reference without changing `static_scene.ply`:

```bash
python scripts/enrich_scene_reference_from_legacy.py \
    --scene_reference_dir scene_reference \
    --first_frame_pose foundation_pose/video_id/first_frame_pose.pickle \
    --human_mask foundation_pose/video_id/human_masks/000000.png \
    --object_mask foundation_pose/video_id/masks/000000.png \
    --object_mesh generation/mesh/object.obj
```

The helper writes `object_init_pose.npy` from `obj_R/obj_t`, records
`obj_scale` in metadata for audit, and copies optional masks/OBJ sidecars. The
scale is not folded into the pose matrix, matching the dynamic object
initialization path.

Inside the full pipeline, set `vggt.scene_reference_dir` to a directory pattern
such as `generation/scene_reference/{origin}`. Relative paths are resolved under
`results_dir`; `{video_id}`, `{origin}`, and `{results_dir}` are supported. For
non-standard layouts, override `vggt.blender_depth_path`, `vggt.blender_K_path`,
`vggt.blender_c2w_path`, and the optional mask/static-scene paths directly.

If the generated first frame is not pixel-aligned with the Blender seed render,
set `vggt.alignment_mode: hybrid` and point `vggt.static_scene_ply` at the
exported `scene_reference/static_scene.ply`. The pipeline first tries the
strong first-frame pixel correspondences, then falls back to static-scene ICP
using SAM2 masks to remove human/object pixels. The ICP fallback uses one
global Sim(3) for the whole video: robust nearest-neighbor/Umeyama updates for
coarse alignment followed by point-to-plane refinement against normals estimated
on the Blender static scene.

Dynamic validation writes camera and alignment diagnostics under
`{output_dir}/{video_id}/validation/`:

- `camera_trajectory_top.png`, `camera_trajectory_side.png`, `camera_speed.png`, `camera_rotation_speed.png`
- `vggt_raw_scene.ply`, `vggt_aligned_scene.ply`, `blend_scene.ply`, and `alignment_overlay.ply` when the corresponding VGGT/alignment/static-scene sources are present; `--skip_done` only trusts validation caches whose artifact files and `metrics.json` artifact flags match current alignment provenance.
- `human_root_trajectory.png`, `human_root_speed.png`, and `human_root_rotation_speed.png` when a dynamic human motion cache is present.
- `foundationpose_overlay.mp4` copied from FoundationPose's `pose_estimation_tracking.mp4` when available, so object-tracker failures can be inspected separately from the final optimizer overlay.
- `camera_motion.json` with per-frame translation/rotation deltas and `camera_jump_frames`
- `metrics.json` with Sim(3), alignment provenance (`alignment_inputs_*`), static-scene, depth, contact, grasp-relative translation/rotation stability, and pre-contact object-stability metrics when the corresponding optimization outputs exist. When `hoi_data.pkl` includes validation mesh samples, human/object depth metrics are recomputed per frame against the saved VGGT observation point clouds instead of relying only on scalar optimizer logs.

## Required environment

`OPENAI_API_KEY` is used by the OpenAI API for contact-joint detection in step 4.
The default vision model is `gpt-4o`.

## Common variants

```bash
# Single video by ID
python -m grail.pipelines.recon_4dhoi --video_id ComAsset/cordless_drill/<video_name> \
    --results_dir results

# Skip already-finished videos
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --skip_done

# Step 4+ only (after rerun of contact detection)
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --skip_step1 --skip_step2 --skip_step3

# Static-object mode (no global object motion expected)
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill \
    --results_dir results --is_static_obj
```

## Configs

Configs are split by **task** (manipulation vs. locomotion/terrain) × **body model** (SMPL-X vs. SOMA):

```{list-table}
:widths: 35 65
:header-rows: 1

* - File
  - Purpose
* - `configs/recon_4dhoi/manip_smplx.yaml`
  - Manipulation / pickup, SMPL-X (G1) body. `filter_object_motion: dynamic_only` drops static-object recons at step 5; FoundationPose runs (object expected to move). Default for `grail.pipelines.recon_4dhoi`.
* - `configs/recon_4dhoi/manip_soma.yaml`
  - Manipulation / pickup, SOMA body. Same params as `manip_smplx.yaml` — only `body_model` + paths differ.
* - `configs/recon_4dhoi/loco_smplx.yaml`
  - Locomotion / terrain / sitting, SMPL-X (G1) body. `is_static_obj: true` bypasses FoundationPose (terrain doesn't move); `filter_object_motion: static_only` keeps only static-object recons.
* - `configs/recon_4dhoi/loco_soma.yaml`
  - Locomotion / terrain / sitting, SOMA body. Same params as `loco_smplx.yaml`.
```

SOMA variants share **all** optimization params with their SMPL-X counterparts — only `body_model` + `hmr_dir` + `output_dir` differ.

## Sharded fan-out

```bash
# Run one shard per worker in your scheduler.
python -m grail.pipelines.recon_4dhoi \
    --dataset ComAsset \
    --results_dir results \
    --job_chunk_idx <i> \
    --num_job_chunks <N>
```

A typical 8-chunk run covers ~24 videos in ~35 minutes wall-clock (parallel).
