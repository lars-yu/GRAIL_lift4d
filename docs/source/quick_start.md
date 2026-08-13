# Quick Start

A 5-minute smoke run that exercises every stage of the pipeline on a single
example asset (`cordless_drill`). Assumes you've completed
[`installation`](installation.md) and have the docker image (or the
`grail` / `hunyuan` conda envs) available.

Pipeline stages are launched as package modules, for example
`python -m grail.pipelines.gen_2dhoi` and
`python -m grail.pipelines.recon_4dhoi`. Do not use project-root wrapper
scripts for pipeline stages.

## 1. Source environment keys

```bash
source .env   # OPENAI_API_KEY, KLING_ACCESS_KEY, KLING_SECRET_KEY, HF_TOKEN
```

`OPENAI_API_KEY` is used by the OpenAI API for prompt refinement and
contact/scale helpers. Chat and vision helpers default to `gpt-4o`.

## 2. Generate a 3D object (procedural — no GPU needed)

```bash
python -m grail.pipelines.gen_terrain --type stairs --num 5 --output_dir data/syn_stairs
```

Outputs `data/syn_stairs/stairs_{000..004}/model.obj` plus `texture.jpg`.

## 3. Generate a 2D HOI video for one object

```bash
python -m grail.pipelines.gen_2dhoi \
    --dataset ComAsset --category cordless_drill \
    --character kid \
    --results_dir results --video_model_api kling-ai
```

The pipeline runs four stages: physics simulation, optional scale
optimization, multi-view Blender rendering, and Kling-AI video
generation. Total wall time: ~6–10 min per object on an L40S.

## 4. Reconstruct 4D motion from the video

```bash
python -m grail.pipelines.recon_4dhoi \
    --dataset ComAsset --category cordless_drill --results_dir results
```

Six stages: GEM-SMPL pose, SAM2 + MoGe preprocess, FoundationPose object
tracking, multi-stage HOI optimization, filter, visualize. Runs in
~30–40 min per video on an L40S.

## 5. Inspect outputs

```bash
ls results/generation/4dhoi_recon_smplx_valid/ComAsset/cordless_drill/*/result_vis/
```

You'll see `recon_result.mp4`, `recon_comparison.mp4`,
`recon_result_top_view.mp4`, and an HTML viewer. The validated motion data
(SMPL-X params + object 6-DOF poses) lives in `hoi_data/hoi_data.pkl`.

## Next steps

| You want to … | Doc |
|---|---|
| Generate Hunyuan3D textured assets | [`gen_3d_assets`](gen_3d_assets.md) |
| Render and re-render scenes with new characters | [`gen_2dhoi`](gen_2dhoi.md) |
| Tune the 4D reconstruction optimizer | [`recon_4dhoi`](recon_4dhoi.md) |
| Retarget to Unitree G1 | [`retargeting`](retargeting.md) |
| Train a SONIC policy | [`tracking`](tracking.md) |
| Browse a motion library in the browser | [`web_visualizer`](web_visualizer.md) |
