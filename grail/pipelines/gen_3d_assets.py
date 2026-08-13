"""
Batch 3D asset generation using Hunyuan3D-2.1.

Reads a list of objects from a YAML or JSON file, generates reference images
via the OpenAI Image API, produces textured 3D meshes with
Hunyuan3D-2.1, and scales them to real-world dimensions.

Usage:
    python generate_3d_assets.py -i config/example_objects.yaml -o batch_outputs
"""

import argparse
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Resolve Hunyuan3D-2.1 submodule paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HUNYUAN_ROOT = PROJECT_ROOT / "imports" / "Hunyuan3D-2.1"

sys.path.insert(0, str(HUNYUAN_ROOT / "hy3dshape"))
sys.path.insert(0, str(HUNYUAN_ROOT / "hy3dpaint"))
sys.path.insert(0, str(HUNYUAN_ROOT))

# Hunyuan3D's DifferentiableRenderer/mesh_utils.py imports `bpy` at module
# level, but the PyPI `bpy` wheels require Python 3.11+ while the `hunyuan`
# conda env is pinned to 3.10. The only consumer, `convert_obj_to_glb`, is
# never called here (grail always passes `save_glb=False`), so register a
# stub so the import succeeds.
if "bpy" not in sys.modules:
    import types

    sys.modules["bpy"] = types.ModuleType("bpy")

# Hunyuan3D imports (available after sys.path setup)
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore
from hy3dshape.rembg import BackgroundRemover  # type: ignore
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

try:
    from torchvision_fix import apply_fix  # type: ignore

    apply_fix()
except (ImportError, Exception):
    pass

from grail.adapters.openai_api import (
    DEFAULT_REASONING_MODEL,
    chat_text,
    chat_with_image,
    generate_image,
)

# ── helpers ────────────────────────────────────────────────────────────────


def slugify(name: str) -> str:
    """Turn an object name into a filesystem-safe folder name."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "object"


def load_object_list(path: Path) -> list[str]:
    """
    Load a list of object names from a YAML or JSON file.

    Expected format -- a plain list of strings:
        - Wooden dining chair with slat back
        - Metal bar stool with wooden seat
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of object names, got {type(data).__name__}")

    names = [str(item) for item in data]
    if not names:
        raise ValueError("Object list is empty")

    return names


# ── image generation ──────────────────────────────────────────────────────

_ENHANCE_SYSTEM_PROMPT = """\
You are an expert at creating prompts for image generation models to produce \
images that will be converted to 3D models for humanoid HOI training.

Enhance user prompts so the resulting image is perfect for 3D reconstruction:
1. The object MUST be a rigid body (no cloth, fabric, or flexible items).
2. Real-world size suitable for human manipulation (typical largest dimension
   5-150 cm). Reject too-small (spoon, paperclip) or too-large (fridge, sofa)
   categories.
3. No articulated parts. Avoid fridges with doors, scissors, foldable items,
   drawers, robot arms with joints -- Hunyuan3D produces a single rigid mesh
   and articulation is lost.
4. Avoid fully rotationally symmetric objects (plain spheres, plain balls,
   generic orbs). Prefer feature-rich surfaces with a clear "front" so
   downstream pose estimators can lock onto a canonical orientation.
5. The object should rest on its natural feet, legs, or contact surface in
   the image. Do NOT add a display plinth, museum pedestal, or flat slab
   base beneath the object unless the user explicitly requested one --
   Hunyuan3D bakes whatever base is in the image into the mesh, and phantom
   slabs clip through tables and floors downstream.
6. Centered, clearly visible, white/neutral background.
7. Product-photography style with studio lighting.
8. Single object only -- no complex scenes.
9. Clear, defined edges and surfaces. Specify a three-quarter viewing angle
   when appropriate.

Return ONLY the enhanced prompt text, nothing else."""


def enhance_prompt(user_prompt: str) -> str:
    """Rewrite *user_prompt* for better 3D-suitable images via the chat adapter."""
    return chat_text(
        prompt_text=f"Enhance this prompt for 3D object generation: {user_prompt}",
        max_tokens=200,
        temperature=0.7,
        system_prompt=_ENHANCE_SYSTEM_PROMPT,
    )


def generate_reference_image(obj_name: str, output_path: Path) -> Image.Image:
    """Generate a reference image for *obj_name* via the OpenAI Image API."""
    prompt = enhance_prompt(obj_name)
    print(f"  Enhanced prompt: {prompt}")

    generate_image(prompt=prompt, output_path=output_path)
    return Image.open(output_path).convert("RGBA")


# ── background removal & resize ──────────────────────────────────────────


def remove_background(
    image: Image.Image,
    rembg: BackgroundRemover,
) -> Image.Image:
    """Remove background from *image*, returning an RGBA image."""
    return rembg(image.convert("RGB"))


def resize_to_square(image: Image.Image, target_size: int = 512) -> Image.Image:
    """
    Resize *image* to fit inside *target_size x target_size* while preserving
    the aspect ratio, then center it on a transparent canvas.
    """
    w, h = image.size
    ratio = w / h
    if ratio > 1:
        new_w = target_size
        new_h = int(target_size / ratio)
    else:
        new_h = target_size
        new_w = int(target_size * ratio)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y), resized if resized.mode == "RGBA" else None)
    return canvas


# ── height estimation ─────────────────────────────────────────────────────


def estimate_height_with_openai_vision(obj_name: str, image_path: str) -> float | None:
    """
    Ask the configured OpenAI vision model to estimate the real-world height of the object in
    meters, using the project's existing ``chat_with_image`` utility.
    """
    system_prompt = (
        "You are an expert at estimating the real-world size of objects from "
        "reference images. You know typical dimensions of furniture, tools, "
        "appliances, and everyday items.\n\n"
        "For the given object, estimate its *height* in meters (bottom to top), "
        "based on typical real-world usage. If ambiguous, use your best guess.\n\n"
        'Respond ONLY with a JSON object: {"height_m": 1.23}\n'
        "No extra text."
    )
    user_prompt = (
        f"This is a reference image for: '{obj_name}'. " "Estimate its real-world height in meters."
    )

    try:
        text = chat_with_image(
            prompt_text=user_prompt,
            image_path=image_path,
            max_tokens=128,
            temperature=0.2,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        print(f"  Warning: height-estimation call failed: {exc}")
        return None

    try:
        data = json.loads(text)
        h = float(data["height_m"])
        return h if h > 0 else None
    except Exception:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            h = float(m.group(1))
            return h if h > 0 else None
        return None


# ── image–prompt verification ─────────────────────────────────────────────

_VERIFY_SYSTEM_PROMPT = (
    "You are a strict quality-assurance reviewer for AI-generated images "
    "destined for 3D reconstruction in a humanoid HOI pipeline. Given an "
    "object description and a generated image, determine whether the image "
    "accurately depicts the described object AND is suitable as a 3D-gen "
    "reference.\n\n"
    "Pay close attention to:\n"
    "- Specific features mentioned (e.g., materials, colors, shapes)\n"
    "- Explicit negations (e.g., 'no handrails' means handrails must be absent)\n"
    "- Object identity (the image must show the correct type of object)\n"
    "- HOI suitability: reject articulated meshes (fridges with doors, "
    "scissors, foldable items), fully rotationally symmetric objects "
    "(plain spheres/balls), objects clearly outside the 5-150 cm size range, "
    "or images showing a phantom flat-slab / display-plinth base beneath the "
    "object that the user did not request.\n\n"
    'Respond ONLY with JSON: {"matches": true, "reason": "..."}\n'
    "Set matches to true only if the image faithfully represents the "
    "description and passes the HOI-suitability checks above."
)


def verify_image_matches_prompt(obj_name: str, image_path: str) -> bool:
    """Return *True* if the generated image matches *obj_name*."""
    try:
        text = chat_with_image(
            prompt_text=f"Object description: '{obj_name}'\nDoes this image accurately depict the described object?",
            image_path=image_path,
            model=DEFAULT_REASONING_MODEL,
            max_tokens=1024,
            temperature=0.2,
            system_prompt=_VERIFY_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        print(f"  Warning: verification call failed: {exc}")
        return True  # fail-open

    # Parse JSON response with fallback
    matches = True
    reason = ""
    try:
        data = json.loads(text)
        raw = data.get("matches", True)
        matches = raw if isinstance(raw, bool) else str(raw).lower() in ("true", "yes", "1")
        reason = data.get("reason", "")
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                data = json.loads(m.group())
                raw = data.get("matches", True)
                matches = raw if isinstance(raw, bool) else str(raw).lower() in ("true", "yes", "1")
                reason = data.get("reason", "")
            except Exception:
                pass
        else:
            matches = "true" in text.lower()

    status = "pass" if matches else "FAIL"
    print(f"  Verification: {status} – {reason}")
    return matches


# ── OBJ scaling ───────────────────────────────────────────────────────────


def load_vertices(obj_path: Path) -> list[tuple[float, float, float]]:
    """Parse vertex positions from an OBJ file."""
    vertices: list[tuple[float, float, float]] = []
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        vertices.append(tuple(map(float, parts[1:4])))  # type: ignore[arg-type]
                    except ValueError:
                        continue
    return vertices


def compute_model_height(vertices: list[tuple[float, float, float]]) -> float:
    """Largest axis-aligned extent (robust when up-axis is unknown)."""
    if not vertices:
        return 0.0
    xs, ys, zs = zip(*vertices)
    return max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def scale_obj_in_place(obj_path: Path, scale: float) -> None:
    """Multiply all vertex positions in an OBJ file by *scale*."""
    lines: list[str] = []
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z = (float(v) * scale for v in parts[1:4])
                        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                        continue
                    except ValueError:
                        pass
            lines.append(line)

    with obj_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch 3D asset generation with Hunyuan3D-2.1",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to YAML or JSON file listing objects to generate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="batch_outputs",
        type=Path,
        help="Root output directory (default: batch_outputs).",
    )
    parser.add_argument(
        "--model-path",
        default="tencent/Hunyuan3D-2.1",
        help="HuggingFace model path for Hunyuan3D-2.1.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-generate objects even if model.obj already exists.",
    )
    parser.add_argument(
        "--max-image-retries",
        type=int,
        default=3,
        help="Max retries for image-prompt alignment verification (0 = no verification).",
    )
    parser.add_argument("--num_job_chunks", type=int, default=1)
    parser.add_argument("--job_chunk_idx", type=int, default=0)
    args = parser.parse_args()

    # --- Validate environment ---
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Source .env first.")

    # --- Load object list ---
    objects = sorted(load_object_list(args.input))
    objects = objects[args.job_chunk_idx :: args.num_job_chunks]
    root: Path = args.output
    root.mkdir(parents=True, exist_ok=True)

    # --- Initialize pipelines (once) ---
    print(
        f"Worker {args.job_chunk_idx}/{args.num_job_chunks} | "
        f"Initializing Hunyuan3D-2.1 for {len(objects)} objects from {args.input}"
    )

    rembg = BackgroundRemover()

    pipeline_shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)

    conf = Hunyuan3DPaintConfig(6, 512)
    conf.realesrgan_ckpt_path = str(HUNYUAN_ROOT / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = str(HUNYUAN_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = str(HUNYUAN_ROOT / "hy3dpaint" / "hunyuanpaintpbr")
    pipeline_paint = Hunyuan3DPaintPipeline(conf)
    print("Pipelines ready.")

    # --- Process each object ---
    for idx, obj_name in enumerate(objects, start=1):
        obj_slug = slugify(obj_name)
        obj_dir = root / obj_slug
        obj_dir.mkdir(parents=True, exist_ok=True)

        textured_mesh_path = obj_dir / "model.obj"

        if args.skip_existing and textured_mesh_path.exists():
            print(f"  [{idx}/{len(objects)}] Skip '{obj_name}' (exists)")
            continue

        print(f"\n[{idx}/{len(objects)}] {obj_name} → {obj_dir}")

        try:
            # 1. Generate reference image (with optional alignment verification)
            raw_image_path = obj_dir / "generated_raw.png"
            image = None
            max_retries = args.max_image_retries
            if max_retries > 0:
                for attempt in range(1, max_retries + 1):
                    image = generate_reference_image(obj_name, raw_image_path)
                    if verify_image_matches_prompt(obj_name, str(raw_image_path)):
                        break
                    print(f"  Retry {attempt}/{max_retries}: image did not match prompt")
                    image = None
                if image is None:
                    print(
                        f"  Skipping '{obj_name}' — failed alignment after {max_retries} attempts"
                    )
                    continue
            else:
                image = generate_reference_image(obj_name, raw_image_path)

            # 2. Remove background & resize
            image_nobg = remove_background(image, rembg)
            processed = resize_to_square(image_nobg)
            processed_path = obj_dir / "generated_512.png"
            processed.save(processed_path)

            # 3. Shape generation
            mesh = pipeline_shape(image=processed)[0]
            mesh_path = obj_dir / "mesh.glb"
            mesh.export(str(mesh_path))

            # 4. Texture generation
            pipeline_paint(
                mesh_path=str(mesh_path),
                image_path=str(processed_path),
                output_mesh_path=str(textured_mesh_path),
                save_glb=False,
            )

            # 5. Scale to real-world dimensions
            target_height: float | None = estimate_height_with_openai_vision(
                obj_name,
                str(raw_image_path),
            )

            if target_height is not None and target_height > 0:
                vertices = load_vertices(textured_mesh_path)
                current_height = compute_model_height(vertices)
                if current_height > 0:
                    scale_factor = target_height / current_height
                    backup = obj_dir / "model_original.obj"
                    if not backup.exists():
                        shutil.copy(textured_mesh_path, backup)
                    scale_obj_in_place(textured_mesh_path, scale_factor)
                    print(f"  Scaled to {target_height:.3f}m (factor {scale_factor:.4f})")

            # 6. Clean up intermediate files
            keep_files = {
                "model.obj",
                "model.mtl",
                "model.jpg",
                "generated_512.png",
                "generated_raw.png",
            }
            for f in obj_dir.iterdir():
                if f.is_file() and f.name not in keep_files:
                    f.unlink()

        except Exception as e:
            print(f"  Error: {obj_name}: {e}\n{traceback.format_exc()}")

    print(f"\nAll {len(objects)} objects processed → {root.resolve()}")


if __name__ == "__main__":
    main()
