#!/usr/bin/env python3
"""
Batch convert USD/USDA files collision approximation to SDF Mesh

Features:
  - Set physics:approximation to "sdf"
  - Apply PhysxSDFMeshCollisionAPI
  - Set SDF resolution parameter

Usage:
  # Single file
  python qben/convert_collision_to_sdf.py --file path/to/model.usd

  # Batch process folder
  python qben/convert_collision_to_sdf.py --dir path/to/folder

  # Recursive process subfolders
  python qben/convert_collision_to_sdf.py --dir path/to/folder --recursive

  # Dry-run mode (preview without modifying)
  python qben/convert_collision_to_sdf.py --file path/to/model.usd --dry-run

  # Custom SDF resolution (default 256)
  python qben/convert_collision_to_sdf.py --file path/to/model.usd --sdf-resolution 512

Example:
  python qben/convert_collision_to_sdf.py \
    --dir data/motion_lib_genhoi/motion_lib_hoi_expt_withcontacts/object_usd \
    --recursive
"""
import argparse
from pathlib import Path

from pxr import Sdf, Usd

try:
    from pxr import PhysxSchema

    HAS_PHYSX = True
except ImportError:
    HAS_PHYSX = False


def convert_to_sdf(file_path: Path, dry_run: bool = False, sdf_resolution: int = 256):
    stage = Usd.Stage.Open(str(file_path))
    if not stage:
        print(f"Failed to open: {file_path}")
        return False

    modified_count = 0

    for prim in stage.Traverse():
        approx_attr = prim.GetAttribute("physics:approximation")
        if not approx_attr or not approx_attr.IsValid():
            continue

        current = approx_attr.Get()
        if current == "sdf":
            continue

        print(f"  {prim.GetPath()}: {current} -> sdf")

        if not dry_run:
            if HAS_PHYSX:
                approx_attr.Set(PhysxSchema.Tokens.sdf)
                sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                sdf_api.CreateSdfResolutionAttr().Set(sdf_resolution)
            else:
                approx_attr.Set("sdf")
                current_schemas = list(prim.GetAppliedSchemas())
                if "PhysxSDFMeshCollisionAPI" not in current_schemas:
                    current_schemas.append("PhysxSDFMeshCollisionAPI")
                    new_listop = Sdf.TokenListOp()
                    new_listop.explicitItems = current_schemas
                    prim.SetMetadata("apiSchemas", new_listop)

                sdf_res_attr = prim.CreateAttribute(
                    "physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int, custom=False
                )
                sdf_res_attr.Set(sdf_resolution)

        modified_count += 1

    if modified_count > 0 and not dry_run:
        stage.GetRootLayer().Save()
        print(f"✓ Saved {file_path} ({modified_count} prims)")

    return True


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=Path)
    group.add_argument("--dir", "-d", type=Path)
    parser.add_argument("--recursive", "-r", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sdf-resolution", type=int, default=256)
    args = parser.parse_args()

    if not HAS_PHYSX:
        print("Warning: PhysxSchema not found, using manual method")

    if args.file:
        convert_to_sdf(args.file, args.dry_run, args.sdf_resolution)
    else:
        pattern = "**/*.usd*" if args.recursive else "*.usd*"
        files = list(args.dir.glob(pattern))
        print(f"Found {len(files)} USD files")
        for f in sorted(files):
            if f.suffix in [".usd", ".usda", ".usdc"]:
                print(f"\n{f}")
                convert_to_sdf(f, args.dry_run, args.sdf_resolution)


if __name__ == "__main__":
    main()
