# Lift4D Palm-Ray Contact And Rendering Fix

- Replaces whole-hand means with semantic wrist/MCP palm centers.
- Adds explicit SMPL-X/G1-SMPL-X palm and finger patch mappings; unsupported
  backends fail fast.
- Uses observed wrist/MCP pixels with recorded per-frame fallback.
- Uses GRAIL renderer intrinsics for palm rays, SMPL-X/object projection, and
  formal rendering; Lift4D intrinsics remain diagnostic-only.
- Adds palm reprojection, depth, 3D target, surface, normal, coverage,
  penetration, path, and temporal constraints.
- Opens hand pose residuals only for the contact hand during Stage C and masks
  gradients outside the post-contact window. Stage B explicitly rejects
  `hand_pose_res`; its approach phase remains body/arm-only as required.
- Adds palm contact/reprojection diagnostics and strict formal gates.
- Preserves Stage A camera-Z-only Lift4D supervision, FoundationPose image-plane
  position and rotation, static locking, and zero contact gradient to object depth.
- Keeps the Stage C hand residual learning rate at `5e-5` and freezes the
  Stage-B `t_move` endpoint by starting hand/body refinement at `t_move + 1`.
