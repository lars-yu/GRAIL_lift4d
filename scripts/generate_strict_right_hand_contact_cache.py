#!/usr/bin/env python3
"""Generate strict right-hand-only contact label caches for pickup_table videos.

Policy:
- GPT is used only to find the first frame where right-hand physical contact starts.
- Contact is true only when the hand/fingers/palm visibly touch or hold the object;
  near-but-not-touching is false.
- The generated cache labels after the first contact frame are forced to ["R_Hand"].
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grail.adapters.openai_api import chat_with_image  # noqa: E402

STRICT_CONTACT_PROMPT = """
You are a very conservative frame-level contact detector for a human-object interaction reconstruction pipeline.

Task: decide whether FIRST PHYSICAL CONTACT has already occurred in this single frame.

Object: the pink intraoral scanner and the white base/tail beneath it. Treat the pink scanner and white base/tail as one single object.
Body part to judge: ONLY the person's RIGHT hand, right fingers, or right palm.

Return in_contact=true ONLY if the RIGHT hand/fingers/palm is visibly physically touching, grasping, or holding the object in this frame.
The hand and object must be extremely close with no visible gap: fingertips/palm should visibly overlap, occlude, press against, wrap around, or clearly touch the object.

Return in_contact=false for all of these cases:
- the right hand is merely approaching, reaching, hovering, pointing, or close but not visibly touching;
- there is any visible gap between the right hand and the object;
- contact is ambiguous because of blur, occlusion, perspective, shadow, or low resolution;
- only the left hand touches or approaches the object;
- the arm/wrist is near the object but the hand/fingers/palm are not clearly touching it;
- the object is moving but the right hand is not clearly touching it.

Be strict: if you are not sure, answer false.
Respond with strictly valid JSON only, using this schema:
{"in_contact": true/false, "reason": "short reason"}
""".strip()


def parse_bool(raw: str) -> tuple[bool, str]:
    try:
        data = json.loads(raw)
        val = data.get("in_contact")
        reason = str(data.get("reason", ""))[:240]
        if isinstance(val, bool):
            return val, reason
        if isinstance(val, str):
            return val.lower() in {"true", "yes", "1"}, reason
    except Exception:
        pass
    low = raw.lower()
    if '"in_contact"' in low and "true" in low:
        return True, raw[:240]
    return False, raw[:240]


def ensure_frames(video_path: Path, frame_dir: Path, max_frames: int | None = None) -> tuple[list[Path], int]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame_count = source_frame_count
    if max_frames is not None and max_frames > 0:
        target_frame_count = min(source_frame_count, max_frames) if source_frame_count > 0 else max_frames

    existing = sorted(frame_dir.glob("*.jpg"))
    if target_frame_count > 0 and len(existing) >= target_frame_count:
        cap.release()
        return existing[:target_frame_count], source_frame_count

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = frame_dir / f"{idx:06d}.jpg"
        if not out.exists():
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        idx += 1
        if max_frames is not None and max_frames > 0 and idx >= max_frames:
            break
    cap.release()

    frames = sorted(frame_dir.glob("*.jpg"))
    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"No frames extracted for {video_path}")
    return frames, source_frame_count or len(frames)


def contact_decider(model: str, max_retries: int, sleep_s: float):
    cache: dict[str, tuple[bool, str, str]] = {}

    def decide(image_path: Path) -> tuple[bool, str, str]:
        key = str(image_path)
        if key in cache:
            return cache[key]
        last_err = ""
        for attempt in range(1, max_retries + 1):
            try:
                raw = chat_with_image(
                    prompt_text=STRICT_CONTACT_PROMPT,
                    image_path=str(image_path),
                    model=model,
                    max_tokens=256,
                    temperature=0.0,
                    system_prompt="You are a strict visual contact classifier. Return JSON only.",
                    response_format={"type": "json_object"},
                )
                val, reason = parse_bool(raw)
                cache[key] = (val, reason, raw[:500])
                return cache[key]
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {str(exc)[:300]}"
                if attempt < max_retries:
                    time.sleep(sleep_s * attempt)
        raise RuntimeError(f"GPT contact check failed for {image_path}: {last_err}")

    return decide


def find_contact_start(frames: list[Path], interval: int, model: str, max_retries: int) -> tuple[int, list[dict]]:
    decide = contact_decider(model=model, max_retries=max_retries, sleep_s=3.0)
    n = len(frames)
    decisions: list[dict] = []

    first_positive = None
    prev_probe = -1
    for idx in range(0, n, interval):
        val, reason, _ = decide(frames[idx])
        decisions.append({"frame": idx, "phase": "coarse", "in_contact": val, "reason": reason})
        if val:
            first_positive = idx
            break
        prev_probe = idx

    if first_positive is None:
        idx = n - 1
        val, reason, _ = decide(frames[idx])
        decisions.append({"frame": idx, "phase": "final", "in_contact": val, "reason": reason})
        if not val:
            return n, decisions
        first_positive = idx

    start_scan = max(0, prev_probe + 1)
    end_scan = min(n - 1, first_positive)
    strict_start = first_positive
    for idx in range(start_scan, end_scan + 1):
        val, reason, _ = decide(frames[idx])
        decisions.append({"frame": idx, "phase": "refine", "in_contact": val, "reason": reason})
        if val:
            strict_start = idx
            break

    return strict_start, decisions


def process_video(video_path: Path, args) -> dict:
    stem = video_path.stem
    video_id = f"{args.dataset}/{args.category}/{stem}"
    frame_dir = args.video_dir / "frames" / stem
    cache_file = args.cache_dir / args.dataset / args.category / f"{stem}.json"
    debug_file = args.debug_dir / f"{stem}.json"

    if cache_file.exists() and not args.overwrite:
        return {"stem": stem, "status": "skip", "cache": str(cache_file)}

    frames, source_frame_count = ensure_frames(video_path, frame_dir, max_frames=args.target_frame_count)
    start_idx, decisions = find_contact_start(
        frames=frames,
        interval=args.interval,
        model=args.model,
        max_retries=args.max_retries,
    )

    if start_idx >= len(frames):
        labels: list[list[str]] = []
    else:
        n_intervals = max(1, math.ceil((len(frames) - start_idx) / args.interval))
        labels = [["R_Hand"] for _ in range(n_intervals)]

    payload = {
        "contact_labels": labels,
        "contact_interval": args.interval,
        "contact_start_idx": int(start_idx),
        "policy": "strict_gpt_start_frame_then_force_all_labels_to_R_Hand",
        "model": args.model,
        "video_id": video_id,
        "frame_count": len(frames),
        "source_frame_count": source_frame_count,
        "truncated_to_frame_count": len(frames),
        "prompt_policy": "Contact only when right hand/fingers/palm visibly touches or holds the object; near/ambiguous is false.",
    }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    debug_file.parent.mkdir(parents=True, exist_ok=True)
    debug_file.write_text(
        json.dumps({**payload, "decisions": decisions}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "stem": stem,
        "status": "ok",
        "start_idx": int(start_idx),
        "frame_count": len(frames),
        "source_frame_count": source_frame_count,
        "n_labels": len(labels),
        "cache": str(cache_file),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pickup_table")
    ap.add_argument("--dataset", default="dl300_delta")
    ap.add_argument("--category", default="dl300")
    ap.add_argument("--model", default=os.environ.get("OPENAI_REASONING_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.5")
    ap.add_argument("--interval", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument(
        "--target-frame-count",
        type=int,
        default=121,
        help="Only use the first N frames for contact labels; set 0 to use full videos.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    args.video_dir = root / "generation" / "videos_wan" / args.dataset / args.category
    args.cache_dir = root / "generation" / "4dhoi_recon_cache" / "contact_labels"
    args.debug_dir = root / "logs_contact_labels_strict"

    videos = sorted(args.video_dir.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No mp4 videos under {args.video_dir}")

    print(
        f"CONTACT_CACHE_START videos={len(videos)} workers={args.workers} model={args.model} interval={args.interval} target_frame_count={args.target_frame_count}",
        flush=True,
    )
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_video, v, args): v for v in videos}
        for fut in as_completed(futs):
            v = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                print(json.dumps(res, ensure_ascii=False), flush=True)
            except Exception as exc:
                err = {"stem": v.stem, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                failures.append(err)
                print(json.dumps(err, ensure_ascii=False), flush=True)
                traceback.print_exc()

    summary = {
        "total": len(videos),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "skip": sum(1 for r in results if r.get("status") == "skip"),
        "fail": len(failures),
        "target_frame_count": args.target_frame_count,
        "results": sorted(results, key=lambda x: x.get("stem", "")),
        "failures": failures,
    }
    summary_path = args.debug_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"CONTACT_CACHE_DONE summary={summary_path} ok={summary['ok']} skip={summary['skip']} fail={summary['fail']}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
