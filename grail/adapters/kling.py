"""Kling AI video generation adapter.

Wraps the Kling image-to-video API with JWT authentication, task polling,
and automatic video download + resize.

Environment variables required:
    KLING_API_KEY: Your Kling API key

Legacy environment variables also supported:
    KLING_ACCESS_KEY: Your Kling API access key
    KLING_SECRET_KEY: Your Kling API secret key
"""

import base64
import hashlib
import hmac
import json
import os
import time

import requests
from PIL import Image

from grail.core.video import download_video, resize_video


def _generate_jwt_token(access_key, secret_key):
    """Generate a JWT token for Kling API authentication."""

    def _base64url_encode(data):
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    header = json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
    now = int(time.time())
    payload = json.dumps(
        {"iss": access_key, "exp": now + 1800, "nbf": now - 5}, separators=(",", ":")
    ).encode()

    h = _base64url_encode(header)
    p = _base64url_encode(payload)
    sig = _base64url_encode(
        hmac.new(secret_key.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    )
    return f"{h}.{p}.{sig}"


def _get_auth_token():
    """Return a Kling auth token, preferring the current single-key API."""
    api_key = os.getenv("KLING_API_KEY")
    if api_key:
        return api_key, False

    access_key = os.getenv("KLING_ACCESS_KEY")
    secret_key = os.getenv("KLING_SECRET_KEY")
    if access_key and secret_key:
        return _generate_jwt_token(access_key, secret_key), True

    raise ValueError(
        "KLING_API_KEY must be set in environment variables. "
        "Legacy KLING_ACCESS_KEY and KLING_SECRET_KEY are also supported."
    )


def generate_video(
    image_path,
    prompt,
    output_dir,
    base_name,
    *,
    model_name="kling-v3",
    mode="std",
    duration="10",
    cfg_scale=0.5,
    negative_prompt="blur, distort, and low quality",
    image_tail_path=None,
    generate_audio=False,
):
    """Generate a video from an image using the Kling AI API.

    Args:
        image_path: Path to the start/input image (first frame).
        prompt: Text prompt for video generation.
        output_dir: Directory to save the generated video.
        base_name: Base name for the output file (without extension).
        model_name: Kling model version (e.g. "kling-v3").
        mode: "std" (standard) or "pro" (professional).
        duration: Video duration in seconds ("5" or "10").
        cfg_scale: Classifier-free guidance scale (0.0–1.0).
        negative_prompt: Negative prompt to avoid certain elements.
        image_tail_path: Optional end-frame image for start→end transitions.
        generate_audio: Whether to generate audio.

    Returns:
        Path to the downloaded video, or None on failure.
    """
    # Read image dimensions and determine aspect ratio
    with Image.open(image_path) as img:
        width, height = img.size
        ratio = width / height
        aspect_ratio = "16:9" if ratio > 1.2 else ("9:16" if ratio < 0.8 else "1:1")

    print(f"Kling: {model_name}/{mode} {duration}s — {width}x{height} ({aspect_ratio})")

    # Encode images to base64
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    tail_b64 = None
    if image_tail_path:
        with open(image_tail_path, "rb") as f:
            tail_b64 = base64.b64encode(f.read()).decode()

    token, token_expires = _get_auth_token()
    base_url = "https://api.klingai.com"
    create_url = f"{base_url}/v1/videos/image2video"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    payload = {
        "model_name": model_name,
        "mode": mode,
        "duration": duration,
        "image": image_b64,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "aspect_ratio": aspect_ratio,
        "generate_audio": generate_audio,
    }
    if tail_b64:
        payload["image_tail"] = tail_b64

    try:
        # Create task
        resp = requests.post(create_url, headers=headers, json=payload)
        data = resp.json()
        if resp.status_code != 200 or data.get("code") != 0:
            raise RuntimeError(f"Task creation failed: {data}")

        task_id = data["data"]["task_id"]
        print(f"  Task {task_id} created, polling...")

        # Poll for completion
        query_url = f"{create_url}/{task_id}"
        max_attempts, interval = 120, 5

        for attempt in range(max_attempts):
            if token_expires and attempt > 0 and attempt % 60 == 0:
                token, _ = _get_auth_token()
                headers["Authorization"] = f"Bearer {token}"

            time.sleep(interval)
            status_resp = requests.get(query_url, headers=headers)
            status_data = status_resp.json()

            if status_resp.status_code != 200 or status_data.get("code") != 0:
                continue

            task_status = status_data["data"]["task_status"]
            if attempt % 6 == 0:
                print(f"  Status: {task_status} ({attempt + 1}/{max_attempts})")

            if task_status == "succeed":
                videos = status_data["data"].get("task_result", {}).get("videos", [])
                video_url = videos[0].get("url") if videos else None
                if not video_url:
                    raise RuntimeError("No video URL in result")

                downloaded = download_video(video_url, output_dir, f"{base_name}.mp4")
                return resize_video(downloaded, width, height)

            if task_status == "failed":
                msg = status_data["data"].get("task_status_msg", "Unknown")
                raise RuntimeError(f"Video generation failed: {msg}")

        raise RuntimeError(f"Timed out after {max_attempts * interval}s")

    except Exception as e:
        print(f"Kling error: {e}")
        return None
