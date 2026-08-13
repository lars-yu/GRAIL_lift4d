"""DashScope Wan 2.7 image-to-video adapter.

Wraps the Alibaba Cloud DashScope Wan 2.7 i2v API with async task submission,
polling, and automatic video download + resize. The submit/poll/download flow
mirrors the standalone ``gen_v2.py`` script (and ``pano_test.py`` which reuses
it), packaged here as a drop-in replacement for ``grail.adapters.kling`` so the
2D-HOI pipeline can switch backends via config (``model_api``).

Environment variables (all optional — defaults mirror ``gen_v2.py``):
    DASHSCOPE_API_KEY:  DashScope API key (falls back to the MaaS-gateway key).
    DASHSCOPE_BASE_URL: API base, e.g. the dedicated MaaS gateway or the public
                        ``https://dashscope.aliyuncs.com/api/v1``.
    DASHSCOPE_MODEL:    Wan model name / dated snapshot.
    DASHSCOPE_VERIFY_SSL: "1" to re-enable TLS verification (default off, because
                        the MaaS-gateway cert SAN doesn't cover the multi-level
                        subdomain — same reasoning as gen_v2.py).
"""

import base64
import mimetypes
import os
import random
import time

import requests
import urllib3
from PIL import Image

from grail.core.video import download_video, resize_video

# ============================================================
# Config (mirrors gen_v2.py:88-115)
# ============================================================
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-63adfd632b764e25b212860441e728f6")

BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://ws-vl85wbglzuzirrwb.cn-beijing.maas.aliyuncs.com/api/v1",
)
SUBMIT_URL = f"{BASE_URL}/services/aigc/video-generation/video-synthesis"
TASK_URL_TMPL = f"{BASE_URL}/tasks/{{task_id}}"

# Default Wan model (dated snapshot pins a reproducible version, like gen_v2.py).
MODEL_NAME = os.environ.get("DASHSCOPE_MODEL", "wan2.7-i2v-2026-04-25")

# The dedicated MaaS gateway serves a cert whose SAN only covers *.aliyuncs.com
# (single-level wildcard), which fails to match ws-xxx.cn-beijing.maas.aliyuncs.com;
# requests then raises SSLError(CertificateError). Skip verification by default.
VERIFY_SSL = os.environ.get("DASHSCOPE_VERIFY_SSL", "0") not in ("0", "false", "False", "")
if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

POLL_INTERVAL_SEC = 5
POLL_TIMEOUT_SEC = 30 * 60


# ============================================================
# DashScope helpers (adapted from gen_v2.py)
# ============================================================


def _image_to_data_uri(path):
    """Base64-encode an image file into a ``data:`` URI."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None or not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _submit_task(payload, label="", max_retries=8):
    """Submit an async video-synthesis task, returning its task_id.

    Retries 429 (Throttling.RateQuota) and transient network/5xx errors with
    jittered exponential backoff. Deterministic 4xx errors (AccessDenied /
    InvalidParameter) are raised immediately.
    """
    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                SUBMIT_URL, headers=headers, json=payload, timeout=120, verify=VERIFY_SSL
            )
        except requests.exceptions.RequestException as e:
            last_err = f"network error {type(e).__name__}: {str(e)[:120]}"
            retryable = True
        else:
            if resp.ok:
                data = resp.json()
                task_id = data.get("output", {}).get("task_id")
                if not task_id:
                    raise RuntimeError(f"Submit failed (no task_id): {data}")
                return task_id
            retryable = resp.status_code == 429 or resp.status_code >= 500
            last_err = f"HTTP {resp.status_code}: {resp.text}"
            if not retryable:
                raise RuntimeError(f"Submit failed {last_err}")

        if attempt >= max_retries:
            break
        wait = min(2 ** attempt, 60) + random.uniform(0, 3)
        print(
            f"  [wan submit retry {attempt}/{max_retries}] {label}: "
            f"{last_err[:120]} (wait {wait:.1f}s)",
            flush=True,
        )
        time.sleep(wait)

    raise RuntimeError(f"Submit failed after {max_retries} retries: {last_err}")


def _poll_until_done(task_id, label="", heartbeat_sec=30):
    """Poll a task until it succeeds, raising on failure/timeout."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = TASK_URL_TMPL.format(task_id=task_id)
    start = time.time()
    last_status = None
    last_print_at = -(10 ** 9)
    while True:
        resp = requests.get(url, headers=headers, timeout=30, verify=VERIFY_SSL)
        if not resp.ok:
            raise RuntimeError(f"Query failed HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        status = data.get("output", {}).get("task_status", "UNKNOWN")
        elapsed = int(time.time() - start)

        if status != last_status or elapsed - last_print_at >= heartbeat_sec:
            print(f"  [wan poll] {label:>22s} {elapsed:>4}s status={status}", flush=True)
            last_status = status
            last_print_at = elapsed

        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            raise RuntimeError(f"Task abnormal [{label}]: {data}")
        if time.time() - start > POLL_TIMEOUT_SEC:
            raise TimeoutError(f"Polling timed out [{label}]: {data}")

        time.sleep(POLL_INTERVAL_SEC)


def _build_payload(media, prompt, model_name, resolution, duration,
                   negative_prompt=None, prompt_extend=False, watermark=False):
    """Assemble the DashScope video-synthesis request payload (see gen_v2.py:594)."""
    input_block = {"prompt": prompt, "media": media}
    if negative_prompt:
        input_block["negative_prompt"] = negative_prompt
    return {
        "model": model_name,
        "input": input_block,
        "parameters": {
            "resolution": resolution,
            "duration": duration,
            # Disable auto prompt rewriting so our carefully written prompt is used as-is.
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        },
    }


# ============================================================
# Public API (Kling-compatible contract)
# ============================================================


def generate_video(
    image_path,
    prompt,
    output_dir,
    base_name,
    *,
    model_name=None,
    duration=5,
    resolution="720P",
    image_tail_path=None,
    negative_prompt=None,
    prompt_extend=False,
    watermark=False,
    **_ignored,
):
    """Generate a video from an image using DashScope Wan 2.7 i2v.

    Mirrors the signature/return contract of ``grail.adapters.kling.generate_video``
    so it is a drop-in backend: it returns the path to the downloaded (and resized
    to the input image's dimensions) video, or ``None`` on failure. The pipeline's
    retry loop depends on the ``None``-on-failure contract.

    Args:
        image_path: Path to the start/input image (first frame).
        prompt: Text prompt for video generation.
        output_dir: Directory to save the generated video.
        base_name: Base name for the output file (without extension).
        model_name: Wan model name; defaults to the module ``MODEL_NAME``.
        duration: Video duration in seconds (int; string like "5" is coerced).
        resolution: Wan resolution tag, e.g. "720P".
        image_tail_path: Optional end-frame image for first+last-frame control.
        negative_prompt: Optional negative prompt (omitted when falsy).
        prompt_extend: Whether to let the service rewrite the prompt (default off).
        watermark: Whether to add a watermark (default off).
        **_ignored: Accepts (and ignores) Kling-only kwargs such as ``mode`` /
            ``cfg_scale`` so call sites can stay symmetrical across backends.

    Returns:
        Path to the downloaded video, or ``None`` on failure.
    """
    model_name = model_name or MODEL_NAME
    duration = int(duration)
    label = f"{os.path.basename(str(base_name))[:24]}"

    try:
        with Image.open(image_path) as img:
            width, height = img.size

        media = [{"type": "first_frame", "url": _image_to_data_uri(image_path)}]
        if image_tail_path:
            media.append({"type": "last_frame", "url": _image_to_data_uri(image_tail_path)})

        mode = "first+last frame" if image_tail_path else "first frame"
        print(f"Wan: {model_name} {resolution} {duration}s — {width}x{height} ({mode})")

        payload = _build_payload(
            media, prompt, model_name, resolution, duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend, watermark=watermark,
        )

        task_id = _submit_task(payload, label=label)
        print(f"  Task {task_id} created, polling...")

        result = _poll_until_done(task_id, label=label)
        video_url = result.get("output", {}).get("video_url")
        if not video_url:
            raise RuntimeError(f"Task succeeded but no video_url: {result}")

        downloaded = download_video(video_url, output_dir, f"{base_name}.mp4")
        return resize_video(downloaded, width, height)

    except Exception as e:
        print(f"Wan error: {e}")
        return None
