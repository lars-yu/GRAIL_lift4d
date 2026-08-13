# Web Visualizer

Generate a self-contained static site that lets anyone browse a GRAIL motion
library in a web browser — hover to play the MP4 preview, per-motion
metadata panel, client-side name filter, dark/light mode.

No server-side code, no WebGL, no shader setup. It's just HTML + JS + a
manifest JSON alongside kinematic-replay MP4s in `<motion_lib>/vis/` and
meta pickles from retarget / data-export.

## Generate

Render kinematic-replay MP4s into `<motion_lib>/vis/` first (if not already
present):

```bash
conda activate sonic    # or GRAIL_SONIC_ENV override
bash grail/visualization/scripts/visualize.sh data/pickup_table
```

Then build the static site:

```bash
python -m grail.web_visualizer.generate_manifest \
    --motion_lib data/pickup_table \
    --output     out/viz/pickup_table
```

Outputs:

```
out/viz/pickup_table/
├── index.html
├── main.js
├── style.css
├── manifest.json            # 41 motions, 41 with video (example)
└── vis/                     # relative symlinks into motion_lib/vis/
    ├── pickup_table__apple_0__000.mp4
    └── ...
```

The `vis/` entries are relative symlinks by default — fast + space-efficient.
Pass `--copy_videos` if you need a self-contained directory (e.g. publishing
to S3).

## Serve

Any static-file host works. For a quick local check:

```bash
cd out/viz/pickup_table
python -m http.server 8000
# open http://localhost:8000/
```

To publish on GitHub Pages, host from a dedicated branch or drop the output
into a path the Pages workflow picks up.

## What the manifest contains

A stable-allowlist projection from the motion library:

| Field        | Source                             | Notes                                   |
|--------------|------------------------------------|-----------------------------------------|
| `name`       | filename stem of `robot/*.pkl`     |                                         |
| `frames`     | robot pkl (joint_pos / body_pos)   | number of frames                          |
| `video`      | `vis/*.mp4`                        | omitted if video missing                |
| `meta.*`     | `meta/*.pkl` (allowlisted keys)    | `object_name`, `table_{pos,quat,size}`  |

Only the allowlisted meta keys are copied into the manifest so the schema
stays stable even as `meta/*.pkl` grows new fields.

## What the viewer does

- Renders a grid of cards, responsive down to mobile width.
- Hover-to-play / mouseleave-to-pause. No autoplay, no audio.
- Type-to-filter: `search` box performs a case-insensitive substring match
  against motion names (client-side, instant).
- Dark mode honours `prefers-color-scheme`; no theme switch widget.
