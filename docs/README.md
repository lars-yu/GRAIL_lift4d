# Documentation

Source for the GRAIL documentation website.

## Building Locally

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

The built site is at `build/html/index.html`.

## Live Preview

```bash
cd docs/build/html
python -m http.server 8000
# → http://localhost:8000
```

## Clean Build

```bash
cd docs
make clean && make html
```

## Deployment

Pushes to the `main` branch trigger `.github/workflows/docs.yml`,
which builds with Sphinx and publishes to GitHub Pages:
**https://nvlabs.github.io/GRAIL/**

## Layout

- `source/conf.py` — Sphinx config (sphinx-book-theme, NVIDIA branding)
- `source/index.rst` — landing page / toctree
- `source/*.md` — content pages (MyST markdown)
- `source/_static/css/custom.css` — NVIDIA-green theme overrides

Style follows the
[GR00T-WholeBodyControl docs](https://nvlabs.github.io/GR00T-WholeBodyControl/).
