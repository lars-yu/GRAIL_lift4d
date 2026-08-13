from glob import glob


def category2object(data_path, category):
    """Resolve the mesh file for ``category`` under ``data_path``.

    Searches common per-category layouts in order and returns the first hit.
    """
    data_path = data_path.rstrip("/")
    patterns = [
        f"{data_path}/{category}/model.obj",
        f"{data_path}/{category}/model.usda",
        f"{data_path}/{category}/{category}.obj",
        f"{data_path}/{category}/mesh.obj",
        f"{data_path}/{category}/*/model.obj",
        f"{data_path}/{category}/*.obj",
        f"{data_path}/{category}/*.usda",
        f"{data_path}/{category}/**/*.obj",
        f"{data_path}/{category}/**/*.usda",
    ]
    for pattern in patterns:
        hits = sorted(glob(pattern, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No mesh (.obj/.usda) for category '{category}' under '{data_path}'"
    )


def scene2blender(scene_name):
    # HACK: Some scenes have a version suffix, so we need to remove it
    scene_name = scene_name.split("-")[0]
    return f"data/Scene/{scene_name}.blend"
