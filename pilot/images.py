"""Turn split files into image lists, and cache PlantDoc crops at training size."""
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES

CROP_MARGIN = 0.10
SHORT_SIDE = 256
SMALLEST_BOX = 32


def plantdoc_rows(split_dir, split):
    with open(split_dir / "plantdoc_split.csv", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == split]


def plantvillage_items(split_dir, plantvillage_dir, split):
    """Return (path, class index, group) for one PlantVillage split."""
    items = []
    with open(split_dir / "plantvillage_split.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == split:
                path = plantvillage_dir / "raw/color" / row["path"]
                items.append((path, CLASSES.index(row["label"]), row["leaf_id"]))
    return items


def open_photo(path, csv_width, csv_height):
    """Boxes follow the stored pixel grid, so do not apply the EXIF rotation."""
    image = Image.open(path).convert("RGB")
    scale_x = image.width / csv_width
    scale_y = image.height / csv_height
    return image, scale_x, scale_y


def clip_box(box, width, height, margin=0.0):
    x0, y0, x1, y1 = box
    pad_x = (x1 - x0) * margin
    pad_y = (y1 - y0) * margin
    return (
        max(0, int(x0 - pad_x)),
        max(0, int(y0 - pad_y)),
        min(width, int(x1 + pad_x)),
        min(height, int(y1 + pad_y)),
    )


def save_small(image, path):
    scale = SHORT_SIDE / min(image.size)
    if scale < 1:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.BILINEAR)
    image.save(path, quality=92)


def plantdoc_folder_path(plantdoc_dir, photo_key):
    path = plantdoc_dir / photo_key
    if path.exists():
        return path
    from urllib.parse import unquote

    return plantdoc_dir / unquote(photo_key)


def cache_plantdoc(split_dir, plantdoc_dir, cache_dir, split):
    """Write one crop per labelled box and one small whole photo for a split.

    Refuses the final partition, which stays unread until the frozen run.
    """
    if split == "pd_final":
        raise ValueError("The final partition is not cached during development.")

    crop_dir = cache_dir / split / "crops"
    whole_dir = cache_dir / split / "whole"
    crop_dir.mkdir(parents=True, exist_ok=True)
    whole_dir.mkdir(parents=True, exist_ok=True)

    crops = []
    wholes = []
    skipped = 0
    for index, row in enumerate(plantdoc_rows(split_dir, split)):
        path = plantdoc_folder_path(plantdoc_dir, row["photo"])
        image, scale_x, scale_y = open_photo(path, int(row["width"]), int(row["height"]))

        whole_path = whole_dir / f"{index:05d}.jpg"
        save_small(image, whole_path)
        wholes.append((whole_path, CLASSES.index(row["photo_label"]), row["group"], row))

        for box_index, (label, box) in enumerate(json.loads(row["boxes"])):
            scaled = [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]
            x0, y0, x1, y1 = clip_box(scaled, image.width, image.height, CROP_MARGIN)
            if x1 - x0 < SMALLEST_BOX or y1 - y0 < SMALLEST_BOX:
                skipped += 1
                continue
            crop_path = crop_dir / f"{index:05d}_{box_index:02d}.jpg"
            save_small(image.crop((x0, y0, x1, y1)), crop_path)
            crops.append((crop_path, CLASSES.index(label), row["group"]))
    return crops, wholes, skipped
