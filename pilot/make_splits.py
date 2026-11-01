"""Freeze the train, dev and final partitions before any model sees the data.

PlantVillage is split by physical leaf. PlantDoc is split by photo, with exact
and near-duplicate photos kept together. Final-partition images are only hashed
here. No model reads them until the frozen final run.

Usage:
    python make_splits.py --plantvillage DIR --plantdoc DIR --out ../splits
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES, PLANTDOC_LABEL_TO_CLASS, PLANTVILLAGE_FOLDER_TO_CLASS

SEED = 539
PV_DEV_SHARE = 0.20
PD_DEV_SHARE = 0.15
PD_FINAL_SHARE = 0.25
NEAR_DUPLICATE_BITS = 5


def plantvillage_leaf_id(folder, filename, leaf_map):
    """Return a group id so that photos of one leaf stay on one side of the split."""
    stem = filename.split("___", 1)[1].rsplit(".", 1)[0]
    for entry in leaf_map.get(stem.lower(), []):
        if entry.startswith(folder + ":::"):
            return entry

    # The creators' map misses about a third of tomato photos. Fall back on the
    # name: "GHLB2 Leaf 117.4" is photo 4 of leaf 117.
    dotted = re.fullmatch(r"(.+ \d+)\.\d+", stem)
    if dotted:
        return f"{folder}:{dotted.group(1)}"

    # Otherwise treat 10 consecutive frame numbers from one source as one leaf.
    numbered = re.fullmatch(r"(.+?)[ _]?(\d+)", stem)
    if numbered:
        return f"{folder}:{numbered.group(1)}:{int(numbered.group(2)) // 10}"
    return f"{folder}:{stem}"


def split_plantvillage(pv_dir, out_dir):
    leaf_map = json.loads((pv_dir / "leaf-map.json").read_text())
    rng = random.Random(SEED)
    rows = []
    for folder, label in PLANTVILLAGE_FOLDER_TO_CLASS.items():
        groups = defaultdict(list)
        for path in sorted((pv_dir / "raw/color" / folder).iterdir()):
            groups[plantvillage_leaf_id(folder, path.name, leaf_map)].append(path.name)

        leaf_ids = sorted(groups)
        rng.shuffle(leaf_ids)
        n_dev = round(len(leaf_ids) * PV_DEV_SHARE)
        dev_ids = set(leaf_ids[:n_dev])
        for leaf_id, names in groups.items():
            split = "pv_dev" if leaf_id in dev_ids else "pv_train"
            for name in names:
                rows.append([f"{folder}/{name}", label, leaf_id, split])

    with open(out_dir / "plantvillage_split.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "leaf_id", "split"])
        writer.writerows(sorted(rows))
    return rows


def difference_hash(path):
    """64-bit perceptual hash: is each pixel brighter than its right neighbour?"""
    gray = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


class DisjointSet:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def read_plantdoc_boxes(pd_dir):
    """Return one record per tomato photo, keyed by its folder and filename."""
    photos = {}
    for folder, csv_name in (("TRAIN", "train_labels.csv"), ("TEST", "test_labels.csv")):
        with open(pd_dir / csv_name, newline="") as fh:
            for row in csv.DictReader(fh):
                label = PLANTDOC_LABEL_TO_CLASS.get(row["class"].strip())
                if label is None:
                    continue
                path = pd_dir / folder / row["filename"]
                if not path.exists():
                    path = pd_dir / folder / unquote(row["filename"])
                key = f"{folder}/{row['filename']}"
                record = photos.setdefault(
                    key,
                    {
                        "key": key,
                        "path": path,
                        "official": folder,
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "boxes": [],
                    },
                )
                box = [int(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                record["boxes"].append((label, box))
    return photos


def split_plantdoc(pd_dir, out_dir):
    photos = read_plantdoc_boxes(pd_dir)
    missing = [k for k, p in photos.items() if not p["path"].exists()]
    for key in missing:
        del photos[key]

    # Group exact and near-duplicate photos.
    keys = sorted(photos)
    groups = DisjointSet(keys)
    by_md5 = {}
    hashes = []
    for key in keys:
        digest = hashlib.md5(photos[key]["path"].read_bytes()).hexdigest()
        if digest in by_md5:
            groups.union(key, by_md5[digest])
        by_md5.setdefault(digest, key)
        hashes.append(difference_hash(photos[key]["path"]))
    hashes = np.array(hashes)
    for i in range(len(keys)):
        distance = (hashes[i + 1 :] != hashes[i]).sum(axis=1)
        for j in np.flatnonzero(distance <= NEAR_DUPLICATE_BITS):
            groups.union(keys[i], keys[i + 1 + j])

    members = defaultdict(list)
    for key in keys:
        members[groups.find(key)].append(key)

    # A group's class is the most common box label across its photos.
    group_label = {}
    for root, group_keys in members.items():
        counts = Counter(lab for k in group_keys for lab, _ in photos[k]["boxes"])
        group_label[root] = counts.most_common(1)[0][0]

    # All official test photos stay in final. Top up each class from the rest
    # until final holds its share, then carve dev out of what remains.
    rng = random.Random(SEED)
    assignment = {}
    for label in CLASSES:
        roots = sorted(r for r in members if group_label[r] == label)
        rng.shuffle(roots)
        total = sum(len(members[r]) for r in roots)

        final = [r for r in roots if any(photos[k]["official"] == "TEST" for k in members[r])]
        rest = [r for r in roots if r not in set(final)]
        n_final = sum(len(members[r]) for r in final)
        while rest and n_final < round(total * PD_FINAL_SHARE):
            root = rest.pop()
            final.append(root)
            n_final += len(members[root])

        n_dev = 0
        dev = []
        while rest and n_dev < round(total * PD_DEV_SHARE):
            root = rest.pop()
            dev.append(root)
            n_dev += len(members[root])

        for root in final:
            assignment[root] = "pd_final"
        for root in dev:
            assignment[root] = "pd_dev"
        for root in rest:
            assignment[root] = "pd_train"

    rows = []
    for root, group_keys in members.items():
        for key in group_keys:
            photo = photos[key]
            labels = {lab for lab, _ in photo["boxes"]}
            rows.append(
                {
                    "photo": key,
                    "group": root,
                    "split": assignment[root],
                    "photo_label": group_label[root],
                    "mixed_labels": int(len(labels) > 1),
                    "width": photo["width"],
                    "height": photo["height"],
                    "boxes": json.dumps(photo["boxes"]),
                }
            )
    rows.sort(key=lambda r: r["photo"])
    with open(out_dir / "plantdoc_split.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    duplicates = sum(len(v) - 1 for v in members.values())
    crossing = sum(
        1 for v in members.values() if len({photos[k]["official"] for k in v}) > 1
    )
    return rows, missing, duplicates, crossing


def summarize(pv_rows, pd_rows, missing, duplicates, crossing):
    summary = {"plantvillage": {}, "plantdoc": {}}
    for split in ("pv_train", "pv_dev"):
        chosen = [r for r in pv_rows if r[3] == split]
        summary["plantvillage"][split] = {
            "images": len(chosen),
            "leaf_groups": len({r[2] for r in chosen}),
            "per_class": dict(Counter(r[1] for r in chosen)),
        }
    for split in ("pd_train", "pd_dev", "pd_final"):
        chosen = [r for r in pd_rows if r["split"] == split]
        summary["plantdoc"][split] = {
            "photos": len(chosen),
            "boxes": sum(len(json.loads(r["boxes"])) for r in chosen),
            "mixed_label_photos": sum(r["mixed_labels"] for r in chosen),
            "per_class": dict(Counter(r["photo_label"] for r in chosen)),
        }
    summary["plantdoc"]["listed_but_missing_files"] = len(missing)
    summary["plantdoc"]["duplicate_photos_grouped"] = duplicates
    summary["plantdoc"]["duplicate_groups_crossing_official_split"] = crossing
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantvillage", type=Path, required=True)
    parser.add_argument("--plantdoc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pv_rows = split_plantvillage(args.plantvillage, args.out)
    pd_rows, missing, duplicates, crossing = split_plantdoc(args.plantdoc, args.out)
    summary = summarize(pv_rows, pd_rows, missing, duplicates, crossing)
    (args.out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
