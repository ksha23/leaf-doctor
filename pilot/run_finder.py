"""Run the frozen text-prompted leaf finder on development photos.

Records each photo's best box and how well it overlaps the labelled leaves.

Usage:
    python run_finder.py --splits ../splits --plantdoc DIR --out DIR
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection

sys.path.insert(0, str(Path(__file__).parent))
from images import open_photo, plantdoc_folder_path, plantdoc_rows

MODEL_ID = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["a photo of a leaf", "a tomato leaf"]
MIN_SCORE = 0.10
MAX_BOXES = 5
OVERLAP_LIMIT = 0.5


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap + 1e-9)


def distinct_boxes(boxes, scores):
    """Keep the best-scoring boxes that do not overlap one another much."""
    kept = []
    for index in np.argsort(-scores):
        box = boxes[index]
        if all(iou(box, other) < OVERLAP_LIMIT for other in kept):
            kept.append(box)
        if len(kept) == MAX_BOXES:
            break
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--plantdoc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="pd_dev", choices=["pd_dev"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(args.device).eval()

    records = []
    seconds = []
    for row in plantdoc_rows(args.splits, args.split):
        path = plantdoc_folder_path(args.plantdoc, row["photo"])
        image, scale_x, scale_y = open_photo(path, int(row["width"]), int(row["height"]))

        start = time.time()
        inputs = processor(text=[PROMPTS], images=image, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        size = torch.tensor([[image.height, image.width]], device=args.device)
        found = processor.post_process_grounded_object_detection(
            outputs, target_sizes=size, threshold=MIN_SCORE, text_labels=[PROMPTS]
        )[0]
        seconds.append(time.time() - start)

        truth = [
            [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
            for _, b in json.loads(row["boxes"])
        ]
        record = {"photo": row["photo"], "found": False, "boxes": "[]", "score": 0.0, "best_iou": 0.0}
        if len(found["scores"]) > 0:
            scores = found["scores"].cpu().numpy()
            boxes = found["boxes"].cpu().numpy().tolist()
            kept = distinct_boxes(boxes, scores)
            record.update(
                found=True,
                boxes=json.dumps([[round(v, 1) for v in box] for box in kept]),
                score=round(float(scores.max()), 3),
                best_iou=round(max(iou(kept[0], t) for t in truth), 3),
            )
        records.append(record)
        if len(records) % 20 == 0:
            print(f"{len(records)} photos, {np.mean(seconds):.2f} s each", flush=True)

    with open(args.out / "finder_dev.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    ious = np.array([r["best_iou"] for r in records])
    summary = {
        "model": MODEL_ID,
        "device": args.device,
        "photos": len(records),
        "no_box_above_threshold": sum(not r["found"] for r in records),
        "best_box_iou_at_least_0.5": float((ious >= 0.5).mean()),
        "mean_best_box_iou": float(ious.mean()),
        "seconds_per_photo": round(float(np.mean(seconds[1:])), 3),
    }
    (args.out / "finder_dev.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
