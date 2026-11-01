"""Score both arms end to end on development photos, and test the triage gate.

Each photo goes finder -> classifier -> gate. Outcomes are what the gardener sees:
right advice, wrong advice, a follow-up question between two candidates, or a
clinic referral. Whole photos and labelled crops are diagnostics for the finder.

Usage:
    python eval_system.py --splits ../splits --plantdoc DIR --runs DIR --cache DIR
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import models
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES, accuracy, bootstrap_ci, macro_f1
from images import clip_box, open_photo, plantdoc_folder_path, plantdoc_rows, save_small
from train_classifier import predict

CROP_MARGIN = 0.10
ADVISE_AT = 0.70
ASK_AT = 0.85


def softmax(logits):
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """Pick the one scalar that best calibrates confidence on labelled dev crops."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 15.01, 0.05):
        probs = softmax(logits / t)
        nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def calibration_error(probs, labels, bins=10):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (confidence > low) & (confidence <= high)
        if chosen.any():
            total += chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean())
    return float(total)


def triage(probs, labels):
    """Route each photo with fixed thresholds and count what the gardener gets."""
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    p1 = probs[np.arange(len(probs)), top1]
    p2 = probs[np.arange(len(probs)), top2]

    advise = p1 >= ADVISE_AT
    ask = ~advise & (p1 + p2 >= ASK_AT)
    refer = ~advise & ~ask

    right = advise & (top1 == labels)
    wrong = advise & (top1 != labels)
    ask_holds_truth = ask & ((top1 == labels) | (top2 == labels))
    return {
        "right_advice": float(right.mean()),
        "wrong_advice": float(wrong.mean()),
        "asked_two_candidates": float(ask.mean()),
        "asked_and_truth_is_a_candidate": float(ask_holds_truth.mean()),
        "referred_to_clinic": float(refer.mean()),
        "precision_when_advising": float(right.sum() / max(1, advise.sum())),
        "top2_accuracy_all_photos": float(((top1 == labels) | (top2 == labels)).mean()),
    }


def finder_items(splits, plantdoc, runs, cache):
    """Return every finder crop, tagged with the photo it came from.

    The first crop of each photo is the finder's best box. A photo with no box
    falls back on the whole photo, so every photo stays in the count.
    """
    with open(runs / "finder_dev.csv", newline="") as fh:
        finder = {r["photo"]: r for r in csv.DictReader(fh)}
    crop_dir = cache / "pd_dev" / "finder"
    crop_dir.mkdir(parents=True, exist_ok=True)

    items = []
    photo_number = 0
    for index, row in enumerate(plantdoc_rows(splits, "pd_dev")):
        if row["mixed_labels"] == "1":
            continue
        path = plantdoc_folder_path(plantdoc, row["photo"])
        image, _, _ = open_photo(path, int(row["width"]), int(row["height"]))
        label = CLASSES.index(row["photo_label"])

        crops = []
        for box in json.loads(finder[row["photo"]]["boxes"]):
            x0, y0, x1, y1 = clip_box(box, image.width, image.height, CROP_MARGIN)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                crops.append(image.crop((x0, y0, x1, y1)))
        if not crops:
            crops = [image]

        for crop_number, crop in enumerate(crops):
            crop_path = crop_dir / f"{index:05d}_{crop_number}.jpg"
            save_small(crop, crop_path)
            items.append((crop_path, label, row["group"], photo_number, crop_number))
        photo_number += 1
    return items


def per_photo(logits, items, temperature=1.0):
    """Reduce crop logits to one row per photo: the best box, and all boxes averaged."""
    probs = softmax(logits / temperature)
    photo_ids = np.array([item[3] for item in items])
    best = []
    mean = []
    for photo in range(photo_ids.max() + 1):
        chosen = np.flatnonzero(photo_ids == photo)
        best.append(probs[chosen[0]])
        mean.append(probs[chosen].mean(axis=0))
    return np.array(best), np.array(mean)


def photo_summary(probs, labels, groups):
    predictions = probs.argmax(axis=1)
    return {
        "n": len(labels),
        "macro_f1": macro_f1(labels, predictions),
        "macro_f1_ci95": list(bootstrap_ci(labels, predictions, groups)),
        "accuracy": accuracy(labels, predictions),
    }


def whole_items(splits, cache):
    """Whole photos cached by training, dropping mixed-label photos to match."""
    items = []
    for index, row in enumerate(plantdoc_rows(splits, "pd_dev")):
        if row["mixed_labels"] == "1":
            continue
        path = cache / "pd_dev" / "whole" / f"{index:05d}.jpg"
        items.append((path, CLASSES.index(row["photo_label"]), row["group"]))
    return items


def labelled_crop_items(splits, cache):
    items = []
    for path in sorted((cache / "pd_dev" / "crops").glob("*.jpg")):
        items.append(path)
    rows = plantdoc_rows(splits, "pd_dev")
    result = []
    for path in items:
        photo_index, box_index = (int(v) for v in path.stem.split("_"))
        row = rows[photo_index]
        label = json.loads(row["boxes"])[box_index][0]
        result.append((path, CLASSES.index(label), row["group"]))
    return result


def load_model(weights, device):
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


def summarize(logits, items):
    labels = np.array([item[1] for item in items])
    predictions = logits.argmax(axis=1)
    groups = [item[2] for item in items]
    return {
        "n": len(items),
        "macro_f1": macro_f1(labels, predictions),
        "macro_f1_ci95": list(bootstrap_ci(labels, predictions, groups)),
        "accuracy": accuracy(labels, predictions),
    }


def per_class_f1(labels, predictions):
    scores = {}
    for index, name in enumerate(CLASSES):
        tp = ((labels == index) & (predictions == index)).sum()
        fp = ((labels != index) & (predictions == index)).sum()
        fn = ((labels == index) & (predictions != index)).sum()
        denom = 2 * tp + fp + fn
        scores[name] = float(2 * tp / denom) if denom else None
    return scores


def confusion(labels, predictions):
    """Rows are true classes and columns are predictions, in CLASSES order."""
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for truth, guess in zip(labels, predictions):
        matrix[truth, guess] += 1
    return matrix.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--plantdoc", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[f"{arm}_seed{seed}" for arm in ("lab", "lab_field") for seed in (0, 1, 2)],
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps"
    finder = finder_items(args.splits, args.plantdoc, args.runs, args.cache)
    whole = whole_items(args.splits, args.cache)
    crops = labelled_crop_items(args.splits, args.cache)
    labels = np.array([item[1] for item in whole])
    groups = [item[2] for item in whole]
    crop_labels = np.array([item[1] for item in crops])
    majority = np.bincount(labels, minlength=len(CLASSES)).argmax()

    report = {
        "dev_photos_scored": len(whole),
        "finder_crops_scored": len(finder),
        "majority_class_floor": {
            "macro_f1": macro_f1(labels, np.full_like(labels, majority)),
            "accuracy": accuracy(labels, np.full_like(labels, majority)),
        },
        "thresholds": {"advise_at": ADVISE_AT, "ask_at": ASK_AT},
    }

    for name in args.arms:
        model = load_model(args.runs / f"{name}.pt", device)
        finder_logits = predict(model, finder, device)
        whole_logits = predict(model, whole, device)
        crop_logits = predict(model, crops, device)

        temperature = fit_temperature(crop_logits, crop_labels)
        best_box, all_boxes = per_photo(finder_logits, finder)
        best_box_cal, all_boxes_cal = per_photo(finder_logits, finder, temperature)
        whole_cal = softmax(whole_logits / temperature)

        report[name] = {
            "whole_photo": photo_summary(softmax(whole_logits), labels, groups),
            "finder_best_box": photo_summary(best_box, labels, groups),
            "finder_all_boxes_averaged": photo_summary(all_boxes, labels, groups),
            "labelled_crop_each_box": summarize(crop_logits, crops),
            "temperature": temperature,
            "calibration_error_before": calibration_error(best_box, labels),
            "calibration_error_after": calibration_error(best_box_cal, labels),
            "triage_finder_best_box": triage(best_box_cal, labels),
            "triage_finder_all_boxes": triage(all_boxes_cal, labels),
            "triage_whole_photo": triage(whole_cal, labels),
            "per_class_f1_whole_photo": per_class_f1(labels, whole_logits.argmax(axis=1)),
            "confusion_whole_photo": confusion(labels, whole_logits.argmax(axis=1)),
        }

    out = args.runs / "system_dev.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
