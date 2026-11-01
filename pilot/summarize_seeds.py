"""Average the per-seed results that eval_system.py wrote.

Usage:
    python summarize_seeds.py --runs DIR --out FILE
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES

SEEDS = (0, 1, 2)
ARMS = ("lab", "lab_field")
OUTCOMES = (
    "right_advice",
    "wrong_advice",
    "asked_two_candidates",
    "asked_and_truth_is_a_candidate",
    "referred_to_clinic",
    "top2_accuracy_all_photos",
)


def mean_std(values):
    return {
        "mean": round(float(np.mean(values)), 3),
        "std": round(float(np.std(values, ddof=1)), 3),
        "per_seed": [round(float(v), 3) for v in values],
    }


def confused_pairs(results, top=5):
    """Count errors between each pair of classes, summed over seeds."""
    total = sum(np.array(r["confusion_whole_photo"]) for r in results)
    errors = int(total.sum() - np.trace(total))
    pairs = []
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            pairs.append((int(total[i, j] + total[j, i]), CLASSES[i], CLASSES[j]))
    pairs.sort(reverse=True)
    return {"errors_over_seeds": errors, "pairs": pairs[:top]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    system = json.loads((args.runs / "system_dev.json").read_text())
    summary = {
        "photos": system["dev_photos_scored"],
        "majority_class_floor": system["majority_class_floor"],
    }
    for arm in ARMS:
        names = [f"{arm}_seed{seed}" for seed in SEEDS]
        runs = [json.loads((args.runs / f"{name}.json").read_text()) for name in names]
        results = [system[name] for name in names]
        entry = {
            "train_minutes": mean_std([r["train_seconds"] / 60 for r in runs]),
            "lab_dev_macro_f1": mean_std([r["lab_dev"]["macro_f1"] for r in runs]),
            "temperature": mean_std([r["temperature"] for r in results]),
        }
        for route in ("whole_photo", "finder_best_box", "finder_all_boxes_averaged"):
            entry[f"{route}_macro_f1"] = mean_std([r[route]["macro_f1"] for r in results])
            entry[f"{route}_accuracy"] = mean_std([r[route]["accuracy"] for r in results])
        for route in ("triage_whole_photo", "triage_finder_best_box"):
            for outcome in OUTCOMES:
                entry[f"{route}_{outcome}"] = mean_std([r[route][outcome] for r in results])
        for label in results[0]["per_class_f1_whole_photo"]:
            entry[f"whole_photo_f1_{label}"] = mean_std(
                [r["per_class_f1_whole_photo"][label] for r in results]
            )
        entry["most_confused_pairs_whole_photo"] = confused_pairs(results)
        summary[arm] = entry

    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
