"""Fine-tune the disease classifier for one arm of the controlled comparison.

The two arms differ only in training data:
    lab        PlantVillage photos
    lab_field  PlantVillage photos plus PlantDoc leaf crops, half of each batch

Everything else is matched: network, update count, schedule, seed, augmentation.

Usage:
    python train_classifier.py --arm lab --seed 0 --splits ../splits \
        --plantvillage DIR --plantdoc DIR --cache DIR --out DIR
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES, accuracy, bootstrap_ci, macro_f1
from images import cache_plantdoc, plantvillage_items

BATCH_SIZE = 64
HEAD_STEPS = 200
FINE_TUNE_STEPS = 600
HEAD_LR = 1e-3
FINE_TUNE_LR = 2e-4
WORKERS = 8

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


class ImageList(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index][0], self.items[index][1]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def balanced_weights(lab_items, field_items):
    """Equal chance per class, and an equal share per domain when both are used."""
    weights = []
    domains = [d for d in (lab_items, field_items) if d]
    for domain in domains:
        counts = Counter(item[1] for item in domain)
        for item in domain:
            weights.append(1.0 / (counts[item[1]] * len(counts) * len(domains)))
    return weights


def next_batches(loader):
    while True:
        yield from loader


def run_steps(model, loader_iter, optimizer, scheduler, steps, device, phase):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for step in range(steps):
        images, labels = next(loader_iter)
        images = images.to(device)
        labels = labels.to(device)
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if (step + 1) % 100 == 0:
            print(f"{phase} step {step + 1}/{steps} loss {loss.item():.3f}", flush=True)


@torch.no_grad()
def predict(model, items, device):
    loader = DataLoader(
        ImageList(items, TEST_TRANSFORM), batch_size=128, num_workers=WORKERS, shuffle=False
    )
    model.eval()
    outputs = [model(images.to(device)).cpu().numpy() for images, _ in loader]
    return np.concatenate(outputs)


def score(logits, items):
    y_true = [item[1] for item in items]
    y_pred = logits.argmax(axis=1)
    groups = [item[2] for item in items]
    low, high = bootstrap_ci(y_true, y_pred, groups)
    return {
        "n": len(items),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci95": [low, high],
        "accuracy": accuracy(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["lab", "lab_field"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--plantvillage", type=Path, required=True)
    parser.add_argument("--plantdoc", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    lab_train = plantvillage_items(args.splits, args.plantvillage, "pv_train")
    lab_dev = plantvillage_items(args.splits, args.plantvillage, "pv_dev")
    field_train, _, _ = cache_plantdoc(args.splits, args.plantdoc, args.cache, "pd_train")
    field_dev_crops, field_dev_whole, skipped = cache_plantdoc(
        args.splits, args.plantdoc, args.cache, "pd_dev"
    )

    extra = field_train if args.arm == "lab_field" else []
    train_items = lab_train + extra
    weights = balanced_weights(lab_train, extra)
    total_steps = HEAD_STEPS + FINE_TUNE_STEPS
    sampler = WeightedRandomSampler(
        weights,
        num_samples=total_steps * BATCH_SIZE,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        ImageList(train_items, TRAIN_TRANSFORM),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=WORKERS,
        persistent_workers=True,
        drop_last=True,
    )
    batches = next_batches(loader)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    model.to(device)

    start = time.time()

    # Stage 1: train only the new head, so a random head does not disturb the backbone.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=HEAD_LR)
    run_steps(model, batches, optimizer, None, HEAD_STEPS, device, "head")

    # Stage 2: fine-tune every layer at a lower rate.
    for parameter in model.features.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_STEPS)
    run_steps(model, batches, optimizer, scheduler, FINE_TUNE_STEPS, device, "fine-tune")

    train_seconds = time.time() - start

    name = f"{args.arm}_seed{args.seed}"
    torch.save(model.state_dict(), args.out / f"{name}.pt")

    start = time.time()
    logits = {
        "lab_dev": predict(model, lab_dev, device),
        "field_dev_crops": predict(model, field_dev_crops, device),
        "field_dev_whole": predict(model, field_dev_whole, device),
    }
    predict_seconds = time.time() - start
    np.savez(args.out / f"{name}_logits.npz", **logits)

    result = {
        "arm": args.arm,
        "seed": args.seed,
        "device": device,
        "updates": total_steps,
        "batch_size": BATCH_SIZE,
        "train_images": {"lab": len(lab_train), "field": len(extra)},
        "train_seconds": round(train_seconds, 1),
        "predict_seconds": round(predict_seconds, 1),
        "field_boxes_skipped_as_too_small": skipped,
        "lab_dev": score(logits["lab_dev"], lab_dev),
        "field_dev_labelled_crops": score(logits["field_dev_crops"], field_dev_crops),
        "field_dev_whole_photos": score(logits["field_dev_whole"], field_dev_whole),
    }
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
