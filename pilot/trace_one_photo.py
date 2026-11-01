"""Trace development photos through all four stages and print each handoff.

Picks one photo the gate sends to advice and one it sends to a question.

Usage:
    python trace_one_photo.py --splits ../splits --plantdoc DIR --runs DIR --cache DIR --out FILE
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Owlv2ForObjectDetection,
)

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASSES
from eval_system import ADVISE_AT, ASK_AT, load_model, softmax
from fact_sheets import SHEETS, uw_passages
from images import open_photo, plantdoc_folder_path, plantdoc_rows
from run_finder import MIN_SCORE, MODEL_ID, PROMPTS
from train_classifier import TEST_TRANSFORM

LANGUAGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
UW_CLASSES = ["early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot", "mosaic_virus"]


def find_leaf(processor, model, image, device):
    inputs = processor(text=[PROMPTS], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    size = torch.tensor([[image.height, image.width]], device=device)
    found = processor.post_process_grounded_object_detection(
        outputs, target_sizes=size, threshold=MIN_SCORE, text_labels=[PROMPTS]
    )[0]
    if len(found["scores"]) == 0:
        return None
    best = int(found["scores"].argmax())
    return {
        "box": [round(float(v), 1) for v in found["boxes"][best].cpu()],
        "score": round(float(found["scores"][best]), 3),
    }


def classify(model, image, device):
    batch = TEST_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(batch).cpu().numpy()


def route(probabilities):
    order = np.argsort(-probabilities)
    p1, p2 = probabilities[order[0]], probabilities[order[1]]
    if p1 >= ADVISE_AT:
        return "advise", [CLASSES[order[0]]]
    if p1 + p2 >= ASK_AT:
        return "ask", [CLASSES[order[0]], CLASSES[order[1]]]
    return "refer", []


def write_text(tokenizer, model, prompt, device):
    messages = [
        {
            "role": "system",
            "content": "You help home gardeners. Use only the text you are given. Do not add facts.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    seconds = time.time() - start
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return answer.strip(), round(seconds, 1)


def advice_prompt(label, care):
    name = label.replace("_", " ")
    return f"Fact sheet text on {name}:\n{care}\n\nWrite 3 short care steps for a home gardener."


def question_prompt(labels, symptoms):
    a, b = (label.replace("_", " ") for label in labels)
    return (
        f"Symptoms of {a}:\n{symptoms[0]}\n\nSymptoms of {b}:\n{symptoms[1]}\n\n"
        f"Write one question a gardener can answer by looking at the leaf that tells {a} from {b}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--plantdoc", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", default="lab_field_seed0")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = args.device
    report = json.loads((args.runs / "system_dev.json").read_text())
    temperature = report[args.weights]["temperature"]

    finder_processor = AutoProcessor.from_pretrained(MODEL_ID)
    finder = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    classifier = load_model(args.runs / f"{args.weights}.pt", device)
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    writer = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL, dtype=torch.float16).to(device)

    wanted = {"advise", "ask"}
    traces = []
    for row in plantdoc_rows(args.splits, "pd_dev"):
        if row["mixed_labels"] == "1" or not wanted:
            continue
        path = plantdoc_folder_path(args.plantdoc, row["photo"])
        image, _, _ = open_photo(path, int(row["width"]), int(row["height"]))

        logits = classify(classifier, image, device)
        probabilities = softmax(logits / temperature)[0]
        decision, candidates = route(probabilities)
        if decision not in wanted or not set(candidates) <= set(UW_CLASSES):
            continue
        wanted.discard(decision)

        trace = {
            "photo": row["photo"],
            "photo_size": [image.width, image.height],
            "true_label": row["photo_label"],
            "stage1_leaf_finder": find_leaf(finder_processor, finder, image, device),
            "stage2_probabilities": {
                c: round(float(p), 3) for c, p in zip(CLASSES, probabilities)
            },
            "stage3_route": decision,
            "stage3_candidates": candidates,
            "sources": [SHEETS[c] for c in candidates],
        }
        if decision == "advise":
            _, care = uw_passages(candidates[0])
            prompt = advice_prompt(candidates[0], care[:2500])
        else:
            symptoms = [uw_passages(c)[0][:1200] for c in candidates]
            prompt = question_prompt(candidates, symptoms)
        trace["stage4_text"], trace["stage4_seconds"] = write_text(tokenizer, writer, prompt, device)
        traces.append(trace)

    args.out.write_text(json.dumps(traces, indent=2) + "\n")
    print(json.dumps(traces, indent=2))


if __name__ == "__main__":
    main()
