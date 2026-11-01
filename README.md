# Leaf Doctor

A tomato disease triage assistant for real garden photos. ECE 539 Fall 2026 Integration Project, team G11.

A home gardener photographs a sick leaf. The system either gives care steps from a UW-Madison Extension fact sheet, asks one question that separates two likely diseases, asks for a better photo, or refers the gardener to the UW Plant Disease Diagnostics Clinic.

## Pipeline

| Stage | Method | Status | Input | Output |
| --- | --- | --- | --- | --- |
| Leaf finder | OWLv2 text-prompted detector | Pretrained, frozen | RGB photo | Best leaf box and score, or "retake photo" |
| Disease classifier | EfficientNet-B0 | ImageNet start, fine-tuned by the team | 3 x 224 x 224 image | 8 logits |
| Triage gate | Temperature scaling and thresholds | One fitted scalar, not neural | 8 logits | Advise, ask, or refer, with the candidate diseases |
| Advice writer | Small language model over retrieved fact sheet passages | Pretrained, frozen | Route, candidates, passages | Care steps or a question, with the source link |

## Repository layout

| Folder | Contents |
| --- | --- |
| `docs/` | Project proposal (PDF and Word source) and its diagram (SVG source and PNG). Member names and GitHub usernames are left out and are only in the Canvas copy |
| `splits/` | Frozen train, development and final partitions |
| `pilot/` | Feasibility pilot code and saved results |
| `P1/` | Data access, baseline, evaluation harness |
| `P2/` | Disease classifier training and the controlled comparison |
| `P3/` | Leaf finder and pipeline integration |
| `P4/` | Triage gate, advice writer, and demo |
| `final/` | Integrated Colab notebook and final results |

## Rebuilding the proposal

To rebuild the PDF, open `docs/proposal.docx` in Word and save as PDF. Check that it stays at 3 pages.

To change the diagram, edit `docs/architecture.svg` in a text editor or Inkscape. Then export a PNG, and in Word right-click the diagram in `proposal.docx` and choose Change Picture. To export with Chrome on a Mac, run this from `docs/`:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --hide-scrollbars \
  --user-data-dir="$(mktemp -d)" --force-device-scale-factor=2 --window-size=760,356 \
  --screenshot=architecture.png "file://$PWD/architecture.svg"
```

## Data access

Neither dataset is stored here.

```bash
# PlantVillage: only the 8 tomato classes we use
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset pv
git -C pv sparse-checkout set --no-cone /leaf-map.json \
  "/raw/color/Tomato___Bacterial_spot/" "/raw/color/Tomato___Early_blight/" \
  "/raw/color/Tomato___Late_blight/" "/raw/color/Tomato___Leaf_Mold/" \
  "/raw/color/Tomato___Septoria_leaf_spot/" "/raw/color/Tomato___Tomato_Yellow_Leaf_Curl_Virus/" \
  "/raw/color/Tomato___Tomato_mosaic_virus/" "/raw/color/Tomato___healthy/"

# PlantDoc with leaf boxes
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset pd
```

## Partitions

`pilot/make_splits.py` wrote the files in `splits/`. They are committed so that every arm and every teammate uses the same partitions. PlantVillage is split by physical leaf. PlantDoc is split by photo, with exact and near-duplicate photos kept together, and every official PlantDoc test photo is in the final partition.

Choices are made on train and development data only. The final partition is read once, after the system is frozen. `pilot/images.py` refuses to cache it.

## Pilot results

These come from development data only. No model has read the final partition.

Three seeds on an RTX 3090, scored on 112 real development photos. Values are the mean, with the standard deviation across seeds after +/-. Resampling photos gives a much wider interval: for seed 0, whole-photo macro-F1 runs from 0.11 to 0.24 for the baseline and from 0.46 to 0.66 for the change. Saved in `pilot/results/`.

| Measure | Majority guess | Baseline: lab photos only | Change: lab photos and real leaf crops |
| --- | --- | --- | --- |
| Macro-F1, whole photo | 0.04 | 0.16 +/- 0.02 | 0.56 +/- 0.01 |
| Macro-F1, finder's best box | | 0.20 +/- 0.01 | 0.55 +/- 0.02 |
| Accuracy, whole photo | 20% | 24% | 59% |
| Macro-F1, 3,033 lab photos | | 0.99 | 0.97 |
| Right advice, whole photo | | 1% | 21% |
| Wrong advice, whole photo | | 0% | 5% |
| Asked one question, whole photo | | 0% | 11% |
| Referred to the clinic, whole photo | | 99% | 63% |

The gate thresholds are first guesses, and the temperature was fitted on labeled crops from the same development photos, so the outcome rates are optimistic. Right advice means the gate advised and the disease was correct. The written advice is checked separately.

The leaf finder's best box overlaps a labeled leaf (IoU of at least 0.5) on 64% of development photos.

`trace_two_photos.json` follows two photos through all four stages on an Apple M4 Pro. It shows two known problems in the advice writer: it changed a disposal step from the fact sheet, and its question asks for book knowledge and not what a gardener can see.

## Running the pilot

Requires PyTorch, torchvision, Pillow and NumPy. The finder and trace also need transformers.

```bash
python pilot/make_splits.py --plantvillage pv --plantdoc pd --out splits   # already committed
for seed in 0 1 2; do
  for arm in lab lab_field; do
    python pilot/train_classifier.py --arm $arm --seed $seed --splits splits \
      --plantvillage pv --plantdoc pd --cache cache --out runs
  done
done
python pilot/run_finder.py --splits splits --plantdoc pd --out runs --device cuda
python pilot/eval_system.py --splits splits --plantdoc pd --runs runs --cache cache
python pilot/summarize_seeds.py --runs runs --out runs/seed_summary.json
python pilot/trace_one_photo.py --splits splits --plantdoc pd --runs runs \
  --out runs/trace.json --weights lab_field_seed0 --device cuda
```

Each training arm took 55 seconds on an RTX 3090 and 4.8 minutes on an Apple M4 Pro (`pilot/results/m4pro/`).
