"""Shared labels and metrics for the pilot scripts."""
import numpy as np

# Our 8 labels, in a fixed order. Each one exists in both datasets.
CLASSES = [
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "mosaic_virus",
    "yellow_leaf_curl_virus",
    "healthy",
]

PLANTVILLAGE_FOLDER_TO_CLASS = {
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus": "mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
    "Tomato___healthy": "healthy",
}

PLANTDOC_LABEL_TO_CLASS = {
    "Tomato leaf bacterial spot": "bacterial_spot",
    "Tomato Early blight leaf": "early_blight",
    "Tomato leaf late blight": "late_blight",
    "Tomato mold leaf": "leaf_mold",
    "Tomato Septoria leaf spot": "septoria_leaf_spot",
    "Tomato leaf mosaic virus": "mosaic_virus",
    "Tomato leaf yellow virus": "yellow_leaf_curl_virus",
    "Tomato leaf": "healthy",
}


def macro_f1(y_true, y_pred, n_classes=len(CLASSES)):
    """Mean F1 over the classes that appear in y_true."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in range(n_classes):
        if not (y_true == c).any():
            continue
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def bootstrap_ci(y_true, y_pred, groups=None, n_rounds=2000, seed=0):
    """95% interval for macro-F1, resampling whole photos when groups are given."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n_rounds):
        picks = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[g] for g in picks])
        values.append(macro_f1(y_true[idx], y_pred[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
