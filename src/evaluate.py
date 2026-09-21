"""Evaluation metrics and baseline calculation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_labels(model, x, threshold: float = 0.5):
    proba = np.asarray(model.predict(x, verbose=0)).reshape(-1)
    labels = (proba >= threshold).astype(np.int8)
    return labels, proba


def predict_labels_sequence(model, sequence, threshold: float = 0.5):
    proba = np.asarray(model.predict(sequence, verbose=0)).reshape(-1)
    labels = (proba >= threshold).astype(np.int8)
    y_true = np.concatenate([sequence[i][1] for i in range(len(sequence))]).astype(np.int8)
    return y_true, labels, proba


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true).astype(np.int8)
    y_pred = np.asarray(y_pred).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    if y_proba is not None and len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "mcc": float(mcc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def accuracy_only(model, x, y, threshold=0.5) -> float:
    pred, _ = predict_labels(model, x, threshold)
    return float(accuracy_score(y, pred))


def sequence_accuracy(model, sequence, threshold=0.5) -> float:
    y_true, pred, _ = predict_labels_sequence(model, sequence, threshold)
    return float(accuracy_score(y_true, pred))


def majority_baseline(y_train, y_val, y_test) -> dict:
    results = {}
    majority = int(np.bincount(np.asarray(y_train).astype(np.int8)).argmax())
    for split, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        y = np.asarray(y).astype(np.int8)
        pred = np.full_like(y, majority)
        m = classification_metrics(y, pred, None)
        results[split] = m
    results["majority_class"] = majority
    return results
