"""Strict exact-match F1 evaluation using seqeval for NER."""

from __future__ import annotations

import numpy as np
import evaluate

from src.alignment.token_mapper import DEFAULT_LABELS, IGNORE_INDEX

metric = evaluate.load("seqeval")


def compute_metrics(eval_prediction) -> dict[str, float]:
    """Compute strict span-level F1, precision, recall via seqeval.

    Filters out padding tokens (-100) and maps integer label IDs
    back to IOB2 string labels before passing to seqeval.
    """
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=2)

    true_predictions = [
        [DEFAULT_LABELS[p] for (p, l) in zip(prediction, label) if l != IGNORE_INDEX]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [DEFAULT_LABELS[l] for (p, l) in zip(prediction, label) if l != IGNORE_INDEX]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
