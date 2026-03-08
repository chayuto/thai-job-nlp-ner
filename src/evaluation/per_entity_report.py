"""Per-entity F1 evaluation and confusion analysis for NER.

Loads a trained model and test dataset, runs inference, then produces
a per-entity-class breakdown of precision/recall/F1 plus a label
confusion matrix showing which entity types get misclassified.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.alignment.token_mapper import DEFAULT_LABELS, IGNORE_INDEX

logger = logging.getLogger(__name__)

metric = evaluate.load("seqeval")


def _decode_labels(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert integer label arrays to IOB2 string sequences, filtering padding."""
    pred_tags = []
    true_tags = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_row = []
        true_row = []
        for p, l in zip(pred_seq, label_seq):
            if l == IGNORE_INDEX:
                continue
            pred_row.append(DEFAULT_LABELS[p])
            true_row.append(DEFAULT_LABELS[l])
        pred_tags.append(pred_row)
        true_tags.append(true_row)

    return pred_tags, true_tags


def per_entity_report(
    pred_tags: list[list[str]],
    true_tags: list[list[str]],
) -> dict:
    """Compute per-entity-class and overall metrics via seqeval.

    Returns a dict with:
      - Per entity type: precision, recall, f1, number (support)
      - overall_precision, overall_recall, overall_f1, overall_accuracy
    """
    results = metric.compute(
        predictions=pred_tags,
        references=true_tags,
        zero_division=0,
    )
    return results


def build_confusion_matrix(
    pred_tags: list[list[str]],
    true_tags: list[list[str]],
) -> dict[str, dict[str, int]]:
    """Build a token-level confusion matrix between entity types.

    Groups B-/I- prefixes into their base entity type (e.g. B-LOCATION -> LOCATION).
    Returns: {true_label: {pred_label: count}}
    """
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred_seq, true_seq in zip(pred_tags, true_tags):
        for p, t in zip(pred_seq, true_seq):
            # Strip IOB prefix to get entity type
            true_type = t.split("-", 1)[1] if "-" in t else t
            pred_type = p.split("-", 1)[1] if "-" in p else p
            confusion[true_type][pred_type] += 1

    return {k: dict(v) for k, v in confusion.items()}


def format_report(results: dict, confusion: dict[str, dict[str, int]]) -> str:
    """Format per-entity metrics and confusion matrix as a readable string."""
    lines = []
    lines.append("=" * 65)
    lines.append("Per-Entity NER Evaluation Report")
    lines.append("=" * 65)

    # Per-entity table
    entity_types = [
        "HARD_SKILL", "PERSON", "LOCATION", "COMPENSATION",
        "EMPLOYMENT_TERMS", "CONTACT", "DEMOGRAPHIC",
    ]

    lines.append(f"\n{'Entity Type':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    lines.append("-" * 50)

    for etype in entity_types:
        key = etype
        if key in results:
            data = results[key]
            lines.append(
                f"{etype:<20} {data['precision']:>6.3f} {data['recall']:>6.3f} "
                f"{data['f1']:>6.3f} {data['number']:>8d}"
            )
        else:
            lines.append(f"{etype:<20} {'--':>6} {'--':>6} {'--':>6} {'0':>8}")

    lines.append("-" * 50)
    lines.append(
        f"{'OVERALL':<20} {results['overall_precision']:>6.3f} "
        f"{results['overall_recall']:>6.3f} {results['overall_f1']:>6.3f}"
    )

    # Confusion matrix
    all_types = sorted(set(
        list(confusion.keys()) +
        [p for row in confusion.values() for p in row.keys()]
    ))

    if len(all_types) > 1:
        lines.append(f"\n{'Confusion Matrix (token-level)':}")
        lines.append("-" * 50)

        # Header
        header = f"{'True \\ Pred':<18}" + "".join(f"{t[:8]:>9}" for t in all_types)
        lines.append(header)

        for true_type in all_types:
            row = f"{true_type:<18}"
            for pred_type in all_types:
                count = confusion.get(true_type, {}).get(pred_type, 0)
                row += f"{count:>9}"
            lines.append(row)

    lines.append("")
    return "\n".join(lines)


def evaluate_model(
    model_dir: Path,
    dataset_dir: Path,
    split: str = "test",
    output_path: Path | None = None,
) -> dict:
    """Load model, run predictions on a split, produce per-entity report.

    Args:
        model_dir: Path to saved model (e.g. results/final/).
        dataset_dir: Path to processed HuggingFace DatasetDict.
        split: Which split to evaluate ("test", "validation", "train").
        output_path: Optional path to save JSON results.

    Returns:
        Dict with per-entity metrics and confusion matrix.
    """
    # Load model and tokenizer
    logger.info(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}...")
    dataset = DatasetDict.load_from_disk(str(dataset_dir))
    eval_data = dataset[split]
    logger.info(f"Evaluating on {split} split: {len(eval_data)} examples")

    # Set up a lightweight Trainer for prediction
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True
    )

    training_args = TrainingArguments(
        output_dir="/tmp/eval_tmp",
        per_device_eval_batch_size=16,
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    # Get predictions
    predictions_output = trainer.predict(eval_data)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    preds = np.argmax(logits, axis=2)

    # Decode to IOB2 strings
    pred_tags, true_tags = _decode_labels(preds, labels)

    # Per-entity report
    results = per_entity_report(pred_tags, true_tags)
    confusion = build_confusion_matrix(pred_tags, true_tags)

    # Print report
    report_str = format_report(results, confusion)
    print(report_str)

    # Save if requested
    if output_path:
        output_data = {
            "per_entity": {
                k: v for k, v in results.items()
                if not k.startswith("overall_")
            },
            "overall": {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            },
            "confusion_matrix": confusion,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            """Convert numpy types to native Python for JSON serialization."""
            if hasattr(obj, "item"):
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=_convert)
        logger.info(f"Results saved to {output_path}")

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Per-entity NER evaluation report")
    parser.add_argument(
        "--model", type=Path, default=Path("results/final"),
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed"),
        help="Path to processed HuggingFace DatasetDict",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save JSON results (e.g. results/per_entity_test.json)",
    )
    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model,
        dataset_dir=args.dataset,
        split=args.split,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
