"""Compare NER evaluation results across multiple model runs.

Reads per_entity_test.json from each result directory and produces
a side-by-side markdown comparison table.

Usage:
    python scripts/compare_models.py results_v3 results/phayathaibert
    python scripts/compare_models.py results_v3 results/phayathaibert --output results/model_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ENTITY_TYPES = [
    "HARD_SKILL", "PERSON", "LOCATION", "COMPENSATION",
    "EMPLOYMENT_TERMS", "CONTACT", "DEMOGRAPHIC",
]


def load_results(result_dir: Path) -> dict:
    """Load per_entity_test.json from a result directory."""
    path = result_dir / "per_entity_test.json"
    if not path.exists():
        raise FileNotFoundError(f"No per_entity_test.json in {result_dir}")
    with open(path) as f:
        return json.load(f)


def format_delta(val: float) -> str:
    """Format a delta value with sign and color hint."""
    if val > 0:
        return f"+{val:.3f}"
    elif val < 0:
        return f"{val:.3f}"
    return "0.000"


def build_comparison_table(model_results: dict[str, dict]) -> str:
    """Build a markdown comparison table from multiple model results."""
    model_names = list(model_results.keys())
    lines = []

    # Header
    header = f"| {'Metric':<22} |"
    separator = f"|{'-' * 24}|"
    for name in model_names:
        header += f" {name:>16} |"
        separator += f"{'-' * 18}|"

    # Add delta column if exactly 2 models
    if len(model_names) == 2:
        header += f" {'Delta':>10} |"
        separator += f"{'-' * 12}|"

    lines.append(header)
    lines.append(separator)

    # Overall metrics
    for metric_key, metric_label in [
        ("f1", "Overall F1"),
        ("precision", "Overall Precision"),
        ("recall", "Overall Recall"),
        ("accuracy", "Overall Accuracy"),
    ]:
        row = f"| **{metric_label:<20}** |"
        values = []
        for name in model_names:
            val = model_results[name]["overall"][metric_key]
            values.append(val)
            row += f" {val:>16.3f} |"

        if len(model_names) == 2:
            delta = values[1] - values[0]
            row += f" {format_delta(delta):>10} |"

        lines.append(row)

    # Separator
    lines.append(f"| {'':22} |" + "|".join([" " * 17 + " |" for _ in model_names]) +
                 (" " + " " * 10 + " |" if len(model_names) == 2 else ""))

    # Per-entity F1
    for etype in ENTITY_TYPES:
        row = f"| {etype + ' F1':<22} |"
        values = []
        for name in model_names:
            entity_data = model_results[name]["per_entity"].get(etype, {})
            val = entity_data.get("f1", 0.0)
            values.append(val)
            row += f" {val:>16.3f} |"

        if len(model_names) == 2:
            delta = values[1] - values[0]
            row += f" {format_delta(delta):>10} |"

        lines.append(row)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NER model evaluation results")
    parser.add_argument(
        "result_dirs", type=Path, nargs="+",
        help="Paths to result directories containing per_entity_test.json",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save comparison markdown (default: stdout only)",
    )
    parser.add_argument(
        "--names", type=str, nargs="+", default=None,
        help="Display names for each model (default: directory names)",
    )
    args = parser.parse_args()

    # Load results
    model_results = {}
    for i, result_dir in enumerate(args.result_dirs):
        name = args.names[i] if args.names and i < len(args.names) else result_dir.name
        model_results[name] = load_results(result_dir)

    # Build and print table
    table = build_comparison_table(model_results)

    print("\n# Model Comparison\n")
    print(table)
    print()

    # Save if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("# Model Comparison\n\n")
            f.write(table)
            f.write("\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
