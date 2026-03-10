"""End-to-end pipeline: raw NER export → HuggingFace Dataset with IOB2 labels.

Orchestrates the full Phase 1 data engineering pipeline:
1. Load raw data (production export or synthetic)
2. Fuzzy-align entity substrings to character offsets
3. Map character offsets to subword token IOB2 labels
4. Split into train/val/test
5. Save as HuggingFace Dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.alignment.fuzzy_matcher import align_post_entities
from src.alignment.token_mapper import (
    DEFAULT_LABELS,
    TokenizedPost,
    align_tokens_to_iob2,
    get_tokenizer,
    verify_iob2_consistency,
)
from src.data.load_dataset import NERPost, collect_data_files, load_and_validate, merge_sources, print_stats

logger = logging.getLogger(__name__)


def process_post(
    post: NERPost,
    tokenizer,
    threshold: float = 85.0,
    max_length: int = 512,
) -> TokenizedPost | None:
    """Process a single post through the full alignment pipeline."""
    # Step 1: Fuzzy-align entities to character offsets
    aligned, unmatched = align_post_entities(
        raw_text=post.raw_text,
        entities=post.entities,
        threshold=threshold,
    )

    if unmatched:
        logger.debug(
            f"Post {post.id}: {len(unmatched)} unmatched entities: "
            f"{[e['text'] for e in unmatched]}"
        )

    if not aligned:
        logger.warning(f"Post {post.id}: no entities aligned, skipping")
        return None

    # Step 2: Map character offsets to subword token IOB2 labels
    tokenized = align_tokens_to_iob2(
        tokenizer=tokenizer,
        raw_text=post.raw_text,
        aligned_entities=aligned,
        max_length=max_length,
    )

    # Step 3: Verify IOB2 consistency
    warnings = verify_iob2_consistency(tokenized.labels, DEFAULT_LABELS)
    for w in warnings:
        logger.warning(f"Post {post.id}: {w}")

    return tokenized


def build_dataset(
    posts: list[NERPost],
    tokenizer,
    threshold: float = 85.0,
    max_length: int = 512,
) -> list[dict]:
    """Process all posts and return list of feature dicts for HuggingFace Dataset."""
    records: list[dict] = []
    skipped = 0

    for post in posts:
        result = process_post(post, tokenizer, threshold, max_length)
        if result is None:
            skipped += 1
            continue

        records.append({
            "input_ids": result.input_ids,
            "attention_mask": result.attention_mask,
            "labels": result.labels,
        })

    logger.info(f"Processed {len(records)} posts, skipped {skipped}")
    return records


def split_dataset(
    records: list[dict],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Split records into train/val/test and return a HuggingFace DatasetDict."""
    # First split: separate test set
    train_val, test = train_test_split(
        records, test_size=test_size, random_state=seed
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction, random_state=seed
    )

    return DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })


def run_pipeline(
    input_paths: list[Path],
    output_dir: Path,
    checkpoint: str = "airesearch/wangchanberta-base-att-spm-uncased",
    threshold: float = 85.0,
    max_length: int = 512,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Run the full IOB2 formatting pipeline end-to-end."""
    # Load and merge data sources
    posts = merge_sources(*input_paths)
    if not posts:
        raise ValueError("No valid posts loaded from any input file")
    logger.info(f"Loaded {len(posts)} total posts")

    # Print stats for each source
    for path in collect_data_files(input_paths):
        _, stats = load_and_validate(path)
        print_stats(stats, source=path.name)

    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {checkpoint}")
    tokenizer = get_tokenizer(checkpoint)

    # Process all posts
    logger.info("Running alignment pipeline...")
    records = build_dataset(posts, tokenizer, threshold, max_length)

    if not records:
        raise ValueError("No posts survived the alignment pipeline")

    # Split and save
    logger.info("Splitting into train/val/test...")
    dataset_dict = split_dataset(records, test_size, val_size, seed)

    logger.info(f"Saving dataset to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    # Print split sizes
    for split_name, ds in dataset_dict.items():
        logger.info(f"  {split_name}: {len(ds)} examples")

    return dataset_dict


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert NER export data to IOB2-tagged HuggingFace Dataset"
    )
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True,
        help="Path(s) to NER export JSON file(s)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"),
        help="Output directory for HuggingFace Dataset",
    )
    parser.add_argument("--threshold", type=float, default=85.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        input_paths=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        max_length=args.max_length,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
