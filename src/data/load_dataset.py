"""Load and validate NER export data from production or synthetic sources."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_LABELS = {
    "HARD_SKILL",
    "PERSON",
    "LOCATION",
    "COMPENSATION",
    "EMPLOYMENT_TERMS",
    "CONTACT",
    "DEMOGRAPHIC",
}


@dataclass
class NERPost:
    id: str
    raw_text: str
    entities: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DatasetStats:
    total_posts: int = 0
    valid_posts: int = 0
    skipped_posts: int = 0
    total_entities: int = 0
    matched_entities: int = 0
    unmatched_entities: int = 0
    label_counts: Counter = field(default_factory=Counter)
    avg_text_length: float = 0.0
    avg_entities_per_post: float = 0.0


def load_json(path: Path) -> list[dict]:
    """Load a JSON file containing NER export data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def validate_post(raw: dict) -> tuple[NERPost | None, list[str]]:
    """Validate a single post entry and return (post, warnings)."""
    warnings: list[str] = []
    post_id = str(raw.get("id", "unknown"))

    raw_text = raw.get("raw_text")
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return None, [f"Post {post_id}: missing or empty raw_text, skipping"]

    entities = raw.get("entities", [])
    valid_entities: list[dict[str, str]] = []

    for ent in entities:
        text = ent.get("text", "")
        label = ent.get("label", "")

        if not text or not label:
            warnings.append(f"Post {post_id}: entity missing text or label, skipping entity")
            continue

        if label not in VALID_LABELS:
            warnings.append(f"Post {post_id}: unknown label '{label}', skipping entity")
            continue

        if text not in raw_text:
            warnings.append(f"Post {post_id}: entity '{text}' not found in raw_text (fuzzy match will handle later)")

        valid_entities.append({"text": text, "label": label})

    return NERPost(id=post_id, raw_text=raw_text, entities=valid_entities), warnings


def load_and_validate(path: Path) -> tuple[list[NERPost], DatasetStats]:
    """Load, validate, and compute stats for an NER export file."""
    raw_data = load_json(path)
    posts: list[NERPost] = []
    stats = DatasetStats()
    text_lengths: list[int] = []

    for raw in raw_data:
        stats.total_posts += 1
        post, warnings = validate_post(raw)

        for w in warnings:
            logger.warning(w)

        if post is None:
            stats.skipped_posts += 1
            continue

        stats.valid_posts += 1
        posts.append(post)
        text_lengths.append(len(post.raw_text))

        for ent in post.entities:
            stats.total_entities += 1
            stats.label_counts[ent["label"]] += 1
            if ent["text"] in post.raw_text:
                stats.matched_entities += 1
            else:
                stats.unmatched_entities += 1

    if text_lengths:
        stats.avg_text_length = sum(text_lengths) / len(text_lengths)
    if stats.valid_posts:
        stats.avg_entities_per_post = stats.total_entities / stats.valid_posts

    return posts, stats


def merge_sources(*paths: Path) -> list[NERPost]:
    """Load and merge multiple NER export files (production + synthetic)."""
    all_posts: list[NERPost] = []
    seen_ids: set[str] = set()

    for path in paths:
        if not path.exists():
            logger.warning(f"File not found, skipping: {path}")
            continue

        posts, stats = load_and_validate(path)
        for post in posts:
            if post.id in seen_ids:
                post.id = f"{path.stem}_{post.id}"
            seen_ids.add(post.id)
            all_posts.append(post)

        logger.info(f"Loaded {stats.valid_posts} posts from {path.name}")

    return all_posts


def print_stats(stats: DatasetStats, source: str) -> None:
    """Print dataset statistics to console."""
    print(f"\n{'=' * 50}")
    print(f"Dataset Stats: {source}")
    print(f"{'=' * 50}")
    print(f"Total posts:          {stats.total_posts}")
    print(f"Valid posts:          {stats.valid_posts}")
    print(f"Skipped posts:        {stats.skipped_posts}")
    print(f"Total entities:       {stats.total_entities}")
    print(f"  Exact match:        {stats.matched_entities}")
    print(f"  Needs fuzzy match:  {stats.unmatched_entities}")
    print(f"Avg text length:      {stats.avg_text_length:.0f} chars")
    print(f"Avg entities/post:    {stats.avg_entities_per_post:.1f}")
    print(f"\nEntity distribution:")
    for label in sorted(VALID_LABELS):
        count = stats.label_counts.get(label, 0)
        pct = (count / stats.total_entities * 100) if stats.total_entities else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<20} {count:>5}  ({pct:5.1f}%) {bar}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Load and validate NER export data")
    parser.add_argument("--input", type=Path, required=True, help="Path to NER export JSON")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"File not found: {args.input}")
        return

    posts, stats = load_and_validate(args.input)
    print_stats(stats, source=args.input.name)


if __name__ == "__main__":
    main()
