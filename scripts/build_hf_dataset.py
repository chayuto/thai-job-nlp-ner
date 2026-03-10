"""Build a publishable HuggingFace NER dataset from synthetic data.

Collates all synthetic data files, converts from span-based format to
CoNLL-style tokens + ner_tags (IOB2), and publishes to HuggingFace Hub.

The output follows HF NER conventions:
- `tokens`: word-level tokenization (pythainlp) — model-agnostic
- `ner_tags`: integer IOB2 labels via ClassLabel

Usage:
    # Build locally for review:
    python scripts/build_hf_dataset.py

    # Build and push to HuggingFace Hub:
    python scripts/build_hf_dataset.py --push --repo-id chayuto/thai-job-ner-dataset

    # Push as private:
    python scripts/build_hf_dataset.py --push --private
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from pythainlp.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from src.alignment.fuzzy_matcher import align_post_entities
from src.data.load_dataset import NERPost, collect_data_files, load_and_validate, print_stats

logger = logging.getLogger(__name__)

# IOB2 label names — must match the order used by the model
LABEL_NAMES = [
    "O",
    "B-HARD_SKILL", "I-HARD_SKILL",
    "B-PERSON", "I-PERSON",
    "B-LOCATION", "I-LOCATION",
    "B-COMPENSATION", "I-COMPENSATION",
    "B-EMPLOYMENT_TERMS", "I-EMPLOYMENT_TERMS",
    "B-CONTACT", "I-CONTACT",
    "B-DEMOGRAPHIC", "I-DEMOGRAPHIC",
]


def word_tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Word-tokenize Thai text and compute character offsets for each token.

    Returns list of (token, char_start, char_end) tuples.
    Whitespace-only tokens are excluded.
    """
    words = word_tokenize(text, engine="newmm", keep_whitespace=True)
    result: list[tuple[str, int, int]] = []
    pos = 0
    for word in words:
        start = pos
        end = pos + len(word)
        if word.strip():
            result.append((word, start, end))
        pos = end
    return result


def spans_to_iob2(
    word_tokens: list[tuple[str, int, int]],
    entities: list[dict],
) -> list[str]:
    """Convert character-level entity spans to word-level IOB2 tags.

    Args:
        word_tokens: List of (token, char_start, char_end) from word_tokenize_with_offsets.
        entities: List of dicts with keys char_start, char_end, label.

    Returns:
        List of IOB2 tag strings, one per word token.
    """
    tags = ["O"] * len(word_tokens)

    for entity in entities:
        ent_start = entity["char_start"]
        ent_end = entity["char_end"]
        label = entity["label"]
        first_token = True

        for i, (_, tok_start, tok_end) in enumerate(word_tokens):
            overlap_start = max(tok_start, ent_start)
            overlap_end = min(tok_end, ent_end)

            if overlap_start < overlap_end:
                if first_token:
                    tags[i] = f"B-{label}"
                    first_token = False
                else:
                    tags[i] = f"I-{label}"

    return tags


def process_post(post: NERPost, threshold: float = 85.0) -> dict | None:
    """Process a single post: align entities, word-tokenize, assign IOB2 tags."""
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

    entity_spans = [
        {"char_start": a.char_start, "char_end": a.char_end, "label": a.label}
        for a in aligned
    ]

    word_tokens = word_tokenize_with_offsets(post.raw_text)
    if not word_tokens:
        logger.warning(f"Post {post.id}: empty after tokenization, skipping")
        return None

    iob2_tags = spans_to_iob2(word_tokens, entity_spans)

    label_set = set(LABEL_NAMES)
    for tag in iob2_tags:
        if tag not in label_set:
            logger.warning(f"Post {post.id}: unknown tag '{tag}', skipping")
            return None

    tokens = [tok for tok, _, _ in word_tokens]
    return {
        "id": post.id,
        "tokens": tokens,
        "ner_tags": iob2_tags,
    }


def build_records(
    posts: list[NERPost],
    threshold: float = 85.0,
) -> list[dict]:
    """Process all posts into CoNLL-style records."""
    records: list[dict] = []
    skipped = 0

    for post in posts:
        result = process_post(post, threshold)
        if result is None:
            skipped += 1
            continue
        records.append(result)

    logger.info(f"Processed {len(records)} posts, skipped {skipped}")
    return records


def records_to_dataset_dict(
    records: list[dict],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Split records and create a DatasetDict with proper ClassLabel features."""
    class_label = ClassLabel(names=LABEL_NAMES)
    features = Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(class_label),
    })

    # Convert string tags to integer IDs
    for record in records:
        record["ner_tags"] = [class_label.str2int(tag) for tag in record["ner_tags"]]

    train_val, test = train_test_split(records, test_size=test_size, random_state=seed)
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_fraction, random_state=seed)

    return DatasetDict({
        "train": Dataset.from_list(train, features=features),
        "validation": Dataset.from_list(val, features=features),
        "test": Dataset.from_list(test, features=features),
    })


def generate_dataset_card(dataset_dict: DatasetDict, repo_id: str) -> str:
    """Generate a HuggingFace dataset card (README.md) with YAML frontmatter."""
    n_train = len(dataset_dict["train"])
    n_val = len(dataset_dict["validation"])
    n_test = len(dataset_dict["test"])
    n_total = n_train + n_val + n_test

    if n_total < 1000:
        size_cat = "n<1K"
    elif n_total < 10000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "10K<n<100K"

    return f"""---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- th
license: cc-by-4.0
multilinguality:
- monolingual
size_categories:
- {size_cat}
source_datasets:
- original
task_categories:
- token-classification
task_ids:
- named-entity-recognition
pretty_name: Thai Job Posting NER (Synthetic)
tags:
- NER
- Thai
- job-posting
- synthetic
- wangchanberta
- IOB2
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Thai Job Posting NER Dataset (Synthetic)

A synthetic Named Entity Recognition dataset for Thai job postings, formatted in IOB2 tagging scheme.

## Dataset Description

This dataset contains **{n_total} synthetically generated Thai job posting texts** annotated with 7 entity types commonly found in informal Thai job advertisements (e.g., Facebook groups, LINE).

The data was generated using GPT-4o with carefully designed prompts to mimic the style and content of real Thai job posts, then aligned using fuzzy matching with Thai Character Cluster (TCC) boundary snapping.

### Entity Types

| Entity | Description | Example |
|--------|-------------|---------|
| `PERSON` | Names, nicknames | พี่แจง, คุณสมชาย |
| `LOCATION` | Places, areas | บางนา, ลาดพร้าว |
| `CONTACT` | Phone, LINE, email | 088-888-8888, @line_id |
| `HARD_SKILL` | Job skills, tasks | ทำบัญชี, ขับรถ |
| `COMPENSATION` | Salary, pay | วันละ 500 บาท |
| `EMPLOYMENT_TERMS` | Job type, hours | งานประจำ, Full-time |
| `DEMOGRAPHIC` | Age, gender, nationality | คนไทย, อายุ 20-35 |

### Tagging Scheme

IOB2 format with 15 labels: `O` + 7 entity types × 2 (`B-` prefix for begin, `I-` for inside).

## Dataset Structure

### Splits

| Split | Examples |
|-------|----------|
| `train` | {n_train} |
| `validation` | {n_val} |
| `test` | {n_test} |

### Features

- `id` (`string`): Unique example identifier
- `tokens` (`list[string]`): Word-tokenized text (pythainlp, newmm engine)
- `ner_tags` (`list[ClassLabel]`): IOB2 integer labels per token

### Example

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0])
# {{'id': 'synthetic_0000', 'tokens': ['คนไทย', 'ที่', 'ทำ', ...], 'ner_tags': [13, 0, 0, ...]}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")

# Access label names
label_names = dataset["train"].features["ner_tags"].feature.names
# ['O', 'B-HARD_SKILL', 'I-HARD_SKILL', 'B-PERSON', ...]

# Decode tags for an example
example = dataset["train"][0]
tags = [label_names[t] for t in example["ner_tags"]]
for token, tag in zip(example["tokens"], tags):
    if tag != "O":
        print(f"  {{token}} → {{tag}}")
```

## Data Generation

- **Generator**: GPT-4o (OpenAI API)
- **Style**: Informal Thai job posts mimicking social media (Facebook groups, LINE)
- **Alignment**: Fuzzy string matching (rapidfuzz) with TCC boundary snapping (pythainlp)
- **Tokenization**: pythainlp word_tokenize (newmm engine), whitespace tokens excluded

## Associated Model

Fine-tuned WangchanBERTa model trained on this dataset:
[chayuto/thai-job-ner-wangchanberta](https://huggingface.co/chayuto/thai-job-ner-wangchanberta)

## License

CC-BY-4.0

## Citation

```bibtex
@misc{{thai-job-ner-dataset,
  title={{Thai Job Posting NER Dataset (Synthetic)}},
  author={{Chayut Orapinpatipat}},
  year={{2026}},
  url={{https://huggingface.co/datasets/{repo_id}}},
}}
```
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Build and publish HuggingFace NER dataset from synthetic data"
    )
    parser.add_argument(
        "--input", type=Path, nargs="+",
        default=[Path("data/raw/synthetic_0.json"), Path("data/raw/synthetic_template.json")],
        help="Path(s) to synthetic NER JSON file(s)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/hf_dataset"),
        help="Local output directory for review before pushing",
    )
    parser.add_argument("--threshold", type=float, default=85.0)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--push", action="store_true",
        help="Push to HuggingFace Hub after building",
    )
    parser.add_argument(
        "--repo-id", type=str, default="chayuto/thai-job-ner-dataset",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Push as private dataset",
    )
    args = parser.parse_args()

    # Step 1: Load and collate synthetic data
    logger.info("Loading synthetic data files...")
    from src.data.load_dataset import merge_sources
    posts = merge_sources(*args.input)
    if not posts:
        raise ValueError("No valid posts loaded from any input file")
    logger.info(f"Collated {len(posts)} synthetic posts")

    for path in collect_data_files(args.input):
        _, stats = load_and_validate(path)
        print_stats(stats, source=path.name)

    # Step 2: Convert to CoNLL-style records
    logger.info("Converting to CoNLL-style tokens + IOB2 tags...")
    records = build_records(posts, threshold=args.threshold)
    if not records:
        raise ValueError("No posts survived the conversion pipeline")

    # Step 3: Split and create DatasetDict
    logger.info("Splitting into train/val/test...")
    dataset_dict = records_to_dataset_dict(
        records,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    for split_name, ds in dataset_dict.items():
        logger.info(f"  {split_name}: {len(ds)} examples")

    # Step 4: Save locally for review
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    logger.info(f"Saved dataset to {output_dir}/")

    card = generate_dataset_card(dataset_dict, args.repo_id)
    card_path = output_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    logger.info(f"Saved dataset card to {card_path}")

    # Print a sample for verification
    print("\n--- Sample (first record) ---")
    example = dataset_dict["train"][0]
    label_names = dataset_dict["train"].features["ner_tags"].feature.names
    tags = [label_names[t] for t in example["ner_tags"]]
    for token, tag in zip(example["tokens"], tags):
        marker = f"  ← {tag}" if tag != "O" else ""
        print(f"  {token:20s} {tag:25s}{marker}")

    # Step 5: Push to Hub (optional)
    if args.push:
        logger.info(f"Pushing to HuggingFace Hub: {args.repo_id}...")
        dataset_dict.push_to_hub(
            args.repo_id,
            private=args.private,
            commit_message="Upload Thai Job NER synthetic dataset (IOB2, word-tokenized)",
        )
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        logger.info(f"Done! Dataset at: https://huggingface.co/datasets/{args.repo_id}")
    else:
        logger.info(
            f"Dataset saved locally at {output_dir}/. "
            f"Run with --push to upload to HuggingFace Hub."
        )


if __name__ == "__main__":
    main()
