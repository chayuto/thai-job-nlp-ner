"""Map character-level entity boundaries to subword token IOB2 labels.

WangchanBERTa's SentencePiece tokenizer replaces spaces with a <_> token
before subword segmentation. This breaks the standard char_to_token() method
because internal character offsets shift. We bypass this entirely by using
the tokenizer's offset_mapping — raw character offset tuples computed before
the <_> replacement — and intersecting them with our TCC-snapped entity spans.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.alignment.fuzzy_matcher import AlignmentResult

# Label index used by PyTorch CrossEntropyLoss to ignore special tokens
IGNORE_INDEX = -100

DEFAULT_CHECKPOINT = "airesearch/wangchanberta-base-att-spm-uncased"


@dataclass
class TokenizedPost:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    tokens: list[str]
    offset_mapping: list[tuple[int, int]]


def get_tokenizer(checkpoint: str = DEFAULT_CHECKPOINT) -> PreTrainedTokenizerBase:
    """Load the WangchanBERTa tokenizer."""
    return AutoTokenizer.from_pretrained(checkpoint)


def build_label_map(label_list: list[str]) -> dict[str, int]:
    """Build label string -> integer ID mapping."""
    return {label: idx for idx, label in enumerate(label_list)}


# Default label list matching configs/config.yaml
DEFAULT_LABELS = [
    "O",
    "B-HARD_SKILL", "I-HARD_SKILL",
    "B-PERSON", "I-PERSON",
    "B-LOCATION", "I-LOCATION",
    "B-COMPENSATION", "I-COMPENSATION",
    "B-EMPLOYMENT_TERMS", "I-EMPLOYMENT_TERMS",
    "B-CONTACT", "I-CONTACT",
    "B-DEMOGRAPHIC", "I-DEMOGRAPHIC",
]

DEFAULT_LABEL_MAP = build_label_map(DEFAULT_LABELS)


def align_tokens_to_iob2(
    tokenizer: PreTrainedTokenizerBase,
    raw_text: str,
    aligned_entities: list[AlignmentResult],
    label_map: dict[str, int] | None = None,
    max_length: int = 512,
) -> TokenizedPost:
    """Map TCC-snapped character boundaries to subword token IOB2 labels.

    Uses offset_mapping from the tokenizer to directly intersect character
    spans with subword positions, bypassing char_to_token() bugs caused
    by WangchanBERTa's <_> space token replacement.

    Args:
        tokenizer: The WangchanBERTa tokenizer instance.
        raw_text: Original post text.
        aligned_entities: List of AlignmentResults with char_start/char_end.
        label_map: Mapping from label strings to integer IDs.
        max_length: Maximum sequence length for truncation.

    Returns:
        TokenizedPost with input_ids, attention_mask, and integer labels.
    """
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    tokenized = tokenizer(
        raw_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    offsets: list[tuple[int, int]] = tokenized["offset_mapping"]
    input_ids: list[int] = tokenized["input_ids"]
    attention_mask: list[int] = tokenized["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Initialize all labels to IGNORE_INDEX (special tokens) or O
    labels = []
    for off_start, off_end in offsets:
        if off_start == 0 and off_end == 0:
            # Special tokens: [CLS], [SEP], padding
            labels.append(IGNORE_INDEX)
        else:
            labels.append(label_map["O"])

    # Assign entity labels by intersecting offset_mapping with char spans
    for entity in aligned_entities:
        entity_started = False
        b_label = f"B-{entity.label}"
        i_label = f"I-{entity.label}"

        if b_label not in label_map:
            continue

        for idx, (off_start, off_end) in enumerate(offsets):
            # Skip special tokens
            if off_start == 0 and off_end == 0:
                continue

            # Check if this subword token falls within the entity span
            if off_start >= entity.char_start and off_end <= entity.char_end:
                if not entity_started:
                    labels[idx] = label_map[b_label]
                    entity_started = True
                else:
                    labels[idx] = label_map[i_label]
            elif entity_started:
                # We've moved past the entity
                break

    return TokenizedPost(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        tokens=tokens,
        offset_mapping=offsets,
    )


def verify_iob2_consistency(labels: list[int], label_list: list[str]) -> list[str]:
    """Check IOB2 tag sequence validity and return any warnings.

    Rules:
    - I-X can only follow B-X or I-X (same entity type)
    - Every entity must start with B-
    """
    warnings: list[str] = []
    prev_label = "O"

    for idx, label_id in enumerate(labels):
        if label_id == IGNORE_INDEX:
            prev_label = "O"
            continue

        label_str = label_list[label_id]

        if label_str.startswith("I-"):
            entity_type = label_str[2:]
            expected_prev = {f"B-{entity_type}", f"I-{entity_type}"}
            if prev_label not in expected_prev:
                warnings.append(
                    f"Token {idx}: {label_str} follows {prev_label} "
                    f"(expected {expected_prev})"
                )

        prev_label = label_str

    return warnings
