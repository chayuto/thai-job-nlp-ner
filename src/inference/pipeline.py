"""NER inference pipeline: load model, tokenize, predict, decode IOB2 to entities.

Converts raw Thai text into structured entity spans with character offsets
and confidence scores. Handles the full lifecycle from text input to
grouped entity output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.alignment.token_mapper import DEFAULT_LABELS, IGNORE_INDEX

logger = logging.getLogger(__name__)


@dataclass
class EntitySpan:
    text: str
    label: str
    start: int
    end: int
    confidence: float


@dataclass
class ExtractionResult:
    entities: list[EntitySpan] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "start": e.start,
                    "end": e.end,
                    "confidence": round(e.confidence, 4),
                }
                for e in self.entities
            ],
        }

    def grouped(self) -> dict[str, list[str]]:
        """Group entity texts by label type."""
        groups: dict[str, list[str]] = {}
        for e in self.entities:
            groups.setdefault(e.label, []).append(e.text)
        return groups


class NERPipeline:
    """End-to-end NER inference pipeline for Thai text.

    Loads a fine-tuned WangchanBERTa model and provides a simple
    `extract(text)` interface that returns structured entities.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        # Resolve device
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load model and tokenizer
        logger.info(f"Loading model from {self.model_dir} on {self.device}...")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(self.model_dir)
        )
        self.model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            str(self.model_dir)
        )
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {i: label for i, label in enumerate(DEFAULT_LABELS)}
        logger.info(f"Model loaded. {len(DEFAULT_LABELS)} labels, device={self.device}")

    def extract(self, text: str) -> ExtractionResult:
        """Extract named entities from Thai text.

        Args:
            text: Raw Thai text (job posting, CV excerpt, etc.)

        Returns:
            ExtractionResult with entity spans, offsets, and confidences.
        """
        if not text or not text.strip():
            return ExtractionResult(raw_text=text)

        # Tokenize with offset mapping for char alignment
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()

        # Move to device and run inference
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        probabilities = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probabilities, dim=-1).cpu().tolist()
        confidences = probabilities.max(dim=-1).values.cpu().tolist()

        # Decode IOB2 predictions to entity spans
        entities = self._decode_iob2(
            pred_ids=pred_ids,
            confidences=confidences,
            offset_mapping=offset_mapping,
            raw_text=text,
        )

        return ExtractionResult(entities=entities, raw_text=text)

    def _decode_iob2(
        self,
        pred_ids: list[int],
        confidences: list[float],
        offset_mapping: list[list[int]],
        raw_text: str,
    ) -> list[EntitySpan]:
        """Convert IOB2 token predictions back to character-level entity spans.

        Merges consecutive B-X / I-X tokens into single entity spans,
        computing character offsets from the tokenizer's offset_mapping
        and averaging confidence scores across constituent tokens.
        """
        entities: list[EntitySpan] = []
        current_entity: dict | None = None

        for idx, (pred_id, conf, (char_start, char_end)) in enumerate(
            zip(pred_ids, confidences, offset_mapping)
        ):
            # Skip special tokens
            if char_start == 0 and char_end == 0:
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, raw_text))
                    current_entity = None
                continue

            label = self.id2label.get(pred_id, "O")

            if label.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, raw_text))

                entity_type = label[2:]
                current_entity = {
                    "label": entity_type,
                    "start": char_start,
                    "end": char_end,
                    "confidences": [conf],
                }

            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    # Continue current entity
                    current_entity["end"] = char_end
                    current_entity["confidences"].append(conf)
                else:
                    # Type mismatch — close current, start new
                    entities.append(self._finalize_entity(current_entity, raw_text))
                    current_entity = {
                        "label": entity_type,
                        "start": char_start,
                        "end": char_end,
                        "confidences": [conf],
                    }

            elif label.startswith("I-") and not current_entity:
                # Orphan I- tag — treat as B-
                entity_type = label[2:]
                current_entity = {
                    "label": entity_type,
                    "start": char_start,
                    "end": char_end,
                    "confidences": [conf],
                }

            else:
                # O label
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, raw_text))
                    current_entity = None

        # Close any remaining entity
        if current_entity:
            entities.append(self._finalize_entity(current_entity, raw_text))

        return entities

    @staticmethod
    def _finalize_entity(entity: dict, raw_text: str) -> EntitySpan:
        """Convert accumulated entity dict to EntitySpan."""
        start = entity["start"]
        end = entity["end"]
        avg_confidence = sum(entity["confidences"]) / len(entity["confidences"])

        return EntitySpan(
            text=raw_text[start:end],
            label=entity["label"],
            start=start,
            end=end,
            confidence=avg_confidence,
        )
