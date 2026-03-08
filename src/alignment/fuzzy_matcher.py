"""Fuzzy string alignment for Thai text using rapidfuzz + pythainlp TCC snapping.

Thai text has no whitespace word boundaries, and entity substrings from silver
labels may contain OCR errors or informal spelling variants. This module finds
entity positions in raw text while respecting Thai Character Cluster (TCC)
boundaries — preventing Unicode grapheme cluster splitting that would break
downstream tokenization.

Algorithm:
1. Segment raw text into TCC units (pythainlp)
2. Find approximate substring match (rapidfuzz partial_ratio)
3. Snap matched boundaries to nearest TCC edges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pythainlp.tokenize import subword_tokenize
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    entity_text: str
    label: str
    char_start: int
    char_end: int
    score: float
    matched_text: str


def _build_tcc_boundaries(text: str) -> list[tuple[int, int]]:
    """Build a list of (start, end) char boundaries for each TCC cluster."""
    clusters = subword_tokenize(text, engine="tcc")
    boundaries: list[tuple[int, int]] = []
    pos = 0
    for cluster in clusters:
        end = pos + len(cluster)
        boundaries.append((pos, end))
        pos = end
    return boundaries


def _snap_to_tcc(
    start: int,
    end: int,
    boundaries: list[tuple[int, int]],
) -> tuple[int, int]:
    """Snap arbitrary char indices to the nearest TCC cluster edges.

    Always expands outward to avoid splitting grapheme clusters.
    """
    safe_start = start
    safe_end = end

    for b_start, b_end in boundaries:
        # Snap start: find the TCC cluster that contains our start index
        if b_start <= start < b_end:
            safe_start = b_start
        # Snap end: find the TCC cluster that contains our end index
        if b_start < end <= b_end:
            safe_end = b_end

    return safe_start, safe_end


def _find_best_window(
    text: str,
    query: str,
    tcc_boundaries: list[tuple[int, int]],
    threshold: float,
) -> tuple[int, int, float, str] | None:
    """Slide a window across TCC-aligned positions to find the best fuzzy match.

    Uses rapidfuzz.fuzz.ratio to score each candidate window against the query.
    Windows are built from contiguous TCC clusters to guarantee safe boundaries.
    """
    query_len = len(query)
    best_score = 0.0
    best_start = 0
    best_end = 0
    best_text = ""

    # Try windows of varying TCC cluster counts
    n_clusters = len(tcc_boundaries)
    for i in range(n_clusters):
        window_start = tcc_boundaries[i][0]

        for j in range(i + 1, min(i + 20, n_clusters + 1)):
            window_end = tcc_boundaries[j - 1][1]
            window_len = window_end - window_start

            # Skip windows that are too short or too long relative to query
            if window_len < query_len * 0.5 or window_len > query_len * 2.0:
                continue

            candidate = text[window_start:window_end]
            score = fuzz.ratio(query, candidate)

            if score > best_score:
                best_score = score
                best_start = window_start
                best_end = window_end
                best_text = candidate

    if best_score >= threshold:
        return best_start, best_end, best_score, best_text
    return None


def find_entity_boundaries(
    raw_text: str,
    entity_text: str,
    label: str,
    threshold: float = 85.0,
) -> AlignmentResult | None:
    """Find TCC-safe character boundaries for an entity in raw text.

    First attempts exact match (fast path). Falls back to fuzzy windowed
    search over TCC-aligned positions if exact match fails.

    Args:
        raw_text: The full post text.
        entity_text: The entity substring to locate.
        label: The NER label for this entity.
        threshold: Minimum fuzzy match score (0-100) to accept.

    Returns:
        AlignmentResult with TCC-snapped boundaries, or None if no match found.
    """
    # Fast path: exact substring match
    idx = raw_text.find(entity_text)
    if idx != -1:
        tcc_boundaries = _build_tcc_boundaries(raw_text)
        safe_start, safe_end = _snap_to_tcc(idx, idx + len(entity_text), tcc_boundaries)
        return AlignmentResult(
            entity_text=entity_text,
            label=label,
            char_start=safe_start,
            char_end=safe_end,
            score=100.0,
            matched_text=raw_text[safe_start:safe_end],
        )

    # Slow path: fuzzy matching over TCC windows
    tcc_boundaries = _build_tcc_boundaries(raw_text)
    if not tcc_boundaries:
        return None

    result = _find_best_window(raw_text, entity_text, tcc_boundaries, threshold)
    if result is None:
        logger.warning(
            f"No match for '{entity_text}' (label={label}) in text "
            f"(best score < {threshold})"
        )
        return None

    start, end, score, matched = result
    logger.debug(
        f"Fuzzy matched '{entity_text}' -> '{matched}' "
        f"(score={score:.1f}, span=[{start}:{end}])"
    )
    return AlignmentResult(
        entity_text=entity_text,
        label=label,
        char_start=start,
        char_end=end,
        score=score,
        matched_text=matched,
    )


def align_post_entities(
    raw_text: str,
    entities: list[dict[str, str]],
    threshold: float = 85.0,
) -> tuple[list[AlignmentResult], list[dict[str, str]]]:
    """Align all entities for a single post.

    Returns:
        Tuple of (aligned entities, unmatched entities).
    """
    aligned: list[AlignmentResult] = []
    unmatched: list[dict[str, str]] = []

    for ent in entities:
        result = find_entity_boundaries(
            raw_text=raw_text,
            entity_text=ent["text"],
            label=ent["label"],
            threshold=threshold,
        )
        if result:
            aligned.append(result)
        else:
            unmatched.append(ent)

    # Sort by position to handle overlaps
    aligned.sort(key=lambda r: r.char_start)

    # Remove overlapping spans — keep the one with higher score
    deduped: list[AlignmentResult] = []
    for result in aligned:
        if deduped and result.char_start < deduped[-1].char_end:
            if result.score > deduped[-1].score:
                deduped[-1] = result
        else:
            deduped.append(result)

    return deduped, unmatched
