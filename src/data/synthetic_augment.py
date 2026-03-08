"""Generate synthetic Thai job posts with NER annotations using OpenAI GPT-4o."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a Thai social media data generator. Your job is to create realistic, \
informal Thai job posts as they would appear on Facebook groups for job seekers.

Rules:
- Write in colloquial Thai as real people post online (mix of formal/informal, \
abbreviations, emojis, occasional typos)
- Vary the post styles: some are employers seeking workers, some are job seekers \
advertising themselves, some are short and some are detailed
- Include a realistic mix of Thai and English words (many Thai job posts mix languages)
- Posts should feel authentic — not like AI-generated template text
- Each post should contain between 2 and 7 entities from the allowed types

Allowed entity types:
- HARD_SKILL: Specific abilities or procedures (e.g., "ทำอาหาร", "CPR", "ขับรถ", "Python")
- PERSON: Names of people (e.g., "คุณสมชาย", "พี่แจน")
- LOCATION: Places (e.g., "สีลม", "ลาดพร้าว", "รพ.รามาฯ")
- COMPENSATION: Money amounts (e.g., "18,000 บาท/เดือน", "วันละ 800")
- EMPLOYMENT_TERMS: Job structure (e.g., "เต็มเวลา", "อยู่ประจำ", "กะดึก")
- CONTACT: Phone, Line, email (e.g., "081-234-5678", "Line: @job123")
- DEMOGRAPHIC: Age, gender, nationality (e.g., "อายุ 30-45", "หญิง", "สัญชาติไทย")\
"""

USER_PROMPT_TEMPLATE = """\
Generate {count} unique Thai job posts. For each post, return a JSON object with:
- "raw_text": the complete post text (realistic Thai social media style)
- "entities": array of objects with "text" (exact substring from raw_text) and "label"

CRITICAL: Every entity "text" value MUST be an exact substring that appears in "raw_text". \
Do not paraphrase or reformat — copy the exact characters.

Return ONLY a JSON array, no markdown formatting or explanation.\
"""


def generate_batch(
    client: OpenAI,
    count: int,
    model: str = "gpt-4o",
) -> list[dict]:
    """Generate a batch of synthetic posts via OpenAI API."""
    batch_size = min(count, 10)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(count=batch_size)},
        ],
        temperature=1.0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from OpenAI API")

    parsed = json.loads(content)

    # Handle both {"posts": [...]} and [...] response formats
    if isinstance(parsed, dict):
        for key in ("posts", "data", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        raise ValueError(f"Unexpected response structure: {list(parsed.keys())}")
    elif isinstance(parsed, list):
        return parsed
    else:
        raise ValueError(f"Expected list or dict, got {type(parsed).__name__}")


def validate_synthetic(posts: list[dict]) -> list[dict]:
    """Validate and fix synthetic data — drop entities not found in raw_text."""
    valid_posts: list[dict] = []

    for i, post in enumerate(posts):
        raw_text = post.get("raw_text", "")
        if not raw_text:
            logger.warning(f"Synthetic post {i}: missing raw_text, skipping")
            continue

        valid_entities = []
        for ent in post.get("entities", []):
            text = ent.get("text", "")
            label = ent.get("label", "")
            if text and label and text in raw_text:
                valid_entities.append({"text": text, "label": label})
            else:
                logger.debug(f"Synthetic post {i}: dropping entity '{text}' (not in raw_text)")

        if valid_entities:
            valid_posts.append({
                "id": f"synthetic_{i:04d}",
                "raw_text": raw_text,
                "entities": valid_entities,
            })

    return valid_posts


def generate_dataset(
    total_count: int,
    output_path: Path,
    model: str = "gpt-4o",
) -> list[dict]:
    """Generate a full synthetic dataset in batches."""
    client = OpenAI()
    all_posts: list[dict] = []
    generated = 0
    batch_num = 0

    # Load existing data if appending
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            all_posts = json.load(f)
        generated = len(all_posts)
        logger.info(f"Loaded {generated} existing posts from {output_path.name}")

    while generated < total_count:
        remaining = total_count - generated
        batch_size = min(remaining, 10)
        batch_num += 1

        logger.info(f"Generating batch {batch_num} ({batch_size} posts, {generated}/{total_count} done)...")

        try:
            raw_batch = generate_batch(client, count=batch_size, model=model)
            valid_batch = validate_synthetic(raw_batch)

            # Re-number IDs to avoid collisions
            for post in valid_batch:
                post["id"] = f"synthetic_{generated:04d}"
                generated += 1

            all_posts.extend(valid_batch)

            # Save after each batch for crash resilience
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_posts, f, ensure_ascii=False, indent=2)

        except Exception:
            logger.exception(f"Batch {batch_num} failed")
            time.sleep(2)
            continue

        # Rate limiting
        if generated < total_count:
            time.sleep(1)

    logger.info(f"Done. {len(all_posts)} total posts saved to {output_path}")
    return all_posts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic Thai NER training data")
    parser.add_argument("--count", type=int, default=400, help="Number of posts to generate")
    parser.add_argument("--output", type=Path, default=Path("data/raw/synthetic.json"))
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(total_count=args.count, output_path=args.output, model=args.model)


if __name__ == "__main__":
    main()
