"""Test script for the Thai NER API.

Sends sample Thai job posts to the local FastAPI server and prints results.

Usage:
    # Start the server first:
    uvicorn src.inference.app:app --host 0.0.0.0 --port 8000

    # Then run this script:
    python scripts/test_api.py
    python scripts/test_api.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

SAMPLE_POSTS = [
    "รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท ต้องทำ CPR ได้ โทร 081-234-5678",
    "ต้องการพี่เลี้ยงเด็ก อายุ 25-40 ปี หญิง ทำอาหารได้ อยู่ประจำ ลาดพร้าว เงินเดือน 15,000 ติดต่อคุณแจน Line @care123",
    "หาคนขับรถส่งของ มีใบขับขี่ พื้นที่บางนา-ศรีนครินทร์ รายได้ 20,000-25,000 บาท/เดือน กะกลางวัน สนใจ โทร 02-345-6789",
    "รับสมัครแม่บ้าน part-time ทำความสะอาด ซักรีด 3 วัน/สัปดาห์ ย่านอารีย์ 500 บาท/วัน ติดต่อพี่นก 089-111-2222",
]


def test_extract(base_url: str, text: str) -> dict | None:
    """Send a POST /extract request and return the response."""
    url = f"{base_url}/extract"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        print(f"  ERROR: {e}")
        return None


def test_health(base_url: str) -> dict | None:
    """Send a GET /health request."""
    url = f"{base_url}/health"
    try:
        with urlopen(url) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        print(f"  ERROR: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Thai NER API")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Base URL of the NER API server",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Health check
    print(f"Testing API at {base_url}\n")
    print("=" * 60)
    print("Health Check")
    print("=" * 60)
    health = test_health(base_url)
    if health is None:
        print("\nServer not reachable. Start it with:")
        print("  uvicorn src.inference.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    print(json.dumps(health, indent=2, ensure_ascii=False))

    # Test extraction
    print(f"\n{'=' * 60}")
    print(f"Testing {len(SAMPLE_POSTS)} sample posts")
    print("=" * 60)

    for i, text in enumerate(SAMPLE_POSTS, 1):
        print(f"\n--- Post {i} ---")
        print(f"Input: {text[:80]}{'...' if len(text) > 80 else ''}")

        result = test_extract(base_url, text)
        if result is None:
            continue

        print(f"Time:  {result['processing_time_ms']:.1f}ms")

        if result["entities"]:
            print("Entities:")
            for ent in result["entities"]:
                print(
                    f"  [{ent['label']:<18}] "
                    f"\"{ent['text']}\" "
                    f"(conf={ent['confidence']:.3f}, "
                    f"chars={ent['start']}-{ent['end']})"
                )
        else:
            print("  No entities found.")

        if result["grouped"]:
            print("Grouped:")
            for label, texts in result["grouped"].items():
                print(f"  {label}: {texts}")

    # Edge cases
    print(f"\n{'=' * 60}")
    print("Edge Cases")
    print("=" * 60)

    edge_cases = [
        ("Empty-ish text", "   "),
        ("Short text", "สวัสดี"),
        ("English text", "Looking for a nurse in Bangkok"),
    ]

    for name, text in edge_cases:
        print(f"\n--- {name} ---")
        print(f"Input: \"{text}\"")
        result = test_extract(base_url, text)
        if result:
            n = len(result["entities"])
            print(f"Result: {n} entities, {result['processing_time_ms']:.1f}ms")


if __name__ == "__main__":
    main()
