"""Upload fine-tuned NER model to HuggingFace Hub.

Usage:
    # Login first:
    huggingface-cli login

    # Upload:
    python scripts/upload_to_hub.py
    python scripts/upload_to_hub.py --repo-id your-username/your-model-name
    python scripts/upload_to_hub.py --private  # upload as private repo
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def upload_model(
    model_dir: Path,
    repo_id: str,
    model_card_path: Path,
    private: bool = False,
) -> str:
    """Upload model artifacts and model card to HuggingFace Hub."""
    api = HfApi()

    # Create repo (no-op if exists)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    # Copy model card into model dir as README.md
    readme_path = model_dir / "README.md"
    shutil.copy2(model_card_path, readme_path)

    # Upload the entire directory
    print(f"Uploading {model_dir} to {repo_id}...")
    url = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        commit_message="Upload fine-tuned Thai Job NER model (v2, F1=0.828)",
    )

    # Clean up copied README
    readme_path.unlink(missing_ok=True)

    print(f"Done! Model available at: https://huggingface.co/{repo_id}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload NER model to HuggingFace Hub")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("results/final"),
        help="Path to model directory",
    )
    parser.add_argument(
        "--repo-id", type=str, default="chayuto/thai-job-ner-wangchanberta",
        help="HuggingFace repo ID (username/model-name)",
    )
    parser.add_argument(
        "--model-card", type=Path, default=Path("model_card.md"),
        help="Path to model card markdown",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Upload as private repo",
    )
    args = parser.parse_args()

    upload_model(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        model_card_path=args.model_card,
        private=args.private,
    )


if __name__ == "__main__":
    main()
