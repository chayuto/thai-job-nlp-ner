# CLAUDE.md — Project Rules

## DATA PROTECTION — HARD RULES (NON-NEGOTIABLE)

### Real-World Data is SECRET
- `data/raw/ner_export.json` contains real-world production data. It is **proprietary and confidential**.
- NEVER display, print, log, or echo contents of real-world data files.
- NEVER include real-world data samples in commits, PRs, dataset cards, READMEs, notebooks, or any public-facing output.
- NEVER push real-world data to HuggingFace Hub, GitHub, or any remote.
- NEVER reference real database schemas, table names, or production endpoints.
- When asked to show data examples, ONLY use synthetic examples or create new fictitious Thai text.

### What Counts as Real-World Data
- `data/raw/ner_export.json` — production database export
- `data/raw/prod-secret/` — proprietary raw data (real posts, annotations, scrapes)
- `data/processed/prod-secret/` — proprietary processed datasets
- Any `**/prod-secret/` directory anywhere in the repo
- Any file with real user-generated Thai job posts (scraped, exported, or manually collected)
- Database connection strings, API keys, credentials
- Any data that came from a real person or real job posting

### What is Safe to Share Publicly
- `data/raw/synthetic_*.json` and `synthetic.jsonl` — LLM-generated, no PII
- `data/hf_dataset/` — built from synthetic data only
- Model weights (`.safetensors`) — trained output, no raw data recoverable
- Code, scripts, notebooks (without real data outputs embedded)

### Before Any Public Action (push, upload, publish)
1. Verify NO real-world data is staged: check `git diff --cached` for any `ner_export` content
2. Verify dataset builds use ONLY synthetic sources
3. Verify notebooks don't have real data in cell outputs
4. When in doubt, ASK the user before proceeding

## Project Commands

```bash
# Activate environment
source .venv/bin/activate

# Train model
python3 scripts/train.py

# Run inference API
python3 -m uvicorn src.api.main:app --reload

# Build HF dataset
python3 scripts/build_hf_dataset.py

# Run tests
python3 -m pytest tests/
```

## Key Architecture
- Model: `wangchanberta-base-att-spm-uncased` (110M params)
- Training: PyTorch MPS backend (Apple Silicon), FP32 only
- 7 entity types: PERSON, LOCATION, CONTACT, HARD_SKILL, COMPENSATION, EMPLOYMENT_TERMS, DEMOGRAPHIC
- Labels defined in `src/alignment/token_mapper.py` (DEFAULT_LABELS, IGNORE_INDEX)
