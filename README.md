# thai-job-nlp-ner

Fine-tuning [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) for Named Entity Recognition on informal Thai job posts.

## What This Does

Extracts structured HR data from unstructured Thai social media text — the kind of posts you'd find in Facebook job groups. Instead of calling an LLM API for every post, we fine-tune a small (110M param) local model that runs fast and costs nothing at inference.

**Input:**
```
รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท ต้องทำ CPR ได้ โทร 081-234-5678
```

**Output:**
```json
{
  "HARD_SKILL": ["ดูแลผู้สูงอายุ", "CPR"],
  "LOCATION": ["สีลม"],
  "COMPENSATION": ["18,000 บาท"],
  "CONTACT": ["081-234-5678"]
}
```

## NER Entity Classes

| Tag | Description | Examples |
|-----|-------------|----------|
| `HARD_SKILL` | Abilities or procedures | ทำอาหาร, CPR, Python |
| `PERSON` | Names | คุณสมชาย, พี่แจน |
| `LOCATION` | Places | สีลม, ลาดพร้าว |
| `COMPENSATION` | Pay amounts | 18,000 บาท/เดือน |
| `EMPLOYMENT_TERMS` | Job structure | เต็มเวลา, กะดึก |
| `CONTACT` | Phone, Line, email | 081-234-5678 |
| `DEMOGRAPHIC` | Age, gender, nationality | อายุ 30-45, หญิง |

## Pipeline

```
Phase 1: Data Engineering
  Raw Thai text + entity metadata
  → Fuzzy alignment (rapidfuzz + pythainlp TCC snapping)
  → Subword token mapping (offset_mapping, bypasses char_to_token bug)
  → IOB2-formatted HuggingFace Dataset

Phase 2: Training
  WangchanBERTa fine-tuning on Apple Silicon (MPS backend, FP32)

Phase 3: Evaluation
  Strict exact-match F1 via seqeval

Phase 4: Deployment
  FastAPI microservice in Docker
```

## Key Technical Decisions

- **PyTorch MPS over MLX** — HuggingFace Trainer integration, mature BERT training kernels
- **FP32 only** — MPS fp16 causes gradient underflow / NaN loss
- **rapidfuzz + pythainlp** — Fast fuzzy matching with Thai Character Cluster boundary safety
- **offset_mapping** — Bypasses WangchanBERTa's `<_>` space token misalignment in `char_to_token()`

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/thai-job-nlp-ner.git
cd thai-job-nlp-ner

# Create environment
python -m venv venv
source venv/bin/activate

# Install
pip install -e ".[dev]"

# Verify MPS
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Usage

```bash
# Load and validate data
python -m src.data.load_dataset --input data/raw/ner_export.json

# Generate synthetic training data (requires OPENAI_API_KEY)
python -m src.data.synthetic_augment --count 400

# Run full alignment pipeline → HuggingFace Dataset
python -m src.alignment.iob2_formatter \
  --input data/raw/ner_export.json data/raw/synthetic.json \
  --output data/processed/
```

## Project Structure

```
├── configs/config.yaml          # Hyperparameters & pipeline settings
├── src/
│   ├── data/
│   │   ├── load_dataset.py      # Load & validate NER export JSON
│   │   └── synthetic_augment.py # GPT-4o synthetic data generation
│   ├── alignment/
│   │   ├── fuzzy_matcher.py     # rapidfuzz + TCC boundary snapping
│   │   ├── token_mapper.py      # Subword token ↔ IOB2 label mapping
│   │   └── iob2_formatter.py    # End-to-end pipeline orchestrator
│   ├── training/                # (Sprint 2)
│   ├── evaluation/              # (Sprint 2)
│   └── inference/               # (Sprint 3)
├── notebooks/
│   └── 01_data_inspection.ipynb # Visual alignment verification
├── docs/
│   ├── research/                # Apple Silicon & Thai NLP research
│   └── study_guide/             # NLP/NER learning materials
└── data/
    ├── raw/                     # Input data (gitignored)
    └── processed/               # IOB2 datasets (gitignored)
```

## Tech Stack

- **Model:** `airesearch/wangchanberta-base-att-spm-uncased` (110M params)
- **Framework:** PyTorch + HuggingFace Transformers
- **Hardware:** Apple Silicon (MPS backend)
- **Thai NLP:** pythainlp (TCC tokenization), rapidfuzz (fuzzy alignment)
- **Evaluation:** seqeval (strict exact-match F1)
- **Deployment:** FastAPI + Docker

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
