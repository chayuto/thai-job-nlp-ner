# thai-job-nlp-ner

Efficient Named Entity Recognition (NER) for informal Thai job postings. Fine-tunes pretrained Thai language models to extract structured HR data from unstructured social media text — replacing expensive LLM API calls with fast local inference.

**Best F1: 0.956 (PhayaThaiBERT)** | **Inference: <100ms on MPS, <300ms on CPU** | **Trained in ~10 min on Apple Silicon**

[PhayaThaiBERT Model](https://huggingface.co/chayuto/thai-job-ner-phayathaibert) | [WangchanBERTa Model](https://huggingface.co/chayuto/thai-job-ner-wangchanberta) | [Dataset](https://huggingface.co/datasets/chayuto/thai-job-ner-dataset)

## Features

- **Domain-Specific Extraction:** Parses informal Thai job posts from Facebook groups, Line chats, and social media
- **7 Entity Types:** Skills, locations, salaries, contacts, employment terms, demographics, person names
- **Multi-Model:** PhayaThaiBERT (F1=0.956) and WangchanBERTa (F1=0.897) — config-driven model swapping
- **Fast Inference:** ~110-122M parameter models — runs locally on CPU or Apple Silicon MPS
- **Production Ready:** FastAPI microservice with Docker, or interactive Gradio demo

**Input:**
```
รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท ต้องทำ CPR ได้ โทร 081-234-5678
```

**Output:**
```json
{
  "entities": [
    {"text": "ดูแลผู้สูงอายุ", "label": "HARD_SKILL", "confidence": 0.98},
    {"text": "สีลม", "label": "LOCATION", "confidence": 0.97},
    {"text": "18,000 บาท", "label": "COMPENSATION", "confidence": 0.99},
    {"text": "081-234-5678", "label": "CONTACT", "confidence": 0.99}
  ]
}
```

## Model Performance

Trained on 1,253 Thai job posts (synthetic silver labels from GPT-4o, fuzzy-aligned to IOB2). Two model variants available:

### Model Comparison

| Metric | WangchanBERTa | PhayaThaiBERT | Delta |
|--------|---------------|---------------|-------|
| **Overall F1** | 0.897 | **0.956** | **+0.059** |
| Overall Precision | 0.850 | 0.939 | +0.089 |
| Overall Recall | 0.949 | 0.974 | +0.025 |

### Per-Entity F1 (PhayaThaiBERT — best model)

| Entity | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| CONTACT | 0.987 | 0.983 | 0.991 |
| PERSON | 0.979 | 0.972 | 0.986 |
| LOCATION | 0.966 | 0.950 | 0.983 |
| EMPLOYMENT_TERMS | 0.966 | 0.943 | 0.990 |
| COMPENSATION | 0.965 | 0.956 | 0.973 |
| HARD_SKILL | 0.946 | 0.919 | 0.974 |
| DEMOGRAPHIC | 0.915 | 0.897 | 0.935 |
| **Overall** | **0.956** | **0.939** | **0.974** |

## NER Entity Classes

| Tag | Description | Examples |
|-----|-------------|----------|
| `HARD_SKILL` | Abilities or procedures | ทำอาหาร, CPR, ดูแลผู้สูงอายุ, Python |
| `PERSON` | Names | คุณสมชาย, พี่แจน, ป้าแมว |
| `LOCATION` | Places | สีลม, ลาดพร้าว, รพ.รามาฯ |
| `COMPENSATION` | Pay amounts | 18,000 บาท/เดือน, วันละ 800 |
| `EMPLOYMENT_TERMS` | Job structure | เต็มเวลา, Part-time, กะดึก |
| `CONTACT` | Phone, Line, email | 081-234-5678, Line @care123 |
| `DEMOGRAPHIC` | Age, gender, nationality | อายุ 30-45, หญิง |

## Quick Start

### Setup

```bash
git clone https://github.com/chayuto/thai-job-nlp-ner.git
cd thai-job-nlp-ner

python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Run the API

```bash
# FastAPI server
uvicorn src.inference.app:app --port 8000

# Test it
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท"}'
```

### Docker

```bash
docker compose up        # API at localhost:8000
```

### Gradio Demo

```bash
python app_demo.py       # Interactive UI at localhost:7860
```

### Python API

```python
from src.inference.pipeline import NERPipeline

pipe = NERPipeline("results/final")
result = pipe.extract("รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท")

for entity in result.entities:
    print(f"{entity.label}: {entity.text} ({entity.confidence:.0%})")
# HARD_SKILL: ดูแลผู้สูงอายุ (98%)
# LOCATION: สีลม (97%)
# COMPENSATION: 18,000 บาท (99%)
```

## Pipeline

```
Phase 1: Data Engineering
  Raw Thai text + GPT-4o entity extractions
  → Fuzzy alignment (rapidfuzz + pythainlp TCC snapping)
  → Subword token mapping (offset_mapping, bypasses char_to_token bug)
  → IOB2-formatted HuggingFace Dataset

Phase 2: Training
  Config-driven model fine-tuning on Apple Silicon (MPS backend, FP32)
  Supports WangchanBERTa and PhayaThaiBERT via YAML config swap
  LR=3e-5, warmup=0.1, 15 epochs, gradient checkpointing for large-vocab models

Phase 3: Evaluation
  Strict exact-match F1 via seqeval + per-entity breakdown

Phase 4: Deployment
  FastAPI microservice + Gradio demo, containerized with Docker
```

## Key Technical Decisions

- **PhayaThaiBERT > WangchanBERTa** — XLM-R-derived vocab (248K) handles Thai-English code-switching better; +0.059 F1
- **Frozen embeddings for large-vocab models** — PhayaThaiBERT's 248K embedding matrix causes MPS OOM; freezing saves ~750MB with no quality loss
- **PyTorch MPS over MLX** — HuggingFace Trainer integration, mature BERT training kernels
- **FP32 only** — MPS fp16 causes gradient underflow / NaN loss
- **rapidfuzz + pythainlp** — Fast fuzzy matching with Thai Character Cluster boundary safety
- **offset_mapping** — Tokenizer-agnostic subword-to-character alignment (works with any HF tokenizer)
- **LR=3e-5 with warmup** — 2e-5 undertrained; higher LR with warmup=0.1 gave +4.3pp F1

## Project Structure

```
├── configs/
│   ├── config.yaml                # WangchanBERTa training config
│   └── config_phayathaibert.yaml  # PhayaThaiBERT training config
├── src/
│   ├── data/
│   │   ├── load_dataset.py        # Load & validate NER export JSON
│   │   └── synthetic_augment.py   # GPT-4o synthetic data generation
│   ├── alignment/
│   │   ├── fuzzy_matcher.py       # rapidfuzz + TCC boundary snapping
│   │   ├── token_mapper.py        # Subword token ↔ IOB2 label mapping
│   │   └── iob2_formatter.py      # End-to-end pipeline orchestrator
│   ├── training/
│   │   ├── train_ner.py           # Fine-tuning script (MPS/CUDA/CPU)
│   │   └── metrics.py            # seqeval compute_metrics for Trainer
│   ├── evaluation/
│   │   └── per_entity_report.py   # Per-entity F1 + confusion matrix
│   └── inference/
│       ├── pipeline.py            # NERPipeline: load → predict → decode
│       └── app.py                 # FastAPI server
├── app_demo.py                    # Gradio interactive demo
├── Dockerfile                     # CPU deployment container
├── docker-compose.yml             # Local dev with health check
├── model_card.md                  # HuggingFace model card (WangchanBERTa)
├── model_card_phayathaibert.md    # HuggingFace model card (PhayaThaiBERT)
├── scripts/
│   ├── test_api.py                # API integration test
│   ├── upload_to_hub.py           # Push model to HuggingFace
│   ├── build_hf_dataset.py       # Build & publish dataset to HuggingFace
│   └── compare_models.py         # Side-by-side model comparison
├── notebooks/
│   └── 01_data_inspection.ipynb   # Visual alignment verification
├── docs/
│   ├── ELI5/                      # Plain-language explanations
│   ├── research/                  # Apple Silicon & Thai NLP research
│   └── study_guide/               # NLP/NER learning materials
└── data/
    ├── raw/                       # Input data (gitignored)
    └── processed/                 # IOB2 datasets (gitignored)
```

## Tech Stack

- **Models:** `clicknext/phayathaibert` (best, ~122M) / `airesearch/wangchanberta-base-att-spm-uncased` (110M)
- **Framework:** PyTorch + HuggingFace Transformers
- **Hardware:** Apple Silicon (MPS backend) / CPU
- **Thai NLP:** pythainlp (TCC tokenization), rapidfuzz (fuzzy alignment)
- **Evaluation:** seqeval (strict exact-match F1)
- **Serving:** FastAPI + Gradio
- **Deployment:** Docker

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
