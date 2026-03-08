# thai-job-nlp-ner

Efficient Named Entity Recognition (NER) for informal Thai job postings. Fine-tunes [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) (110M params) to extract structured HR data from unstructured social media text — replacing expensive LLM API calls with fast local inference.

**Test F1: 0.828** | **Inference: <100ms on MPS, <300ms on CPU** | **Trained in ~4 min on Apple Silicon**

## Features

- **Domain-Specific Extraction:** Parses informal Thai job posts from Facebook groups, Line chats, and social media
- **7 Entity Types:** Skills, locations, salaries, contacts, employment terms, demographics, person names
- **Fast Inference:** 110M parameter model — runs locally on CPU or Apple Silicon MPS
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

Trained on 758 Thai job posts (synthetic silver labels from GPT-4o, fuzzy-aligned to IOB2).

| Entity | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| CONTACT | 0.957 | 0.928 | 0.987 |
| PERSON | 0.892 | 0.892 | 0.892 |
| LOCATION | 0.861 | 0.816 | 0.912 |
| EMPLOYMENT_TERMS | 0.850 | 0.915 | 0.793 |
| COMPENSATION | 0.819 | 0.782 | 0.859 |
| DEMOGRAPHIC | 0.776 | 0.760 | 0.792 |
| HARD_SKILL | 0.761 | 0.697 | 0.838 |
| **Overall** | **0.828** | **0.799** | **0.859** |

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
  WangchanBERTa fine-tuning on Apple Silicon (MPS backend, FP32)
  LR=3e-5, warmup=0.1, batch=8, grad_accum=2, 15 epochs

Phase 3: Evaluation
  Strict exact-match F1 via seqeval + per-entity breakdown

Phase 4: Deployment
  FastAPI microservice + Gradio demo, containerized with Docker
```

## Key Technical Decisions

- **PyTorch MPS over MLX** — HuggingFace Trainer integration, mature BERT training kernels
- **FP32 only** — MPS fp16 causes gradient underflow / NaN loss
- **rapidfuzz + pythainlp** — Fast fuzzy matching with Thai Character Cluster boundary safety
- **offset_mapping** — Bypasses WangchanBERTa's `<_>` space token misalignment in `char_to_token()`
- **LR=3e-5 with warmup** — 2e-5 undertrained; higher LR with warmup=0.1 gave +4.3pp F1

## Project Structure

```
├── configs/config.yaml            # Hyperparameters & pipeline settings
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
├── model_card.md                  # HuggingFace Hub model card
├── scripts/
│   ├── test_api.py                # API integration test
│   └── upload_to_hub.py           # Push model to HuggingFace
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

- **Model:** `airesearch/wangchanberta-base-att-spm-uncased` (110M params)
- **Framework:** PyTorch + HuggingFace Transformers
- **Hardware:** Apple Silicon (MPS backend) / CPU
- **Thai NLP:** pythainlp (TCC tokenization), rapidfuzz (fuzzy alignment)
- **Evaluation:** seqeval (strict exact-match F1)
- **Serving:** FastAPI + Gradio
- **Deployment:** Docker

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
