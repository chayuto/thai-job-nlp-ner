---
language:
  - th
license: mit
library_name: transformers
tags:
  - token-classification
  - ner
  - thai
  - phayathaibert
  - job-posting
  - apple-silicon
datasets:
  - chayuto/thai-job-ner-dataset
metrics:
  - f1
  - precision
  - recall
pipeline_tag: token-classification
model-index:
  - name: thai-job-ner-phayathaibert
    results:
      - task:
          type: token-classification
          name: Named Entity Recognition
        metrics:
          - name: F1
            type: f1
            value: 0.956
          - name: Precision
            type: precision
            value: 0.939
          - name: Recall
            type: recall
            value: 0.974
---

# Thai Job NER — Fine-tuned PhayaThaiBERT

Named Entity Recognition model for extracting structured HR data from informal Thai job postings (e.g., Facebook groups, Line chats). Fine-tuned from [PhayaThaiBERT](https://huggingface.co/clicknext/phayathaibert) (~122M params).

## Model Description

This model extracts 7 entity types from Thai job-related text:

| Entity | Description | Example |
|--------|-------------|---------|
| `HARD_SKILL` | Skills or procedures | ดูแลผู้สูงอายุ, CPR, Python |
| `PERSON` | Names | คุณสมชาย, พี่แจน |
| `LOCATION` | Places | สีลม, ลาดพร้าว, บางนา |
| `COMPENSATION` | Pay amounts | 18,000 บาท/เดือน |
| `EMPLOYMENT_TERMS` | Job structure | part-time, กะกลางวัน |
| `CONTACT` | Phone, Line, email | 081-234-5678, @care123 |
| `DEMOGRAPHIC` | Age, gender | อายุ 25-40, หญิง |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "chayuto/thai-job-ner-phayathaibert"
ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

text = "รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท โทร 081-234-5678"
results = ner(text)
for entity in results:
    print(f"{entity['entity_group']}: {entity['word']} ({entity['score']:.2%})")
```

## Training

- **Base model:** `clicknext/phayathaibert` (CamemBERT architecture, ~122M params, XLM-R-derived vocabulary)
- **Training data:** 1,253 Thai job posts (synthetic silver labels from GPT-4o, fuzzy-aligned to IOB2) — [Dataset on HuggingFace](https://huggingface.co/datasets/chayuto/thai-job-ner-dataset)
- **Hardware:** Apple Silicon MPS backend, FP32
- **Hyperparameters:** LR=3e-5, warmup=0.1, batch=2, grad_accum=8, 15 epochs, gradient checkpointing, frozen embeddings
- **Training time:** ~10 min

### Data Pipeline

Raw Thai text + GPT-4o entity extractions → fuzzy alignment with rapidfuzz + pythainlp TCC boundary snapping → subword token mapping via offset_mapping → IOB2-formatted HuggingFace Dataset.

## Evaluation

### Overall (Test Set, 126 examples)

| Metric | Score |
|--------|-------|
| **F1** | **0.956** |
| Precision | 0.939 |
| Recall | 0.974 |

### Per-Entity F1

| Entity | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| CONTACT | 0.987 | 0.983 | 0.991 |
| PERSON | 0.979 | 0.972 | 0.986 |
| LOCATION | 0.966 | 0.950 | 0.983 |
| EMPLOYMENT_TERMS | 0.966 | 0.943 | 0.990 |
| COMPENSATION | 0.965 | 0.956 | 0.973 |
| HARD_SKILL | 0.946 | 0.919 | 0.974 |
| DEMOGRAPHIC | 0.915 | 0.897 | 0.935 |

### Comparison vs WangchanBERTa

| Entity | WangchanBERTa | PhayaThaiBERT | Delta |
|--------|---------------|---------------|-------|
| **Overall F1** | 0.897 | **0.956** | **+0.059** |
| COMPENSATION | 0.764 | **0.965** | **+0.200** |
| PERSON | 0.907 | **0.979** | **+0.072** |
| HARD_SKILL | 0.903 | **0.946** | **+0.043** |
| EMPLOYMENT_TERMS | 0.926 | **0.966** | +0.040 |
| DEMOGRAPHIC | 0.875 | **0.915** | +0.041 |
| CONTACT | 0.962 | **0.987** | +0.025 |
| LOCATION | 0.959 | **0.966** | +0.008 |

PhayaThaiBERT improves on every entity type, with the most dramatic gain on COMPENSATION (+0.200 F1).

## Links

- **Model:** [chayuto/thai-job-ner-phayathaibert](https://huggingface.co/chayuto/thai-job-ner-phayathaibert)
- **WangchanBERTa variant:** [chayuto/thai-job-ner-wangchanberta](https://huggingface.co/chayuto/thai-job-ner-wangchanberta)
- **Dataset:** [chayuto/thai-job-ner-dataset](https://huggingface.co/datasets/chayuto/thai-job-ner-dataset)
- **Source Code:** [github.com/chayuto/thai-job-nlp-ner](https://github.com/chayuto/thai-job-nlp-ner)

## Limitations

- Trained on synthetic data — may underperform on real-world posts with heavy emoji usage, OCR errors, or extreme colloquialism
- Embeddings were frozen during training (MPS memory constraint) — unfreezing on a larger GPU may yield further gains
- 512 token max sequence length
- Larger model file size due to 248K vocabulary (vs WangchanBERTa's 25K)

## Technical Notes

- **FP16 is broken on MPS** — always use FP32 for Apple Silicon training
- PhayaThaiBERT's 248K vocab (XLM-R-derived) requires frozen embeddings + gradient checkpointing to fit on 18GB MPS
- Uses `offset_mapping` for tokenizer-agnostic subword-to-character alignment
- Thai Character Cluster (TCC) boundary snapping prevents Unicode grapheme splitting during alignment

## License

MIT
