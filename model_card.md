---
language:
  - th
license: mit
library_name: transformers
tags:
  - token-classification
  - ner
  - thai
  - wangchanberta
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
  - name: thai-job-ner-wangchanberta
    results:
      - task:
          type: token-classification
          name: Named Entity Recognition
        metrics:
          - name: F1
            type: f1
            value: 0.897
          - name: Precision
            type: precision
            value: 0.850
          - name: Recall
            type: recall
            value: 0.859
---

# Thai Job NER — Fine-tuned WangchanBERTa

Named Entity Recognition model for extracting structured HR data from informal Thai job postings (e.g., Facebook groups, Line chats). Fine-tuned from [wangchanberta-base-att-spm-uncased](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) (110M params).

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

model_name = "chayuto/thai-job-ner-wangchanberta"
ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

text = "รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท โทร 081-234-5678"
results = ner(text)
for entity in results:
    print(f"{entity['entity_group']}: {entity['word']} ({entity['score']:.2%})")
```

## Training

- **Base model:** `airesearch/wangchanberta-base-att-spm-uncased` (CamemBERT architecture, 110M params)
- **Training data:** 1,253 Thai job posts (synthetic silver labels from GPT-4o, fuzzy-aligned to IOB2) — [Dataset on HuggingFace](https://huggingface.co/datasets/chayuto/thai-job-ner-dataset)
- **Hardware:** Apple Silicon MPS backend, FP32
- **Hyperparameters:** LR=3e-5, warmup=0.1, batch=8, grad_accum=2, 15 epochs, class-weighted loss, label smoothing=0.05
- **Training time:** ~3 min 48 sec

### Data Pipeline

Raw Thai text + GPT-4o entity extractions → fuzzy alignment with rapidfuzz + pythainlp TCC boundary snapping → subword token mapping via offset_mapping → IOB2-formatted HuggingFace Dataset.

## Evaluation

### Overall (Test Set, 126 examples)

| Metric | Score |
|--------|-------|
| **F1** | **0.897** |
| Precision | 0.850 |
| Recall | 0.949 |

### Per-Entity F1

| Entity | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| CONTACT | 0.962 | 0.942 | 0.983 |
| LOCATION | 0.959 | 0.928 | 0.991 |
| EMPLOYMENT_TERMS | 0.926 | 0.870 | 0.990 |
| PERSON | 0.907 | 0.861 | 0.958 |
| HARD_SKILL | 0.903 | 0.873 | 0.936 |
| DEMOGRAPHIC | 0.875 | 0.827 | 0.928 |
| COMPENSATION | 0.764 | 0.673 | 0.884 |

## Links

- **Model:** [chayuto/thai-job-ner-wangchanberta](https://huggingface.co/chayuto/thai-job-ner-wangchanberta)
- **Dataset:** [chayuto/thai-job-ner-dataset](https://huggingface.co/datasets/chayuto/thai-job-ner-dataset)
- **Source Code:** [github.com/chayuto/thai-job-nlp-ner](https://github.com/chayuto/thai-job-nlp-ner)

## Limitations

- Trained on synthetic data — may underperform on real-world posts with heavy emoji usage, OCR errors, or extreme colloquialism
- Thai-specific: limited English entity extraction capability
- 512 token max sequence length
- HARD_SKILL has the lowest F1 (0.761) due to open vocabulary and complex boundaries

## Technical Notes

- **FP16 is broken on MPS** — always use FP32 for Apple Silicon training
- Uses `offset_mapping` to bypass WangchanBERTa's `<_>` space token misalignment in `char_to_token()`
- Thai Character Cluster (TCC) boundary snapping prevents Unicode grapheme splitting during alignment

## License

MIT
