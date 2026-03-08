# Multi-Model Synthetic Data Generation

## Should we use different models to generate synthetic data?

Currently `src/data/synthetic_augment.py` uses a single model (default: `gpt-4o`) to generate all synthetic Thai job posts. The question is whether mixing multiple LLMs would improve NER training quality.

## Why multiple models help

- **More diverse output**: Each LLM has different "writing style" biases. GPT-4o might favor certain sentence structures, Claude might produce different entity formats. Mixing them reduces the risk of the NER model overfitting to one model's patterns.
- **Better entity coverage**: One model might always generate compensation as "18,000 บาท/เดือน" while another uses "วันละ 800" or "เงินเดือน 25k". More variety = better generalization.
- **Cost optimization**: Generate some portion with cheaper models (GPT-4o-mini, Haiku) and reserve expensive models for quality-critical batches.

## When it's worth the effort

For the current scale (**400 posts**, 7 entity types), a single model is probably fine. Multi-model generation pays off when:

- Scaling to **1,000+ posts**
- The NER model overfits to synthetic data patterns (e.g., always predicting the same compensation format)
- Per-entity F1 evaluation shows low recall on certain entity styles that one model tends to miss

## How to do it (already supported)

The CLI already accepts a `--model` flag, so you can run multiple batches:

```bash
# Generate with different OpenAI models
python -m src.data.synthetic_augment --count 200 --model gpt-4o --output data/raw/synthetic_gpt4o.json
python -m src.data.synthetic_augment --count 200 --model gpt-4o-mini --output data/raw/synthetic_mini.json
```

To use Claude or Gemini, you would need to swap the OpenAI client in `generate_batch()` for the respective SDK — the prompt and validation logic can stay the same.

## Recommendation: fix entity imbalance first

Before adding model diversity, check the per-entity F1 scores from the current experiment. If specific entity types (e.g., DEMOGRAPHIC, EMPLOYMENT_TERMS) have low recall, it's more effective to:

1. Generate **targeted examples** for those specific types (e.g., add a `--focus-entity` flag)
2. Adjust the prompt to emphasize underrepresented entities

This gives a bigger quality improvement per additional post than blindly switching models.

## Relevant code

- Generator: `src/data/synthetic_augment.py`
- Prompt template: `src/data/synthetic_augment.py:15-47`
- Model config: `configs/config.yaml` → `synthetic.openai_model`
