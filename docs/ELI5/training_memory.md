# Training Memory Management

## How does training load data — all at once or in batches?

**Both**, but at different layers:

1. **Dataset (RAM)**: Loaded via `DatasetDict.load_from_disk()` using Apache Arrow memory-mapping. The full dataset is accessible but Arrow reads lazily from disk — it doesn't copy everything into Python memory upfront.

2. **Training (MPS/GPU)**: Processed in **batches**. The HuggingFace `Trainer` creates a PyTorch `DataLoader` that samples `batch_size` examples at a time. Only one batch of tensors lives on the GPU/MPS device at any moment.

## Memory Breakdown

| Component | Where | Size | When |
|-----------|-------|------|------|
| Dataset (Arrow) | RAM (memory-mapped) | ~300KB for 500 posts | At load time |
| Model weights (WangchanBERTa, FP32) | MPS/GPU | ~440MB | Entire training |
| Gradients | MPS/GPU | ~440MB | Entire training |
| AdamW optimizer states | MPS/GPU | ~880MB | Entire training |
| Activations (1 batch) | MPS/GPU | ~140MB (batch=8, ~50 tokens) | Per training step |
| **Total on device** | | **~1.9GB** | |

## What controls memory usage?

The device memory is dominated by **model + gradients + optimizer** (~1.76GB), which is **constant** regardless of how many posts you have. The variable part is activation memory, which depends on:

- `batch_size` (config: 8)
- `sequence_length` (your posts average 49 tokens, max 343)
- `gradient_accumulation_steps` (config: 2) — processes 2 mini-batches before updating weights, effective batch = 16

The number of posts does **not** affect device memory.

## First experiment stats (Sprint 2)

| Split | Posts | Arrow file |
|-------|-------|-----------|
| Train | 406 | 260KB |
| Val | 51 | 33KB |
| Test | 51 | 30KB |
| **Total** | **508** | **323KB** |

Sequence lengths: min 24, max 343, mean 49.4, median 39.0. Only 2 posts exceeded 256 tokens. None exceeded 512.

Training completed in **199.7 seconds** (10 epochs) on Apple Silicon MPS.

## Scaling: will more posts exceed hardware limits?

**No.** Post count is not the bottleneck.

| Posts | Dataset in RAM | MPS Memory | Est. Training Time (10 epochs) |
|-------|---------------|------------|-------------------------------|
| 500 (current) | ~300KB | ~1.9GB | ~3.3 min |
| 5,000 | ~3MB | ~1.9GB (same) | ~33 min |
| 50,000 | ~30MB | ~1.9GB (same) | ~5.5 hrs |

MPS memory stays constant because only one batch is on device at a time. Even a Mac with 8GB unified memory has ~5GB available for ML workloads.

## What would actually blow MPS memory?

- Increasing `batch_size` significantly (e.g., 32+)
- Very long sequences close to `max_length=512` across the whole dataset
- Switching to a larger model (e.g., XLM-RoBERTa-large at 560M params)

## Relevant config (`configs/config.yaml`)

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
  mps_high_watermark_ratio: "1.5"  # controls MPS memory ceiling
model:
  max_length: 512
```

## Key code paths

- Dataset loading: `src/training/train_ner.py:64`
- Batch collation: `src/training/train_ner.py:81` (`DataCollatorForTokenClassification`)
- MPS watermark setting: `src/training/train_ner.py:58`
