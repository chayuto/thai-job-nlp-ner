"""Fine-tune a transformer model for Thai NER on Apple Silicon (MPS)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.alignment.token_mapper import DEFAULT_LABELS
from src.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Select the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA backend")
        return torch.device("cuda")
    else:
        logger.info("Using CPU backend")
        return torch.device("cpu")


def train(
    dataset_dir: Path,
    output_dir: Path,
    config_path: Path = Path("configs/config.yaml"),
) -> None:
    """Run the full training pipeline."""
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Set MPS memory watermark
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(train_cfg.get("mps_high_watermark_ratio", "1.5"))

    device = get_device()

    # Load processed dataset
    logger.info(f"Loading dataset from {dataset_dir}...")
    dataset = DatasetDict.load_from_disk(str(dataset_dir))
    logger.info(f"  train: {len(dataset['train'])}, val: {len(dataset['validation'])}, test: {len(dataset['test'])}")

    # Load tokenizer and model
    checkpoint = model_cfg["checkpoint"]
    model_name = checkpoint.split("/")[-1]
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading model: {checkpoint}")
    logger.info(f"Output directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=model_cfg["num_labels"],
        id2label={i: l for i, l in enumerate(DEFAULT_LABELS)},
        label2id={l: i for i, l in enumerate(DEFAULT_LABELS)},
    )
    # Freeze embeddings for large-vocab models to reduce memory
    if train_cfg.get("freeze_embeddings", False):
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        logger.info("Froze embedding layer to reduce memory usage")

    model.to(device)

    # Dynamic padding collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=None,
    )

    # Training arguments — FP32 only on MPS
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg["epochs"],
        weight_decay=train_cfg["weight_decay"],
        fp16=False,
        bf16=False,
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=True,
        logging_steps=10,
        report_to=train_cfg.get("report_to", "none"),
        save_total_limit=3,
        remove_unused_columns=False,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
    )

    callbacks = []
    patience = train_cfg.get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log training results
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")

    # Save final model
    final_dir = output_dir / "final"
    logger.info(f"Saving final model to {final_dir}...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    logger.info("Done.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for Thai NER")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed"), help="Processed dataset directory")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory for checkpoints")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Training config YAML")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
