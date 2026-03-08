"""FastAPI server for Thai NER extraction.

Endpoints:
  POST /extract  — Extract entities from Thai text
  GET  /health   — Model status and version info
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.pipeline import NERPipeline

logger = logging.getLogger(__name__)

# Configuration via environment variables
MODEL_DIR = os.environ.get("NER_MODEL_DIR", "results/final")
DEVICE = os.environ.get("NER_DEVICE", None)  # auto-detect if not set
MAX_LENGTH = int(os.environ.get("NER_MAX_LENGTH", "512"))
MAX_TEXT_LENGTH = int(os.environ.get("NER_MAX_TEXT_LENGTH", "10000"))

app = FastAPI(
    title="Thai NER API",
    description="Named Entity Recognition for informal Thai job postings",
    version="0.1.0",
)

# Global pipeline instance — loaded on startup
pipeline: NERPipeline | None = None


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Thai text to extract entities from",
        json_schema_extra={
            "examples": ["รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท"]
        },
    )


class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class ExtractResponse(BaseModel):
    entities: list[EntityResponse]
    grouped: dict[str, list[str]]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_dir: str
    device: str
    num_labels: int
    max_length: int


@app.on_event("startup")
async def load_model() -> None:
    """Load NER model on server startup."""
    global pipeline
    logger.info(f"Loading NER model from {MODEL_DIR}...")
    pipeline = NERPipeline(
        model_dir=MODEL_DIR,
        device=DEVICE,
        max_length=MAX_LENGTH,
    )
    logger.info("Model loaded and ready.")


@app.post("/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest) -> ExtractResponse:
    """Extract named entities from Thai text.

    Returns entity spans with labels, character offsets, confidence scores,
    and entities grouped by type.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Text too long ({len(text)} chars). Max: {MAX_TEXT_LENGTH}",
        )

    start_time = time.perf_counter()
    result = pipeline.extract(text)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return ExtractResponse(
        entities=[
            EntityResponse(
                text=e.text,
                label=e.label,
                start=e.start,
                end=e.end,
                confidence=round(e.confidence, 4),
            )
            for e in result.entities
        ],
        grouped=result.grouped(),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check model status and server health."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_dir=str(pipeline.model_dir),
        device=str(pipeline.device),
        num_labels=len(pipeline.id2label),
        max_length=pipeline.max_length,
    )
