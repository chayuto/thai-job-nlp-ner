"""Gradio demo for Thai NER extraction.

Paste Thai text, see highlighted entities in a browser UI.

Usage:
    python app_demo.py
    # Opens at http://localhost:7860
"""

from __future__ import annotations

import os

import gradio as gr

from src.inference.pipeline import NERPipeline

MODEL_DIR = os.environ.get("NER_MODEL_DIR", "results/final")

# Entity type → color mapping for highlighted text
ENTITY_COLORS = {
    "HARD_SKILL": "#FF6B6B",
    "PERSON": "#4ECDC4",
    "LOCATION": "#45B7D1",
    "COMPENSATION": "#96CEB4",
    "EMPLOYMENT_TERMS": "#FFEAA7",
    "CONTACT": "#DDA0DD",
    "DEMOGRAPHIC": "#98D8C8",
}

EXAMPLES = [
    "รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท ต้องทำ CPR ได้ โทร 081-234-5678",
    "ต้องการพี่เลี้ยงเด็ก อายุ 25-40 ปี หญิง ทำอาหารได้ อยู่ประจำ ลาดพร้าว เงินเดือน 15,000 ติดต่อคุณแจน Line @care123",
    "หาคนขับรถส่งของ มีใบขับขี่ พื้นที่บางนา-ศรีนครินทร์ รายได้ 20,000-25,000 บาท/เดือน กะกลางวัน สนใจ โทร 02-345-6789",
    "รับสมัครแม่บ้าน part-time ทำความสะอาด ซักรีด 3 วัน/สัปดาห์ ย่านอารีย์ 500 บาท/วัน ติดต่อพี่นก 089-111-2222",
]


def extract_entities(text: str) -> tuple[dict, str]:
    """Run NER extraction and return highlighted text + entity table."""
    if not text or not text.strip():
        return {"text": "", "entities": []}, ""

    result = pipeline.extract(text)

    # Build highlighted text output for Gradio
    entities_for_highlight = []
    for e in result.entities:
        entities_for_highlight.append({
            "entity": e.label,
            "start": e.start,
            "end": e.end,
            "score": e.confidence,
        })

    highlighted = {
        "text": text,
        "entities": entities_for_highlight,
    }

    # Build summary table
    if not result.entities:
        summary = "No entities found."
    else:
        rows = []
        for e in result.entities:
            color = ENTITY_COLORS.get(e.label, "#CCCCCC")
            rows.append(
                f"| <span style='color:{color}'>**{e.label}**</span> "
                f"| {e.text} | {e.confidence:.1%} | {e.start}-{e.end} |"
            )
        summary = "| Label | Text | Confidence | Span |\n"
        summary += "|-------|------|------------|------|\n"
        summary += "\n".join(rows)

    return highlighted, summary


# Load model at module level
pipeline = NERPipeline(MODEL_DIR)

demo = gr.Interface(
    fn=extract_entities,
    inputs=gr.Textbox(
        label="Thai Text",
        placeholder="วางข้อความภาษาไทยที่นี่...",
        lines=4,
    ),
    outputs=[
        gr.HighlightedText(
            label="Extracted Entities",
            combine_adjacent=True,
            color_map=ENTITY_COLORS,
        ),
        gr.Markdown(label="Entity Details"),
    ],
    title="Thai Job NER Extraction",
    description=(
        "Extract structured entities (skills, locations, salaries, contacts, etc.) "
        "from informal Thai job postings using a fine-tuned WangchanBERTa model (110M params)."
    ),
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
