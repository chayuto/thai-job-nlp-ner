# Incremental Synthetic Data Generation Prompt

**Purpose:** Use this prompt with any advanced LLM (like Claude, GPT-4, or Gemini) directly in the chat interface to generate high-quality, linguistically diverse Thai job posts in small batches. 

**Why this approach?**
- **Quality over Quantity:** By generating 10-20 posts at a time, the LLM can focus on maximizing variety, realistic typos, and complex language mixing without hitting output token limits or getting lazy.
- **Easy Appending:** The output format is **JSONL (JSON Lines)**. Every line is a complete, independent JSON object. You can just copy-paste the new batch at the bottom of your existing `data/raw/synthetic.jsonl` file. No need to worry about missing commas or breaking an array structure!
- **No API Key Required:** You act as the orchestrator. Just paste the prompt into the web UI.

**Output location:** Append the resulting code blocks to `data/raw/synthetic.jsonl` in the `thai-job-nlp-ner` repo.

---

## The Prompt

Copy and paste the text below to your LLM of choice:

> Act as an expert data generator for Thai Natural Language Processing (NLP). Your task is to generate a batch of **20 highly realistic, linguistically diverse Thai job posts** for Named Entity Recognition (NER) training.
> 
> ### What to generate
> 
> Create 20 unique Thai social media job posts simulating real posts from Facebook employment groups (e.g., finding a caregiver, offering housekeeping services, looking for a part-time nurse). 
> 
> ### Requirements for Realism
> 
> The posts MUST feel like real Thai social media. Follow these strictly:
> 1. **Mix of Thai and English** — freely mix languages ("ต้องมี exp.", "รับสมัคร Part-time", "ฟีดอาหาร", "suction เสมหะ").
> 2. **Informal tone** — use casual particles (คะ/ค่ะ/คับ/ค้าบ), abbreviations, and incomplete sentences.
> 3. **Varied formats** — include short one-liners, detailed multi-paragraph descriptions, bullet points, and job seeker self-introductions ("หนูชื่อแจน อายุ 28 รับเฝ้าไข้ค่ะ").
> 4. **Realistic typos** — occasional misspellings (e.g., วันเสา, ผช.), missing tone marks, or autocorrect artifacts. Don't overdo it, but make it look human.
> 5. **Emojis** — sprinkle realistically (🙏, 📞, 💰, 🏥), but not in every post.
> 6. Every post must contain **2 to 7 entities** from the labels below.
> 
> ### The 7 Entity Labels
> 
> | Label | What it captures | Example substrings |
> |---|---|---|
> | `HARD_SKILL` | Abilities, procedures, certifications | ทำอาหาร, CPR, ขับรถ, ดูแลผู้ป่วยติดเตียง, เจาะเลือด |
> | `PERSON` | Names of people | คุณสมชาย, พี่แจน, ป้าจู, หมอเก่ง |
> | `LOCATION` | Places, areas, hospitals | สีลม, รพ.รามาฯ, เชียงใหม่, พุทธมณฑลสาย 4, กทม. |
> | `COMPENSATION` | Pay, salary, budget | 18,000 บาท/เดือน, วันละ 800, 25k-30k, เหมา 5000 |
> | `EMPLOYMENT_TERMS` | Job structure, schedule | เต็มเวลา, อยู่ประจำ, กะดึก, Part-time, ไป-กลับ, freelance |
> | `CONTACT` | Phone, Line, email, inbox | 081-234-5678, Line: @job123, ib มาเลย, โทร 08x-xxx-xxxx |
> | `DEMOGRAPHIC` | Age, gender, nationality, patient type | อายุ 30-45, หญิง, สัญชาติไทย, ผู้ชาย, คนแก่, ผู้ป่วยอัมพฤกษ์ |
> 
> ### Critical Validation Rules
> 
> Every entity `"text"` value MUST be an **exact substring** that appears character-for-character in `"raw_text"`. 
> - If the post says "วันละ800" without a space, the entity text MUST be "วันละ800". 
> - If the post has a typo like "suctioon", the entity text MUST be "suctioon".
> - **DO NOT** paraphrase, reformat, or fix typos in the entity text.
> 
> ### Output Format (JSONL)
> 
> Output the result as valid JSON Lines (JSONL) inside a single standard markdown code block. Every line must be a single, complete JSON object. Do not output a JSON array `[]`.
> 
> **Example line structure:**
> `{"id": "batch_<BATCH_NUMBER>_<INDEX>", "raw_text": "text here", "entities": [{"text": "exact substring", "label": "HARD_SKILL"}]}`

---

## How to use this workflow

1. Create an empty file at `data/raw/synthetic.jsonl` if it doesn't exist.
2. Paste the prompt above into an AI chat (ChatGPT, Claude, Gemini).
3. Copy the output code block.
4. Paste it directly at the bottom of `data/raw/synthetic.jsonl`.
5. Repeat by telling the AI: **"Generate another batch of 20, focusing more on [LABEL_NAME] this time."**
6. **Validating:** After adding a batch, run the permanent validation script to ensure every entity is an exact substring:
   ```bash
   python3 scripts/validate_synthetic.py --file data/raw/synthetic.jsonl
   ```
   This will check for errors and provide a breakdown of entities by label.
