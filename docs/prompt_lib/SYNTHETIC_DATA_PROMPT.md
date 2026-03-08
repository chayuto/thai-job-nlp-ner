# Synthetic Data Generation Prompt

**Purpose:** Give this prompt to any AI coding agent with OpenAI API access to generate synthetic Thai job posts for NER training. Re-run whenever you need more training data.

**When to re-run:**
- Initial dataset creation (target 400+ posts)
- When a specific entity class is underrepresented (pass the optional balancing instructions)
- When you want more variety in post styles
- Before a retraining cycle

**Output location:** Save the resulting file to `data/raw/synthetic.json` in the `thai-job-nlp-ner` repo.

---

## The Prompt

> You have access to the OpenAI API (the `openai` Python package is installed and `OPENAI_API_KEY` is set). Generate synthetic Thai job post data for NER model training.
>
> ### What to generate
>
> Create **{COUNT}** unique Thai social media job posts. These simulate real posts from Facebook groups where people seek or offer employment. Each post must include the raw text and a list of tagged entities.
>
> ### Output format
>
> Save to `{OUTPUT_PATH}` as a JSON array:
>
> ```json
> [
>   {
>     "id": "synthetic_0000",
>     "raw_text": "the complete Thai post text",
>     "entities": [
>       {"text": "exact substring from raw_text", "label": "HARD_SKILL"}
>     ]
>   }
> ]
> ```
>
> ### The 7 entity labels
>
> | Label | What it captures | Example substrings |
> |---|---|---|
> | `HARD_SKILL` | Abilities, procedures, certifications | ทำอาหาร, CPR, ขับรถ, Python, ดูแลผู้สูงอายุ |
> | `PERSON` | Names of people | คุณสมชาย, พี่แจน, น้องมิ้น |
> | `LOCATION` | Places, areas, hospitals | สีลม, ลาดพร้าว, รพ.รามาฯ, เชียงใหม่ |
> | `COMPENSATION` | Pay, salary, budget | 18,000 บาท/เดือน, วันละ 800, 25k-30k |
> | `EMPLOYMENT_TERMS` | Job structure, schedule | เต็มเวลา, อยู่ประจำ, กะดึก, Part-time, สัญญา 1 ปี |
> | `CONTACT` | Phone, Line, email | 081-234-5678, Line: @job123, somchai@gmail.com |
> | `DEMOGRAPHIC` | Age, gender, nationality | อายุ 30-45, หญิง, สัญชาติไทย, เพศชาย |
>
> ### Style requirements
>
> The posts MUST feel like real Thai social media. Follow these patterns:
>
> 1. **Mix of Thai and English** — real posts freely mix languages ("ต้องมี exp. อย่างน้อย 2 ปี", "รับสมัคร Part-time")
> 2. **Informal tone** — abbreviations (ค่ะ/ครับ → คะ/คับ), casual particles, incomplete sentences
> 3. **Emojis** — sprinkle realistically (🙏, 📞, 💰, ✅, not every post)
> 4. **Varied formats:**
>    - Short one-liners ("หาคนขับรถ ย่านบางนา โทร 089-xxx-xxxx")
>    - Detailed multi-paragraph job descriptions
>    - Job seeker self-introductions ("หนูชื่อแจน อายุ 28 ทำอาหารได้ค่ะ")
>    - Copy-paste style with bullet points
> 5. **Realistic typos** — occasional misspellings, missing tone marks, autocorrect artifacts (but not too many — keep it readable)
> 6. **Each post should have 2-7 entities** from different label types. Not every post needs all 7 labels
>
> ### Critical validation rule
>
> Every entity `"text"` value MUST be an **exact substring** that appears character-for-character in `"raw_text"`. Do not paraphrase, reformat, or clean up the entity text. If the post says "วันละ800" with no space, the entity text must be "วันละ800" not "วันละ 800".
>
> ### How to call the API
>
> Use GPT-4o with `temperature=1.0` for maximum variety. Generate in batches of 10 posts per API call. After each batch:
> 1. Parse the JSON response
> 2. Validate every entity text exists in raw_text (drop any that don't)
> 3. Append valid posts to the output file
> 4. Sleep 1 second between batches for rate limiting
>
> Use `response_format={"type": "json_object"}` to ensure valid JSON output.
>
> ### Post-generation validation
>
> After all posts are generated, print a summary:
> - Total posts generated
> - Total entities
> - Entity count per label (check for severe imbalance)
> - Number of entities dropped during validation
> - Average entities per post

---

## Optional: Balancing Instructions

If a specific label is underrepresented, append this to the prompt:

> **Balancing requirement:** The current dataset is low on `{LABEL}` entities. For this batch, ensure that at least 70% of generated posts contain at least one `{LABEL}` entity. Vary the examples — don't repeat the same patterns.

---

## Default Values

| Parameter | Default | Notes |
|---|---|---|
| `{COUNT}` | 400 | Increase for better model performance |
| `{OUTPUT_PATH}` | `data/raw/synthetic.json` | Relative to repo root |

---

## Example Usage

Just tell the agent:

> "Run the synthetic data generation prompt from `docs/internal/SYNTHETIC_DATA_PROMPT.md`. Generate 400 posts, save to `data/raw/synthetic.json`."

Or for balancing:

> "Run the synthetic data prompt but we need more DEMOGRAPHIC entities. Generate 100 extra posts focused on DEMOGRAPHIC."
