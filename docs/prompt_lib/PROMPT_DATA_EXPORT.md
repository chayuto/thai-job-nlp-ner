# Data Export Prompt for Production Repo Agent

**Purpose:** Copy-paste this prompt to the coding agent in the Happy Care Connect repo whenever you need a fresh data export for NER training.

**When to re-run:**
- Initial export to kick off Phase 1
- After significant new posts accumulate in production
- After schema changes in the production database
- When retraining the model on fresh data

**Output location:** Save the resulting `ner_export.json` to `data/raw/` in this repo (that path is gitignored — never commit real data).

---

## The Prompt

> Export raw post data from the production database into a JSON format that our NER fine-tuning pipeline can consume. The output file should be saved as `ner_export.json`.
>
> ### Output Schema
>
> ```json
> [
>   {
>     "id": "unique-post-id",
>     "raw_text": "the original raw Thai post text exactly as submitted",
>     "entities": [
>       {
>         "text": "the exact substring as it appears in raw_text",
>         "label": "HARD_SKILL"
>       }
>     ]
>   }
> ]
> ```
>
> ### Label Mapping
>
> Map your internal extracted fields to these 7 generic labels:
>
> | Your internal field(s) | Map to label | Notes |
> |---|---|---|
> | skills_required, medical_conditions, certifications | `HARD_SKILL` | Any specific ability or procedure |
> | applicant name, employer name, patient name | `PERSON` | Any named person |
> | location, address, hospital, district | `LOCATION` | Any place reference |
> | salary, daily_rate, budget, compensation | `COMPENSATION` | Any monetary amount with context |
> | job_type, shift, schedule, contract_type | `EMPLOYMENT_TERMS` | How the job is structured |
> | phone, line_id, email, social_media | `CONTACT` | Any contact info |
> | age, gender, nationality | `DEMOGRAPHIC` | Personal characteristics |
>
> ### Requirements
>
> 1. Only include posts where `raw_text` is not null/empty
> 2. Only include entities where the `text` substring actually exists in `raw_text` (do a simple `entity_text in raw_text` check — our pipeline handles fuzzy matching for near-misses later)
> 3. Strip any internal IDs, timestamps, or schema-specific metadata that would expose the production system
> 4. If a field maps to multiple entities (e.g., skills_required is an array), create a separate entity object for each value
> 5. Target: export ALL available posts. If that's too many, prioritize posts that have at least 3 different entity labels present
> 6. Save to `ner_export.json` in the project root
>
> ### Example Output
>
> ```json
> [
>   {
>     "id": "post-001",
>     "raw_text": "รับสมัครคนดูแลผู้สูงอายุ ย่านสีลม เงินเดือน 18,000 บาท ต้องทำ CPR ได้ โทร 081-234-5678",
>     "entities": [
>       {"text": "ดูแลผู้สูงอายุ", "label": "HARD_SKILL"},
>       {"text": "สีลม", "label": "LOCATION"},
>       {"text": "18,000 บาท", "label": "COMPENSATION"},
>       {"text": "CPR", "label": "HARD_SKILL"},
>       {"text": "081-234-5678", "label": "CONTACT"}
>     ]
>   }
> ]
> ```
>
> Do NOT modify the raw_text in any way — preserve it exactly as stored, including emojis, typos, and informal Thai.
