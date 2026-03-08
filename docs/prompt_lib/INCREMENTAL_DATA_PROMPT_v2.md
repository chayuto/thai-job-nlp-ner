# Synthetic Thai Job Post Generation — v2

**Purpose:** Generate high-quality synthetic Thai job posts for training a Named Entity Recognition (NER) model. This is a self-contained guide — any LLM (Claude, GPT-4, Gemini) can produce batches of 20 posts directly in the chat interface with no API key or code required.

**Output file:** `data/raw/synthetic.jsonl` (JSON Lines — one JSON object per line, append-friendly)

---

## Quick Start

Paste the prompt from Section 1 into any LLM chat. Copy the output code block and append it to `synthetic.jsonl`. Repeat with the batch rotation schedule from Section 4.

After generating enough data:
```bash
python -m src.alignment.iob2_formatter --input data/raw/ --output data/processed
python -m src.training.train_ner --dataset data/processed --output results
```

---

## 1. The Prompt

Copy everything below and paste into your LLM:

---

> Act as an expert Thai NLP data generator. Generate a batch of **20 realistic Thai social media job posts** for NER training. Each post simulates a real Facebook group post about employment — hiring, job seeking, or forwarding a job ad.
>
> ### Output Format (JSONL)
>
> Output **only** a single markdown code block containing 20 lines. Each line is one complete JSON object:
>
> ```
> {"id": "batch_XX_0001", "raw_text": "...", "entities": [{"text": "...", "label": "..."}]}
> ```
>
> Replace `XX` with the batch number I tell you (or start at `01`).
>
> ### The 7 Entity Labels
>
> | Label | What to tag | Good examples | Tricky examples to include |
> |---|---|---|---|
> | `HARD_SKILL` | Specific abilities, procedures, certifications | ทำอาหาร, CPR, ขับรถ, ดูแลผู้ป่วยติดเตียง, suction เสมหะ, เจาะเลือด, Python | พิมพ์คอมได้, ยกตัวผู้ป่วย, ให้อาหารทางสายยาง, เปลี่ยนผ้าอ้อม |
> | `PERSON` | Names of people (employers, workers, patients, references) | คุณสมชาย, พี่แจน, ป้าแมว, น้องมิ้น, หมอเก่ง | คุณอาร์ม, เจ๊หน่อย, ลุงเล็ก, Khun May |
> | `LOCATION` | Places, areas, hospitals, provinces | สีลม, ลาดพร้าว, รพ.รามาฯ, กทม., เชียงใหม่ | พุทธมณฑลสาย 4, ซ.รามอินทรา 40, BTS อนุสาวรีย์, ใกล้เซ็นทรัลเวสเกต |
> | `COMPENSATION` | Pay amounts, salary ranges, budgets | 18,000 บาท/เดือน, วันละ 800, 25k-30k | เหมา 5,000/ทริป, ค่าแรง 400-500, negotiate ได้, เริ่ม 15,000 ขึ้นไป |
> | `EMPLOYMENT_TERMS` | Job structure, schedule, contract type | เต็มเวลา, อยู่ประจำ, กะดึก, Part-time | ไป-กลับ, freelance, สัญญา 6 เดือน, 3 วัน/สัปดาห์, จ-ศ 08:00-17:00, ทดลองงาน 1 เดือน |
> | `CONTACT` | Phone numbers, Line IDs, email, inbox references | 081-234-5678, Line: @job123 | ib มาเลย, โทรหาพี่แจน 089-xxx-xxxx, แอดไลน์ @care99, inbox ได้ค่ะ, tel. 02-345-6789 |
> | `DEMOGRAPHIC` | Age, gender, nationality, physical/health descriptions | อายุ 30-45, หญิง, สัญชาติไทย | เพศชาย, ไม่จำกัดเพศ, วัยทำงาน, น้ำหนักไม่เกิน 60 kg, สุขภาพดี |
>
> ### Critical Rules
>
> 1. **Exact substring:** Every entity `"text"` must appear **character-for-character** in `"raw_text"`. If the post says `วันละ800` with no space, the entity is `วันละ800` not `วันละ 800`. If there's a typo, keep the typo.
>
> 2. **Realism:** Posts must feel like real Thai social media:
>    - Mix Thai and English freely ("ต้องมี exp.", "caregiver ดูแล 24 ชม.")
>    - Informal particles (คะ/ค่ะ/คับ/ค้าบ/นะ/จ้า), abbreviations (ผช. = ผู้ช่วย, บ. = บาท)
>    - Emojis where natural (🙏📞💰🏥✅) — not every post
>    - Occasional typos (วันเสา, suctioon) — not every post
>
> 3. **Entity count varies realistically:**
>    - 30% of posts: **2-4 entities** (short, casual posts like "หาคนขับรถ บางนา โทร 089-xxx")
>    - 50% of posts: **5-7 entities** (typical job ads with details)
>    - 20% of posts: **8-12 entities** (detailed posts or multi-role listings)
>    - Do NOT give every post 10+ entities — real short posts exist
>
> 4. **Post format variety** — every batch should include a mix of:
>    - **Employer posts** ("รับสมัคร...", "ต้องการ...", "หา...")
>    - **Job-seeker posts** ("หนูชื่อแจน อายุ 28 รับเฝ้าไข้ค่ะ", "ว่างรับงาน...")
>    - **Forwarded posts** ("ฝากประชาสัมพันธ์ค่ะ 🙏 มีคนรู้จักหา...")
>    - **Short one-liners** ("หาคนดูแลผู้สูงอายุ ลาดพร้าว 15k โทร 081-xxx-xxxx")
>    - **Structured ads** (with bullet points, line breaks, headers)
>
> 5. **Hard negatives — include text that looks like an entity but ISN'T:**
>    - "ดูแลอย่างดี" — NOT a HARD_SKILL (it's a general descriptor, not a specific ability)
>    - "มาสมัครกันนะ" — NOT a PERSON (even though it contains a common name-like syllable)
>    - "ราคาไม่แพง" — NOT COMPENSATION (not a specific amount)
>    - "อยู่ใกล้ๆ" — NOT LOCATION (not a specific place)
>    - Do NOT tag these as entities. Just let them be part of raw_text with no entity annotation. This teaches the model to distinguish real entities from similar-looking text.

---

## 2. What the Model Currently Struggles With

The current model (v2, F1=0.828) has specific weaknesses. New data should **deliberately target** these patterns:

### HARD_SKILL (F1=0.761 — weakest class)

**Problem:** The model over-predicts HARD_SKILL boundaries. It tags surrounding words as part of a skill when they're not.

**What helps:**
- Skills embedded in flowing sentences, not just bullet lists: "ต้องเป็นคนที่**ทำอาหาร**ได้คล่องและดูแลบ้านเป็น" — only "ทำอาหาร" is the skill, not "ทำอาหารได้คล่อง"
- Multi-word skills with clear boundaries: "**ดูแลผู้ป่วยติดเตียง**" is ONE skill, "**ทำแผล**และ**เจาะเลือด**" is TWO skills
- English skills in Thai context: "ใช้ **Excel** กับ **SAP** ได้"
- Medical procedure names: "**suction เสมหะ**", "**วัด vital signs**", "**ให้อาหารทางสายยาง**"

### DEMOGRAPHIC (F1=0.776)

**Problem:** Gets confused with HARD_SKILL ("ใบขับขี่" — skill or demographic?) and sometimes LOCATION.

**What helps:**
- Clear demographic phrases near skill phrases: "**หญิง** **อายุ 25-40** ที่**ทำอาหารไทย**ได้" — first two are DEMOGRAPHIC, last is HARD_SKILL
- Patient descriptions (these ARE demographics): "**ผู้ป่วยอัมพาต**", "**คนชรา 80 ปี**", "**เด็ก 3 ขวบ**"
- Qualification requirements that sound demographic: "**วุฒิ ม.3 ขึ้นไป**" — this is DEMOGRAPHIC, not a skill

### EMPLOYMENT_TERMS (recall=0.793 — model misses 20% of these)

**Problem:** The model fails to spot subtle employment terms, especially schedule formats and contract details.

**What helps:**
- Varied schedule formats: "**จ-ศ 08:00-17:00**", "**เวร 3 กะ**", "**ทำงาน 5 วัน หยุด 2 วัน**"
- Contract details: "**ทดลองงาน 3 เดือน**", "**สัญญา 1 ปี**", "**ต่อสัญญาได้**"
- Live-in vs commute: "**อยู่ประจำ**" vs "**ไป-กลับ**" vs "**นอนเฝ้า 24 ชม.**"
- English employment terms in Thai: "**full-time**", "**Part-time**", "**freelance**"

### PERSON (support=37 — fewest test examples)

**Problem:** Low data volume, model sometimes misses names entirely.

**What helps:** More posts with Thai names, nicknames, and informal references:
- Formal: "ติดต่อ**คุณสมศรี**", "นาย**ประยุทธ์**"
- Nicknames: "**พี่แจน**", "**น้องมิ้น**", "**ป้าแมว**", "**เจ๊หน่อย**"
- Patient names: "ดูแล**คุณยายทองดี**", "ผู้ป่วยชื่อ**ลุงเล็ก**"
- Mixed: "โทรหา**Khun May** ได้เลย"
- Self-intros: "**หนูชื่อแจน** อายุ 28..."

---

## 3. Entity Boundary Guidelines

These rules prevent the most common alignment failures:

### DO tag as entity:
```
"เงินเดือน 18,000 บาท"     → COMPENSATION (include the unit)
"โทร 081-234-5678"          → CONTACT (include "โทร" prefix)
"Line @care123"              → CONTACT (include "Line" prefix)
"อายุ 25-40 ปี"              → DEMOGRAPHIC (include "อายุ" and "ปี")
"Part-time"                  → EMPLOYMENT_TERMS
"ดูแลผู้ป่วยติดเตียง"          → HARD_SKILL (entire procedure name)
"สีลม"                       → LOCATION (place name only)
```

### DO NOT tag:
```
"รับสมัคร"                   → NOT an entity (it's a verb meaning "hiring")
"ต้องการ"                    → NOT an entity (it's a verb meaning "need")
"สนใจ"                      → NOT an entity (it's a verb meaning "interested")
"ด่วน"                       → NOT an entity (it's "urgent")
"มีประสบการณ์"                → NOT an entity (generic qualifier, not a specific skill)
"ทำงานดี"                    → NOT an entity (general praise, not a skill)
"เงินดี"                      → NOT COMPENSATION (not a specific amount)
"ใกล้ BTS"                   → only "BTS" could be LOCATION if a specific station follows
```

### When in doubt — be conservative:
- Tag only the specific noun phrase, not the surrounding verb/particles
- "ต้อง**ทำอาหาร**ได้" → tag "ทำอาหาร" not "ต้องทำอาหารได้"
- "เงินเดือน **18,000 บาท**" → tag "18,000 บาท" or "เงินเดือน 18,000 บาท", just be consistent

---

## 4. Batch Rotation Schedule

LLMs get repetitive within a session. Rotate your re-prompt after each batch to keep the data diverse. After the first batch, just say:

```
Generate another batch of 20 posts. [FOCUS DIRECTIVE]. Use batch number XX.
```

### Phase 1: Entity-focused rotation (batches 01-15)

| Batches | Re-prompt directive |
|---------|---------------------|
| 01-03 | *(base prompt as-is)* |
| 04-05 | "Focus on HARD_SKILL boundaries. Include multi-word medical skills like 'ให้อาหารทางสายยาง', 'วัด vital signs', 'suction เสมหะ'. Put skills in flowing sentences, not just bullet lists." |
| 06-07 | "Focus on PERSON names. Every post must include at least one Thai name — mix formal (คุณสมศรี), nicknames (พี่แจน, ป้าแมว), and patient names (คุณยายทองดี)." |
| 08-09 | "Focus on DEMOGRAPHIC and EMPLOYMENT_TERMS together. Include age ranges, gender preferences, AND schedule details in each post. Include subtle terms like 'ทดลองงาน 3 เดือน', 'ไป-กลับ'." |
| 10-11 | "Focus on COMPENSATION variety. Include Thai formats (เหมา 5,000, วันละ 800), English formats (25k-30k), and ranges (15,000-18,000 บาท/เดือน). Some posts should NOT have any compensation." |
| 12-13 | "Focus on CONTACT variety. Include Line IDs (@care123), 'ib มาเลย', phone with โทร prefix, email addresses, and 'inbox ได้ค่ะ'." |
| 14-15 | "Focus on LOCATION specificity. Include hospital names (รพ.รามาฯ), soi numbers (ซ.รามอินทรา 40), BTS stations (BTS อนุสาวรีย์), malls (ใกล้เซ็นทรัลเวสเกต), and provinces." |

### Phase 2: Post style rotation (batches 16-30)

| Batches | Re-prompt directive |
|---------|---------------------|
| 16-18 | "Make ALL 20 posts **job-seeker self-introductions** ('หนูชื่อ...อายุ...รับงาน...'). These are people offering their services, not employers posting ads." |
| 19-21 | "Make ALL 20 posts **very short — 1-2 lines max**. Heavy abbreviations and slang. Like 'หาคนดูแลคนแก่ ลาดพร้าว 15k โทร 089xxx'. Most should have only 2-4 entities." |
| 22-24 | "Make ALL 20 posts **long and detailed — 3-5 lines**. Structured with bullet points or emoji markers. Include 6-10 entities per post covering multiple entity types." |
| 25-27 | "**Medical/nursing/elderly care** scenarios only. Hospitals, home care, rehab. Use medical Thai-English mixing ('suction เสมหะ', 'วัด O2 sat', 'เปลี่ยน tracheostomy')." |
| 28-30 | "**Non-care jobs** only. Restaurant (ร้านอาหาร), factory (โรงงาน), construction (ก่อสร้าง), delivery (ขับรถส่งของ), domestic (แม่บ้าน). Vary the industry." |

### Phase 3: Hard negatives & edge cases (batches 31-40)

| Batches | Re-prompt directive |
|---------|---------------------|
| 31-33 | "Include **hard negatives** in every post. Text that looks like entities but isn't: 'ดูแลอย่างดี' (not a skill), 'ราคาไม่แพง' (not compensation), 'อยู่ใกล้ๆ' (not a location). Do NOT tag these." |
| 34-36 | "**Thai-English code-switching** posts. 'รับสมัคร caregiver ประสบการณ์ min 2 yrs', 'ต้อง drive ได้ มี license'. Mix English words naturally." |
| 37-38 | "**Forwarded/reposted** style. Start with 'ฝากประชาสัมพันธ์ค่ะ 🙏' or 'ส่งต่อจากกลุ่ม...' or '📢 ด่วน!! มีคนรู้จักหา...'" |
| 39-40 | "**Ambiguous boundary** posts. Include entities where the start/end is tricky: 'เงินเดือนเริ่ม 15,000 ขึ้นไป' (is 'เริ่ม' part of compensation?), 'คนดูแลผู้สูงอายุและทำอาหาร' (two skills or one?)." |

---

## 5. Post-Generation Checklist

After generating all batches, verify before training:

```bash
# Count posts and check entity distribution
python -c "
import json
from collections import Counter
with open('data/raw/synthetic.jsonl') as f:
    posts = [json.loads(l) for l in f if l.strip()]
labels = Counter(e['label'] for p in posts for e in p['entities'])
print(f'Posts: {len(posts)}')
print(f'Entities: {sum(labels.values())}')
for l, c in labels.most_common():
    print(f'  {l:<20} {c:>5} ({c/sum(labels.values())*100:.1f}%)')

# Check for substring violations
bad = 0
for p in posts:
    for e in p['entities']:
        if e['text'] not in p['raw_text']:
            bad += 1
            print(f'  VIOLATION: {p[\"id\"]} — \"{e[\"text\"]}\" not in raw_text')
print(f'Substring violations: {bad}')
"
```

### Target distribution

Aim for these proportions (exact match not required):

| Label | Target % | Why |
|-------|----------|-----|
| HARD_SKILL | 20-25% | High volume + diverse boundaries needed |
| DEMOGRAPHIC | 14-17% | Currently confused with HARD_SKILL |
| CONTACT | 12-15% | Already strong, maintain coverage |
| LOCATION | 12-15% | Include varied formats (soi, BTS, hospital) |
| COMPENSATION | 11-14% | Include range formats and Thai abbreviations |
| EMPLOYMENT_TERMS | 11-14% | Include subtle schedule formats |
| PERSON | 8-12% | Boost from current 7% — more names needed |

### Red flags to watch for

- **Every post has 10+ entities** → need more short posts (2-4 entities)
- **PERSON < 5%** → add more batches with name-focused directive
- **Substring violations > 0** → fix or discard those posts before training
- **All posts are employer-style** → add job-seeker self-intro batches
- **No English mixing** → add code-switching batches

---

## 6. Current Model Weaknesses (Reference)

These are the v2 model's per-entity scores. New data should push the weakest classes up:

```
CONTACT            F1=0.957  ✅ Strong — maintain, don't over-index
PERSON             F1=0.892  ⬆ Good but low test support — need more volume
LOCATION           F1=0.861  ⬆ Good — keep including diverse locations
EMPLOYMENT_TERMS   F1=0.850  ⚠ Misses subtle terms — schedule formats, contracts
COMPENSATION       F1=0.819  ⚠ Confused by range formats — need more variety
DEMOGRAPHIC        F1=0.776  ⚠ Confused with HARD_SKILL — need clearer boundaries
HARD_SKILL         F1=0.761  ❌ Weakest — over-predicts boundaries, needs flowing context
```

### Confusion patterns to break

| True → Predicted | Count | What's happening |
|--|--|--|
| O → HARD_SKILL | 61 | Model tags non-skill words as skills |
| EMPLOYMENT_TERMS → O | 35 | Model misses employment terms |
| O → DEMOGRAPHIC | 31 | Model tags non-demographic text as demographic |
| PERSON → O | 24 | Model misses person names |
| DEMOGRAPHIC → HARD_SKILL | 11 | "ใบขับขี่" — skill or qualification? |

**Fix:** Include posts where:
- Non-skill text appears near skills (teaches O vs HARD_SKILL boundary)
- Employment terms are phrased subtly (teaches model to spot them)
- Demographics and skills appear in the same sentence with clear separation
- Person names appear in varied contexts (not always after "ติดต่อ")
