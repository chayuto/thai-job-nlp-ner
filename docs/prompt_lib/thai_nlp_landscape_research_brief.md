# Thai NLP Landscape Research Brief

## Objective

Deep research into the WangchanBERTa ecosystem, equivalent Thai language models, and the broader Thai NLP landscape. Identify gaps, underserved use cases, and high-impact portfolio project opportunities.

---

## Research Agent Instructions

### Agent 1: WangchanBERTa & Thai Encoder Models

**Goal:** Map the full WangchanBERTa family and competing Thai encoder models.

**Search targets:**
- HuggingFace Hub: search `wangchanberta`, `thai-bert`, `phayathaibert`, filter by language=th
- GitHub: `airesearch-in-th`, `vistec-AI`, `PyThaiNLP` orgs
- Google Scholar / Semantic Scholar: `"WangchanBERTa"`, `"PhayaThaiBERT"`, `"Thai BERT"`
- ArXiv: `thai language model`, `thai NLP pretrained`

**Collect for each model:**
| Field | Detail |
|-------|--------|
| Model name & HF ID | e.g. `airesearch/wangchanberta-base-att-spm-uncased` |
| Architecture | RoBERTa-base, DeBERTa-v3, etc. |
| Pretraining data | Size, source (Wisesight, Oscar, MC4, ThaiCC, etc.) |
| Tokenizer | SentencePiece, WordPiece, BPE — vocab size |
| Downstream benchmarks | NER, POS, sentiment, text classification — scores |
| Known fine-tuned checkpoints | Any community fine-tunes on HF |
| Last updated | Is the model actively maintained? |
| Paper / citation | Link to paper if available |

**Models to investigate:**
1. `airesearch/wangchanberta-base-att-spm-uncased` (original)
2. `airesearch/wangchanberta-base-wiki-spm` (wiki variant)
3. `clicknext/phayathaibert` — how does it compare?
4. `xlm-roberta-base` / `xlm-roberta-large` — multilingual baseline for Thai
5. `bert-base-multilingual-cased` (mBERT) — still used?
6. Any DeBERTa-v3 or ELECTRA variants trained on Thai
7. Any Thai-specific sentence transformers (for embeddings)

---

### Agent 2: Thai LLMs & Generative Models

**Goal:** Map the Thai large language model (LLM) space — decoder models, instruction-tuned, chat models.

**Search targets:**
- HuggingFace Hub: `thai`, `openthaigpt`, `typhoon`, `seallm`, language=th, sort by downloads
- Web search: "Thai LLM benchmark 2024 2025", "OpenThaiGPT", "SCB 10X Typhoon"
- GitHub: repos with Thai LLM fine-tuning, RLHF, or eval

**Models to investigate:**
1. **OpenThaiGPT** — versions, benchmarks, training data, license
2. **SCB 10X Typhoon** — 1.5, 2.0 variants, instruct versions, what tasks?
3. **SeaLLM** (DAMO Academy) — multilingual SEA model, Thai performance
4. **Qwen-2.5 / Llama-3 fine-tunes for Thai** — any community efforts?
5. **Google Gemma Thai** — any Thai fine-tunes?
6. **WangchanGLM / WangchanX** — any generative models from AIResearch?

**Collect:**
- Model size (params), base model, training data
- Thai-specific benchmarks (ThaiExam, M3Exam, ONET, etc.)
- License / commercial usability
- Gaps: what CAN'T these models do well in Thai?

---

### Agent 3: Thai NLP Tasks, Datasets & Benchmarks

**Goal:** Catalog all major Thai NLP tasks, available datasets, and current SOTA.

**Search targets:**
- HuggingFace Datasets: language=th
- GitHub: `PyThaiNLP/`, `thai-nlp-datasets`, `LST20`
- Papers With Code: Thai language tasks
- Web: "Thai NLP benchmark", "Thai NER dataset", "Thai text classification dataset"

**Task landscape to map:**

| Task | Datasets | SOTA Model | SOTA Score | Gap/Opportunity |
|------|----------|------------|------------|-----------------|
| **NER** | LST20, ThaiNER (1.4, 2.0), THAI-NEST | ? | ? | Domain-specific NER? |
| **POS Tagging** | LST20, Orchid | ? | ? | |
| **Sentiment Analysis** | Wisesight, Wongnai, PyThaiNLP sentiment | ? | ? | |
| **Text Classification** | Wisesight, Prachathai67k, TruthfulQA-TH | ? | ? | |
| **Question Answering** | iApp Thai QA, XQuAD-TH, TyDiQA | ? | ? | |
| **Machine Translation** | OPUS, SCB-MT-EN-TH, NECTEC | ? | ? | |
| **Summarization** | ThaiSum, XL-Sum TH | ? | ? | |
| **Word Segmentation** | BEST, LST20 | DeepCut, Attacut, newmm | ? | |
| **Dependency Parsing** | Thai-PUD, LST20 | ? | ? | |
| **Relation Extraction** | ??? | ? | ? | Likely a GAP |
| **Coreference Resolution** | ??? | ? | ? | Likely a GAP |
| **Aspect-Based Sentiment** | ??? | ? | ? | Likely a GAP |
| **Hate Speech / Toxicity** | Thai-Toxicity? | ? | ? | |
| **Semantic Similarity** | STS-TH? Thai-STS? | ? | ? | |
| **Speech (ASR/TTS)** | CommonVoice-TH, Gowajee | ? | ? | |

---

### Agent 4: Gap Analysis & Portfolio Project Ideas

**Goal:** Identify underserved areas in Thai NLP that are portfolio-worthy and technically feasible.

**Evaluation criteria for each opportunity:**
- **Demand:** Is there real-world need? (industry, government, social good)
- **Gap:** Is this underserved? Few models/datasets available?
- **Feasibility:** Can it be done with WangchanBERTa + consumer hardware (Apple Silicon, 16-32GB)?
- **Differentiator:** Would this stand out on a portfolio / resume?
- **Data availability:** Can training data be sourced, generated, or scraped?

**Specific areas to investigate:**

#### NER Extensions (build on our existing pipeline)
1. **Thai Legal NER** — extract entities from Thai legal documents (court rulings, contracts). Entities: LAW_REFERENCE, COURT_NAME, PARTY_NAME, DATE, MONETARY_AMOUNT, LEGAL_TERM
2. **Thai Medical/Health NER** — extract from health forums, news. Entities: DISEASE, SYMPTOM, DRUG, DOSAGE, BODY_PART
3. **Thai E-commerce NER** — product listings, reviews. Entities: PRODUCT, BRAND, PRICE, SPEC, MATERIAL
4. **Thai Address Parsing** — structured extraction from unstructured Thai addresses (province, district, subdistrict, postal code)
5. **Thai Social Media NER** — handle informal text, slang, code-switching (Thai-English)

#### Beyond NER (new task types)
6. **Thai Relation Extraction** — given entities, extract relationships. E.g., (Company, HIRES, Person), (Drug, TREATS, Disease)
7. **Thai Question Answering** — extractive QA on Thai documents, fine-tune WangchanBERTa for SQuAD-style task
8. **Thai Aspect-Based Sentiment Analysis (ABSA)** — fine-grained sentiment on Thai product reviews (food, hotels)
9. **Thai Text-to-SQL** — natural language to SQL for Thai business queries
10. **Thai Document Layout + NER** — combine OCR + NER for Thai receipts, invoices, ID cards
11. **Thai Keyword/Keyphrase Extraction** — unsupervised or supervised, useful for SEO/content
12. **Thai Fake News / Misinformation Detection** — classification + evidence extraction
13. **Thai Resume Parsing** — structured extraction from Thai resumes (closely related to our job posting NER)
14. **Thai Multimodal NER** — NER from Thai text in images (social media posts, screenshots)

---

### Agent 5: Competitive & Community Analysis

**Goal:** Understand who is building what, and where the community is active.

**Search:**
- GitHub trending repos for Thai NLP (sort by stars, recent activity)
- HuggingFace spaces with Thai demos
- Thai NLP community: PyThaiNLP Discord/GitHub, AI Builders, Bualabs
- Thai AI conferences: TNLP, Thai-SNLP, relevant AACL/ACL papers
- Kaggle competitions involving Thai text
- Industry players: LINE Thailand, Agoda, SCB, True, AIS — what NLP are they doing?

**Questions to answer:**
1. What Thai NLP projects have the most GitHub stars? What's popular?
2. Are there active Thai NLP Kaggle competitions or challenges?
3. What are Thai companies hiring for in NLP? (LinkedIn, JobThai, etc.)
4. What Thai NLP papers were published at ACL/EMNLP/AACL in 2024-2025?
5. Is there a standard Thai NLP leaderboard (like GLUE for English)?

---

## Output Format

Each agent should produce:

```markdown
## [Agent Name] Findings

### Key Discoveries
- Bullet-point summary of top 5-10 findings

### Detailed Table
(structured data as specified above)

### Gaps Identified
- What's missing, outdated, or underserved

### Recommended Actions
- Specific project ideas with estimated effort (S/M/L)
- Links to relevant resources
```

---

## Priority Ranking Criteria

After all agents report, synthesize findings into a ranked list of project opportunities:

| Rank | Project | Gap Score (1-5) | Feasibility (1-5) | Portfolio Impact (1-5) | Total |
|------|---------|----------------|--------------------|----------------------|-------|
| 1 | ? | ? | ? | ? | ? |
| 2 | ? | ? | ? | ? | ? |
| ... | | | | | |

**Gap Score:** 5 = no existing solution, 1 = well-covered
**Feasibility:** 5 = can do in 1-2 weeks with existing tools, 1 = needs massive data/compute
**Portfolio Impact:** 5 = highly impressive & demonstrable, 1 = generic/common

---

## Context: What We Already Have

- **Current project:** Thai Job Posting NER (WangchanBERTa fine-tune, F1=0.897)
- **Entities:** PERSON, LOCATION, CONTACT, HARD_SKILL, DEMOGRAPHIC, EMPLOYMENT_TERMS, COMPENSATION
- **Pipeline:** Synthetic data generation → fuzzy alignment to IOB2 → MPS training → seqeval eval → FastAPI + Gradio demo
- **Published:** HuggingFace Hub (model + dataset)
- **Tech stack:** Python, PyTorch (MPS), HuggingFace Transformers, FastAPI, Docker

Any new project should ideally **reuse or extend** this pipeline where possible.
