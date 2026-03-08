# 🕵️‍♂️ Research Agent Instructions: Thai NLP on Apple Silicon M5
**Target Repository:** `thai-job-nlp-ner` (Portfolio Project)
**Current Phase:** Pre-implementation Research

Hello Research Agent. You are tasked with performing deep technical research before our ML Engineering team begins writing the Python pipeline for fine-tuning `wangchanberta` for Named Entity Recognition (NER). 

We are specifically targeting a local-first workflow on the newly released **Apple Silicon M5 chip**.

Your objective is to produce a highly technical, code-focused research report detailing the absolute latest best practices, libraries, and gotchas for this exact stack. Please format your final output as a comprehensive Markdown document with code snippets.

---

## 🔬 Research Area 1: Apple Silicon M5 PyTorch Optimization (MPS)
The primary training hardware is an Apple MacBook Pro with an M5 chip. Historically, Apple's Metal Performance Shaders (MPS) backend in PyTorch has had severe issues with mixed-precision training (`fp16`) causing gradient underflow and `NaN` losses.

Your task is to investigate the *current* state of the art for M5 ML execution:
1.  **PyTorch MPS Status:** What is the most stable version of PyTorch (Nightly vs. Stable) to use in early 2026 for Apple Silicon? Has the `fp16` gradient underflow bug been resolved in newer `torch` builds, or do we still strictly need to force `fp32` in our Hugging Face `TrainingArguments`?
2.  **Apple MLX Framework:** Apple recently released their own native ML framework (`mlx`). Is it currently viable/recommended to fine-tune a Hugging Face BERT model (Token Classification) using `mlx` instead of PyTorch? If so, provide a high-level pros/cons list of `mlx` vs PyTorch `mps` for a beginner ML engineer.
3.  **Memory Management:** Given the unified memory architecture of the M5, what are the best practices for managing RAM during a Hugging Face `Trainer` loop (e.g., specific environment variables like `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, batch size limits for a 110M parameter model)?

## 🔬 Research Area 2: Thai Fuzzy String Alignment Libraries
Our data pipeline relies on "Silver Labels." We have raw Thai text and a substring (e.g., "ซีพีอาร์") that we know exists somewhere in the text. We need to find the exact character start and end indices to convert it into IOB2 tags.
Thai text has no spaces, and OCR/social media typos are rampant.

Your task is to find the absolute best Python library and algorithm for this specific sub-task:
1.  **FuzzyWuzzy vs RapidFuzz:** `rapidfuzz` is generally faster, but does it handle Thai Unicode character boundaries correctly (especially dealing with Thai vowels and tone marks stacking on the same character index)?
2.  **PyThaiNLP Integration:** Does the standard Thai NLP library (`pythainlp`) have a native, modern function for substring alignment or error-tolerant searching that we should use instead of a generic Levenshtein distance library?
3.  **The "Token Fragmentation" Risk:** When we find the character boundaries (e.g., `[15:20]`) and pass them to the `wangchanberta` `SentencePiece` tokenizer, what is the safest way to ensure the tokenizer doesn't truncate or misalign the IOB2 array? Review the `tokenizers` library `char_to_token()` method and document any known gotchas for Thai text.

## 🔬 Research Area 3: Modern seqeval Alternatives (Optional)
We plan to evaluate the model using strict exact-match F1 scoring (requiring perfect `B-TAG` and `I-TAG` boundary prediction). `seqeval` has been the industry standard for 5+ years.
1.  Is `seqeval` still the standard in 2026, or has the Hugging Face `evaluate` library introduced a newer, faster, or more robust native method for strict NER evaluation?

---

**Output Requirements:**
Do not write the actual training scripts. Your output should be a strategic research document referencing specific library versions, environment variables, PyTorch Github discussion threads (if applicable to the MPS bugs), and concrete Python snippet examples demonstrating the alignment math.
