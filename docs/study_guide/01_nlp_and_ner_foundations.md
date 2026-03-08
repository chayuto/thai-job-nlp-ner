# Chapter 1: The Foundations of NLP & Named Entity Recognition

Welcome to the new domain of Machine Learning Engineering! As a software engineer, you already know how to build systems, manage databases, and write logic. Machine Learning (ML) is just another tool in your toolkit, but instead of writing explicit `if/else` rules, we write code that learns the rules from data.

This chapter will break down the fundamental concepts you need to understand the `thai-job-nlp-ner` project.

---

## 1. What is Natural Language Processing (NLP)?
Natural Language Processing is the branch of Artificial Intelligence concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.

Historically, NLP relied on complex dictionaries and grammar trees. Today, modern NLP relies entirely on **Neural Networks** and **Transformers**.

### The Transformer Revolution
In 2017, Google published a paper called "Attention Is All You Need." It introduced the **Transformer** architecture, which changed the world forever. Before transformers, AI read text linearly, word by word. Transformers read the *entire sentence all at once* and use a mathematical concept called **Self-Attention** to figure out which words relate to each other.

For example, in the sentence: *"The bank of the river"*, the AI pays "attention" to "river" to understand that "bank" means land, not a financial institution.

GPT-4, Claude, and WangchanBERTa are all built on the Transformer architecture.

---

## 2. Generative Models (GPT) vs. Encoder Models (BERT)
There are two main families of Transformers that you need to know:

### A. Generative Models (GPT - Generative Pre-trained Transformer)
*   **How they work:** They are basically extreme autocomplete. They read text and predict the *next word*.
*   **Examples:** GPT-4, Llama 3, Typhoon, Claude.
*   **Pros:** They can hold conversations, write code, and answer questions.
*   **Cons:** They are massive (Billions of parameters), slow, and expensive to run.

### B. Encoder Models (BERT - Bidirectional Encoder Representations from Transformers)
*   **How they work:** They look at a sentence from left-to-right AND right-to-left simultaneously. They don't generate new text; they *understand and classify* existing text.
*   **Examples:** RoBERTa, **WangchanBERTa**.
*   **Pros:** They are tiny (Millions of parameters), extremely fast, and perfect for specific classification tasks. They can easily run on your MacBook.
*   **Cons:** You cannot chat with them. They will not write an essay for you.

**Why we chose WangchanBERTa:** For our project, we don't need the AI to write a new job post. We already have the post. We just need to classify the words inside it. Therefore, an Encoder model (BERT) is 100x more efficient and cheaper than a Generative model (GPT).

---

## 3. What is Named Entity Recognition (NER)?
NER is a specific task within NLP. It is the process of locating and classifying named entities in unstructured text into pre-defined categories.

**Unstructured Text:**
> "รับสมัครคนดูแลผู้ป่วยกึ่งแม่บ้าน ย่านวัชรพล เงินเดือน 18,000 บาท สามารถทำแผลและฉีดอินซูลีนได้ โทร 081-234-5678"

**Named Entities Extracted:**
*   `[LOCATION]`: วัชรพล
*   `[COMPENSATION]`: 18,000 บาท
*   `[HARD_SKILL]`: ทำแผล, ฉีดอินซูลีน
*   `[CONTACT]`: 081-234-5678

NER is fundamentally a **Token Classification** problem. We ask the model to look at every single word (token) and assign it a class tag.

---

## 4. The Tokenization Problem (Why Thai is Hard)
Before a neural network can read text, the text must be converted into numbers. This is called **Tokenization**.

In English, tokenization is easy. We use spaces:
`"I love code"` -> `["I", "love", "code"]`

In Thai, there are no spaces between words:
`"รับสมัครคนดูแล"` -> Where do we split this?

### Dictionary Tokenization (The Old Way)
Libraries like `PyThaiNLP` use dictionaries to try and split the words:
`["รับสมัคร", "คน", "ดูแล"]`
*Problem:* If someone uses medical slang that isn't in the dictionary (e.g., "ซีพีอาร์" for CPR), the tokenizer panics and splits it into random letters, destroying the meaning.

### Subword Tokenization (The Modern ML Way)
WangchanBERTa uses a tool called **SentencePiece**. It doesn't use a dictionary of whole words. Instead, it uses a statistical dictionary of "subword chunks" (like syllables). 
If it sees a new, weird slang word, it simply breaks it down into familiar syllables instead of single letters. This makes it incredibly robust when reading messy social media posts with typos and abbreviations.

---

## Summary of Chapter 1
1. We are using **NLP** to teach a computer to understand Thai text.
2. We are using an **Encoder Model (WangchanBERTa)** because it is small, fast, and optimized for classification, unlike giant Chatbots (GPT).
3. We are performing **Named Entity Recognition (NER)** to highlight specific categories in the text.
4. We rely on **SentencePiece subword tokenization** to handle the fact that Thai social media text has no spaces and uses weird slang.

*Next, read Chapter 2 to learn about ML Data Engineering and the IOB2 tagging format!*
