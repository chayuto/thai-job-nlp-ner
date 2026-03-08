# Chapter 2: ML Data Engineering & The "Silver Label" Pipeline

In the previous chapter, we established that we want to train WangchanBERTa to play a "highlighting game" called Named Entity Recognition (NER). 

However, in Machine Learning, the hardest part is almost never writing the neural network code. The hardest part is **building the dataset**. If you feed a model garbage data, it will learn garbage rules. (Garbage In, Garbage Out).

This chapter covers the ML Engineering pipeline necessary to build a high-quality dataset without doing thousands of hours of manual labor.

---

## 1. The Supervised Learning Problem
To train WangchanBERTa, we are using **Supervised Learning**. This means we must provide the model with perfectly labeled examples so it can learn the pattern.

We need about 500 perfectly labeled Thai job posts. 
*   **The Hard Way:** Hire 3 humans to sit at a computer for weeks, reading 500 posts, dragging their mouse over words to highlight them, and saving the files. This is expensive and slow.
*   **The ML Engineer Way:** Programmatically generate the labels using scripts and existing data. We call these **"Silver Labels"** (Because they aren't "Gold" labels perfectly verified by humans, but they are generated automatically and are 95% accurate).

---

## 2. Synthetic Data Augmentation (The Teacher Model)
We don't have enough varied social media posts in the Happy Care Connect database yet. If we train our model on only 50 examples, it will **Overfit** (it will just memorize those 50 posts and fail when it sees a new one).

To get 400+ examples, we use **Data Augmentation**:
1. We write a detailed prompt to **GPT-4o**.
2. We ask GPT-4o to act as 100 different personas (a stressed daughter looking for care, a nursing agency, etc.) and write fake Thai Facebook posts.
3. Crucially, we force GPT-4o to output a JSON object containing the exact substrings of the entities it embedded in the text.

```json
{
  "raw_text": "ด่วนหาคนดูแลพ่อเป็นเบาหวาน แถวลาดพร้าว ให้ 18000 จ้า ติดต่อ 0812345678",
  "entities": {
    "HARD_SKILL": ["เบาหวาน"],
    "LOCATION": ["ลาดพร้าว"],
    "COMPENSATION": ["18000"],
    "CONTACT": ["0812345678"]
  }
}
```
*Why this is genius:* You just paid API costs once to generate 400 perfectly labeled training examples. GPT-4o acts as the "Teacher" to generate data that will train your free "Student" (WangchanBERTa). This is called **Knowledge Distillation**.

---

## 3. Fuzzy String Matching (The Alignment Script)
We now have raw text and a JSON list of the substrings we want to highlight. But the neural network needs to know the *exact character positions* (index 0 to index 5) of those words in the text.

We use **Fuzzy Matching** (like the Levenshtein Distance algorithm). 
The Python script takes the string `"เบาหวาน"` and searches the raw text `"ด่วนหาคนดูแลพ่อเป็นเบาหวาน แถวลาดพร้าว"`. 
It finds that `"เบาหวาน"` starts at character index 20 and ends at index 27.

We now have absolute mathematical boundaries for our labels.

---

## 4. The IOB2 Tagging Format (How Models Read)
Neural networks do not understand "highlight character index 20 to 27". They only understand arrays of numbers (Tokens) mapped to arrays of Labels.

We must convert our fuzzy matched boundaries into a format called **IOB2** (Inside-Outside-Beginning). This is the absolute industry standard for NER.

Here is how IOB2 logic works:
*   **`O` (Outside):** This token is just a normal word. Not an entity.
*   **`B-` (Beginning):** This token is the FIRST word of a named entity.
*   **`I-` (Inside):** This token is a continuation of a named entity.

Let's look at the phrase: `"แถวลาดพร้าว ให้ 18000 บาท"`

1. First, the text is split into subword tokens by SentencePiece.
2. Then, our script applies the IOB2 tags based on the character boundaries we found earlier:

| Token | Translation | IOB2 Tag | Reason |
| :--- | :--- | :--- | :--- |
| แถว | Around | `O` | Just a normal word |
| ลาดพร้าว | Lat Phrao | **`B-LOCATION`** | The beginning of a location |
| ให้ | Give | `O` | Normal word |
| 18 | 18 | **`B-COMPENSATION`** | The beginning of salary |
| 000 | 000 | **`I-COMPENSATION`** | Continuation of the salary |
| บาท | Baht | **`I-COMPENSATION`** | Continuation of the salary |

### Why `B-` and `I-`?
Why don't we just tag them all as `LOCATION`? 
Imagine a post: `"ทำงานที่ กรุงเทพ เชียงใหม่"` (Work at Bangkok Chiang Mai).
If we tagged them as `[LOCATION, LOCATION]`, the computer would think there is one giant city called "Bangkok Chiang Mai".
By using `[B-LOCATION, B-LOCATION]`, the `B-` tag forces the computer to understand that these are two completely separate, distinct entities standing next to each other!

---

## Summary of Chapter 2
1. **Supervised Learning** requires hundreds of meticulously labeled examples.
2. We use **GPT-4o to generate Synthetic Data**, acting as a teacher to generate thousands of examples cheaply.
3. We use **Fuzzy Matching algorithms** to find exactly where the entities live inside the messy Thai strings.
4. We format the data into **IOB2 Tags**, breaking the text into arrays of subwords labeled `O`, `B-TAG`, or `I-TAG` so the PyTorch model can mathematically process it.

*Next, read Chapter 3 to learn how we actually fine-tune the model on Apple Silicon and deploy it as a microservice!*
