# Training Report Glossary

A plain-language guide to every technical term that appears in our training logs. Read this before (or alongside) the experiment reports.

---

## The Scoreboard: How We Measure "Is the Model Good?"

### F1 Score

The **single most important number** in our reports. It's the overall "grade" for the model.

Imagine the model reads a Thai job post and tries to highlight all the phone numbers, locations, salaries, etc. F1 asks: "How much did the model's highlights overlap with the correct answers?"

- **F1 = 1.0** means perfect — every entity found, no mistakes
- **F1 = 0.0** means total failure
- **Our v2 model: F1 = 0.828** means ~83% overlap with the correct answers

F1 is the *balance* between Precision and Recall (explained next). It prevents the model from gaming the score by either being too cautious or too aggressive.

**Why not just use accuracy?** Because most tokens in a sentence are "O" (not an entity). A model that labels everything as "O" gets 93% accuracy but is completely useless. F1 only cares about the entities.

### Precision

**"When the model highlights something, how often is it correct?"**

- High precision = few false alarms
- Our v2: 0.799 = when it says "this is a LOCATION", it's right ~80% of the time
- Low precision means the model is *over-eager* — highlighting things that aren't actually entities

Think of a fire alarm. High precision = it only goes off when there's a real fire.

### Recall

**"Of all the real entities in the text, how many did the model find?"**

- High recall = few missed entities
- Our v2: 0.859 = it finds ~86% of all real entities
- Low recall means the model is *too cautious* — missing things it should have found

Back to the fire alarm: high recall = it catches every fire (but might also go off for burnt toast).

### The Precision-Recall Trade-off

You can't usually max out both. Making the model more aggressive (higher recall) tends to cause more false alarms (lower precision), and vice versa. F1 finds the sweet spot.

Our model leans toward **high recall, lower precision** — it finds most entities but sometimes highlights extra text. This is actually the safer direction for our use case: it's better to extract "สีลม" plus a bit of surrounding text than to miss it entirely.

### Support

**How many examples of this entity type existed in the test set.** It's not a score — it tells you how reliable the other scores are.

- HARD_SKILL: 148 support = lots of test examples, the F1 score is reliable
- PERSON: 37 support = fewer examples, the F1 score could fluctuate more

### Accuracy

**What percentage of individual tokens were labeled correctly.** We report it but mostly ignore it (see "Why not just use accuracy?" above). Our 93% accuracy sounds great but hides mistakes on the rare entity tokens that matter most.

---

## The Knobs: Hyperparameters

Hyperparameters are the settings you choose *before* training starts. The model can't learn these — you have to pick them. Tuning them is like adjusting an oven: wrong temperature or time ruins the food even with perfect ingredients.

### Learning Rate (LR)

**How big of a step the model takes when correcting its mistakes.**

After seeing each batch of examples, the model adjusts its internal numbers (weights) to reduce errors. The learning rate controls how much it adjusts each time.

- **Too high (LR = 0.01):** The model overcorrects wildly, never settling on a good answer. Like trying to park by flooring the accelerator.
- **Too low (LR = 1e-6):** The model barely changes, learning so slowly it never gets good. Like inching forward 1mm at a time.
- **Our sweet spot: LR = 3e-5 (0.00003).** Tiny adjustments, but fast enough to converge.

**Key finding:** Changing from 2e-5 to 3e-5 improved F1 by 4.3 points — the single biggest improvement in Sprint 3. The original LR was *too cautious*.

### Warmup Ratio

**Gradually increase the learning rate from 0 to the target over the first X% of training.**

Imagine you're warming up before exercise — you don't sprint immediately. The model does the same: it starts with near-zero LR, then linearly ramps up to the full LR.

- `warmup_ratio: 0.1` = spend the first 10% of training ramping up
- This prevents the model from making wild updates before it's "seen enough data" to know what's going on
- Especially important with higher learning rates

### Epochs

**How many times the model sees the entire training dataset.**

- 1 epoch = the model has read every training post once
- 10 epochs = it has read every post 10 times
- More epochs = more chances to learn, but too many = the model memorizes the training data instead of learning general patterns (overfitting)

Our v2 ran for 11 epochs (out of a planned 15) before early stopping kicked in.

### Batch Size

**How many examples the model looks at before making one update.**

The model doesn't update after every single post — it accumulates information from a *batch* of posts, then makes one averaged update. Like reading 8 essays before giving feedback, rather than commenting after every sentence.

- **batch_size: 8** = look at 8 posts at a time
- Bigger batches = more stable updates but need more memory
- We use batch_size=16 but it crashed (OOM), so we use 8

### Gradient Accumulation Steps

**A trick to simulate larger batches without needing more memory.**

Since batch_size=16 crashed our hardware, we instead:
1. Process 8 posts, remember the corrections needed (but don't apply them yet)
2. Process another 8 posts, accumulate those corrections too
3. *Now* apply all corrections at once

`gradient_accumulation_steps: 2` with `batch_size: 8` = effective batch of 16, but only 8 posts in memory at a time. Same learning behavior, half the memory.

### Weight Decay

**A gentle penalty that prevents any single parameter from becoming too large.**

Without it, some internal numbers in the model can grow huge and dominate predictions. Weight decay slowly shrinks all values toward zero, encouraging the model to spread importance across many parameters.

- `weight_decay: 0.01` = very light regularization
- Think of it as a "stay humble" rule for the model's internal values

### Early Stopping (Patience)

**Stop training when the model stops improving, even if there are epochs left.**

After each epoch, we check the validation F1. If it hasn't improved for `patience` consecutive epochs, we stop — continuing would just overfit.

- `patience: 5` = give the model 5 chances to improve before giving up
- Our v2 peaked at epoch 6, didn't improve for 5 more, stopped at epoch 11

**Why not just train forever?** After a point, more training makes the model memorize the training data ("the phone number in post #47 is 081-xxx") instead of learning general patterns ("phone numbers look like 0XX-XXX-XXXX"). Early stopping catches this.

---

## The Optimizer: AdamW

### What is an Optimizer?

The optimizer is the algorithm that decides *how* to adjust the model's weights after each batch. "Learning rate" says how *much* to adjust; the optimizer says *in which direction*.

### Why AdamW Specifically?

AdamW (Adam with Weight decay) is the standard for fine-tuning language models. It's smart about adjustments:

- **Keeps a running average** of past corrections, so it doesn't overreact to any single batch
- **Adapts per-parameter** — parameters that rarely get updated get bigger steps, frequently updated ones get smaller steps
- **Decouples weight decay** — applies the "stay humble" penalty separately from the gradient updates

You almost never change the optimizer. It's like the engine of the car — LR and epochs are the gas pedal and trip length.

### Optimizer States

AdamW stores 2 extra numbers for every model parameter (the running averages mentioned above). This is why optimizer memory is 2x the model size:

- Model: ~440MB
- Optimizer states: ~880MB (2 × 440MB)

This is a fixed cost you can't avoid with AdamW.

---

## The Data: Train / Val / Test

### Train Set

The data the model actually learns from. It sees these posts over and over (once per epoch). **606 posts** in our v2 run.

### Validation Set (Val)

Data the model **never trains on** but we evaluate after each epoch. This tells us if the model is actually learning *general* patterns or just memorizing training data.

- If train F1 goes up but val F1 goes down → overfitting
- We pick the best model based on the highest **val F1**

**76 posts** in our v2 run.

### Test Set

Data the model **never sees** until the very end. The final, unbiased grade. Neither training nor early stopping decisions are based on this — it's the "real exam" after all the practice tests (validation).

**76 posts** in our v2 run.

### Why Three Sets?

Because the validation set gets "used up" by influencing our decisions (when to stop training, which hyperparameters to pick). The test set provides a truly unbiased final score.

---

## The Losses

### Train Loss

**How wrong the model's predictions are on training data.** Starts high, should go down over epochs. This is the number the model is directly trying to minimize.

- Epoch 1: loss ~3.0 (very wrong)
- Epoch 10: loss ~0.3 (mostly right)

It's measured using *cross-entropy* — roughly, how surprised the model is by the correct answer. Lower = less surprised = better predictions.

### Eval Loss (Validation Loss)

Same measurement but on the validation set. If eval loss starts going *up* while train loss keeps going down, the model is memorizing training data (overfitting).

---

## Hardware Terms

### MPS (Metal Performance Shaders)

Apple's GPU computing framework. It lets us run PyTorch training on the Mac's GPU instead of the CPU. Think of it as "Apple's version of NVIDIA CUDA."

- Training on MPS: ~3-4 minutes
- Training on CPU would take: ~30+ minutes

### FP32 / FP16 / BF16

These describe the **precision** of numbers the model uses internally.

- **FP32 (32-bit floating point):** Full precision. Every number uses 32 bits of storage. Most accurate but uses the most memory.
- **FP16 (16-bit):** Half precision. Saves memory and can be faster, but some numbers become too small to represent (underflow). **Broken on MPS** — causes NaN (Not a Number) errors that corrupt training.
- **BF16 (brain float 16):** A compromise format that keeps the range of FP32 but with less precision. Not yet supported on MPS.

We force **FP32** because FP16 literally breaks on Apple Silicon. This costs us ~2x memory but guarantees correct training.

### UMA (Unified Memory Architecture)

Apple Silicon shares one pool of memory between CPU and GPU. Our Mac has 16GB total — that's shared between the OS, apps, *and* model training. This is why memory management matters so much.

### OOM (Out of Memory)

When training tries to use more memory than available. The program crashes. Our batch_size=16 caused OOM — that's why we dropped to 8.

### PYTORCH_MPS_HIGH_WATERMARK_RATIO

A safety valve that limits how much memory PyTorch can use on MPS.

- Set to 1.5 = PyTorch can use up to 1.5x its initial allocation before triggering cleanup
- Set to 0.0 = no limit (dangerous — can cause kernel panics / system freezes)
- Think of it as setting a credit limit on a credit card

---

## NER-Specific Terms

### IOB2 Format

The labeling scheme we use for Named Entity Recognition. Every token gets one of three prefixes:

- **B-** (Begin): First token of an entity. "B-LOCATION" = this is where a location name starts
- **I-** (Inside): Continuation of the same entity. "I-LOCATION" = this location name continues
- **O** (Outside): Not part of any entity. Most tokens are "O"

Example:
```
รับสมัคร   คน   ดูแล   ผู้สูงอายุ   ย่าน   สีลม
O          O    B-SKILL I-SKILL     O      B-LOC
```

### 15 Label Classes

We have 7 entity types × 2 prefixes (B and I) + 1 "O" class = 15 total labels:
```
O, B-HARD_SKILL, I-HARD_SKILL, B-PERSON, I-PERSON, B-LOCATION, I-LOCATION,
B-COMPENSATION, I-COMPENSATION, B-EMPLOYMENT_TERMS, I-EMPLOYMENT_TERMS,
B-CONTACT, I-CONTACT, B-DEMOGRAPHIC, I-DEMOGRAPHIC
```

### IGNORE_INDEX (-100)

A special label value meaning "don't grade this token." Used for:
- Special tokens like [CLS] and [SEP] (added by the tokenizer)
- Padding tokens (added to make all sequences the same length in a batch)

PyTorch's loss function automatically skips any token labeled -100.

### Seqeval (Strict Exact-Match)

The library we use to compute F1. "Strict exact-match" means:
- The predicted entity must have the **exact same start position, end position, AND label** as the gold answer
- Getting "สีล" instead of "สีลม" = wrong (boundary doesn't match)
- Getting LOCATION instead of PERSON for "สมชาย" = wrong (label doesn't match)

This is intentionally harsh — partial credit is not given.

### Confusion Matrix

A table showing **what the model predicted vs. what was correct**. Each row is a true label, each column is a predicted label. The diagonal (top-left to bottom-right) shows correct predictions. Off-diagonal cells show mistakes.

Our matrix revealed that the main error is **O → entity** (the model highlights text that shouldn't be highlighted), not entity-type confusion (e.g., calling a LOCATION a PERSON).

---

## Training Curve Vocabulary

### Convergence

When the model's scores stop improving meaningfully. The learning curve "flattens out." Our v2 converged around epoch 6 (val F1 peaked at 0.812).

### Plateau

A flat region in the learning curve where scores barely change despite continued training. Usually means the model has learned everything it can from this amount of data with these settings.

### Overfitting

When the model performs great on training data but poorly on new data. It has "memorized the test" rather than "learned the subject."

Signs: train loss keeps dropping but val loss starts rising.

Our models show minimal overfitting — the val/test gap is only ~2-3%.

### Underfitting

The opposite — the model hasn't learned enough. Scores are low on *both* training and validation. Usually caused by too little training (few epochs) or a learning rate that's too low.

Our v1 with LR=2e-5 was slightly underfitting — raising to 3e-5 fixed it.

---

## Reading the Reports: A Quick Cheat Sheet

When you see a training log, here's what to look at:

1. **Test F1** — the final grade. Higher is better. Our target is 0.85+
2. **Per-entity F1** — which entity types are strong vs. weak
3. **Precision vs Recall** — is the model too aggressive (low precision) or too cautious (low recall)?
4. **Val F1 curve** — did it converge? Peak early or late?
5. **Train loss** — is it going down? If it stops decreasing, something is wrong
6. **Training time** — how long did it take? (Ours: ~3-4 min per run)

Everything else (optimizer details, memory stats, warnings) is plumbing — important for debugging but not for understanding "how good is the model."
