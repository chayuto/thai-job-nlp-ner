# Chapter 3: Model Training, MPS Hardware & Production Deployment

In Chapters 1 and 2, we discussed the theory of NLP and how to use Data Engineering to programmatically create perfectly labeled IOB2 training data. 

Now, we have 500 perfectly labeled examples. It is time to train the model and deploy it to the internet!

---

## 1. What actually happens during "Fine-Tuning"?
When we download WangchanBERTa from Hugging Face, we are getting a mathematically pre-trained "brain" that understands the Thai language perfectly. However, the final output layer of that brain is empty. 

During training, we flow our 500 examples through the model. 
*   **Forward Pass:** The model looks at the sentence `"เงินเดือน 18000 บาท"` and blindly guesses that "18000" is `[O]` (Not an entity).
*   **Loss Calculation:** The computer checks the answer key (our IOB2 Tags) and sees that the answer was actually `[B-COMPENSATION]`. A mathematical error value (Loss) is calculated.
*   **Backpropagation:** The computer sends a signal *backwards* through the billions of connections inside the model, slightly adjusting the math weights so that next time, it is more likely to guess `[B-COMPENSATION]`.

We repeat this process (Epochs) until the Loss value gets as close to zero as possible. This usually takes about 10-15 minutes for 500 examples on a modern computer.

---

## 2. Training on Apple Silicon (MPS vs CUDA)
Historically, to train Machine Learning models, you had to rent expensive Nvidia GPUs from AWS or Google Cloud. Nvidia GPUs use a software layer called **CUDA**.

Recently, Apple completely changed the game by introducing **Unified Memory Architecture** in their M1/M2/M3 chips, and a software layer called **MPS (Metal Performance Shaders)**.

### Why MPS is a Big Deal:
A standard Nvidia PC has two separate pools of RAM: System RAM (32GB) and GPU Memory (8GB). Moving data back and forth between them over a PCIe cable is incredibly slow.
Your Mac has **Unified Memory**. The GPU and the CPU share the exact same pool of memory right on the chip.

### The PyTorch MPS Catch (Mixed Precision Underflow)
While PyTorch now natively supports Apple Silicon via `backend='mps'`, there is a famous bug ML Engineers must watch out for!
Normally, to speed up training on Nvidia, engineers use **FP16 (Half-Precision)** math. However, Apple's MPS architecture currently has issues with FP16 accumulating "Gradient Underflow" (The math numbers get so small they round down to exactly zero `0.000`!). This causes the model to break and output `NaN` errors.

**The Fix:** When writing our Python training script, we must explicitly write `fp16=False` in the Hugging Face `TrainingArguments` to force the Apple GPU to use standard FP32 math. The training will take slightly longer, but it will be mathematically safe.

---

## 3. Strict Evaluation (seqeval)
Once the model is trained, we have to prove to the world that it works. We hold out 50 examples that the model never saw during training (The Test Layer). 

We don't use simple "Accuracy." If a post has 95 normal words (`O`) and 5 entity words (`[B-SKILL]`, `[I-SKILL]`), a model that guesses `O` for every single word is still technically "95% accurate." That is terrible!

Instead, we use a library called **seqeval** which measures **Strict Exact Match F1-Score**.
A strict match means: If the ground truth is `[B-COMPENSATION, I-COMPENSATION]` ("18000", "บาท"), but the model only highlights the `[B-COMPENSATION]` ("18000"), **it fails the entire entity.** It must get the boundaries perfectly right.

---

## 4. The Microservice Deployment Architecture
You have a highly accurate, 110-million parameter Safetensors model file sitting on your Mac. How does your Next.js website actually use it?

You cannot upload a 500MB PyTorch model directly to Vercel/Next.js. 

Instead, we build a **Microservice**:
1.  **FastAPI Server:** We wrap our model in a 50-line Python script using a framework called FastAPI. This script turns our model into a standard HTTP URL (e.g., `https://my-thai-nlp.onrender.com/extract`).
2.  **Containerization (Docker):** We write a `Dockerfile` that packages Python, PyTorch, our FastAPI script, and our Model Weights into a single virtual box.
3.  **Hosting:** We host that Docker box on a cheap cloud backend like Render, Railway, or Google Cloud Run for roughly $5/month.
4.  **Supabase Triggers:** When a user types a job post on Happy Care Connect, Next.js saves it to Supabase. Supabase automatically sends an HTTP request behind the scenes to our `https://my-thai-nlp.../extract` URL.
5.  **The Result:** The Python server runs the text through the model, translates the `<HARD_SKILL>` tag back into your proprietary database schema (e.g., `blood_draw`), and returns it to Supabase. 

The user on the frontend never sees this happening and never waits for a loading screen!

---

## Final Review
You are now fully caught up on the domain!
You understand: 
1. Why we use Transformers/BERT.
2. How we automate creating data so we don't have to highlight by hand.
3. How IOB2 tagging works.
4. How to train on your Mac GPU safely without `NaN` errors.
5. How we deploy the PyTorch model without breaking your Next.js application.

Whenever you're ready, we can start writing the actual Python code for Chapter 2: The Data Engineering Pipeline!
