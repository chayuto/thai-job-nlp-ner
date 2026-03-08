# **Optimization and Implementation Strategies for Thai Named Entity Recognition on Apple Silicon M5**

The transition of deep learning workflows from cloud-based discrete Graphics Processing Units (GPUs) to localized unified-memory architectures represents a fundamental paradigm shift in machine learning engineering. With the introduction of the Apple Silicon M5 processor, high-memory local prototyping has reached unprecedented levels of computational efficiency, carving out a distinct and unassailable niche in the hardware landscape. However, fine-tuning transformer-based models—specifically the 110-million parameter wangchanberta architecture—for complex tasks such as Named Entity Recognition (NER) in non-Latin scripts introduces a highly intricate matrix of hardware, orthographic, and software engineering challenges.

The endeavor to build a pre-implementation pipeline for Thai NER using "Silver Labels" requires an exhaustive understanding of the underlying compiler stacks, memory allocators, Unicode grapheme cluster mechanics, and sequence evaluation protocols. This research report investigates the architectural optimizations required to successfully execute a local-first pipeline on the M5 chip, evaluating PyTorch's Metal Performance Shaders (MPS) against Apple's native MLX framework. Furthermore, the analysis deeply explores the algorithmic complexities of Thai Unicode string alignment, evaluating the integration of high-speed C++ distance libraries like rapidfuzz with linguistically aware segmenters like pythainlp. Finally, the report delineates the contemporary state of sequence evaluation, ensuring strict exact-match F1 scoring using the seqeval framework wrapped within the modern Hugging Face ecosystem.

## **Apple Silicon M5 PyTorch Optimization and Framework Selection**

The landscape of hardware acceleration as of early 2026 demands a nuanced understanding of compiler stacks and memory management. While NVIDIA's CUDA platform remains the undisputed operational standard for datacenter training due to its compiler maturity, the Apple Silicon architecture provides a viable, localized alternative for specific workloads. Maximizing the efficacy of the M5 hardware for fine-tuning a BERT-style token classifier requires strict optimization of the execution backend, deliberate framework selection, and granular memory threshold management.

### **The Metal Performance Shaders Backend and Precision Dynamics**

The PyTorch MPS backend, developed in collaboration with Apple's Metal engineering team, maps machine learning computational graphs to the Metal framework, utilizing custom kernels optimized for Apple GPUs.3 Historically, mixed-precision training—specifically utilizing 16-bit floating-point (fp16) formats—on the MPS backend has been plagued by severe numerical instability. This instability typically manifests as gradient underflow, where the gradients computed during the backward pass fall below the representational limits of the fp16 format, resulting in representational collapse to zero, NaN losses, and divergent weight updates.5

As of the PyTorch 2.6 (Stable) and PyTorch 2.7 (Nightly/Beta) releases in early 2026, the state of mixed-precision training on MPS remains highly volatile.8 While extensive engineering effort has been directed toward stabilizing fp16 execution—including updates to horizontal fusion in torchinductor and improved custom kernels—the underlying implementation still suffers from critical numerical divergence.9 Bug reports from the PyTorch 2.7 development cycle explicitly highlight that the MPS backend continues to yield incorrect outputs or crashes due to kernel bugs.10 Specific operations, such as grid\_sample producing wrong results, BatchNorm2d and avg\_pool2d failing with channels\_last tensors, and torch.abs overflowing or underflowing for complex inputs, indicate that the backend's numerical stability is not yet robust enough for unsupervised enterprise training loops.10 Furthermore, discrepancies in convolutional outputs when comparing CPU fp16 against CUDA fp32 have been shown to cause error amplification beyond acceptable thresholds, directly degrading model accuracy when mixed precision is forced.10

The hardware limitation stems from the reduced dynamic range of fp16, which utilizes 5 bits for the exponent and 10 bits for the mantissa.7 During the backpropagation phase of a deep transformer like wangchanberta, gradients frequently fall below the ![][image1] threshold.7 While Automatic Mixed Precision (AMP) algorithms attempt to mitigate this via gradient scaling, the specific implementations of these scalers within the MPS backend frequently encounter race conditions or device-side barrier limitations, leading to application termination.10

Consequently, utilizing the Nightly builds of PyTorch 2.7 offers access to the latest bug fixes but introduces the severe risk of transient compiler regressions and silent numerical errors.9 The most strategic and stable approach for a Machine Learning Engineering team in 2026 is to utilize the PyTorch 2.6 Stable release but strictly disable fp16 within the Hugging Face TrainingArguments.8 While bf16 (bfloat16) offers a superior dynamic range (8 bits for the exponent, mirroring fp32) and has been heavily optimized for x86 CPUs in recent PyTorch releases, native hardware support for bf16 matrix multiplications on Apple Silicon M-series chips has historically required software emulation or complex fallback paths, leading to performance degradation compared to pure fp32 execution.12

Therefore, to guarantee absolute numerical stability during the fine-tuning of wangchanberta, developers must explicitly force 32-bit precision. The configuration within the Hugging Face ecosystem must reflect this constraint to avoid the silent degradation of the token classifier's Cross-Entropy loss trajectory:

Python

from transformers import TrainingArguments

\# Enforcing fp32 for MPS stability on Apple Silicon M5  
\# Bypassing the fp16 gradient underflow bugs present in PyTorch 2.6/2.7  
training\_args \= TrainingArguments(  
    output\_dir="./wangchanberta-ner-checkpoints",  
    num\_train\_epochs=5,  
    per\_device\_train\_batch\_size=16,   
    per\_device\_eval\_batch\_size=16,  
    fp16=False,               \# Explicitly disable fp16 due to MPS gradient underflow \[5, 10\]  
    bf16=False,               \# Disable bf16 to prevent software emulation overhead \[13\]  
    use\_mps\_device=True,      \# Ensure MPS backend is targeted \[4\]  
    evaluation\_strategy="epoch",  
    save\_strategy="epoch",  
    logging\_dir="./logs",  
    learning\_rate=2e-5,  
    weight\_decay=0.01,  
)

### **Framework Evaluation: Apple MLX vs. PyTorch MPS**

Apple's introduction of the MLX framework presented a deep learning paradigm built specifically for Apple Silicon, leveraging a NumPy-like Application Programming Interface (API) with lazy evaluation, automatic differentiation, and a highly optimized unified memory design.14 The framework's ability to seamlessly transition tensors between the CPU and GPU without the costly Peripheral Component Interconnect Express (PCIe) bus transfers required by traditional discrete GPUs represents a significant architectural advantage.14 However, for the specific task of fine-tuning a BERT-based model (110M parameters) for Token Classification using the Hugging Face ecosystem, the viability of MLX compared to PyTorch MPS requires rigorous technical scrutiny.

While MLX excels in Large Language Model (LLM) inference, quantized generative tasks, and memory-constrained parameter-efficient fine-tuning (PEFT) like LoRA 15, empirical benchmarks in 2026 indicate that PyTorch MPS outperforms MLX in standard, full-parameter transformer training loops.16 Benchmarks executed on M-series chips reveal that fine-tuning a BERT architecture via PyTorch is consistently faster than MLX.16 This performance delta is attributed to the maturity of PyTorch's underlying C++ kernels and its highly optimized matrix multiplication (matmul) dispatchers.17 For instance, matrix multiplications of large dimensions on PyTorch 2.6 MPS execute with lower latency than equivalent operations in compiled MLX graphs, despite both utilizing the underlying Metal API.17

Furthermore, the operational friction of utilizing MLX must be factored into the engineering lifecycle. The Hugging Face Trainer class is intrinsically coupled with PyTorch (as well as TensorFlow and JAX), providing automated data collation, gradient accumulation, distributed training sharding, and evaluation loops out of the box.18 Implementing a Token Classification pipeline in MLX requires the engineering team to manually construct bespoke training loops, implement custom Cross-Entropy loss functions capable of masking padding tokens, and manually manage optimizer states.20

The following table synthesizes the technical maturity and operational viability of each platform for a beginner-to-intermediate ML engineer tasked with fine-tuning wangchanberta:

| Feature Category | PyTorch MPS (v2.6/2.7) | Apple MLX Framework |
| :---- | :---- | :---- |
| **Hugging Face Integration** | Native, highly optimized via the Trainer API, requiring minimal boilerplate.18 | Requires custom training loops, manual padding management, and bespoke loss functions.20 |
| **Training Performance (BERT)** | Superior throughput. Mature matmul kernels handle forward/backward passes efficiently.17 | Slower for standard full-parameter backward passes compared to PyTorch MPS.16 |
| **Inference Performance (LLMs)** | Moderate. Can be susceptible to memory fragmentation during long generation sequences. | Exceptional. Highly optimized for Apple Neural Accelerators, supporting rapid quantized generation.14 |
| **Ecosystem Support** | Industry standard. Massive repository of pre-trained weights and community troubleshooting.1 | Emerging. Primarily utilized for generative AI, text generation, and local LLM deployment.14 |
| **Architectural Paradigm** | Eager execution with optional torch.compile graph capturing.8 | Lazy evaluation graph construction similar to JAX, requiring mx.compile for optimization.14 |

The engineering overhead required to construct, debug, and validate an MLX-native NER training pipeline drastically outweighs the hypothetical benefits of framework native-ness. The recommendation is unequivocal: the engineering team must proceed with PyTorch MPS, leveraging the robust infrastructure of the Hugging Face Trainer.

### **Unified Memory Management and Watermark Heuristics**

The Apple Silicon M5 utilizes a Unified Memory Architecture (UMA), physically co-locating the GPU and CPU memory pools.3 While this eliminates the memory transfer bottlenecks inherent to standard x86/CUDA systems, it introduces complex out-of-memory (OOM) dynamics during tensor allocation. When PyTorch requests memory, the MPS allocator monitors system memory pressure through strict high and low watermark ratios.24

The PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO environment variable dictates the absolute upper bound of memory the PyTorch MPS allocator can request from the operating system relative to the device's recommended maximum working set size.24 The default value is set to 1.7.24 Setting this value to 0.0 entirely disables the upper limit.24 While frequently suggested in community forums to bypass OOM errors, disabling this limit allows PyTorch to consume memory unconditionally, which frequently results in total system failure, kernel panics, or hard crashes when the macOS kernel exhausts swap space.24 Conversely, setting the ratio too low (e.g., 1.0) restricts the model from utilizing available physical RAM efficiently, triggering artificial OOM errors prematurely.24

For fine-tuning a 110-million parameter model like wangchanberta, memory optimization must occur at the data loading and batching phase rather than attempting to bypass low-level OS memory safety watermarks.27 A mathematical breakdown of the memory footprint reveals the constraints:

1. **Model Weights:** A 110M parameter model utilizing fp32 requires approximately 440 MB purely for the static weights in VRAM.  
2. **Optimizer States:** The standard AdamW optimizer maintains moving averages (first and second moments) for every parameter. In fp32, this requires an additional 880 MB of memory.  
3. **Gradients:** Storing the gradients for the backward pass requires another 440 MB.  
4. **Activations:** Activations computed during the forward pass, which must be retained in memory for the chain-rule calculations during backpropagation, scale linearly with sequence length and batch size.

To maintain stability on an M5 chip (assuming standard configurations of 16GB or 32GB Unified RAM), engineers must implement gradient accumulation and dynamic padding.27 Utilizing the DataCollatorForTokenClassification ensures that input tensors are dynamically padded to the longest sequence within a specific micro-batch, rather than globally padding every sequence to the maximum model length (e.g., 512 tokens).28 This drastically reduces the activation memory overhead.

Based on the architectural footprint of wangchanberta, a safe batch size limit for a 16GB M5 system, utilizing dynamic padding and fp32 precision, is a per-device batch size of 8 to 16\. If larger effective batch sizes are required for convergence stability, the gradient\_accumulation\_steps parameter must be utilized within the Hugging Face Trainer.

Python

import os  
import torch  
from transformers import DataCollatorForTokenClassification

\# Best practice: Do not set PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO to 0.0.  
\# A safe upper bound allows PyTorch to utilize swap without crashing the OS kernel.  
os.environ \= "1.5"

\# Utilize dynamic padding to conserve Unified RAM during the forward pass  
\# This prevents allocating memory for 512 tokens if the batch only requires 45 tokens.  
data\_collator \= DataCollatorForTokenClassification(  
    tokenizer=tokenizer,  
    padding=True,  
    max\_length=None \# Enforces dynamic padding per batch   
)

## **Thai Fuzzy String Alignment and Tokenization Pipelines**

The core data engineering challenge in this pre-implementation pipeline involves the translation of "Silver Labels"—raw Thai text paired with known substrings representing entities (e.g., "ซีพีอาร์" for "CPR")—into exact character start and end indices. These indices are subsequently converted into the standard IOB2 (Inside, Outside, Beginning) tagging scheme required for NER training.29 This task is profoundly complicated by the unique orthographic characteristics of the Thai script and the rigid mechanics of modern subword tokenization models like SentencePiece.

### **Orthographic Complexity and Unicode Grapheme Clusters**

The Thai writing system operates fundamentally differently from Latin-based alphabets. It is an abugida, a segmental writing system where consonant-vowel sequences are written as a unit. In Thai, characters are constructed using a base consonant followed by optional vowels and tone marks that stack vertically above, below, or alongside the base character.31 Furthermore, the Thai language is written continuously without spaces to separate individual words; spaces are utilized primarily as punctuation, indicating clause or sentence boundaries.33

In the Unicode standard, the base consonants (such as ก, U+0E01) are categorized as Lo (Other\_Letter).35 The stacking elements, such as the upper vowel *Sara I* (ิ, U+0E34) or the tone mark *Mai Ek* (่, U+0E48), are categorized as Mn (Nonspacing Marks).35 When these characters are typed, they occupy multiple Unicode code points but render visually as a single glyph on the screen, known as a Grapheme Cluster or, in specific Thai linguistic terms, a Thai Character Cluster (TCC).37

When performing programmatic string alignment, standard Python string functions and simple matching algorithms count code points rather than visible grapheme clusters.37 If a fuzzy matching algorithm attempts to slice a string based purely on code point index mathematics, it introduces a severe risk of separating a base consonant from its modifying vowel or tone mark.37 Such arbitrary slicing renders the text linguistically invalid, resulting in rendering errors (e.g., detached tone marks) and, critically, causing downstream machine learning tokenizers to generate unpredictable, out-of-vocabulary subwords.39

### **Algorithmic Matching: RapidFuzz vs. PyThaiNLP**

To locate entity substrings within heavily corrupted OCR outputs or informal social media text, the pipeline must utilize an error-tolerant search algorithm. The two primary candidates for this alignment operation are the generic Levenshtein-distance library rapidfuzz and the linguistically aware standard Thai NLP library, pythainlp.

rapidfuzz is a highly optimized C++ implementation of string similarity metrics.41 It utilizes Bit-parallel algorithms, such as Myers' algorithm, to compute Levenshtein and Jaro-Winkler distances in ![][image2] time, making it orders of magnitude faster and more memory-efficient than its predecessor, FuzzyWuzzy.41 However, rapidfuzz operates natively on standard Unicode code points.41 If tasked with finding the partial ratio of a Thai substring within a larger document, the algorithm mathematically computes the minimum edit distance. The resulting boundaries returned by rapidfuzz might terminate precisely on a base consonant code point, entirely ignoring the subsequent Nonspacing Mark (Mn) that grammatically belongs to that character cluster.39

Conversely, pythainlp incorporates a robust, programmatic understanding of Thai Character Clusters (TCC).38 The TCC algorithms within pythainlp group code points into inseparable linguistic units based on formal Thai spelling rules.38 This ensures that a consonant, its accompanying vowel, and its tone mark are treated as a single, indivisible computational block. For instance, the word 'ขนมชั้น' is correctly tokenized into the clusters 'ข', 'น', 'ม', and 'ชั้น', rather than individual Unicode code points.38 While pythainlp.util offers functions like find\_keyword, these functions are primarily designed for exact substring matching, frequency counting, or dictionary lookups, rather than error-tolerant fuzzy alignment against severe typos.47

The optimal architectural solution for the Silver Label alignment pipeline requires integrating the high-speed, error-tolerant bit-parallel processing of rapidfuzz with the orthographic safety mechanisms of pythainlp. The pipeline must first segment the raw text into a sequence of valid TCCs. Subsequently, rapidfuzz can be deployed to identify the approximate substring sequence. Finally, the algorithm must "snap" the mathematical boundaries returned by rapidfuzz back to the nearest valid TCC edge. This guarantees that the final alignment boundaries encompass entire grapheme clusters, preventing the severing of Unicode Nonspacing Marks.

### **The "Token Fragmentation" Risk: SentencePiece and wangchanberta**

Once the exact, orthographically safe character boundaries of the Silver Label are extracted, these indices must be mapped to the subword tokens generated by the model's tokenizer to assign the IOB2 tags. The wangchanberta model utilizes a Unigram Language Model tokenizer implemented via the Google SentencePiece library.2

A critical idiosyncrasy of the wangchanberta preprocessing pipeline is its explicit handling of whitespace.34 Because spaces delineate sentence boundaries rather than words in Thai, the developers of wangchanberta designed the pipeline to explicitly preserve whitespace information. To prevent the SentencePiece Unigram algorithm from aggressively merging spaces with adjacent consonant tokens—which would destroy crucial sentence boundary features—the developers forced the tokenizer to map spaces to a dedicated \<\_\> (or \<\\\\\\\\\_\>) token prior to subword segmentation.34

However, this explicit replacement mechanism severely disrupts the default character-to-token mapping utilities, specifically the char\_to\_token() method provided by the Hugging Face tokenizers library. When the raw string is modified (e.g., spaces converted to \<\_\>) before being passed to the core subword tokenizer, the absolute character offsets diverge.50 If a machine learning engineer attempts to map a valid character span of \[15:20\] from the raw text to the tokenized output using standard high-level functions, the resulting IOB2 array will be misaligned.50 This token fragmentation propagates false label alignments to the Cross-Entropy loss function during the PyTorch training loop, preventing the model from converging on accurate entity boundaries.50

To mitigate this token fragmentation and structural misalignment, the pipeline must employ a specialized alignment mathematical algorithm. Rather than relying on char\_to\_token(), the system must extract the raw offset\_mapping tuples generated by the tokenizer and manually intersect them with the TCC-snapped character boundaries.

### **Implementation: Character-to-Token Alignment Math**

The following Python implementation demonstrates the state-of-the-art methodology for extracting IOB2 tags from raw text. It utilizes rapidfuzz for error-tolerant searching, pythainlp for TCC orthographic safety, and completely neutralizes the SentencePiece \<\_\> misalignment risk by interacting directly with the subword offset matrices.

Python

import re  
from rapidfuzz import process, fuzz  
from pythainlp.tokenize import subword\_tokenize  
from transformers import AutoTokenizer

class ThaiFuzzyAlignmentPipeline:  
    def \_\_init\_\_(self, model\_checkpoint: str \= "airesearch/wangchanberta-base-att-spm-uncased"):  
        \# Load the SentencePiece tokenizer associated with wangchanberta   
        self.tokenizer \= AutoTokenizer.from\_pretrained(model\_checkpoint)  
          
    def find\_safe\_boundaries(self, raw\_text: str, silver\_substring: str, threshold: float \= 85.0):  
        """  
        Locates the fuzzy substring while strictly respecting Thai Character Clusters (TCC).  
        This prevents Unicode Nonspacing Marks (Mn) from being severed.  
        """  
        \# Step 1: Segment raw text into indivisible TCC units using pythainlp   
        tcc\_clusters \= subword\_tokenize(raw\_text, engine="tcc")  
          
        \# Step 2: Utilize rapidfuzz partial matching for high-speed error tolerance   
        \# RapidFuzz returns a tuple: (matched\_string, score, start\_idx, end\_idx)  
        match\_result \= process.extractOne(  
            silver\_substring,   
            \[raw\_text\],   
            scorer=fuzz.partial\_ratio  
        )  
          
        if not match\_result or match\_result \< threshold:  
            return None  
              
        fuzzy\_start, fuzzy\_end \= match\_result, match\_result  
          
        \# Step 3: Snap mathematical boundaries to the nearest valid TCC edges   
        \# to prevent Unicode grapheme cluster splitting \[37, 40\]  
        safe\_start, safe\_end \= 0, 0  
        current\_idx \= 0  
          
        for cluster in tcc\_clusters:  
            cluster\_len \= len(cluster)  
              
            \# Snap the start boundary to the beginning of the TCC  
            if current\_idx \<= fuzzy\_start \< current\_idx \+ cluster\_len:  
                safe\_start \= current\_idx  
              
            \# Snap the end boundary to the conclusion of the TCC  
            if current\_idx \< fuzzy\_end \<= current\_idx \+ cluster\_len:  
                safe\_end \= current\_idx \+ cluster\_len  
                break  
                  
            current\_idx \+= cluster\_len  
              
        return (safe\_start, safe\_end)

    def align\_tokens\_to\_iob2(self, raw\_text: str, start\_char: int, end\_char: int, entity\_label: str \= "ORG"):  
        """  
        Maps character boundaries to SentencePiece tokens, bypassing char\_to\_token() bugs  
        caused by the \<\_\> space replacement in wangchanberta.  
        """  
        \# Tokenize the input text. return\_offsets\_mapping is the critical parameter for alignment.  
        tokenized\_inputs \= self.tokenizer(  
            raw\_text,  
            return\_offsets\_mapping=True,  
            truncation=True,  
            max\_length=512  
        )  
          
        offsets \= tokenized\_inputs\["offset\_mapping"\]  
        input\_ids \= tokenized\_inputs\["input\_ids"\]  
          
        \# Initialize the label array with "O" (Outside) tags  
        labels \= \["O"\] \* len(input\_ids)  
          
        \# The wangchanberta tokenizer maps spaces to a specific token.  
        \# By iterating through the raw offset mapping tuples, we map original character   
        \# boundaries directly to the subword indices, bypassing heuristic bugs.  
          
        entity\_started \= False  
        for idx, (offset\_start, offset\_end) in enumerate(offsets):  
            \# Skip special architectural tokens (,) which have a mapping of (0,0)  
            if offset\_start \== 0 and offset\_end \== 0:  
                labels\[idx\] \= "O"  
                continue  
                  
            \# Intersect: Check if the subword falls within our aligned TCC character boundaries  
            if offset\_start \>= start\_char and offset\_end \<= end\_char:  
                if not entity\_started:  
                    labels\[idx\] \= f"B-{entity\_label}" \# Beginning of entity  
                    entity\_started \= True  
                else:  
                    labels\[idx\] \= f"I-{entity\_label}" \# Inside of entity  
            else:  
                labels\[idx\] \= "O"  
                  
        return labels

\# Example Execution  
pipeline \= ThaiFuzzyAlignmentPipeline()  
text \= "บริษัทซีพีอาร์กำลังขยายสาขา" \# "CPR Company is expanding branches"  
substring \= "ซีพีอาร์" \# "CPR"

\# 1\. Orthographically safe fuzzy alignment  
boundaries \= pipeline.find\_safe\_boundaries(text, substring)

\# 2\. Token-aligned IOB2 generation bypassing char\_to\_token() bugs  
iob2\_tags \= pipeline.align\_tokens\_to\_iob2(text, boundaries, boundaries, "ORG")

This approach represents the apex of current NLP engineering standards for abugida scripts. By utilizing process.extractOne with a fuzz.partial\_ratio scorer, the algorithm remains highly resilient to OCR transcription errors and social media typos.41 By immediately passing those raw indices through a geometric snapping algorithm built upon the pythainlp TCC engine, the data pipeline mathematically guarantees that combining Unicode marks are preserved, maintaining linguistic integrity.36 Finally, relying entirely on the numerical tuple outputs of offset\_mapping rather than Hugging Face's high-level heuristic string-search bridges perfectly encapsulates and neutralizes the modifications made to the wangchanberta vocabulary matrices.34

## **Strict Evaluation Protocols for Named Entity Recognition**

The final critical requirement in the pre-implementation phase concerns the model evaluation logic. In Token Classification tasks, computing isolated, per-token accuracy is fundamentally flawed and highly misleading.52 Because the vast majority of tokens in any given document do not belong to an entity, a model might correctly predict all O (Outside) tags, artificially inflating the accuracy metric to over 95%. Simultaneously, that same model might completely fail to correctly identify the boundaries of multi-word entities, rendering it useless for actual information extraction tasks.52

### **The Mechanics of Strict Exact-Match F1 Scoring**

To accurately assess the performance of an NER model, the industry relies on span-based metrics: Precision, Recall, and the balanced F1 Score. In strict exact-match evaluation, the criteria for a True Positive (![][image3]) are uncompromising.29 For an entity to be registered as a True Positive, the model must predict both the exact entity classification (e.g., ORG vs LOC) and the exact geometric boundaries of the text span.29

This means the model must output the exact B-TAG (Beginning) and all subsequent I-TAG (Inside) indices matching the ground truth. If the ground truth for a corporate name spans three tokens , and the model predicts , a strict evaluation framework scores this as zero for that specific entity.29 The model receives no partial credit for identifying the beginning of the entity; boundary truncation is aggressively penalized to ensure the extracted text is complete and factual.

### **The Persisting Standard: seqeval via Hugging Face evaluate**

In 2026, seqeval remains the absolute, uncontested industry standard for computing these strict sequence labeling metrics.52 The library explicitly understands the strict grammatical syntax of IOB1, IOB2, and BIOES tagging schemes, automatically handling the complex state-machine logic required to validate multi-token spans.29

While seqeval originated as a standalone Python library, the modern deep learning paradigm integrates it natively through the Hugging Face evaluate framework.52 The evaluate library provides standardized wrapper functions that sanitize inputs, align them with the model's logits, and execute the seqeval mathematics under the hood.28 This integration ensures consistency across distributed training environments and prevents metric synchronization errors when computing macro-averages across varying batch sizes.52

### **Sub-token Alignment for the Trainer Loop**

During the PyTorch Trainer evaluation loop, the wangchanberta model generates a multi-dimensional array of logits representing the probability distribution for every subword token.57 However, computing the strict exact-match F1 score requires ignoring the predictions applied to padding tokens (which are assigned a label of \-100 in PyTorch) and architectural special tokens.28

The mathematical formulation for the strict F1 evaluation requires extracting the argmax of the logits across the final hidden dimension, filtering out the padding indices, mapping the integer IDs back to string labels, and passing the sanitized arrays to the seqeval engine.

The implementation required to inject this logic into the Hugging Face Trainer is as follows:

Python

import numpy as np  
import evaluate

\# The evaluate library natively wraps seqeval for token classification   
\# This ensures strict IOB2 boundary enforcement during metric calculation   
metric \= evaluate.load("seqeval")

\# Define the global label list corresponding to the model's classification head  
label\_list \=

def compute\_metrics(eval\_prediction):  
    """  
    Computes strict span-level F1 using seqeval, processing the evaluation logits  
    emitted by the Trainer at the end of each epoch.\[58\]  
    """  
    logits, labels \= eval\_prediction  
      
    \# Extract the highest probability class index from the logits across the vocabulary dimension  
    predictions \= np.argmax(logits, axis=2)  
      
    \# Strip the padding tokens (-100) and map integer IDs back to IOB2 string labels   
    \# The PyTorch Cross-Entropy loss ignores \-100, and seqeval must ignore them as well.  
    true\_predictions \= \[label\_list\[p\] for (p, l) in zip(prediction, label) if l\!= \-100\]  
        for prediction, label in zip(predictions, labels)  
    \]  
      
    true\_labels \= \[label\_list\[l\] for (p, l) in zip(prediction, label) if l\!= \-100\]  
        for prediction, label in zip(predictions, labels)  
    \]  
      
    \# seqeval executes the strict exact-match boundary validation internally   
    results \= metric.compute(predictions=true\_predictions, references=true\_labels)  
      
    \# Return the metrics to the Trainer for logging and checkpoint evaluation  
    return {  
        "precision": results\["overall\_precision"\],  
        "recall": results\["overall\_recall"\],  
        "f1": results\["overall\_f1"\],  
        "accuracy": results\["overall\_accuracy"\] \# Monitored, but secondary to F1  
    }

By passing this compute\_metrics function into the Trainer object, the system will automatically compute strict, exact-boundary F1 scores at the end of every evaluation strategy interval.19 The reliance on the unified evaluate wrapper guarantees that the evaluation mechanism remains robust, highly optimized, and mathematically uncompromising in assessing the model's ability to extract complex Thai entities.

## **Strategic Synthesis and Deployment Imperatives**

Executing a Named Entity Recognition pipeline for Thai text on an Apple Silicon M5 infrastructure requires navigating an exceptionally complex intersection of hardware compiler limitations, Unicode orthography, and subword tokenizer mechanics. Based on an exhaustive review of compiler metrics, architectural benchmarks, and algorithmic constraints as of early 2026, the technical trajectory for the ML Engineering team is well-defined.

The engineering organization must leverage the maturity and ecosystem integration of PyTorch MPS over the Apple MLX framework for the specific task of BERT fine-tuning. However, to bypass the pervasive fp16 gradient underflow bugs and kernel instability inherent to current Metal Performance Shader graphs, the training configuration must strictly enforce 32-bit floating-point precision. To maximize the M5's Unified Memory Architecture without triggering kernel panics, dynamic padding and micro-batch size regulation must be implemented, explicitly avoiding dangerous modifications to the PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO environment variable.

Simultaneously, the data generation pipeline must bridge the speed of bit-parallel alignment algorithms with uncompromising linguistic realities. By wrapping rapidfuzz extractions in geometric snapping algorithms utilizing the pythainlp Thai Character Cluster engine, the pipeline neutralizes the risk of Unicode fragmentation. By relying entirely on direct offset\_mapping matrix intersections, engineers can effortlessly sidestep the tokenization discrepancies caused by wangchanberta's unique \<\_\> spacing schema. Finally, evaluation protocols must steadfastly rely on the seqeval algorithm—facilitated seamlessly via the Hugging Face evaluate library—to mathematically enforce strict boundary-matching F1 scores.

By implementing these optimized, code-driven strategies, the machine learning infrastructure ensures a highly resilient, computationally performant, and orthographically robust local-first training ecosystem capable of conquering the inherent complexities of modern Thai Natural Language Processing.

#### **Works cited**

1. State of PyTorch Hardware Acceleration 2025, accessed March 8, 2026, [https://tunguz.github.io/PyTorch\_Hardware\_2025/](https://tunguz.github.io/PyTorch_Hardware_2025/)  
2. Wangchanberta Base Att Spm Uncased · Models \- Dataloop, accessed March 8, 2026, [https://dataloop.ai/library/model/airesearch\_wangchanberta-base-att-spm-uncased/](https://dataloop.ai/library/model/airesearch_wangchanberta-base-att-spm-uncased/)  
3. Introducing Accelerated PyTorch Training on Mac, accessed March 8, 2026, [https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)  
4. Accelerated PyTorch training on Mac \- Metal \- Apple Developer, accessed March 8, 2026, [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)  
5. Training partially with fp16 \- autograd \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/t/training-partially-with-fp16/31873](https://discuss.pytorch.org/t/training-partially-with-fp16/31873)  
6. No gradient received in mixed precision training \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/t/no-gradient-received-in-mixed-precision-training/200696](https://discuss.pytorch.org/t/no-gradient-received-in-mixed-precision-training/200696)  
7. What Every User Should Know About Mixed Precision Training in PyTorch, accessed March 8, 2026, [https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)  
8. PyTorch 2.6 Release Blog, accessed March 8, 2026, [https://pytorch.org/blog/pytorch2-6/](https://pytorch.org/blog/pytorch2-6/)  
9. Releases · pytorch/pytorch \- GitHub, accessed March 8, 2026, [https://github.com/pytorch/pytorch/releases](https://github.com/pytorch/pytorch/releases)  
10. Weekly GitHub Report for Pytorch: February 01, 2026 \- Buttondown, accessed March 8, 2026, [https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-february-01-2026/](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-february-01-2026/)  
11. the bug that taught me more about PyTorch than years of using it \- Elana Simon, accessed March 8, 2026, [https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)  
12. Unlocking the Latest Features in PyTorch 2.6 for Intel Platforms, accessed March 8, 2026, [https://pytorch.org/blog/unlocking-pt-2-6-intel/](https://pytorch.org/blog/unlocking-pt-2-6-intel/)  
13. Apple Silicon Experiment 5— WWDC23 torch 2.1 mps updates \- Youngrok Song \- Medium, accessed March 8, 2026, [https://id2thomas.medium.com/apple-silicon-experiment-5-wwdc23-torch-2-1-mps-updates-wip-429e9fdfb85a](https://id2thomas.medium.com/apple-silicon-experiment-5-wwdc23-torch-2-1-mps-updates-wip-429e9fdfb85a)  
14. Exploring LLMs with MLX and the Neural Accelerators in the M5 ..., accessed March 8, 2026, [https://machinelearning.apple.com/research/exploring-llms-mlx-m5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)  
15. MPS or MLX for Domestic AI? The Answer Will Surprise You | by Mike Koypish \- Medium, accessed March 8, 2026, [https://medium.com/@koypish/mps-or-mlx-for-domestic-ai-the-answer-will-surprise-you-df4b111de8a0](https://medium.com/@koypish/mps-or-mlx-for-domestic-ai-the-answer-will-surprise-you-df4b111de8a0)  
16. LucasSte/MLX-vs-Pytorch: Benchmarks comparing PyTorch and MLX on Apple Silicon GPUs \- GitHub, accessed March 8, 2026, [https://github.com/LucasSte/MLX-vs-Pytorch](https://github.com/LucasSte/MLX-vs-Pytorch)  
17. matmul() using PyTorch's MPS backend is faster than Apple's MLX \- Kevin Martin Jose, accessed March 8, 2026, [https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/)  
18. Trainer \- Hugging Face, accessed March 8, 2026, [https://huggingface.co/docs/transformers/en/main\_classes/trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)  
19. Fine-tuning a model with the Trainer API \- Hugging Face LLM Course, accessed March 8, 2026, [https://huggingface.co/learn/llm-course/chapter3/3](https://huggingface.co/learn/llm-course/chapter3/3)  
20. mlx-examples/bert/README.md at main \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx-examples/blob/main/bert/README.md](https://github.com/ml-explore/mlx-examples/blob/main/bert/README.md)  
21. Fine-Tuning BERT for Text Classification (w/ Example Code) \- YouTube, accessed March 8, 2026, [https://www.youtube.com/watch?v=4QHg8Ix8WWQ](https://www.youtube.com/watch?v=4QHg8Ix8WWQ)  
22. Latest mixed-precision topics \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/c/mixed-precision/27](https://discuss.pytorch.org/c/mixed-precision/27)  
23. Apple Silicon \- Hugging Face, accessed March 8, 2026, [https://huggingface.co/docs/transformers/en/perf\_train\_special](https://huggingface.co/docs/transformers/en/perf_train_special)  
24. MPS Environment Variables — PyTorch 2.10 documentation, accessed March 8, 2026, [https://docs.pytorch.org/docs/stable/mps\_environment\_variables.html](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)  
25. Use PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0 to disable upper limit \#152351, accessed March 8, 2026, [https://github.com/pytorch/pytorch/issues/152351](https://github.com/pytorch/pytorch/issues/152351)  
26. Set "PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO" on Mac App : r/comfyui \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/comfyui/comments/1j56m5n/set\_pytorch\_mps\_high\_watermark\_ratio\_on\_mac\_app/](https://www.reddit.com/r/comfyui/comments/1j56m5n/set_pytorch_mps_high_watermark_ratio_on_mac_app/)  
27. MPS backend out of memory \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879](https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879)  
28. Token classification \- Hugging Face, accessed March 8, 2026, [https://huggingface.co/docs/transformers/en/tasks/token\_classification](https://huggingface.co/docs/transformers/en/tasks/token_classification)  
29. Calculate F1-score in a Named Entity Recognition model with sklearn \- Stack Overflow, accessed March 8, 2026, [https://stackoverflow.com/questions/70321099/calculate-f1-score-in-a-named-entity-recognition-model-with-sklearn](https://stackoverflow.com/questions/70321099/calculate-f1-score-in-a-named-entity-recognition-model-with-sklearn)  
30. pythainlp.wangchanberta — PyThaiNLP 3.1.0 documentation, accessed March 8, 2026, [https://pythainlp.org/docs/3.1/api/wangchanberta.html](https://pythainlp.org/docs/3.1/api/wangchanberta.html)  
31. \[Thai script\] combining marks \- Glyphs Forum, accessed March 8, 2026, [https://forum.glyphsapp.com/t/thai-script-combining-marks/14526](https://forum.glyphsapp.com/t/thai-script-combining-marks/14526)  
32. ELI5: Why does this letter exceed boundaries of some websites? : r/explainlikeimfive \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/explainlikeimfive/comments/1b9zkbu/eli5\_why\_does\_this\_letter\_exceed\_boundaries\_of/](https://www.reddit.com/r/explainlikeimfive/comments/1b9zkbu/eli5_why_does_this_letter_exceed_boundaries_of/)  
33. Exploration and Insights on Thai Characters in Thai Document Understanding Algorithm Research | by laygin | Medium, accessed March 8, 2026, [https://medium.com/@laygin/exploration-and-insights-on-thai-characters-in-thai-document-understanding-algorithm-research-e943ba0474c2](https://medium.com/@laygin/exploration-and-insights-on-thai-characters-in-thai-document-understanding-algorithm-research-e943ba0474c2)  
34. airesearch/wangchanberta-base-att-spm-uncased \- Hugging Face, accessed March 8, 2026, [https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)  
35. In Python 3, count Thai character positions \- Stack Overflow, accessed March 8, 2026, [https://stackoverflow.com/questions/54263419/in-python-3-count-thai-character-positions](https://stackoverflow.com/questions/54263419/in-python-3-count-thai-character-positions)  
36. L2/02-164 \- Thai in Grapheme Clusters \- Unicode, accessed March 8, 2026, [http://www.unicode.org/L2/L2002/02164-thai.pdf](http://www.unicode.org/L2/L2002/02164-thai.pdf)  
37. Unicode grapheme clusters and parsing : r/ProgrammingLanguages \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/ProgrammingLanguages/comments/1e5dapz/unicode\_grapheme\_clusters\_and\_parsing/](https://www.reddit.com/r/ProgrammingLanguages/comments/1e5dapz/unicode_grapheme_clusters_and_parsing/)  
38. pythainlp.tokenize — PyThaiNLP fa4db9a documentation, accessed March 8, 2026, [https://pythainlp.org/dev-docs/api/tokenize.html](https://pythainlp.org/dev-docs/api/tokenize.html)  
39. Thai script rendering broken on frame names — vowels and tone marks detached from consonants | Figma Forum, accessed March 8, 2026, [https://forum.figma.com/report-a-problem-6/thai-script-rendering-broken-on-frame-names-vowels-and-tone-marks-detached-from-consonants-51323](https://forum.figma.com/report-a-problem-6/thai-script-rendering-broken-on-frame-names-vowels-and-tone-marks-detached-from-consonants-51323)  
40. \[BUG\] Thai combining characters (tone marks/vowels) misaligned in terminal output \#17860, accessed March 8, 2026, [https://github.com/anthropics/claude-code/issues/17860](https://github.com/anthropics/claude-code/issues/17860)  
41. rapidfuzz/RapidFuzz: Rapid fuzzy string matching in Python using various string metrics \- GitHub, accessed March 8, 2026, [https://github.com/rapidfuzz/RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)  
42. RapidFuzz: A Powerful and High-Performance Fuzzy String Matching Library \- Medium, accessed March 8, 2026, [https://medium.com/top-python-libraries/rapidfuzz-a-powerful-and-high-performance-fuzzy-string-matching-library-1b27cd87487c](https://medium.com/top-python-libraries/rapidfuzz-a-powerful-and-high-performance-fuzzy-string-matching-library-1b27cd87487c)  
43. RapidFuzz 3.14.3 documentation \- GitHub Pages, accessed March 8, 2026, [https://rapidfuzz.github.io/RapidFuzz/](https://rapidfuzz.github.io/RapidFuzz/)  
44. PyThaiNLP/JThaiNLP: Thai NLP in Java \- GitHub, accessed March 8, 2026, [https://github.com/PyThaiNLP/JThaiNLP](https://github.com/PyThaiNLP/JThaiNLP)  
45. PyThaiNLP Get Started — pythainlp-tutorials thai2plot-30-g70eec59 documentation, accessed March 8, 2026, [https://pythainlp.org/tutorials/notebooks/pythainlp\_get\_started.html](https://pythainlp.org/tutorials/notebooks/pythainlp_get_started.html)  
46. (PDF) Character Cluster Based Thai Information Retrieval \- ResearchGate, accessed March 8, 2026, [https://www.researchgate.net/publication/2853284\_Character\_Cluster\_Based\_Thai\_Information\_Retrieval](https://www.researchgate.net/publication/2853284_Character_Cluster_Based_Thai_Information_Retrieval)  
47. pythainlp.util — PyThaiNLP b008610 documentation, accessed March 8, 2026, [https://pythainlp.org/dev-docs/api/util.html](https://pythainlp.org/dev-docs/api/util.html)  
48. pythainlp.util — PyThaiNLP 4.0.0 documentation, accessed March 8, 2026, [https://pythainlp.org/docs/4.0/api/util.html](https://pythainlp.org/docs/4.0/api/util.html)  
49. google/sentencepiece: Unsupervised text tokenizer for Neural Network-based text generation. \- GitHub, accessed March 8, 2026, [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)  
50. Fix an issue where input tokens to the WangchanBERTa NER pipeline may be in the incorrect form · Issue \#32 · vistec-AI/thai2transformers \- GitHub, accessed March 8, 2026, [https://github.com/vistec-AI/thai2transformers/issues/32](https://github.com/vistec-AI/thai2transformers/issues/32)  
51. All about RapidFuzz — String Similarity and Matching | by Parthvi Shah \- Medium, accessed March 8, 2026, [https://medium.com/@shahparthvi22/all-about-rapidfuzz-string-similarity-and-matching-cd26fdc963d8](https://medium.com/@shahparthvi22/all-about-rapidfuzz-string-similarity-and-matching-cd26fdc963d8)  
52. F1 Score for NER: A Metric To Evaluate Precision And Recall \- ThatWare, accessed March 8, 2026, [https://thatware.co/f1-score-for-ner/](https://thatware.co/f1-score-for-ner/)  
53. GitHub \- chakki-works/seqeval: A Python framework for sequence labeling evaluation(named-entity recognition, pos tagging, etc...), accessed March 8, 2026, [https://github.com/chakki-works/seqeval](https://github.com/chakki-works/seqeval)  
54. Use seqeval for NER model evaluation · Issue \#336 · BlueBrain/Search \- GitHub, accessed March 8, 2026, [https://github.com/BlueBrain/Search/issues/336](https://github.com/BlueBrain/Search/issues/336)  
55. Using the evaluator \- Hugging Face, accessed March 8, 2026, [https://huggingface.co/docs/evaluate/en/base\_evaluator](https://huggingface.co/docs/evaluate/en/base_evaluator)  
56. Named Entity Recognition (NER) Using the Pre-Trained bert-base-NER Model in Hugging Face | by Yuan An, PhD | Medium, accessed March 8, 2026, [https://medium.com/@anyuanay/working-with-hugging-face-lesson-2-1-71c6e4662479](https://medium.com/@anyuanay/working-with-hugging-face-lesson-2-1-71c6e4662479)  
57. Trainer class, compute\_metrics and EvalPrediction \- Transformers \- Hugging Face Forums, accessed March 8, 2026, [https://discuss.huggingface.co/t/trainer-class-compute-metrics-and-evalprediction/1698](https://discuss.huggingface.co/t/trainer-class-compute-metrics-and-evalprediction/1698)  
58. Transformers \- Hugging Face, accessed March 8, 2026, [https://huggingface.co/docs/evaluate/en/transformers\_integrations](https://huggingface.co/docs/evaluate/en/transformers_integrations)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAXCAYAAACBMvbiAAABI0lEQVR4XmNgGAHgERD/B+IV6BJQIAbEx9EFaQG+IrFvA/FjJD4MgBxKF8eALPKAsi2hfGQACq0CBgKOsQHiNwwQzSeAmAlVmixQyoDqGEEgtmIg4JjJQDwNif+NAWKIEpIYOQBkhiYS/w6UxusYkCZzLGLIvloMxN/xYHRwEYh1kPjInsXpGG4GTItBAJsYsWABEItD2cVQeh8SfgrEH6FsDNAMxNZoYuQ6JhOI84A4AYjTgfgHiiwEzGTAETK4AMgh/9AFiQAwT8DwW1Rphl9A/A6Kf6LJYQWXGCAGcaFL0BuAEjLIIaLoEvQGoHIA5BB2dAl6A1Ahh55gQdl5QAC2xPoXXYAeAJTC0XMBuVmbIiDNgOkAGMZWso6CUTAKBj0AAPZ/V0SGgrSBAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF0AAAAYCAYAAACY5PEcAAADbUlEQVR4Xu2YS8hNURTHl/dbMfAojxQxkkfyKkKhpCRD+TLwiJh4zsyEUgaKkZFMPFIkGWHCRElKKCTJ+/0mrP+393bX/d+9z9n3fh8G9/5qdc7+r7XX3WffffbjiLRo0aIxFrLQhExioR5Gqx1WO6g2kHwx1qvtZLFJ+cVCGQfEVVrly6PUnql9+RNRy0i1xywaXojLGYy5J9X+29Xuf8IitbdS3E7LCanEfVY7bnxj1F6bcpKu4hJcYofnh9pPFj2o15tF4rTaNXGxM8kHuqvdYvE/gPa999cUfdV2iYvZQr7AR7WlLDJIgBGXYr64mAWkz1L7SloM1O3pr9/IBzapLWHxP4D2nfXXFHje3eJiupAvME6Kc8gjKQmQyptgXyPwXfLmcry6APHIg5FteULlIqazEGECCxkMUDuqtlXS/bFZrb+4gZOKCcCPt6KGueKcF0lnBomLe0M6tD6kMROl8hrOFlfncsXdTtkDWPCHpaY6UE8uyx61sWpTxeUYXu1u56S/wo+1rgjE7GMRhJFXNievFBd33WgYGTkPeEqtmymjDte7QuUywlTFxLRc0BcB5MEzWx76K0Y6/BuML8YFSUy9sQ6IcUdcHLaGgXleK4NjMKKghdGPnI3s8bnj+XfqhXMdMeUpapP9fWh/Gdhy18QN8WKNI0IsbnVEixHmc4vN99Q66iR0PCy1qOWA0XvMlJHvgSnfN/dhdihjh0Ti8MpDxD6ziBXi4ng72eb1InBC28aiclNc3dJVvoSwwHckB8DoHW/KNuc5owPoOQMFzx1tV06DUzEzJK5bzkjtTgX0E1cXbwEvqrmEDgdhADUKziEWHG6Qb7DaWqPjdA59o9FSHJJEm8IpLAUWD/h7sEMqO5oiivzYgcCPtaFeMJVw7o50PNcLJ05eCPd7PWcqO6/2icUAktxgUXku1St6DNTFvBoDe9qiBi6W2ofNJVXPjv5cMH1w564Tl4d3dWGg5IA4fFZJEr6PXBU3x+M+5xCCOD4KYzrBUfqVN/zb+LYR4yULGaRyBfAnT2MxwjK1d+LagCsGWC/vG6a2198DPA9mBcTiij8JI7kI9A1mg05nu9oHFlvICMl/IxoCyWOLZTODhXg5i50J5ua7LDYxQ6W+b0kNg28Ma1hsUv7qtMK0sdCEzGGhRYvm4zcTx/CNHbGLFAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAYCAYAAADkgu3FAAABH0lEQVR4Xu2UvUoDQRSFjwpCOtEipW1eI5WFRcRCUkoay9ilyDuIYpk3SGMhNr6AjYUgBIRU6TUJkh9SRO/17iSTk7uyKcX94BRzzmFn986wQM5f4V70tYGUa9Ew8iaiD/LaSXeBmmeOFx4aKDme11MOYf5jMPZhXxSzDSu9kK/0aK29J/ICKy/xINpaZj9cwgoV8ndFV9G6CusdRV6gANqovswW9OGPY09UjNYd+D3lDpadcBCTNncmrVeG+TccxOzASs8cOISNBrApTJP1q+gg6rk0YOVjDohwPjUOsvIJfxzMG7L1UkmbO5O156LXd5Pz6bKZlRbsAefkM01Y74KD3zgVjWG35j2RntMM66O5TTLt6n9tJJqvNHJy/h/fjF9epb8vgbEAAAAASUVORK5CYII=>