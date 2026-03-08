# **State of Machine Learning Training on Apple Silicon: 2026 Infrastructure Architecture Report**

## **Executive Summary**

The landscape of deep learning hardware infrastructure in 2026 presents a fundamental architectural dichotomy for systems engineers and machine learning practitioners. The historical paradigm of monolithic Compute Unified Device Architecture (CUDA) dominance has fractured, yielding a heterogeneous ecosystem where architectural decisions can no longer be decoupled from compiler stack maturity. Within this fractured ecosystem, Apple Silicon has established a formidable, albeit highly specialized, position in the local machine learning lifecycle.1

This report delivers an exhaustive technical analysis of training and fine-tuning Machine Learning models natively on Apple Silicon, specifically focusing on the M3, M4, and M5 processor architectures. The evaluation encompasses the maturation of Apple’s proprietary Machine Learning eXplore (MLX) framework, the current operational state and compiler maturity of PyTorch’s Metal Performance Shaders (MPS) backend, and the terminal hardware limits of Apple’s Unified Memory Architecture (UMA). The objective is to provide infrastructure architects with empirical data to determine the viability of local Apple Silicon clusters versus continued reliance on cloud-provisioned Nvidia H100 instances for specific training workloads. The prevailing analysis indicates that while Apple Silicon serves as an unassailable environment for high-memory local prototyping and parameter-efficient fine-tuning, memory bandwidth constraints and compiler stack immaturity isolate it from large-scale, datacenter-grade pre-training workflows.1

## **Vector 1: The Apple MLX Framework Architecture**

Apple’s introduction and subsequent iterative enhancement of the MLX array framework represents a strategic pivot toward aggressive hardware-software co-design. MLX is purposefully engineered to exploit the unique characteristics of Apple’s System on a Chip (SoC) architecture, directly interfacing with the Unified Memory Architecture and the dedicated Neural Accelerators distributed across the GPU cores.2

### **Adoption, Maturity, and Core Design Paradigms**

As of early 2026, the MLX framework has transitioned from a specialized research tool for inference into a highly mature, production-ready framework capable of sophisticated model training and custom architectural design.2 The framework is characterized by a Python Application Programming Interface (API) structurally analogous to NumPy, supported by robust C++, C, and Swift bindings that permit lower-level hardware orchestration.2

The fundamental architectural divergence between industry-standard eager execution frameworks and MLX lies in the construction and evaluation of computation graphs. MLX utilizes a lazy computation graph, mathematically and operationally akin to JAX, paired with dynamic graph construction.1 Operations define a sequence of transformations but are not materialized in memory until explicitly evaluated via an eval() call. This design paradigm yields profound memory efficiency advantages during training optimizations. For instance, when instantiating a massive language model and updating it with lower precision weights, eager execution frameworks temporarily spike memory usage by retaining both the initialized standard precision tensors and the newly loaded lower precision copies. MLX's lazy evaluation natively minimizes this memory footprint, ensuring that the maximum consumed unified memory aligns strictly with the evaluated graph, effectively halving the memory overhead during state initialization.6

MLX supports composable function transformations for automatic differentiation (mx.grad), automatic vectorization (mx.vmap), and computation graph optimization (mx.compile).5 The ecosystem has matured to support complex network topologies well beyond standard Transformer encoder-decoder architectures. This maturity is evidenced by the release and production deployment of libraries such as mlx-snn, which provides a complete backpropagation-through-time training pipeline for Spiking Neural Networks (SNNs) natively on Apple hardware.7 The library provides sophisticated biological neuron models and surrogate gradient functions, validating MLX's capability to handle highly custom, non-standard gradient flows that previously required deep CUDA kernel modifications.7

Furthermore, parameter-efficient fine-tuning (PEFT) techniques, notably Low-Rank Adaptation (LoRA) and Quantized Low-Rank Adaptation (QLoRA), are fully integrated into the mlx-lm subpackage, enabling robust local fine-tuning of Large Language Models and Vision Language Models.2 However, integrating MLX into enterprise pipelines introduces a significant ecosystem fragmentation risk. For infrastructure engineers, utilizing MLX requires departing from the established PyTorch ecosystem, thereby breaking the "write once, run anywhere" operational methodology.1 Codebases heavily optimized for MLX’s lazy evaluation graph cannot be trivially ported to CUDA environments for scaled cluster training, forcing development teams to maintain bifurcated training scripts.1

### **Performance Benchmarks: MLX versus PyTorch MPS**

The performance delta between MLX and PyTorch’s MPS backend is highly dependent on the mathematical nature of the specific machine learning workload. Benchmarks executed across the M3, M4, and the recently announced M5 silicon reveal a complex interplay between framework overhead, compiler maturity, and hardware utilization.

The M5 architecture, supported natively in macOS 26.2 (Tahoe), exposes dedicated Neural Accelerators within each GPU core directly to the MLX runtime via Tensor Operations (TensorOps) and Apple's Metal 4 performance primitives.2 In autoregressive generation and specific phases of Transformer training, processing the initial prompt or batch (Time-to-First-Token) is entirely compute-bound. In this regime, MLX leverages the M5 Neural Accelerators to achieve up to a four-fold speedup over the preceding M4 baseline.2 The subsequent operations, such as sequential token decoding or gradient accumulation across deep layers, transition from being compute-bound to being strictly bottlenecked by memory bandwidth. The M5 Max processor provides an approximate 19% to 27% performance boost in this phase, correlating directly and linearly to its increased unified memory bandwidth, which scales from 120 GB/s on base variants up to 307.2 GB/s on Pro configurations and higher on Ultra variants.2

While MLX consistently outperforms PyTorch MPS in Large Language Model inference generation and quantized fine-tuning—often recording speeds two to three times faster—pure dense training benchmarks reveal a striking inversion for certain classical architectures.1 Synthetic benchmarks evaluating pure matrix multiplication (matmul) throughput demonstrate that PyTorch's MPS backend can execute raw tensor multiplications faster than Apple's MLX.13 Furthermore, when training standard spatial architectures like Convolutional Neural Networks (CNNs) such as ResNet, empirical tests show PyTorch MPS training executing approximately ten to eleven times faster than MLX.14

This distinct performance discrepancy arises from underlying software architecture. PyTorch's MPS backend relies on heavily optimized Metal Performance Shaders provided directly by the macOS kernel, which have been historically fine-tuned for standard spatial tensor operations and dense matrix arithmetic common in computer vision. MLX, conversely, optimizes its graph compiler specifically for the unique memory access patterns of autoregressive models, dynamic sequence lengths, and low-bit quantized operations.1

| Framework & Backend | Hardware Generation | Metric / Workload | Observed Performance |
| :---- | :---- | :---- | :---- |
| **PyTorch (MPS)** | M1 Max (16GB) | ResNet CNN Training | \~10-11x faster than MLX 15 |
| **MLX** | M3 Max (40-core) | 7B Q4\_0 LLM Inference | \~66 tokens/sec 16 |
| **MLX** | M4 Max (40-core) | 7B Q4\_0 LLM Inference | \~83 tokens/sec 16 |
| **MLX** | M5 Max (Estimated) | 7B Q4\_0 LLM Inference | \~90-95 tokens/sec 16 |
| **MLX** | M5 Architecture | Compute-Bound TTFT | Up to 4.0x faster than M4 2 |

Therefore, for training traditional dense vision models or performing classical numerical simulations, PyTorch MPS remains the mathematically superior runtime. Conversely, for training, fine-tuning, and executing Transformer-based architectures, Mixture of Experts (MoE), and quantized networks, MLX utterly dominates the Apple Silicon ecosystem.

### **Hugging Face Ecosystem Integration and Zero-Copy Safetensors**

The integration between MLX and the broader open-source Hugging Face ecosystem is exceptionally mature, driven primarily by the official mlx-lm library and the dedicated mlx-community model hub.2 A critical and highly optimized advantage of MLX is its native, low-level support for the .safetensors weight formatting standard.

In historical hardware ecosystems, transitioning pre-trained PyTorch fp16 weights into local arrays required translating the tensors into generic NumPy arrays, a process fraught with data-type incompatibilities, specifically regarding the handling of bfloat16 types which NumPy natively lacks.18 MLX circumvents this entirely by implementing direct .safetensors parsing in its core C++ API. The framework directly parses the \_\_metadata\_\_ JSON headers within the .safetensors file and memory-maps the raw byte offsets directly into the Unified Memory.18

This zero-copy mapping approach capitalizes on the physical design of the Unified Memory Architecture. Because the CPU and GPU share the exact same physical memory sectors, the CPU can read the .safetensors file from the NVMe SSD directly into a memory address, and the GPU can immediately execute tensor operations against that address without requiring a secondary bus transfer.6 This radically accelerates model load times and allows developers to easily swap model weights loaded via the Hugging Face hub into custom MLX architectures for training.

The integration allows engineers to instantiate a model with uninitialized memory, apply dynamic chat templates, and stream the weights directly to the Metal backend using minimal syntax:

Python

import mlx.core as mx  
from mlx\_lm import load, generate

\# The load function directly memory-maps the safetensors from the Hugging Face Hub  
\# leveraging the unified memory to bypass traditional CPU-to-VRAM copy overhead.  
model, tokenizer \= load(  
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",  
    tokenizer\_config={"trust\_remote\_code": True}  
)

\# Apply standard Hugging Face chat templates natively  
messages \= \[{"role": "user", "content": "Analyze the MLX graph compilation."}\]  
prompt \= tokenizer.apply\_chat\_template(messages, add\_generation\_prompt=True)

\# Generate utilizing MLX's lazy evaluation for peak memory efficiency  
text \= generate(model, tokenizer, prompt=prompt, verbose=True)

9

### **Distributed Training via RDMA over Thunderbolt 5**

A paradigm-shifting capability introduced in macOS 26.2 (Tahoe) is native operating system support for Remote Direct Memory Access (RDMA) over Thunderbolt 5 interfaces.10 Historically, the limitation of Apple Silicon was the inability to scale beyond a single node's maximum memory capacity (typically 128GB to 512GB). The RDMA implementation effectively allows multiple Mac Studios to be clustered together to perform distributed machine learning training via MLX, acting as a single logical computation unit with a massive pooled memory space.22

By leveraging the JACCL backend and the Open MPI standard, MLX can distribute tensor operations across nodes using sophisticated pipeline and tensor parallelism methodologies.21 Thunderbolt 5 hardware provides 80 Gigabits per second (Gbps) of bidirectional bandwidth. In production MLX clusters, this enables sustained tensor synchronization and file transfer rates of 3.5 to 3.8 Gigabytes per second (GB/s), which is approximately 23 times faster than traditional TCP/IP based synchronization over the same physical link.10 This architecture effectively pools the Unified Memory of multiple machines, allowing engineers to combine three 512GB Mac Studios to yield a 1.5 Terabyte contiguous memory pool capable of training trillion-parameter models.22

However, implementing this requires navigating specific low-level macOS API constraints. Standard mx.distributed.send and mx.distributed.recv commands default to the GPU Metal execution stream. Because the Apple Metal framework enforces a strict, unconfigurable command buffer timeout limit of approximately five seconds to prevent infinite GPU hangs, RDMA receive operations that wait for network synchronization will trigger a Command buffer execution failed: Caused GPU Timeout Error and crash the entire training process.21 Infrastructure engineers must implement a critical workaround by forcing the CPU stream to handle all inter-node communication, thereby bypassing the Metal watchdog timer:

Python

import mlx.core as mx  
import mlx.distributed as dist

\# CRITICAL WORKAROUND: Force CPU stream for RDMA to prevent Metal command buffer timeout  
execution\_stream \= mx.cpu 

\# Distribute tensor data from node 0 using the CPU stream bypass  
mx.eval(dist.send(tensor\_data, dst=1, stream=execution\_stream))

\# Receive operation on node 1 using the CPU stream bypass  
received\_tensor \= dist.recv(src=0, stream=execution\_stream)  
mx.eval(received\_tensor)

21

## **Vector 2: PyTorch MPS (Metal Performance Shaders) Backend**

For enterprise organizations enforcing a strict "PyTorch-only" policy to maintain continuous pipeline parity between local developer environments and upstream cloud clusters, the mps backend remains the sole mechanism for accessing Apple Silicon GPU acceleration without rewriting models in MLX.25 While hardware support has broadened significantly since its inception, the backend remains hampered by systemic compiler immaturity, missing fundamental operations, and historical numerical precision idiosyncrasies that require specific architectural mitigations.

### **The Mixed Precision Nightmare: The Mathematics of fp16 versus bf16**

Historically, attempting mixed-precision training on Apple Silicon utilizing standard 16-bit floating-point formats, either through torch.autocast(device\_type="mps", dtype=torch.float16) or Hugging Face's standard Trainer(fp16=True) argument, resulted in catastrophic training collapse. This collapse was characterized by sudden, unrecoverable NaN (Not a Number) loss values early in the training loop.26

This failure mode is rooted deeply in the mathematical limitations of the IEEE 754 half-precision (fp16) format combined with default optimizer hyperparameters. The fp16 format allocates 1 sign bit, 5 bits for the exponent, and 10 bits for the mantissa. This structural allocation establishes a minimum positive normal value of approximately ![][image1] and a maximum upper bound of ![][image2]. During the backpropagation phase of deep neural networks, particularly in deep Transformers, the computed gradient magnitudes frequently fall below the ![][image3] threshold. In fp16, these microscopic values suffer from severe gradient underflow, snapping instantly to absolute zero.29

The critical point of failure occurs within the optimizer state. Modern adaptive optimizers, notably AdamW, utilize a default epsilon value of ![][image4] in their denominator to maintain numerical stability and prevent division by zero. When the MPS backend evaluates this equation in strict fp16, the ![][image5] epsilon underflows and rounds down to zero.28 Consequently, the optimizer attempts to divide by zero during the weight update step, which instantly corrupts the specific tensor with a NaN value. This corruption propagates forward through the network in the next step, rendering the entire model state invalid.28

In Nvidia datacenter environments, this mathematical constraint is mitigated via dynamic gradient scaling (loss scaling), which artificially multiplies the loss value before backpropagation to push gradients into a safer numerical range. However, the definitive architectural fix implemented for PyTorch in 2026 on Apple hardware relies on bypassing fp16 entirely in favor of the Brain Floating Point (bfloat16 or bf16) format.28 The bfloat16 format sacrifices substantial mantissa precision (reducing from 10 bits to 7 bits) in order to retain an 8-bit exponent. This expansion of the exponent identically matches the dynamic range of standard 32-bit floating-point (fp32), allowing values down to approximately ![][image6] and completely eliminating the underflow division-by-zero vulnerability.29

A critical bug historically existed in PyTorch's Inductor codegen specifically for Apple Silicon, which caused NaN generation during bfloat16 reductions on CPU and MPS backends due to numerical instability in compiled scaled\_dot\_product\_attention operations.26 As of the PyTorch 2.5 and 2.6 release cycles, this has been officially patched by forcing internal fp32 accumulation for all bf16 reduction operations at the compiler level.26 Therefore, the current, absolute industry consensus for safe mixed-precision training on a Mac requires strictly enforcing bf16 through the torch.amp context manager:

Python

import torch

\# Dynamically map tensors to the Apple GPU if available  
device \= torch.device("mps" if torch.backends.mps.is\_available() else "cpu")  
model \= MyCustomArchitecture().to(device)

\# Initialize the GradScaler (typically inactive for bf16, but best practice for pipeline compatibility)  
scaler \= torch.amp.GradScaler('mps', enabled=False) 

\# Execute the forward pass within the bfloat16 autocast context  
\# This preserves the 8-bit exponent, preventing AdamW epsilon underflow  
with torch.amp.autocast(device\_type="mps", dtype=torch.bfloat16):  
    outputs \= model(inputs.to(device))  
    loss \= criterion(outputs, targets.to(device))

loss.backward()  
optimizer.step()

25

It is critical for infrastructure architects to note that Apple Silicon entirely lacks native hardware logic for emerging sub-byte formats, specifically FP8 or FP4 training configurations, which are becoming standard on Nvidia Blackwell and AMD MI300X systems. Any PyTorch operations explicitly cast to FP8 on a Mac are merely emulated in software via upcasting to bfloat16. This emulation completely negates the memory footprint reductions and matrix throughput acceleration typically associated with sub-byte precision, rendering the Mac unsuitable for explicit FP8 training regimes.1

### **Missing Operations and the Compiler Stack Gap**

The integration of PyTorch with Apple Silicon suffers from what analysts categorize as a "Low" compiler maturity rating.1 Unlike NVIDIA and AMD, which leverage the highly optimized, open-source Triton compiler stack for kernel generation, Apple relies on its proprietary Metal Shading Language (MSL) and the internal MPSGraph frameworks.1 Consequently, PyTorch 2.5+ torch.compile support for the MPS backend remains in its infancy. Complex kernel fusions that easily compile on CUDA frequently fail on MPS, resulting in operations running as unfused, generic Metal kernels, which severely degrades training throughput.1

A persistent and highly disruptive bottleneck is the ongoing absence of comprehensive operator support. While the open-source community and Apple actively track missing operations (e.g., PyTorch GitHub Issues \#77764, \#141287, and \#154052) 32, numerous ops utilized in advanced network architectures remain unimplemented. Furthermore, the MPS backend entirely lacks support for float64 (double precision) operations. The Apple GPU hardware natively omits FP64 arithmetic logic units; therefore, any algorithms relying on FP64—which is highly common in scientific computing, genomic analysis, and classical ML simulations—will instantly fail or crash the runtime.34

When the PyTorch execution engine encounters an unimplemented MPS operation (such as aten::\_linalg\_solve\_ex.result or aten::isin) during the forward or backward pass, it halts execution and raises a NotImplementedError.36 To bypass this hard failure, engineers must define a specific environment variable to force an implicit CPU fallback:

Bash

export PYTORCH\_ENABLE\_MPS\_FALLBACK=1

36

While this environment variable prevents the Python script from crashing, it induces a catastrophic performance penalty that ripples through the training loop. The tensor must be synchronized, serialized, copied from the GPU domain to the CPU domain across the unified memory architecture, processed by the CPU cores, and synchronized back to the GPU. In iterative training loops, a single fallback operation occurring inside the core model architecture can increase overall step latency by multiple orders of magnitude.

Additionally, PyTorch MPS completely lacks native hardware support for FlashAttention (FA2/FA3) kernels. While PyTorch on Apple Silicon uses Apple's internal Metal implementation of Scaled Dot Product Attention (SDPA), this fallback fails to support the advanced features of FlashAttention, such as processing variable sequence lengths within a single batch without massive memory padding.1 This architectural gap makes Transformer training significantly less efficient and more memory-hungry than on equivalent CUDA architectures.1

## **Vector 3: Memory Architecture Limitations (UMA)**

Apple's Unified Memory Architecture (UMA) is simultaneously the defining operational advantage and the primary systemic vulnerability of the platform. In traditional x86 computing architectures, the system is hampered by the Peripheral Component Interconnect Express (PCIe) bus bottleneck. On an NVIDIA setup, gigabytes of training data must traverse the PCIe bus from the system's volatile RAM to the dedicated GPU VRAM before computation can occur. On Apple Silicon, the CPU, GPU, and Neural Engine are fabricated onto the same silicon die and share a single, massive, high-bandwidth memory pool.3

This unified approach allows a fully specified Mac Studio with 512GB of RAM to comfortably load and execute inference on a heavily quantized 405-billion parameter model—a feat that is physically impossible on a single $30,000 NVIDIA H100 GPU, which is strictly limited to 80GB of High Bandwidth Memory 3 (HBM3) VRAM.1

### **VRAM Allocation Limits and the High Watermark Ratio**

Despite the hardware lacking physical silicon boundaries between CPU and GPU memory banks, the macOS kernel enforces strict software-level guardrails to prevent the GPU execution streams from entirely consuming the unified memory and starving the operating system’s critical background tasks.

By default, the PyTorch MPS allocator respects a kernel-defined "High Watermark Ratio," which artificially caps the total percentage of unified memory the GPU is permitted to allocate. The default limit acts as a hard ceiling, typically allowing the MPS backend to address up to the device's recommended maximum working set size.37 During intensive machine learning training—particularly when dealing with large batch sizes, extensive context windows, and Adam optimizer states (which require tracking both first and second moments for every single network parameter, tripling the memory requirement)—memory consumption can rapidly hit this artificial ceiling. When the cap is breached, PyTorch throws a fatal RuntimeError: MPS backend out of memory.38

Engineers can explicitly override this macOS safeguard by manipulating the memory allocator's environment variables prior to executing the Python runtime. Setting the high watermark ratio to a float greater than 1.0 extends the limit, while setting it to exactly 0.0 disables the upper limit entirely:

Bash

\# Explicitly disables the macOS unified memory allocation ceiling  
export PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0

37

While deploying this variable allows the training script to fully saturate the physical unified memory, it is classified as a highly perilous operation. Circumventing the kernel's native memory management removes the system's safety buffer. Without this buffer, unexpected memory spikes will bypass standard Out-Of-Memory (OOM) exceptions and frequently lead to hard system crashes, unrecoverable kernel panics, or the triggering of the catastrophic swapping mechanism.37 Additionally, PyTorch provides a PYTORCH\_MPS\_LOW\_WATERMARK\_RATIO environment variable (default 1.4), which acts as a soft limit to trigger aggressive garbage collection and command buffer commits before the hard ceiling is reached.37

### **The Swapping Death and NVMe SSD Degradation**

The most critical and hardware-destructive failure mode of Apple Silicon ML infrastructure is documented by practitioners as the "Unified Memory Swap Death".1

macOS utilizes a highly dynamic virtual memory management system. When the physical unified memory approaches exhaustion, the operating system first attempts to aggressively compress inactive memory pages. If memory pressure remains critical, macOS begins paging (swapping) memory blocks out to the internal Non-Volatile Memory Express (NVMe) Solid State Drive.44

In traditional consumer computing contexts (e.g., web browsing or video editing), swapping elegantly prevents application crashes and the user barely notices. In machine learning training, swapping is fatal to the computational pipeline. The unified memory on an M4 or M5 Ultra chip achieves internal bandwidths exceeding 800 GB/s.1 The internal NVMe SSD, while exceptionally fast for standard storage protocols, peaks at approximately 7 GB/s. When a training loop's batch size forces the KV cache, model weights, or optimizer states into the SSD swap space, the GPU execution units must stall and wait for data to be retrieved across the storage controller.

This vast latency mismatch causes inference and training throughput to collapse geometrically. Inference generation speeds have been observed plummeting from an acceptable 60 tokens per second down to 3 tokens per second, or in severe cases, 0.01 tokens per second, effectively hanging the process indefinitely.1

Beyond the immediate destruction of training throughput, excessive and continuous swapping presents a severe hardware degradation vector. Large ML workloads that constantly thrash the virtual memory can generate terabytes of swap writes in a matter of hours.46 Because the NAND flash storage on all modern Apple Silicon devices is soldered directly to the logic board and cannot be physically replaced or upgraded, sustained ML training that frequently breaches physical memory limits will rapidly exhaust the SSD's Total Bytes Written (TBW) endurance rating. Reaching the TBW limit leads to premature, unrepairable hardware failure of the entire machine.46

**Architectural Best Practices for UMA Management:**

To prevent the Swapping Death and protect the underlying hardware, systems architects must enforce the following protocols during local Apple Silicon training:

1. **Monitor Memory Pressure, Not Capacity:** Relying on the standard "Memory Used" metric in macOS Activity Monitor is highly misleading due to the operating system's aggressive file caching behavior. Engineers must exclusively monitor the "Memory Pressure" graph; sustained yellow or red pressure during the training loop indicates dangerous swap utilization, meaning the batch size must be reduced immediately.43  
2. **Strict Batch Size and Parameter Control:** Batch sizes, sequence lengths, and gradient accumulation steps must be rigorously profiled prior to initiating long training runs. Peak memory utilization must be tuned to remain at least 10% to 15% below the physical hardware limit to allow for temporary OS spikes without triggering swap.38  
3. **Wired Memory Enforcement:** For massive model orchestration where swapping must be prevented at the kernel level, system-level wired memory limits can be adjusted using the terminal command sudo sysctl iogpu.wired\_limit\_mb=N (where N is the limit in megabytes). This pins memory pages to the physical RAM and prevents macOS from swapping them to the SSD, though this requires careful tuning to avoid kernel panics.51

## **Strategic Infrastructure Conclusions**

The exhaustive technical evaluation of Apple Silicon for Machine Learning infrastructure in 2026 yields clear, unambiguous operational delineations. The architectural decision to invest in local Mac hardware fleets versus cloud-provisioned NVIDIA H100 instances must be driven entirely by the mathematical nature, precision requirements, and scale of the specific ML workload.

**The Architectural Case for Apple Silicon (Mac Studio / MacBook Pro):**

Apple hardware remains peerless as a local, high-capacity prototyping, discovery, and inference platform. The Unified Memory Architecture provides an unparalleled cost-to-memory-capacity ratio. Organizations requiring the local execution, prompt engineering, or lightweight parameter-efficient fine-tuning (such as LoRA and QLoRA) of massive models (e.g., 70B to 400B parameters) can achieve this locally on a single $5,000 Mac Studio. Achieving the same memory capacity on an NVIDIA setup demands complex multi-GPU configurations costing upward of $30,000. For these specific workflows, Apple's MLX framework is the definitively superior software stack, offering deep hardware integration, native Neural Accelerator optimization, excellent zero-copy .safetensors support, and robust quantization capabilities that PyTorch MPS currently lacks.

**The Architectural Case for NVIDIA H100 (Cloud/Datacenter):** For full-parameter model pre-training, large-scale distributed fine-tuning, highly custom spatial architectures, and high-throughput batch processing, Apple Silicon remains fundamentally inadequate. While Apple has successfully introduced Thunderbolt 5 RDMA clustering to scale memory, the raw memory bandwidth of a flagship M5 Ultra (\~800 GB/s) simply cannot compete with the 3.35 TB/s HBM3 bandwidth of a single NVIDIA H100, let alone an interconnected NVLink cluster.1 Furthermore, PyTorch's MPS backend continues to suffer from persistent compiler immaturity, the absence of FlashAttention optimizations, missing native FP8/FP4 support, and the severe operational risk of unoptimized CPU fallback bottlenecks.

In summary, infrastructure architects should deploy Apple Silicon hardware exclusively for local model discovery, highly quantized inference, and privacy-constrained local fine-tuning utilizing the MLX framework. For heavy, compute-intensive distributed training pipelines prioritizing raw matrix throughput and total time-to-convergence, the NVIDIA CUDA ecosystem, powered by H100 and emerging B200 instances, remains the definitive, unassailable industry standard.

#### **Works cited**

1. State of PyTorch Hardware Acceleration 2025, accessed March 8, 2026, [https://tunguz.github.io/PyTorch\_Hardware\_2025/](https://tunguz.github.io/PyTorch_Hardware_2025/)  
2. Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU, accessed March 8, 2026, [https://machinelearning.apple.com/research/exploring-llms-mlx-m5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)  
3. Apple Silicon vs NVIDIA CUDA: AI Comparison 2025, Benchmarks, Advantages and Limitations \- Consultant freelance Jean-Jerome Levy, accessed March 8, 2026, [https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/](https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/)  
4. Apple MLX Explained: Run & Optimize ML on Apple Silicon \- F22 Labs, accessed March 8, 2026, [https://www.f22labs.com/blogs/what-is-mlx-a-beginners-guide-to-apples-machine-learning/](https://www.f22labs.com/blogs/what-is-mlx-a-beginners-guide-to-apples-machine-learning/)  
5. ml-explore/mlx: MLX: An array framework for Apple silicon \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)  
6. Dive into MLX: Performance & Flexibility for Apple Silicon | by Pranay Saha | Medium, accessed March 8, 2026, [https://medium.com/@pranaysaha/dive-into-mlx-performance-flexibility-for-apple-silicon-651d79080c4c](https://medium.com/@pranaysaha/dive-into-mlx-performance-flexibility-for-apple-silicon-651d79080c4c)  
7. mlx-snn: Spiking Neural Networks on Apple Silicon via MLX \- arXiv, accessed March 8, 2026, [https://arxiv.org/html/2603.03529v1](https://arxiv.org/html/2603.03529v1)  
8. MLX Community \- Hugging Face, accessed March 8, 2026, [https://huggingface.co/mlx-community](https://huggingface.co/mlx-community)  
9. ml-explore/mlx-lm: Run LLMs with MLX \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  
10. macOS Tahoe 26.2 will give M5 Macs a giant machine learning speed boost \- AppleInsider, accessed March 8, 2026, [https://appleinsider.com/articles/25/11/18/macos-tahoe-262-will-give-m5-macs-a-giant-machine-learning-speed-boost](https://appleinsider.com/articles/25/11/18/macos-tahoe-262-will-give-m5-macs-a-giant-machine-learning-speed-boost)  
11. Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU \#2816 \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx/discussions/2816](https://github.com/ml-explore/mlx/discussions/2816)  
12. Apple unveils M5 Pro and M5 Max, citing up to 4× faster LLM prompt processing than M4 Pro and M4 Max : r/LocalLLaMA \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple\_unveils\_m5\_pro\_and\_m5\_max\_citing\_up\_to\_4/](https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple_unveils_m5_pro_and_m5_max_citing_up_to_4/)  
13. matmul() using PyTorch's MPS backend is faster than Apple's MLX \- Kevin Martin Jose, accessed March 8, 2026, [https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/)  
14. PyTorch and MLX for Apple Silicon \- Towards Data Science, accessed March 8, 2026, [https://towardsdatascience.com/pytorch-and-mlx-for-apple-silicon-4f35b9f60e39/](https://towardsdatascience.com/pytorch-and-mlx-for-apple-silicon-4f35b9f60e39/)  
15. PyTorch (MPS) is faster than MLX for training and inference for ResNets and Transformers (tested on 2 tasks) \#243 \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx/issues/243](https://github.com/ml-explore/mlx/issues/243)  
16. Apple M5 Pro & M5 Max just announced. Here's what it means for local AI \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1rk7n3u/apple\_m5\_pro\_m5\_max\_just\_announced\_heres\_what\_it/](https://www.reddit.com/r/LocalLLaMA/comments/1rk7n3u/apple_m5_pro_m5_max_just_announced_heres_what_it/)  
17. Apple shows how much faster the M5 runs local LLMs compared to the M4 \- 9to5Mac, accessed March 8, 2026, [https://9to5mac.com/2025/11/20/apple-shows-how-much-faster-the-m5-runs-local-llms-compared-to-the-m4/](https://9to5mac.com/2025/11/20/apple-shows-how-much-faster-the-m5-runs-local-llms-compared-to-the-m4/)  
18. Safetensor support · Issue \#182 · ml-explore/mlx \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx/issues/182](https://github.com/ml-explore/mlx/issues/182)  
19. \[Feature request\] Add metadata to safetensors saving and loading · Issue \#633 · ml-explore/mlx \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx/issues/633](https://github.com/ml-explore/mlx/issues/633)  
20. Building On-Device AI Machine Learning | by Badarinath Venkatnarayansetty | Medium, accessed March 8, 2026, [https://badrinathvm.medium.com/building-on-device-ai-machine-learning-1524f6636d3e](https://badrinathvm.medium.com/building-on-device-ai-machine-learning-1524f6636d3e)  
21. Guide: RDMA file transfer over Thunderbolt 5 with JACCL (3.5+ GB/s) \#3207 \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx/issues/3207](https://github.com/ml-explore/mlx/issues/3207)  
22. 1.5 TB of VRAM on Mac Studio \- RDMA over Thunderbolt 5 \- Jeff Geerling, accessed March 8, 2026, [https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/)  
23. macOS 26.2 enables fast AI clusters with RDMA over Thunderbolt | Hacker News, accessed March 8, 2026, [https://news.ycombinator.com/item?id=46248644](https://news.ycombinator.com/item?id=46248644)  
24. Distributed LoRA through MLX \- Medium, accessed March 8, 2026, [https://medium.com/@dutingzhen/distributed-lora-through-mlx-035c48597848](https://medium.com/@dutingzhen/distributed-lora-through-mlx-035c48597848)  
25. Accelerated PyTorch training on Mac \- Metal \- Apple Developer, accessed March 8, 2026, [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)  
26. torch.compile results in loss nan while it converges without torch.compile on mac cpu and mps · Issue \#171764 · pytorch/pytorch \- GitHub, accessed March 8, 2026, [https://github.com/pytorch/pytorch/issues/171764](https://github.com/pytorch/pytorch/issues/171764)  
27. Training with mixed precision: loss is NaN despite finite output in forward pass \- autograd, accessed March 8, 2026, [https://discuss.pytorch.org/t/training-with-mixed-precision-loss-is-nan-despite-finite-output-in-forward-pass/162937](https://discuss.pytorch.org/t/training-with-mixed-precision-loss-is-nan-despite-finite-output-in-forward-pass/162937)  
28. NaN Loss Issues with Precision 16 in PyTorch Lightning GAN Training, accessed March 8, 2026, [https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369](https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369)  
29. Why bf16 do not need loss scaling? \- mixed-precision \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/t/why-bf16-do-not-need-loss-scaling/176596](https://discuss.pytorch.org/t/why-bf16-do-not-need-loss-scaling/176596)  
30. Introducing Mixed Precision Training in Opacus \- PyTorch, accessed March 8, 2026, [https://pytorch.org/blog/introducing-mixed-precision-training-in-opacus/](https://pytorch.org/blog/introducing-mixed-precision-training-in-opacus/)  
31. BFloat16 training \- explicit cast vs autocast \- mixed-precision \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/t/bfloat16-training-explicit-cast-vs-autocast/202618](https://discuss.pytorch.org/t/bfloat16-training-explicit-cast-vs-autocast/202618)  
32. General MPS op coverage tracking issue \#77764 \- GitHub, accessed March 8, 2026, [https://github.com/pytorch/pytorch/issues/77764](https://github.com/pytorch/pytorch/issues/77764)  
33. MPS operator coverage tracking issue (2.6+ version) \#141287 \- GitHub, accessed March 8, 2026, [https://github.com/pytorch/pytorch/issues/141287](https://github.com/pytorch/pytorch/issues/141287)  
34. How to check mps availability? \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/t/how-to-check-mps-availability/152015](https://discuss.pytorch.org/t/how-to-check-mps-availability/152015)  
35. Apple Silicon (mps) compatibility with PyTorch's operations \[D\] \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/pytorch/comments/17mqc6h/apple\_silicon\_mps\_compatibility\_with\_pytorchs/](https://www.reddit.com/r/pytorch/comments/17mqc6h/apple_silicon_mps_compatibility_with_pytorchs/)  
36. How do i fix the MPS NotImplemented Error for m1 macbook air ? : r/pytorch \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/pytorch/comments/1c3kwwg/how\_do\_i\_fix\_the\_mps\_notimplemented\_error\_for\_m1/](https://www.reddit.com/r/pytorch/comments/1c3kwwg/how_do_i_fix_the_mps_notimplemented_error_for_m1/)  
37. MPS Environment Variables — PyTorch 2.10 documentation, accessed March 8, 2026, [https://docs.pytorch.org/docs/stable/mps\_environment\_variables.html](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)  
38. MPS backend out of memory \- PyTorch Forums, accessed March 8, 2026, [https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879](https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879)  
39. Set "PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO" on Mac App : r/comfyui \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/comfyui/comments/1j56m5n/set\_pytorch\_mps\_high\_watermark\_ratio\_on\_mac\_app/](https://www.reddit.com/r/comfyui/comments/1j56m5n/set_pytorch_mps_high_watermark_ratio_on_mac_app/)  
40. Building a Local LLM Mental Health Chatbot on a Mac — All the GPU Pain & How I Survived, accessed March 8, 2026, [https://medium.com/@blue1357a/building-a-local-llm-mental-health-chatbot-on-a-mac-all-the-gpu-pain-how-i-survived-caac2aca9102](https://medium.com/@blue1357a/building-a-local-llm-mental-health-chatbot-on-a-mac-all-the-gpu-pain-how-i-survived-caac2aca9102)  
41. MPS backend out of memory (MPS allocated: 16.91 GB, other allocations: 2.77 MB, max allowed: 18.13 GB). Tried to allocate 1.28 GB on private pool. Use PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0 to · Issue \#7171 · Comfy-Org/ComfyUI \- GitHub, accessed March 8, 2026, [https://github.com/Comfy-Org/ComfyUI/issues/7171](https://github.com/Comfy-Org/ComfyUI/issues/7171)  
42. MPS memory issue, MPS backend out of memory, but works if I empty the MPS cache \#105839 \- GitHub, accessed March 8, 2026, [https://github.com/pytorch/pytorch/issues/105839](https://github.com/pytorch/pytorch/issues/105839)  
43. Stop Running Models Too Big for Your Mac (Memory Trap Explained) \- YouTube, accessed March 8, 2026, [https://www.youtube.com/shorts/vdkMNy5LPho](https://www.youtube.com/shorts/vdkMNy5LPho)  
44. New Analysis Finds 8GB Mac Still Enough in 2026 \- FindArticles, accessed March 8, 2026, [https://www.findarticles.com/new-analysis-finds-8gb-mac-still-enough-in-2026/](https://www.findarticles.com/new-analysis-finds-8gb-mac-still-enough-in-2026/)  
45. Loading models with mmap · ml-explore mlx · Discussion \#615 \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx/discussions/615](https://github.com/ml-explore/mlx/discussions/615)  
46. macOS swap tuning on M1 causing high wear on SSDs : r/apple \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/apple/comments/lkdmwn/macos\_swap\_tuning\_on\_m1\_causing\_high\_wear\_on\_ssds/](https://www.reddit.com/r/apple/comments/lkdmwn/macos_swap_tuning_on_m1_causing_high_wear_on_ssds/)  
47. Concerns Over SSD Swap Usage: Did I Make … \- Apple Support Communities, accessed March 8, 2026, [https://discussions.apple.com/thread/255604027](https://discussions.apple.com/thread/255604027)  
48. Most M1 Macs appear to have a serious SSD wear defect | Hacker News, accessed March 8, 2026, [https://news.ycombinator.com/item?id=26152161](https://news.ycombinator.com/item?id=26152161)  
49. What is a safe amount of swap memory? \- Apple Support Communities, accessed March 8, 2026, [https://discussions.apple.com/thread/255326605](https://discussions.apple.com/thread/255326605)  
50. macOS Using Swap Memory While RAM is not Fully Occupied \- Reddit, accessed March 8, 2026, [https://www.reddit.com/r/MacOS/comments/1p3rawn/macos\_using\_swap\_memory\_while\_ram\_is\_not\_fully/](https://www.reddit.com/r/MacOS/comments/1p3rawn/macos_using_swap_memory_while_ram_is_not_fully/)  
51. ml-explore/mlx-examples: Examples in the MLX framework ... \- GitHub, accessed March 8, 2026, [https://github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)  
52. GPU Cluster for AI: 2026 Buyer's Guide With Benchmarks & TCO Calculator \- io.net, accessed March 8, 2026, [https://io.net/blog/gpu-cluster](https://io.net/blog/gpu-cluster)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAAAXCAYAAAD0v0pBAAAC4klEQVR4Xu2YTchNQRjHH5/5iCiSBa+VKAuRRFY2ip2iZKOkKCSJFEmUQhaSDdnIAhHKStnIwtubfNuw8pFi4yNkgef/zsy5c/93nnPP6V5nofnV0/vO7zkz9zTPuTNnrkgm85/zWeOUj2+UyzTAL40/Gpc4kWmGsyzKmKTxWlzFHlKuCkMsiKvixn6hMYFyTbNa4zLLiIMaXzS+a2ymXB1QgAsatzVGUa6NTeImZ4xvHxK3fnXjgbh+IVJgTORm+zZuBO2ZxRXNsFfjq7Tu9Up7ugAPyJ2o/UzjftSuw5Pof2t+ZLq45LjIlU1oit1iX39P4y25k2Jf3wRWASZL+r7gpkTtHyVxK7ouBmNsYQlSk113iSgrADyvhcu978ZSFgnmsaiAVYBHkr4vuPMsu3BU42fUxhh7onYBEk/9/8vE7QV1sQoQlpsD5Ae8X0ueGa3xm2VE6jOrYBUg9TACy5exTWNO1Eb/sMQXzPcJvCY91piocca7OlgFWCjOIx8Tlr195FOMlfTYKVeVJgoAnovb7LE0LaLcMBskPTieOnSqilWAleL8TvJTvT9H3oKLkPqsOqA/3sqY1FwAy/fMGnEDvyd/1/uqWAWYK87vIj/N+yPkywhFQIygXF0wxjWWYk+05XtmQNzAF8ljJ4dfQt7CKgAmCn4/+VnebyRfxkjp30RgjOssxR7f8n0BA/OhBAcHeDzBVbAKAOCtt6CqZ4Ew+SBs7L2A/jdYSuucwMC9ZNkvMPgrcqnXsXUaM8gFygqA/SQ+kAAciqzrmfAtium1COh7k6WyXtLjwi1m2S8WSOeHon04aodJ4OsCx8Xl8HbDrJDOfmifJmfBfQPxt6IOYS+xTrfIbY3aJ7z7p+wQ9yHht6Bj7elh8JXlgwR+7fsg7qT7xv/9KJ17SnjisdShD34fqcIqFgQejKr71HaNTxrvpHWvuHd+2xsv7l4Hxb2a4zDV66afyWQymUwmkyn4C7tp4TH3P96XAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAXCAYAAACxvufDAAAClklEQVR4Xu2XS6iNURTH/97vJDHxLBJFMaCEKAaKiUeKIkOKDI0uKcXEqzwGEhKlDKQojwy8TUSSiIFnoTyKSML637X2uetb59vnnsMdcDu/+nfO+v/X2ffb5+y9v+8CTZo0+ZcZIHoq+iW6EzLyUbRSNEg0ULRU9KHQUR/PRQtF/URDRGtE3wsdyljRbej1XApZDvZmWQ1t6GH1ZtGnSqowjxpe6KiPOAbVs9ABzDY/MTnUZTxBjR5+mwx7Oy/9cQ/rbaL9orkhawSOs1O0SzQpZAn2rA0ef+1bwUvMFD1D9TVXKJtQ31CT2POntDfOUGgPXz0XzC/jh2gd8nlrcN/eT4fuzTKyAzRIe+NsQnnPYZT7D6GrMDvJCdDguOge9DDYa16E3mPRA9EN6LfXvdBRHxyHh9h10SNU7/3TKP/7+1DtTxHttvfZSS6HBjH8KfoaPPb0cvU58xolfuZ98K6EOrEH6g9zHq8zkZ3kAmjwOviXza/FeGhPSwwahLcQjjPL6hNWR9IKS6vnJopbKzvJUdDgWPDPmD/Ved3ce9IV2sM98TfMgY5zwOrcnjyENn+M6Ehb1Ep2koTByeCdNX+c1eke5G8z/c275rz24PKKF8IHA3pbrZ5hda3TdZnoatBby1NdgAEn4blrfoJPKV9cTeZDe1Y4r49og6sj7H8VvO3mj3Qe68WuJp+h+zdHbpm3MhHVIestrh4BfeTzfEP54URNC35iCfT09LA/Lvnz0NM70QXaN9p5kYvQHm6jUtZDG9KzK59sIqvQ9kvwlbeACL995kdj4NgB/fwbez1YjCvwlsbVcwraN68YV1gkeid6KXoBPUTjqutweOJtjGZno9av2CkYjPr/Lfpv8U9ETZp0IL8BARDEzP/2s0AAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAXCAYAAABnGz2mAAABN0lEQVR4Xu2VvS4FURSFN6LQiMIDeAbexQuIxE8ICaVOR5Qq0YmIWqXWqBA/D0AjKj9BNKyVc1z7LrNncpO5biLzJStz9tozZ6/MTGbMGn7xCG1lvUivp3xAn9CeNnrNthpVrECzajrWoCfoFZqSXicw2C50BA1Ir8W+/dxaaq693eIaOnb1JXTi6k64cGvOrCQKNmzFG9Abyet+6K1Eq/k8hXtMq6lEwc4sDrajZgXr0LuruQdfoVKiYN+PWYn8MvgOj7ma1w+6upC/CEauoANLj3hceoVwyLyaFgeI/NrhkAU1LQ4Q+bXDIYtqWhwg8muHQ5bUBM9WHIDejZrdgIOW1QSTFgebULNuRi0N2tRGhr0ZV29kr2scQg/QHXSbj/eWflOeIUtBTqFzSx/JvrYzGhoa/ilfELFfYZzMhUcAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAAB00lEQVR4Xu2WTSgFURTHj29ZKEVSvsrHxsbGRhayYW2lZKUsJUTKQtnJhoWUCG8lZWFlK2SlpChZWPiIUrJSEs5x7rx353h33nvmNTP17q9+Nfd/7sx977yZ+wbAYvFgCX1EL2Uh15lDC7Txu3ac89ygTdr4WzuONM8yEOwCf5krtEzU0qUb+Br76Da4GxU5zoE/rGMyioBr9WpMjwWNa+IzMmMPEutViVokoV/S1Jwj9F5ki+CePwK8f5gsVfNO0XJ1fAbmNSOFV3MoXxFZp8ozRZ5zi+aLzEUx2of2ykKAmJrjPEKzIm9Qeb/IUyHXeBXjOC3Ak+lWm0SH3OVAMTWnHTgfFzntFZRPizwVTrNX0Te00VVVDABPqpYFD2IebqGb6Aa6jq6hzb9npYepOT3A+ajIK1RO62QduvA1WqgZJqbmtALnYyKvVPm8yH3TAYnH6UAzTEzNyQPOZ0Rep/JBkftmAvjCnrt0EhYytI1PSwtTcwjKTf9W/33XMeJsciWyECJezflCL0Q2Beb5vvlET7Qx3UW12jhojsH8g3XB30bQeFlkWYX2GVqEPAR+voOG3l6fgN+A79AH9AUd1idB4k7ZQT+A/xUtFovFYrFYdH4ABy9/NabEsCEAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAXCAYAAABnGz2mAAABSUlEQVR4Xu2VPS9EURCGh2g0olBRUiglm/gL6lX4AxofISSUfgEV3UajUugUGlqEikSImhDRkfiKhHdyzm5m3z2zssm9NtncJ3myZ945uTO5xV6RggZG4SP8hCXqtZUdc34w57YyBE9MXYE9pk6yAmc5NKzBF/gGp6nXCj/wGw7DferV2IVfEi6rc/XtGtfw0NRX8NjUraBvrTpvi3pJvMX6JPQYzfrjuRu+N3E13huEe/E8KeEZE7F28Ra7EH+xbQ7/4JbqEXhEWQPeYtXXznh5M/TNjZl6HJZNneQ/FlPO4AE8hZvUS6JD5jkUfwEvzxwdssCh+At4eebokEUOxV/AyzNHhyxxCF4lvYBmNxzmgQ5a5hBMib9Y7h/hAQmDNrgR0d6Mqddjlhv6L/wM7+Fd/H2S8Jmy9EpY5Bxewg/YVXejoKCgQ/kFtk5fv+wIHE0AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAXCAYAAABwOa1vAAABfElEQVR4Xu2WOy9EURSFN1HQCInGX1BJdP6GRKnReISQUNJNg5JGRIlELSSi1KhIhEblEfGMV7xCWCvnDHv2zL6D4o7ifsmXe/Y6M/eum7mTGZGMH9MJ3+ABbDF7q/AEzpq8YtTCUzV/wKq4vlU5b2RCzRXlWq1ZuNSarJi5iBHYa0PFGLyDj7Db7P2Fabim5g0Jpbuk8KYKWISvEl5I+wq3v9iD62rehZtq/i05eA4bTf4ioceDyUviFa6X4o+LMGuI62r4lOBofJ2F52iN6/d4rJFQvGxpr/C2+IXnbFiGDrikZp6Dn1wznFF5fi8Rr3D+cbF4eRJ8Ti/VzPe3x/Whygm/K4mkUZiw8AK8guMqb4t78/BG5S68eL8NxS/m5anBiw/YUPxiXp4avPigDcUv5uWpwYsP2RDcS+lizPZtmCYsMGxDCX9WvML8olSEJgkFpuxGhHs9ap6MWeoswwt4DI/i8UzCz7WmTkLBLbgDn+X7X1ZGRkbGP+ATVhN0Tv+Ie8IAAAAASUVORK5CYII=>