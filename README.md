# Mamba-1: Pre-training & Downstream Benchmarks on WikiText-103

A lightweight implementation and empirical evaluation of the **Mamba-1 (Selective State Space Model)** architecture trained from scratch on `WikiText-103-raw-v1`. This repository contains scripts for raw pre-training, inference profiling across context lengths, validation perplexity evaluation, zero-shot assessment, and supervised fine-tuning (SFT) on downstream GLUE/SuperGLUE tasks.

---

## 📌 Architecture & Model Specifications

The network employs the Mamba-1 selective scan mechanism ($S6$) with hardware-aware fused CUDA kernels (`mamba-ssm`, `causal-conv1d`) alongside a native pure-PyTorch sequential fallback for CPU environments.

| Hyperparameter                        | Value                           |
| ------------------------------------- | ------------------------------- |
| **Parameters**                        | ~34.7M                          |
| **Layers (`n_layer`)**                | 16                              |
| **Model Dimension (`d_model`)**       | 384                             |
| **State Dimension (`d_state`)**       | 16                              |
| **Expansion Factor (`expand`)**       | 2 ($d_{inner} = 768$)           |
| **Max Context Window (`block_size`)** | 512 / 1024                      |
| **Tokenizer**                         | GPT-2 (`vocab_size = 50,257`)   |
| **Weight Tying**                      | LM Head tied to Embedding table |

---

## 🚀 Pre-training Setup

* **Dataset:** `wikitext` (`wikitext-103-raw-v1`) streamed and packed into continuous 512-token chunks using an EOS delimiter.

* **Optimizer:** AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$, and `weight_decay = 0.1`.

* **Learning Rate Schedule:** `OneCycleLR` with a peak learning rate of $3 \times 10^{-4}$ and a 500-step linear warmup.

* **Precision:** Mixed-precision training using `torch.bfloat16` with gradient scaling.

* **Hardware:** Multi-GPU training using PyTorch `DataParallel` with a batch size of 32.

---


## 📊 Benchmarks & Empirical Results

### 1. Inference Latency & Memory Scaling (NVIDIA T4 GPU)

Benchmarked at batch size 1 across increasing sequence lengths:

| Sequence Length | GPU Latency | GPU Throughput | Peak VRAM |    CPU Fallback Latency |
| --------------- | ----------: | -------------: | --------: | ----------------------: |
| **128**         |    13.70 ms |  9,345.8 tok/s |  167.5 MB |  361.3 ms (354.3 tok/s) |
| **256**         |    15.38 ms | 16,645.2 tok/s |  193.1 MB |  740.5 ms (345.7 tok/s) |
| **512**         |    16.03 ms | 31,937.3 tok/s |  241.7 MB | 1451.7 ms (352.7 tok/s) |
| **1024**        |    22.27 ms | 45,985.2 tok/s |  340.6 MB | 3093.8 ms (331.0 tok/s) |

**Key Takeaway:** Latency scales sub-linearly on GPU due to Mamba's linear time complexity, avoiding the quadratic $O(N^2)$ attention bottleneck of standard Transformers.

---

### 2. Language Modeling Evaluation (WikiText-103)

Calculated using a strided sliding-window cross-entropy loss on the full validation split:

| Metric                          | Checkpoint (Step 10,500) |
| ------------------------------- | -----------------------: |
| **Validation Loss**             |                    ~4.39 |
| **Validation Perplexity (PPL)** |                **80.76** |

---

### 3. Downstream Task Benchmarks (GLUE & SuperGLUE)

#### Zero-Shot Baseline (Prompt Logit Scoring)

* **GLUE SST-2:** `50.69%` (Random baseline: 50.0%)
* **SuperGLUE BoolQ:** `59.30%` (Majority baseline: 62.2%)

#### Supervised Fine-Tuning (Task-Specific Classification Heads)

Replacing the generative `lm_head` with a linear classifier over the final non-padded token state:

| Benchmark Task      | Metric   | Majority Baseline | Mamba-1 (35M) SFT Score |
| ------------------- | -------- | ----------------: | ----------------------: |
| **GLUE SST-2**      | Accuracy |             50.0% |    **86.81%** (757/872) |
| **SuperGLUE BoolQ** | Accuracy |             62.2% |  **64.10%** (2096/3270) |

---

## 📝 Qualitative Text Generation

Sample outputs from the base pre-trained model (temperature = 0.8):

**Prompt:** `The history of machine learning began with`

> *The history of machine learning began with the first of his two best books, The Washing Show regulars: The Liberal Party and The War of the Lions, in which the young American Devil is playing a comic role. The novel, entitled The Story of Newfoundlanders (1809),*

**Prompt:** `In a surprising turn of events, the global economy`

> *In a surprising turn of events, the global economy is gradually broken up. The 1981 season was her highest in MLB history but one of his most successful years, at the time he was a professional. In 1931 Lou Reed was named to the All @-@ Star Game, although he*

---
