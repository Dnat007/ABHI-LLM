# 🚀 ABHILLM — Medical Domain LLM

> **A locally fine-tuned, medical-focused language model built from Qwen3-4B using QLoRA and PEFT on a consumer GPU.**

ABHILLM is my personal LLM project focused on understanding the complete lifecycle of building and adapting a domain-specific language model — from dataset preparation and fine-tuning to evaluation, inference, RAG, memory, and deployment.

The project started with a simple question:

> **Can I fine-tune and run a useful LLM locally with limited GPU resources?**

Instead of relying only on hosted APIs, I wanted to understand what actually happens behind an LLM and gain hands-on experience with the complete pipeline.

---

## 🎯 Project Goals

The main goals of ABHILLM are:

* Build and fine-tune a domain-focused LLM
* Understand LLM fine-tuning at a practical level
* Work with large-scale medical QA datasets
* Experiment with parameter-efficient fine-tuning
* Run LLM experiments on limited GPU hardware
* Improve domain-specific knowledge and instruction following
* Build a foundation for RAG and conversational memory
* Understand the complete LLM development lifecycle

---

## 🧠 Base Model

**Base Model:** Qwen3-4B

The base model is loaded using **4-bit quantization** to make local experimentation possible on limited VRAM.

Instead of modifying the complete model, ABHILLM uses **LoRA adapters** for parameter-efficient fine-tuning.

---

## 🔧 Technical Stack

| Component                    | Technology                |
| ---------------------------- | ------------------------- |
| Base LLM                     | Qwen3-4B                  |
| Fine-tuning                  | QLoRA                     |
| PEFT                         | LoRA                      |
| Quantization                 | 4-bit                     |
| Framework                    | PyTorch                   |
| LLM Framework                | Hugging Face Transformers |
| Parameter Efficient Training | PEFT                      |
| Dataset Processing           | Python                    |
| GPU                          | NVIDIA RTX 3050 6GB       |
| OS                           | Windows                   |
| Language                     | Python                    |

---

## 📚 Training Data

The initial training experiments use medical question-answering datasets:

### MedMCQA

MedMCQA contains multiple-choice medical questions covering a wide range of medical subjects.

The dataset was processed through a custom pipeline involving:

```text
Raw Dataset
     ↓
Data Validation
     ↓
Quality Filtering
     ↓
Cleaning
     ↓
Format Conversion
     ↓
Train / Validation Split
     ↓
Training-Ready Dataset
```

After quality filtering, more than **180K medical QA samples** were retained for the processed dataset.

### PubMedQA

PubMedQA is used as an additional biomedical question-answering dataset to expose the model to medical research-oriented questions and answers.

---

## 🧹 Dataset Engineering

A significant portion of the project focuses on data quality rather than simply downloading datasets.

The preprocessing pipeline performs:

* JSON/JSONL processing
* Invalid record detection
* Empty-field detection
* Question validation
* Answer validation
* Quality filtering
* Text normalization
* Conversational format conversion
* Train/validation preparation
* Dataset statistics and analysis

The goal is to provide the model with **consistent and high-quality training examples**.

---

## ⚡ Fine-Tuning Approach

Full fine-tuning of a 4B parameter model is not practical on a 6GB GPU.

Therefore, ABHILLM uses:

```text
Qwen3-4B
     ↓
4-bit Quantization
     ↓
QLoRA
     ↓
LoRA Adapters
     ↓
Parameter-Efficient Fine-Tuning
     ↓
ABHILLM
```

### Why QLoRA?

QLoRA makes it possible to fine-tune large language models with significantly lower GPU memory requirements.

Instead of updating all model parameters:

* The base model remains frozen
* The model is loaded in 4-bit precision
* LoRA adapters are added
* Only the adapter parameters are trained
* The resulting adapter can be used with the base model during inference

This approach makes experimentation possible on a **consumer RTX 3050 6GB GPU**.

---

## ⚙️ Current Training Configuration

The current experiments use approximately:

```text
Base Model       : Qwen3-4B
Quantization     : 4-bit
Fine-Tuning      : QLoRA
LoRA Rank        : 16
LoRA Alpha       : 32
LoRA Dropout     : 0.05
Micro Batch Size : 1
Gradient Accum.  : 16
Max Sequence     : 1024
```

The configuration may change as experiments continue.

---

## 🖥️ Hardware Constraints

One of the interesting parts of this project is that it is being developed with limited hardware.

### Current Hardware

```text
GPU : NVIDIA RTX 3050 Laptop GPU
VRAM: 6GB
```

Because of the limited VRAM, memory optimization is an important part of the project.

This includes:

* 4-bit quantization
* QLoRA
* Gradient accumulation
* Small micro-batches
* Efficient model loading
* Local dataset processing

The goal is to understand how far practical LLM experimentation can be pushed with consumer hardware.

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────┐
                    │   Medical Data   │
                    │ MedMCQA / PubMedQA│
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Data Processing  │
                    │ Cleaning/Filter  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Training Dataset │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    Qwen3-4B      │
                    │   Base Model     │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 4-bit Quantization│
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │      QLoRA       │
                    │  LoRA Adapters   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │     ABHILLM      │
                    │ Fine-tuned Model │
                    └────────┬─────────┘
                             ↓
                 ┌───────────┴───────────┐
                 ↓                       ↓
          ┌──────────────┐       ┌──────────────┐
          │   Inference  │       │  Evaluation  │
          └──────┬───────┘       └──────────────┘
                 ↓
          ┌──────────────┐
          │ Future RAG   │
          │ + Memory     │
          └──────────────┘
```

---

## 📂 Project Structure

```text
ABHILLM/
│
├── models/
│   └── Qwen3-4B/
│
├── datasets/
│   └── medical/
│       ├── raw/
│       │   ├── medmcqa/
│       │   └── pubmedqa/
│       │
│       └── processed/
│           ├── medmcqa_clean/
│           └── pubmedqa/
│
├── training/
│   ├── train.py
│   ├── config.py
│   └── utils.py
│
├── inference/
│   └── inference.py
│
├── evaluation/
│   └── evaluate.py
│
├── outputs/
│   └── checkpoints/
│
├── rules.yaml
│
└── README.md
```

---

## 🔬 Development Pipeline

The current development process follows:

```text
1. Dataset Collection
        ↓
2. Dataset Analysis
        ↓
3. Data Cleaning
        ↓
4. Quality Filtering
        ↓
5. Conversational Formatting
        ↓
6. Train/Validation Split
        ↓
7. 4-bit Model Loading
        ↓
8. QLoRA Fine-Tuning
        ↓
9. Checkpoint Generation
        ↓
10. Evaluation
        ↓
11. Local Inference
```

---

## 🧪 Evaluation

Evaluation is an ongoing part of ABHILLM.

The objective is not only to measure whether the model can reproduce training patterns, but also to evaluate:

* Medical question answering
* Answer consistency
* Instruction following
* Domain-specific reasoning
* Generalization to unseen questions
* Response quality
* Hallucination behavior
* Inference performance

Future evaluation will include comparisons between:

```text
Base Qwen3-4B
       vs
Fine-Tuned ABHILLM
       vs
ABHILLM + RAG
```

---

## 🔮 Future Roadmap

ABHILLM is still an ongoing project.

### Phase 1 — Foundation

* [x] Select base model
* [x] Prepare medical datasets
* [x] Build data preprocessing pipeline
* [x] Quality filtering
* [x] Convert datasets into training format
* [x] 4-bit model loading
* [x] QLoRA configuration
* [ ] Complete fine-tuning experiments
* [ ] Benchmark model performance

### Phase 2 — Model Improvement

* [ ] Improve medical reasoning
* [ ] Improve instruction following
* [ ] Experiment with different LoRA configurations
* [ ] Compare training configurations
* [ ] Improve evaluation methodology
* [ ] Analyze failure cases

### Phase 3 — RAG

```text
ABHILLM
   +
Medical Knowledge Base
   ↓
Embedding Model
   ↓
Vector Database
   ↓
Retriever
   ↓
Reranker
   ↓
ABHILLM
```

The objective is to combine learned domain knowledge with external and updatable medical information.

### Phase 4 — Memory

A conversational memory layer is planned to maintain useful user context across interactions.

The planned architecture is:

```text
User
 ↓
Conversation
 ↓
Memory Extraction
 ↓
Memory Store
 ↓
Relevant Memory Retrieval
 ↓
LLM Context
 ↓
Response
```

### Phase 5 — Deployment

Future deployment experiments may include:

* Local inference API
* FastAPI
* Docker
* GPU inference optimization
* Streaming responses
* Monitoring
* Model versioning
* Production-oriented architecture

---

## ⚠️ Important Disclaimer

ABHILLM is an experimental research and engineering project.

It is **not a medical device and should not be used for diagnosis, treatment decisions, prescriptions, or other clinical decision-making**.

The model can produce incorrect, incomplete, outdated, or hallucinated information.

Medical information generated by the model should be independently verified using qualified medical professionals and authoritative medical sources.

---

## 💡 What I’m Learning Through This Project

This project is helping me move beyond simply using LLM APIs and understand the underlying engineering involved in building and adapting language models.

Key areas include:

* LLM architecture
* Transformer models
* Tokenization
* Quantization
* QLoRA
* PEFT
* Fine-tuning
* Dataset engineering
* Model evaluation
* Prompt engineering
* RAG
* Vector databases
* Conversational memory
* Local inference
* MLOps
* LLM deployment

---

## 👨‍💻 About the Project

**ABHILLM** is a personal learning and research project focused on exploring practical LLM development under real-world hardware constraints.

The project is being developed incrementally, with each experiment focused on understanding one part of the LLM lifecycle rather than treating the model as a black box.

> **The goal is not just to use an LLM.
> The goal is to understand how to build, adapt, evaluate and deploy one.**

---

## 📌 Project Status

**Status:** 🚧 Active Development

**Current Focus:** Medical domain fine-tuning with QLoRA on Qwen3-4B

**Hardware:** NVIDIA RTX 3050 6GB

**Next Milestone:** Fine-tuning evaluation → RAG integration → conversational memory → deployment

---

## 🛠️ Core Technologies

`Python` `PyTorch` `Hugging Face` `Transformers` `PEFT` `QLoRA` `LoRA` `BitsAndBytes` `NLP` `LLM` `GenAI` `RAG` `MLOps` `FastAPI` `Docker` `Machine Learning` `Deep Learning`
