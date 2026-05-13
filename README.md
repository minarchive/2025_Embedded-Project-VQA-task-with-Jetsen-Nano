# Embedded-Project-VQA-task-with-Jetsen-Nano
Embedded Project class of 2025 : Lightweight Real-Time VQA on Jetson Nano

---

# Overview

This project aims to implement a **real-time Visual Question Answering (VQA)** system on the resource-constrained embedded environment of the **Jetson Nano**.

Unlike simple image classification tasks, VQA requires simultaneous understanding of both:

* **Visual information** from images
* **Natural language questions**

As a result, VQA models are generally computationally expensive and memory intensive.

To address this challenge, we preserved the strong representational power of a **ResNet152-based VQA architecture** while introducing a 3-stage lightweight optimization pipeline:

1. **Structural Pruning**
2. **Knowledge Distillation**
3. **FP16 Quantization**

Through this pipeline, we successfully built a **general-purpose VQA model** capable of answering diverse questions in real time under limited hardware resources.

---

# Base Architecture

### Paper Baseline

> **Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering**

---

## 1. Image Encoder — ResNet152

We utilized a pretrained **ResNet152** backbone to extract rich visual representations from input images.

Instead of using only classification logits, we extracted the final convolutional feature map to preserve spatial information.

### Output Feature

* Shape: `14 × 14 × 2048`
* Maintains object location and contextual spatial structure

This enables the model to reason about:

* object positions
* relationships
* backgrounds
* counting tasks

---

## 2. Text Encoder — LSTM

To understand user questions, we adopted a multi-layer **LSTM** network.

### Process

1. Tokenize question sentence
2. Convert into embedding vectors
3. Pass through multi-layer LSTM
4. Use final hidden state as question representation

The resulting vector represents the semantic intent of the question.

---

## 3. Attention-based Multimodal Fusion

The image and question features are combined through an attention mechanism.

### Pipeline

1. Duplicate question vector across spatial image locations
2. Compute attention map conditioned on the question
3. Focus on relevant image regions
4. Suppress irrelevant background information
5. Predict final answer from candidate vocabulary

This allows the model to answer questions such as:

* `"What is on the background?"`
* `"How many people are there?"`
* `"What color is the object?"`

---

# Model Variants

We implemented and compared four models:

| Model               | Description                 |
| ------------------- | --------------------------- |
| (1) Base Model      | Original FP32 ResNet152 VQA |
| (2) Pruned Model    | + 50% Structural Pruning    |
| (3) Distilled Model | + Knowledge Distillation    |
| (4) Final Model     | + FP16 Quantization         |

---

# Benchmark Methodology

To ensure fair performance evaluation, we implemented a repeated inference benchmark instead of measuring single inference latency.

### Benchmark Details

* Warmup iterations applied
* Repeated inference: `repeat = 20`
* Average inference time reported

---

# (1) Base Model

## ResNetLayer4 — Visual Encoder

```python
self.r_model = resnet152(pretrained=True)
```

We used a pretrained ResNet152 trained on ImageNet for strong visual representation learning.

Although highly accurate, ResNet152 is computationally heavy, making it a primary target for lightweight optimization.

---

## VQAResNetModel — Full VQA Pipeline

### Multimodal Processing

```python
forward(self, v, q, q_len)
```

Inputs:

* `v`: image
* `q`: question
* `q_len`: question length

### Question Encoding

```python
self.text(q, ...)
```

Transforms the natural language question into vector representation.

### Image Encoding

```python
self.resnet_layer4(v)
```

Extracts spatial visual features.

---

## Attention Mechanism

```python
a = self.attention(v, q)
v = apply_attention(v, a)
```

The model learns where to focus depending on the question.

Example:

* `"What color is the car?"`
  → attention focuses on the car region

---

## Final Answer Prediction

```python
combined = torch.cat([v, q], dim=1)
```

Image and question features are fused and passed into the classifier for answer prediction.

---

# (2) Structural Pruning (50%)

We applied **L1-Norm based Structured Pruning** to physically remove low-importance convolution filters.

Unlike unstructured sparsity, structured pruning reduces:

* actual computation
* memory usage
* inference latency

---

## Filter Pruning Strategy

### Importance Metric

* L1 Norm of convolution filters

### Procedure

1. Rank filters by L1 magnitude
2. Remove bottom 50%
3. Keep high-importance filters only

Because ResNet bottleneck blocks contain wide intermediate channels, pruning significantly reduces computation.

---

## Key Functions

### `prune_bottleneck_layer`

* Computes filter importance
* Removes low-ranked filters
* Generates `keep_indices`

### `apply_structured_pruning`

Applied pruning to:

* all 50 bottleneck blocks
* Conv1 / BN1
* Conv2 / BN2
* Conv3 layers

---

# Pruning Results

| Metric         | Original   | Pruned     | Improvement  |
| -------------- | ---------- | ---------- | ------------ |
| Parameters     | 80,075,210 | 45,043,888 | ↓ 43.7%      |
| Memory         | 306.04 MB  | 172.32 MB  | ↓ 43.7%      |
| Inference Time | 851.45 ms  | 460.98 ms  | 1.85× Faster |

---

## Limitation

Although pruning improved efficiency, accuracy degradation occurred because important representational capacity was partially removed.

To recover performance, we introduced **Knowledge Distillation**.

---

# (3) Knowledge Distillation

## Teacher–Student Framework

| Role    | Model                  |
| ------- | ---------------------- |
| Teacher | Original ResNet152 VQA |
| Student | 50% Pruned Model       |

Dataset:

* VQA v2
* MS COCO 2014 Validation Set

---

## Distillation Loss

Implemented in:

```python
distillation.py
```

The loss combines:

### 1. Soft Loss

KL-Divergence between:

* Teacher softmax distribution
* Student softmax distribution

```python
Temperature T = 4.0
```

This transfers inter-class relational knowledge.

---

### 2. Hard Loss

CrossEntropy with ground truth labels.

---

## Final Loss

```python
alpha * hard_loss + (1 - alpha) * soft_loss
```

This allows the student model to:

* remain lightweight
* recover original model performance

---

## Distillation Results

### Advantages

* No additional memory increase
* No inference slowdown
* Significant accuracy recovery

Teacher knowledge successfully compensated for pruning-induced degradation.

---

# (4) FP16 Quantization

Finally, we applied **FP16 Quantization** to reduce numerical precision:

```python
torch.float32 → torch.float16
```

Implemented via:

```python
model.half()
```

---

## Benefits

FP16 reduces:

* memory bandwidth
* GPU memory usage
* computational overhead

Converted weights were stored in:

```python
fp16_state
```

---

# Final Performance Comparison

| Model              | Inference Time | Memory    | Parameters | Notes                 |
| ------------------ | -------------- | --------- | ---------- | --------------------- |
| Original (FP32)    | 851.45 ms      | 306.04 MB | 80,075,210 | Baseline              |
| Pruned (50%)       | 460.98 ms      | 172.32 MB | 45,043,888 | 1.85× Faster          |
| Pruned + Distilled | 460.50 ms      | 172.32 MB | 45,043,888 | Accuracy Recovered    |
| Final (FP16)       | 483.24 ms      | 86.10 MB  | 45,043,888 | ~72% Memory Reduction |

---

# Demo System

We implemented a Flask-based demo interface with automatic HTML template generation.

---

## 1. Camera Module

To utilize Jetson Nano hardware acceleration:

* OpenCV + GStreamer pipeline used
* Stateless capture design:

  * Open
  * Capture
  * Release

This avoids:

* memory leaks
* camera conflicts

Users can:

* capture images directly
* upload external images

---

## 2. Microphone Module

Speech input pipeline:

* gTTS converts text → MP3
* pydub converts MP3 → WAV

Recognized speech becomes the VQA question input.

---

## 3. Speaker Module

The predicted answer is converted back into speech through a TTS pipeline.

```python
play_audio()
```

This enables immediate auditory feedback.

---

# Demo Modes

Implemented demo systems:

1. **4-model comparison demo**

   * Original
   * Pruned
   * Distilled
   * Quantized

2. **Final optimized model demo**

---

# Final Analysis

This project successfully demonstrated that even a computationally intensive **general-purpose VQA task** can operate efficiently on embedded hardware through systematic model compression.

The 3-stage optimization pipeline achieved:

* significant latency reduction
* major memory savings
* preserved inference capability

Most importantly, the final model maintained strong reasoning ability for:

* spatial understanding
* background reasoning
* object counting
* multimodal question answering

Examples:

* `"What is on the background?"`
* `"How many objects are there?"`

---

# Key Achievements

* Real-time embedded VQA system on Jetson Nano
* Structural pruning with real speedup
* Knowledge distillation for accuracy recovery
* FP16 quantization for large memory reduction
* End-to-end multimodal demo system
* General-purpose visual reasoning capability

---

# Conclusion

Through Structural Pruning, Knowledge Distillation, and FP16 Quantization, we successfully transformed a heavy ResNet152-based VQA architecture into a lightweight embedded AI system suitable for real-world deployment.

The final model achieved:

* approximately **half the inference latency**
* approximately **one-fourth the memory usage**
* while preserving strong VQA performance

This demonstrates the feasibility of deploying advanced multimodal AI systems even in severely resource-constrained environments such as the Jetson Nano.
