# 📘 ASSIGNMENT-102303493  
## Learning Probability Density Functions using a Roll-Number Parameterized GAN Model

---

## 📌 Project Overview

This project extends PDF learning by using a **Generative Adversarial Network (GAN)** to learn the probability distribution of a non-linearly transformed air pollution feature.

Instead of manually fitting a Gaussian distribution, we use a neural network-based generative model to approximate the underlying data distribution.

---

## 📂 Dataset

Dataset: `data.csv`  
Feature used: **NO₂ (Nitrogen Dioxide)**

---

# 🔁 STEP 1 — Roll Number Based Non-Linear Transformation

Roll number:

```
r = 102303493
```

Derived parameters:

```
a_r = 0.5 * (r % 7)
b_r = 0.3 * ((r % 5) + 1)
```

Transformation applied:

\[
z = x + a_r \sin(b_r x)
\]

---

## Convert to PyTorch Tensor

```python
z_tensor = torch.FloatTensor(z).reshape(-1,1)
```

This prepares the data for neural network training.

---

# 🧠 STEP 2 — GAN Architecture

We build two neural networks:

## 🎯 Generator (PDFGEn)

Architecture:

```
Input: 1 (latent noise)
→ Linear(1 → 128)
→ LeakyReLU
→ Linear(128 → 256)
→ LeakyReLU
→ Linear(256 → 1)
Output: Generated sample
```

The generator maps random noise to synthetic NO₂ values.

---

## 🛡 Discriminator (PDFDisc)

Architecture:

```
Input: 1
→ Linear(1 → 256)
→ LeakyReLU
→ Linear(256 → 128)
→ LeakyReLU
→ Linear(128 → 1)
→ Sigmoid
Output: Probability (Real/Fake)
```

The discriminator outputs a value between 0 and 1.

---

# 🔥 STEP 3 — Training the GAN

Training parameters:

```
Iterations: 15000
Batch size: 64
```

Training process per iteration:

### 1️⃣ Train Discriminator

- Sample real batch from dataset
- Generate fake samples from Generator
- Compute:
  - Real loss
  - Fake loss
- Update discriminator weights

### 2️⃣ Train Generator

- Generate fake samples
- Try to fool discriminator
- Update generator weights

---

## Training Output

Example loss logs:

```
Iter: 0      D Loss: 1.587   G Loss: 0.708
Iter: 3000   D Loss: 1.419   G Loss: 0.874
Iter: 6000   D Loss: 1.391   G Loss: 0.771
Iter: 9000   D Loss: 1.385   G Loss: 0.783
Iter: 12000  D Loss: 1.399   G Loss: 0.679
```
---

**Author:** Roll Number 102303493  
Course Assignment – Probability Density Function Learning using GAN
