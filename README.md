# Phoneme Classification with MLP — CMU 11-685

> **Academic Competition / Project** for **11-685: Introduction to Deep Learning (Spring 2026)**  
> Carnegie Mellon University  
> Kaggle Competition: [Frame-Level Classification of Speech](https://www.kaggle.com/competitions/hw-1-p-2-spring-2026-student-competition/leaderboard)

---

## Overview

This project tackles **frame-level phoneme state classification** from Mel spectrogram features using a feedforward Multi-Layer Perceptron (MLP). The dataset consists of Wall Street Journal (WSJ) speech recordings pre-processed into 28-dimensional MFCC frames, each labelled with one of **40 phoneme state classes** (120 subphoneme states total).

The model receives a frame of interest along with surrounding **context frames** and predicts the phoneme state label for that frame.

**Competition Result:** Rank **123 / 268** · Private Score: **0.86132** · [View Leaderboard](https://www.kaggle.com/competitions/hw-1-p-2-spring-2026-student-competition/leaderboard)

---

## Model Architecture

A **inverted pyramid MLP** — wide in the beginning, narrowing towards the end. This allows the network to first expand its representational capacity to capture complex patterns, then progressively compress them into a discriminative output space.

```
Input  →  [2240]  →  [2048]  →  [1024]  →  [512]  →  [256]  →  [128]  →  Output [40]
          Linear     Linear     Linear     Linear     Linear     Linear
          BN+GeLU    BN+GeLU    BN+GeLU    BN+GeLU    BN+GeLU    Dropout
```

> Input size = `(1 + 2 × context) × 28` = `(1 + 2 × 40) × 28 = 2268`  
> 7 layers total · GeLU activations · BatchNorm after each linear layer · Dropout regularization

```
         ┌─────────────┐
         │  Input      │  2268
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Linear     │  2048  ← BatchNorm + GeLU
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Linear     │  1024  ← BatchNorm + GeLU
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Linear     │  512   ← BatchNorm + GeLU  (widest / bottleneck)
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Linear     │  256   ← BatchNorm + GeLU
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Linear     │  128   ← BatchNorm + GeLU
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Dropout    │  0.3
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Output     │  40  (phoneme state classes)
         └─────────────┘
```

---

## Configuration

| Hyperparameter       | Value                                      |
|----------------------|--------------------------------------------|
| Architecture         | Diamond MLP (7 layers)                     |
| Activation           | GeLU                                       |
| Context Size         | 40                                         |
| Input Dimension      | `(1 + 2×40) × 28 = 2268`                  |
| Hidden Dims          | 2048 → 1024 → 512 → 256 → 128             |
| Output Classes       | 40                                         |
| Batch Normalization  | After every linear layer                   |
| Dropout              | 0.3 (before final layer)                   |
| Optimizer            | AdamW                                      |
| Loss Function        | CrossEntropyLoss (label smoothing = 0.1)   |
| LR Scheduler         | ReduceLROnPlateau (factor=0.5, patience=2) |
| Freq Mask Param      | 8                                          |
| Time Mask Param      | 20                                         |
| Batch Size           | 2048                                       |

---

## 📊 Results

| Split          | Score   |
|----------------|---------|
| Public Score   | 0.86132 |
| Private Score  | 0.86132 |
| Competition Rank | **123 / 268** |

---

## Key Learnings

### 1. Train on a smaller subset first
Running early experiments on a fraction of `train-clean-100` saves significant compute time. Once the best configuration is found, retrain on the full dataset.

### 2. Deeper, diamond-shaped architectures work best
Inverse-pyramid and diamond architectures (wide base, narrow output) consistently outperformed shallow or cylinder-shaped networks. The expanding-then-contracting structure helps the model build rich intermediate representations before making final predictions.

### 3. GeLU outperforms ReLU and other activations
GeLU (Gaussian Error Linear Unit) is smoother than ReLU — it doesn't hard-zero negative activations but instead down-weights them probabilistically. This allows gradients to flow more smoothly during training, avoids "dead neuron" problems common in ReLU, and empirically improves convergence and final accuracy on this task.

### 4. BatchNorm + Dropout are essential
Adding Batch Normalization after every linear layer stabilizes training by normalizing activations, reducing internal covariate shift, and enabling higher learning rates. Dropout adds regularization that prevents the model from overfitting to specific training features, improving generalization on the validation and test sets.

### 5. SpecAugment (Frequency & Time Masking) helps generalization
Randomly masking frequency bands (`freq_mask_param=8`) forces the model to not over-rely on any single frequency feature in the mel spectrogram — similar to how dropout regularizes neurons. Randomly masking time steps (`time_mask_param=20`) teaches the model to be robust to missing or corrupted temporal context, mimicking real-world noise conditions. Together, these augmentations act as strong data regularizers that reduce overfitting without requiring more data.

---

## Dataset

The dataset is sourced from the **LibriSpeech corpus** (Wall Street Journal read-aloud articles) and is provided via Kaggle. All audio has been pre-processed into **28-dimensional Mel-frequency cepstral coefficient (MFCC) frames** — no raw waveform processing is required.

### Structure

```
data/
├── train-clean-100/
│   ├── mfcc/        # 28,539 utterances → each of shape (T, 28)
│   └── transcript/  # 28,539 label sequences → each of shape (T,)
├── dev-clean/
│   ├── mfcc/        # 2,703 utterances
│   └── transcript/  # 2,703 label sequences
└── test-clean/
    └── mfcc/        # 2,620 utterances (no labels — Kaggle submission)
```

### Key Specifics

| Property | Detail |
|---|---|
| Feature type | Mel-frequency cepstral coefficients (MFCCs) |
| Feature dimension | 28 per frame |
| Frame length | 25 ms |
| Frame stride | 10 ms (→ 100 frames/second of speech) |
| Labels | Phoneme state indices, integers in `[0, 39]` |
| Total phonemes | 40 (e.g. AA, AE, SIL, +BREATH+, …) |
| Phoneme states | 3 per phoneme → 120 subphoneme states modelled via HMM; task predicts the 40-class mapping |
| Utterance length | Variable — each MFCC is of shape `(T, 28)` where T varies per recording |
| Train set size | 28,539 utterances |
| Dev set size | 2,703 utterances |
| Test set size | 2,620 utterances (unlabelled) |

### Preprocessing & Context

Each frame is classified using a **sliding window of context frames** around it. With a context size of 40, each input to the MLP is a window of 81 frames flattened to `81 × 28 = 2268` features. The boundaries of each utterance are zero-padded so every frame (including the first and last) has a full context window.

Cepstral mean-variance normalisation can optionally be applied per utterance to remove channel effects from the MFCC coefficients.

---

## How to Run

1. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/).
2. Open `assignment-1-irene_on_time.ipynb` in Kaggle or Google Colab (GPU recommended).
3. Set your hyperparameters in the config section.
4. Run all cells sequentially to train, evaluate, and generate predictions.

---
