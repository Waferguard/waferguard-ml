# Wafer Map Defect Classification — Architecture & Model Comparison

## Pipeline Overview

```mermaid
flowchart TD
    A[Mixed-Type Wafer Map Dataset<br/>52x52 images, 38 defect patterns] --> B[Preprocessing<br/>One-hot encode pixels → 52x52x3]
    B --> C[Train / Val / Test Split<br/>70% / 15% / 15%<br/>Iterative stratification]
    C --> D[Data Augmentation<br/>Rotations, flips, shifts<br/>Training only]
    D --> E[Focal Loss + Class Weights<br/>Handle class imbalance]

    E --> F[Phase 1: Multi-Label<br/>8 base defect types]
    E --> G[Phase 2: Multi-Class<br/>38 defect patterns]

    F --> F1[Custom CNN]
    F --> F2[MobileNetV2<br/>Transfer Learning]

    F1 -.->|Freeze backbone<br/>Replace head| G1[Phase 2 CNN]
    F2 -.->|Freeze backbone<br/>Replace head| G2[Phase 2 MobileNetV2]

    G --> G1
    G --> G2
```

## Model Architectures

### Phase 1 — Multi-Label Classification (8 defect types)

```mermaid
flowchart LR
    subgraph CNN ["Custom CNN (424K params)"]
        direction LR
        I1[Input<br/>52x52x3] --> C1[Conv2D 32<br/>+BN+ReLU<br/>+MaxPool]
        C1 --> C2[Conv2D 64<br/>+BN+ReLU<br/>+MaxPool]
        C2 --> C3[Conv2D 128<br/>+BN+ReLU<br/>+MaxPool]
        C3 --> C4[Conv2D 256<br/>+BN+ReLU<br/>+MaxPool]
        C4 --> GAP1[GlobalAvgPool]
        GAP1 --> D1[Dense 128]
        D1 --> DR1[Dropout 0.5]
        DR1 --> O1[Dense 8<br/>sigmoid]
    end
```

```mermaid
flowchart LR
    subgraph TL ["MobileNetV2 Transfer Learning (2.6M params)"]
        direction LR
        I2[Input<br/>52x52x4] --> P[Conv2D 1x1<br/>Project → 3ch]
        P --> R[Resize<br/>→ 224x224]
        R --> MN[MobileNetV2<br/>Frozen ImageNet<br/>backbone]
        MN --> GAP2[GlobalAvgPool]
        GAP2 --> D2[Dense 128]
        D2 --> DR2[Dropout 0.5]
        DR2 --> O2[Dense 8<br/>sigmoid]
    end
```

### Phase 2 — Multi-Class Classification (38 patterns)

Reuses Phase 1 convolutional backbones as frozen feature extractors with a new classification head.

```mermaid
flowchart LR
    subgraph P2 ["Phase 2 Model (both variants)"]
        direction LR
        I3[Input<br/>52x52x3] --> BB[Phase 1 Backbone<br/>Frozen conv layers]
        BB --> D3[Dense 128 + ReLU]
        D3 --> DR3[Dropout]
        DR3 --> O3[Dense 38<br/>softmax]
    end
```

**Two-stage training:**
1. **Stage A** — Head-only (backbone frozen), lr=1e-3, ~30 epochs
2. **Stage B** — Unfreeze last 2 conv blocks, fine-tune end-to-end, lr=1e-5, ~30 epochs

## Training Configuration

| Setting | Value |
|---|---|
| Loss (Phase 1) | Binary Focal Loss (gamma=2.0, per-label weights) |
| Loss (Phase 2) | Sparse Categorical Crossentropy + class weights |
| Optimizer | Adam (lr=1e-3) |
| Callbacks | ReduceLROnPlateau (patience=5), EarlyStopping (patience=10) |
| Max Epochs | 100 (Phase 1), 30+30 (Phase 2 stages A+B) |

## Results Comparison

| Model | Macro F1 | Weighted F1 |
|---|---|---|
| **Phase 1 — Custom CNN (8-label)** | **0.9779** | **0.9905** |
| Phase 1 — MobileNetV2 (8-label) | 0.8998 | 0.8847 |
| Phase 2 — Custom CNN (38-class) | 0.9736 | 0.9735 |
| Phase 2 — MobileNetV2 (38-class) | 0.9667 | 0.9658 |

### Phase 1 — Per-Label F1 (8 base defect types)

| Defect Type | Custom CNN | MobileNetV2 |
|---|---|---|
| Center | 1.00 | 0.97 |
| Donut | 1.00 | 1.00 |
| Edge_Loc | 0.99 | 0.75 |
| Edge_Ring | 0.99 | 0.88 |
| Loc | 0.99 | 0.89 |
| Near_Full | 0.90 | 0.88 |
| Scratch | 0.98 | 0.84 |
| Random | 0.97 | 0.99 |

### Key Takeaways

- The **Custom CNN consistently outperforms MobileNetV2** across both phases, likely because wafer map patterns (discrete categorical pixels) differ significantly from natural images that ImageNet features were trained on.
- The lightweight Custom CNN (424K params) beats MobileNetV2 (2.6M params) — **6x fewer parameters, better results**.
- Phase 2 models maintain strong performance when scaling from 8 labels to 38 classes, with the Custom CNN backbone achieving 0.97 Macro F1 on all 38 defect patterns.
