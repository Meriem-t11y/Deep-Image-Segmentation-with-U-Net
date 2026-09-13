# Deep Image Segmentation with U-Net

<p align="center">
  <b>Semantic Image Segmentation • PyTorch • U-Net • Computer Vision</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Torchvision-Computer%20Vision-orange">
  <img src="https://img.shields.io/badge/NumPy-Data%20Processing-013243?logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c">
</p>

## Overview

This project explores **semantic image segmentation** using a U-Net architecture implemented with PyTorch.

The objective is to segment pets from their surrounding background at the pixel level using the **Oxford-IIIT Pet Dataset**.

Rather than focusing only on model training, the project investigates how different training strategies affect segmentation performance through quantitative evaluation and qualitative error analysis.

### Key Focus

* U-Net implementation from scratch
* Pixel-level semantic segmentation
* BCE and Dice-based optimization
* Data augmentation
* Dice and IoU evaluation
* Qualitative prediction analysis
* Error analysis on difficult samples

---

## Dataset

### Oxford-IIIT Pet

The Oxford-IIIT Pet Dataset contains images of cats and dogs together with pixel-level segmentation annotations.

**Dataset configuration**

| Property          | Configuration           |
| ----------------- | ----------------------- |
| Dataset           | Oxford-IIIT Pet         |
| Task              | Semantic Segmentation   |
| Training split    | 80%                     |
| Validation split  | 20%                     |
| Input resolution  | 256 × 256               |
| Number of classes | Foreground / Background |

Images were resized using bilinear interpolation, while segmentation masks were resized using nearest-neighbor interpolation to preserve their discrete labels.

---

## Model Architecture

The segmentation model is a custom **U-Net** implemented in PyTorch.

```mermaid
flowchart TD
    In[Input Image] --> Enc1[Encoder Block 1]
    
    subgraph Contracting_Path ["Contracting Path (Encoder)"]
        Enc1 -->|Downsample| Enc2[Encoder Block 2]
        Enc2 -->|Downsample| Enc3[Encoder Block 3]
    end

    Enc3 -->|Downsample| Bottleneck[Bottleneck]

    subgraph Expanding_Path ["Expanding Path (Decoder)"]
        Dec3 -->|Upsample| Dec2[Decoder Block 2]
        Dec2 -->|Upsample| Dec1[Decoder Block 1]
    end

    Bottleneck -->|Upsample| Dec3[Decoder Block 3]
    Dec1 --> Out[Segmentation Mask]

    %% Skip Connections
    Enc1 -.->|Skip Connection| Dec1
    Enc2 -.->|Skip Connection| Dec2
    Enc3 -.->|Skip Connection| Dec3

    %% Styling
    style In fill:#1a202c,stroke:#e2e8f0,color:#fff
    style Out fill:#22c55e,stroke:#4ade80,color:#fff
    style Bottleneck fill:#8b5cf6,stroke:#c084fc,color:#fff
    style Enc1 fill:#0284c7,stroke:#38bdf8,color:#fff
    style Enc2 fill:#0284c7,stroke:#38bdf8,color:#fff
    style Enc3 fill:#0284c7,stroke:#38bdf8,color:#fff
    style Dec1 fill:#0d9488,stroke:#2dd4bf,color:#fff
    style Dec2 fill:#0d9488,stroke:#2dd4bf,color:#fff
    style Dec3 fill:#0d9488,stroke:#2dd4bf,color:#fff

```

The network contains:

* Convolutional encoder blocks
* Max-pooling layers
* Bottleneck representation
* Transposed convolutions for upsampling
* Skip connections
* Convolutional segmentation head

Skip connections allow the decoder to recover spatial information lost during downsampling.

---

## Training Strategy

The project investigates the effect of different training configurations.

### Loss Function

The main segmentation objective combines:

``` text
Loss = BCE Loss + Dice Loss
```

**Binary Cross-Entropy** provides pixel-level supervision, while **Dice Loss** encourages better overlap between predicted and ground-truth regions.

### Data Augmentation

The final experiment uses random:

* Horizontal flipping
* Vertical flipping

Augmentation is applied only during training.

---

## Experiments

### Experiment 1 — BCE-only Baseline

A U-Net model was trained using only Binary Cross-Entropy.

| Metric          | Result |
| --------------- | -----: |
| Validation Dice | 0.0383 |
| Validation IoU  | 0.0224 |

The very low overlap scores indicate that BCE-only training was not effective for this segmentation setup.

---

### Experiment 2 — BCE + Dice + Augmentation

The final configuration combines:

```text
U-Net
+
BCE Loss
+
Dice Loss
+
Data Augmentation
```

Results:

| Metric          |     Result |
| --------------- | ---------: |
| Validation Dice | **0.6571** |
| Validation IoU  | **0.5057** |

### Final Comparison

| Configuration             |       Dice |        IoU |
| ------------------------- | ---------: | ---------: |
| BCE-only                  |     0.0383 |     0.0224 |
| BCE + Dice + Augmentation | **0.6571** | **0.5057** |

The final configuration substantially improved segmentation overlap compared with the BCE-only baseline.

---

## Qualitative Results

The model predictions are evaluated visually by comparing:

```text
Original Image → Ground Truth → Prediction
```

Example predictions and difficult cases are provided in the `results/` directory.

### Prediction Examples

![Segmentation Predictions](results/predictions.png)

### Error Analysis

The model's difficult cases were identified according to their Dice scores.

![Error Analysis](results/error_analysis.png)

This analysis helps identify cases where segmentation quality decreases because of challenging object shapes, boundaries, or background conditions.

---

## Evaluation Metrics

### Dice Score

Dice measures the overlap between the predicted segmentation and the ground-truth mask.

Higher values indicate better segmentation overlap.

### Intersection over Union

IoU measures the ratio between the intersection and union of the predicted and ground-truth regions.

Higher values indicate better segmentation quality.

---

## Technologies

| Category            | Tools           |
| ------------------- | --------------- |
| Language            | Python          |
| Deep Learning       | PyTorch         |
| Computer Vision     | Torchvision     |
| Numerical Computing | NumPy           |
| Visualization       | Matplotlib      |
| Model               | U-Net           |
| Dataset             | Oxford-IIIT Pet |

---

## Project Structure

```text
deep-image-segmentation-unet/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebook/
│   └── segmentation.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── evaluate.py
│
└── results/
    ├── predictions.png
    └── error_analysis.png
```

---

## Reproducibility

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/deep-image-segmentation-unet.git
cd deep-image-segmentation-unet
pip install -r requirements.txt
```

The complete experimental workflow is available in the Jupyter notebook.

---

## Key Takeaways

This project demonstrates practical experience with:

* Semantic image segmentation
* U-Net architecture design
* Encoder-decoder CNNs
* Skip connections
* Segmentation-specific loss functions
* Data augmentation
* Dice and IoU evaluation
* Quantitative model comparison
* Qualitative error analysis
* PyTorch implementation

---

## Future Work

Possible extensions include:

* Transfer-learning-based segmentation
* Stronger geometric and photometric augmentation
* Weighted BCE + Dice objectives
* Learning-rate scheduling
* Early stopping and model checkpointing
* Comparison with modern segmentation architectures
* Evaluation across different object categories

---

## Author

**Meriem Tafraoui**

AI Engineering Student 
• Computer Vision • Deep Learning • Artificial Intelligence  • machine learning
