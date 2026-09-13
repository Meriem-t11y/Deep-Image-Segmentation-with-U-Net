# Deep Image Segmentation with U-Net

A deep learning project focused on semantic image segmentation using a U-Net architecture. The project investigates the effect of loss functions and data augmentation on segmentation performance and includes quantitative evaluation and qualitative error analysis.

## Overview

Image segmentation requires a model to assign a class to each pixel of an image. In this project, a U-Net model is trained to segment pets from their background using the Oxford-IIIT Pet dataset.

The project focuses on three main aspects:

* Building a U-Net segmentation model with PyTorch
* Evaluating different training strategies using Dice and IoU
* Analyzing segmentation errors through qualitative predictions

## Dataset

**Oxford-IIIT Pet Dataset**

The dataset contains images of cats and dogs together with pixel-level segmentation annotations.

For this project:

* 80% of the data was used for training
* 20% was used for validation
* Images were resized to `256 × 256`
* Masks were resized using nearest-neighbor interpolation to preserve label values

## Methodology

### U-Net Architecture

The segmentation model is a U-Net consisting of:

* Encoder for extracting spatial features
* Bottleneck for high-level representation
* Decoder for recovering spatial resolution
* Skip connections between encoder and decoder layers
* Final convolution layer producing the segmentation mask

The model was implemented from scratch using PyTorch.

### Loss Function

The main training objective combines:

* Binary Cross-Entropy (BCE)
* Dice Loss

The combined objective is:

```text
Loss = BCE Loss + Dice Loss
```

BCE provides pixel-level supervision, while Dice Loss directly encourages better overlap between predicted and ground-truth segmentation masks.

### Data Augmentation

To improve generalization, training images were randomly augmented using:

* Horizontal flipping
* Vertical flipping

The validation set was kept unchanged to ensure a consistent evaluation protocol.

## Experiments

### Experiment 1 — BCE-only Baseline

The first experiment trained the U-Net using only Binary Cross-Entropy.

| Metric          |  Score |
| --------------- | -----: |
| Validation Dice | 0.0383 |
| Validation IoU  | 0.0224 |

The very low segmentation scores indicate that BCE alone was not sufficient for obtaining good foreground-background overlap in this experiment.

### Experiment 2 — BCE + Dice + Data Augmentation

The final configuration combined BCE and Dice Loss with data augmentation.

| Metric          |      Score |
| --------------- | ---------: |
| Validation Dice | **0.6571** |
| Validation IoU  | **0.5057** |

Compared with the BCE-only experiment, the final configuration produced a substantial improvement in both segmentation metrics.

## Results

| Configuration             |       Dice |        IoU |
| ------------------------- | ---------: | ---------: |
| BCE-only                  |     0.0383 |     0.0224 |
| BCE + Dice + Augmentation | **0.6571** | **0.5057** |

The results demonstrate the importance of selecting an appropriate segmentation objective and improving training diversity through augmentation.

## Error Analysis

Beyond numerical evaluation, qualitative predictions were examined using:

* Original image
* Ground-truth segmentation mask
* Predicted segmentation mask

The worst predictions were identified according to their Dice scores. This provides insight into cases where the model struggles with object boundaries, shape variations, or complex visual backgrounds.

## Qualitative Results

Example predictions are included in the `results/` directory.

Each visualization compares:

```text
Original Image | Ground Truth | Prediction
```

Additional examples focus on low-Dice predictions for error analysis.

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

## Project Structure

```text
deep-image-segmentation-unet/
│
├── README.md
├── notebook/
│   └── segmentation.ipynb
│
├── results/
│   ├── predictions.png
│   └── error_analysis.png
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── evaluate.py
│
├── requirements.txt
└── .gitignore
```

## Reproducibility

The experiments were developed using PyTorch and can be reproduced by installing the required dependencies and running the provided notebook.

```bash
pip install -r requirements.txt
```

The notebook contains the complete data preparation, model implementation, training, evaluation, and visualization workflow.

## Key Takeaways

This project provided practical experience with:

* Semantic image segmentation
* U-Net architecture
* Encoder-decoder networks
* Skip connections
* BCE and Dice-based objectives
* Data augmentation
* Dice and IoU evaluation
* Qualitative error analysis
* PyTorch model implementation and experimentation

## Future Improvements

Potential extensions include:

* Stronger geometric and photometric augmentation
* Transfer-learning-based segmentation models
* Dice + BCE weighting experiments
* Learning-rate scheduling
* Early stopping and checkpoint selection
* More extensive evaluation across different object categories
* Comparison with modern segmentation architectures
