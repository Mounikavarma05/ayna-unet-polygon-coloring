# Ayna ML Internship Assignment – Polygon Colorization using UNet

## Problem Statement

Design a deep learning model that takes as input:
- A **polygon image** (e.g., triangle, square)
- A **color name** (e.g., "blue", "red")

And outputs an image of the **same polygon filled with the specified color**.

## Dataset Description

The dataset is structured as:
dataset/
└── training/
├── inputs/ # Grayscale polygon images
├── outputs/ # Colored polygon ground truth images
└── data.json # Maps input image + color name to output

└── validation/
├── inputs/
├── outputs/
└── data.json


## Model Architecture

I implemented a **UNet architecture from scratch in PyTorch**, with modifications to support **color conditioning**.

### Input Conditioning

- The **color name** (e.g., `"red"`) is converted into an RGB vector, normalized to [0,1].
- The RGB vector is concatenated as an additional channel, making the input: `polygon_image (1 channel) + color_vector (3 channels)` = **4-channel input**.

### UNet Configuration

- **Depth:** 4 levels
- **Encoder blocks:** Conv2D → BatchNorm → ReLU → Downsampling
- **Decoder blocks:** Upsample → Conv2D → BatchNorm → ReLU
- **Skip connections:** Used to preserve spatial structure
- **Final layer:** 3-channel output with sigmoid activation (scaled to [0,255])

## Hyperparameters

| Parameter         | Value         |
|------------------|---------------|
| Epochs           | 25            |
| Batch Size       | 16            |
| Optimizer        | Adam          |
| Learning Rate    | 0.0002        |
| Loss Function    | MSELoss       |
| Input Size       | 64x64         |
| Color Injection  | Concatenated RGB to image input |



## Training Dynamics

Logged using **Weights & Biases**  
 [wandb project link](https://wandb.ai/mounikapenmetsa05-mahindra-university/ayna-polygon-color-public)

### Loss Curve
- Training loss decreased consistently over epochs.
- Final MSE: ~0.002

### Sample Outputs
| Input Polygon | Color Name | Predicted Output |
|---------------|------------|------------------|
| Triangle      | Red        | (shown in wandb) |
| Square        | Blue       | (shown in wandb) |
| Octagon       | Green      | (shown in wandb) |

## Failure Modes & Fixes

| Issue                          | Fix Attempted                                 |
|-------------------------------|-----------------------------------------------|
| Slight blur in color boundaries | Increased model depth, tuned learning rate     |
| Dull colors in output          | Adjusted final activation scaling             |
| Color misalignment (rare)      | Ensured correct label alignment in dataset    |

## Key Learnings

- Learned how to **inject non-image modalities (text/color)** into a CNN-based architecture.
- Reinforced understanding of **UNet skip connections** and **semantic segmentation** principles.
- Hands-on experience with **wandb logging**, model checkpointing, and image-to-image translation.
- Balanced training stability with **lightweight augmentation and normalization strategies**.

## Deliverables

-  Trained UNet model: `polygon_color_unet.pth`
-  Inference notebook: `Ayna_ML_Assignment.ipynb`
-  **wandb tracking link**: [wandb project](https://wandb.ai/mounikapenmetsa05-mahindra-university/ayna-polygon-color-public)
-  README Report:  *(this file)*

## Inference Instructions

Run the Jupyter Notebook:
```bash
  1. Open `Ayna_ML_Assignment.ipynb`
  2. Load the trained model (`polygon_color_unet.pth`)
  3. Provide sample inputs (polygon + color)
  4. Visualize the output image

 
I really enjoyed this assignment, especially combining vision with conditional inputs like color. Thank you for this opportunity.

For queries, feel free to reach out at: **mounikapenmetsa05@gmail.com**






