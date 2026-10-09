# Face Swap Machine Learning Pipeline

This directory contains the complete face swap machine learning training pipeline implemented using PyTorch.

## Architecture Overview

The face swap system implements an image-to-image transformation model that takes:
- **Source Face**: The face whose identity should be transferred
- **Target Image**: The image containing the background/content where the identity will be swapped

Model architecture follows a U-Net like encoder-decoder with skip connections and proper input handling for both source and target faces.

## Files Structure

```
ml/
├── models/
│   └── face_swap_model.py    # Contains the main model implementation
├── data/
│   └── dataset.py            # Dataset preparation and augmentation
├── training/
│   ├── train.py              # Main training script
│   ├── smoke_test.py         # Smoke test to verify model works correctly
│   └── inference.py          # Inference script for using trained models
├── requirements.txt          # Dependencies needed for the ML pipeline
└── README.md                 # This file
```

## Model Details

### Input/Output Specification:

**Inputs:**
1. `source_face` - [B, 3, H, W] - Source face image tensor (3-channel RGB)
2. `target_face` - [B, 3, H, W] - Target face image tensor (3-channel RGB)

**Output:**
- `swapped_image` - [B, 3, H, W] - Face-swapped result

### Key Features:

1. **Dual Input Architecture**: Model accepts both source and target images as inputs
2. **Conditional Generation**: Uses source identity to influence the swap while preserving target scene/content
3. **Image-to-Image Transformation**: Direct pixel-level transformation between faces\
4. **Skip Connections**: U-Net style architecture for better feature preservation

## Implementation Details

### Model Architecture:

The `SimpleFaceSwapModel` implements:
- Encoder with down-sampling layers (Conv2D)
- Bottleneck layer
- Decoder with up-sampling and skip connections (ConvTranspose2D)
- Fusion of source and target features via concatenation
- Tanh activation on final output for normalized RGB values

### Training Process:

1. **Data Preparation**:
   - Source and target face pairs are created from input data
   - Images are resized to 256x256 pixels
   - Augmentation applied during training

2. **Loss Function**:
   - L1 reconstruction loss (between generated output and target)
   - Identity preservation losses to maintain source characteristics
   - Optimizer: Adam with learning rate 0.0002

3. **Training Pipeline**:
   - Batch processing of face pairs
   - Forward pass with both source and target inputs
   - Backward propagation through shared decoder
   - Model checkpoint saving after training epochs

## Usage Instructions

### For Training:

```bash
# Install dependencies\
pip install -r ml/requirements.txt

# Run training
cd ml/training/
python train.py
```

### For Inference:

```bash
# Load and run inference with a trained model
python inference.py --checkpoint-path path/to/checkpoint.pth --image-path path/to/image.jpg
```

## Dataset Requirements

The dataset layer is designed for the real CelebA dataset with identity annotations:
1. `ml/data/celeba/identity_CelebA.txt`
2. `ml/data/celeba/img_align_celeba/*.jpg`

Validate dataset presence with:
```bash
python ml/training/prepare_dataset.py --dataset-root ml/data/celeba
```

## Requirements

See `requirements.txt` for dependencies. Core:
- torch >= 2.0.0\
- torchvision >= 0.15.0
- opencv-python >= 4.7.0
- Pillow >= 9.5.0
- numpy >= 1.24.0

## License\

See LICENSE file in repository root. The face swap pipeline itself is implemented under MIT license.
