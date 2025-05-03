# SRGAN for Satellite Image Super-Resolution

This repository contains a PyTorch implementation of SRGAN (Super-Resolution Generative Adversarial Network) specifically adapted for satellite imagery. The model can upscale low-resolution satellite images by a factor of 4x while preserving and enhancing details.

## Overview

SRGAN uses a generative adversarial network to produce photo-realistic super-resolved images from low-resolution inputs. This implementation is based on:
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

## Requirements

* Python 3.x
* PyTorch (latest stable version)
* torchvision
* tensorboard_logger
* tqdm
* CUDA-compatible GPU (recommended)

Install dependencies:

```bash
pip install torch torchvision tensorboard_logger tqdm
```

## Dataset Structure

The code expects the following directory structure:

```
data/
  ├── train/   # Training images (HR)
  ├── dev/     # Validation images (HR)
  └── val/     # Test images (HR)
```

- Only high-resolution images are needed; low-resolution counterparts are generated on-the-fly
- Images should be in common formats (.jpg, .png, .jpeg)
- For optimal results with default settings, images should be at least 96×96 pixels

## Training

### Basic Training

```bash
python train.py --train_set data/train --crop_size 96 --batch_size 16 --num_epochs 1000
```

The training process consists of:
1. Pre-training phase: Generator only with MSE loss (2 epochs)
2. Adversarial training: Generator and Discriminator with combined losses

### Parameters

- `--crop_size`: Size of training patches (default: 128)
- `--num_epochs`: Training epochs (default: 1000)
- `--batch_size`: Batch size (default: 64)
- `--train_set`: Training data directory (default: data/train)
- `--check_point`: Resume from checkpoint (default: -1)

### Training Features

- Label smoothing and flipping for improved GAN stability
- TensorBoard logging of losses and gradients
- Regular checkpoint saving (every epoch for generator, every 5 epochs for others)
- Visualization of intermediate results

## Evaluation

Evaluate trained models with:

```bash
python eval.py --val_set data/val --start 1000 --end 1000
```

### Parameters

- `--val_set`: Test data directory (default: data/val)
- `--start`: Starting epoch for evaluation (default: 1)
- `--end`: Ending epoch for evaluation (default: 100)
- `--interval`: Epoch interval for evaluation (default: 1)

### Evaluation Metrics

The script reports:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Inference time per image
- GPU memory usage
- Model size and parameters

## Super-Resolution Inference

For individual images:

```bash
python sr.py --lr path/to/lowres_image.jpg
```

## Implementation Details

- **Generator**: ResNet-based architecture with skip connections
- **Discriminator**: VGG-style convolutional network
- **Loss Function**: Combination of MSE loss and adversarial loss
- **Preprocessing**: Images are cropped and randomly augmented during training
- **Hardware**: Developed and tested on NVIDIA GPUs

## Results

Evaluation results include:
- SR output compared with bicubic interpolation
- Visual comparisons between:
  - Bicubic upsampling
  - Ground truth HR images
  - SRGAN output
  - Baseline model output

## Performance Considerations

- Memory usage scales with image and batch size
- For smaller GPUs, reduce batch size or crop size
- Training a full model (1000 epochs) takes several hours on modern GPUs

## References

- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

## Visual Results

Below are some example results from the SRGAN model compared to bicubic interpolation:

![Urban Area Comparison](sr_results/epoch_1000_index_1.png)

*1: Bicubic upsampling | 2: Ground truth | 3: SRGAN output (our model) | 4: Baseline Model*

![Rural Comparison](sr_results/epoch_1000_index_2.png)

*1: Bicubic upsampling | 2: Ground truth | 3: SRGAN output (our model) | 4: Baseline Model*

![Details Comparison](sr_results/epoch_1000_index_3.png)

*1: Bicubic upsampling | 2: Ground truth | 3: SRGAN output (our model) | 4: Baseline Model*

The SRGAN model (right image) produces sharper details and more defined edges compared to standard bicubic upsampling (left image), resulting in super-resolved images that more closely match the ground truth (middle image).

### Performance Metrics

| Method | PSNR (dB) | SSIM | Inference Time (s) |
|--------|-----------|------|-------------------|
| Bicubic | ~28.31 | ~0.8057 | - |
| SRGAN (ours) | ~31.014 | ~0.8829 | ~0.0173 |

## License

MIT License