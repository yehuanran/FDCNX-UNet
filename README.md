# FDCNX-UNet Frequency Dynamic ConvNeXt UNet with Linear Attention for Building Segmentation

This is an official PyTorch implementation of "[**FDCNX-UNet Frequency Dynamic ConvNeXt UNet with Linear Attention for Building Segmentation**]".

# Introduction
With the rapid development of remote sensing technology, acquiring high-resolution remote sensing images has become increasingly straightforward. These images hold significant value in the field of building segmentation. However, their high resolution, complex scenes, and scale variations introduce challenges in their utilization. To address these issues, we propose a residual attention U-Net based on a diffusion model, integrating the diffusion model with structures such as U-Net, residual models and attention mechanisms to enhance the segmentation accuracy of buildings in remote sensing images. Leveraging the characteristics of diffusion models, we mitigate the influence of irrelevant background factors on segmentation results. A dual-branch Feature Extraction Block(FEB) is designed for noisy images and RGB images to achieve multi-modal information fusion. The self-attention mechanisms are embedded within the down-sample and up-sample to strengthen the comprehension of global features. The Channel Attention Block(CAB) guides high-level features to select low-level features, thus greatly ensuring the consistency of features.
<center> 
<img src="DRAU-Net.png" width="auto" height="auto">
</center>

# Image segmentation

## 1. Requirements
```
# Environments:
cuda==12.6
python==3.10
# Dependencies:
pip install torch==2.7.1 torchvision==0.22.1
pip install einops==0.6.1 imageio==2.28.1   albumentations   Torchmetrics==0.11.4
```

## 2. Data Preparation

```
│inria/
├──austin1/
│  ├── images
│  │   ├── austin1.jpg
│  │   ├── ......
│  ├── binary_masks
├──austin2/
│  ├── images
│  │   ├── austin2.jpg
│  │   ├── ......
│  ├── ......
```

## 3. Train

python train.py --dataset inria --n_timesteps 16

## 4. Validation

python test.py --load_checkpoint output/checkpoints/20250401-0413_unet/20250401-0413_unet_e1.pt

