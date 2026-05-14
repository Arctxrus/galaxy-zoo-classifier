"""Image transform pipelines for Galaxy10 DECaLS.

Galaxies are rotationally symmetric on the sky, so flips and arbitrary
rotations are valid augmentations. Inputs are 256x256 uint8 PIL images;
we crop to 224x224 to match ImageNet pretrained backbones, then
normalise to ImageNet statistics.
"""

from torchvision import transforms

# ImageNet statistics. Pretrained backbones expect inputs normalised
# this way, even though galaxy pixel distributions differ.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Random transformations to each image so the model learns not memorises
def get_train_transform():
    """Stochastic augmentation pipeline for training."""
    return transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])


def get_eval_transform():
    """Deterministic pipeline for validation and test."""
    return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])

