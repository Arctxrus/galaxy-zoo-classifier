"""ResNet50 model for Galaxy10 DECaLS classification.

Two-stage transfer learning strategy:

  Stage 1: backbone frozen. Only the new 10-class classifier head trains.
           Fast (most of the network is in eval mode for gradients) and
           gets the head into a sensible region of weight space before
           you start moving the backbone.

  Stage 2: layer4 (last residual block) unfrozen. Full end-to-end training
           at a smaller learning rate. The high-level features adapt to
           galaxies; early layers keep their general-purpose ImageNet
           filters (edges, textures, simple shapes), which transfer fine.

We use IMAGENET1K_V2 weights, which are torchvision's improved ResNet50
weights (better than the original V1 from the He et al. paper). They are
the right default for any new transfer learning project on top of ResNet50.
"""

from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

NUM_CLASSES = 10

def build_model(num_classes = NUM_CLASSES, pretrained = True):
    """Return a ResNet50 with a fresh classifier head."""
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace the 1000-class ImageNet head with our num_classes head.
    in_features = model.fc.in_features  # 2048 for ResNet50
    # fc refers to output layer, making it connect to 10 neurosn in there instead of 1000
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model):
    """Stage 1: freeze all parameters except the classifier head."""
    # Only trains head, learns to take already good features from res net and produce sensible class scores
    # Fast and stable
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_layer4(model):
    """Stage 2: unfreeze layer4 and keep the head trainable.

    ResNet50 has four residual stages: layer1, layer2, layer3, layer4.
    Unfreezing only layer4 strikes a good balance between adaptability
    and avoiding catastrophic forgetting of low-level pretrained features.
    """
    # Do this after stage 1, now that head is reasonable, let layer 4 (detecting more big parts of image)
    # adjust to galaxies, other layers stay frozen since their features still apply
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("layer4.") or name.startswith("fc.")
        )


def count_trainable_params(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total