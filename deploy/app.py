"""Gradio interface for the Galaxy10 morphology classifier.

Run locally:
    cd deploy
    python app.py

Then open http://localhost:7860 in a browser.
"""

import gradio as gr
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round smooth",
    "In-between round smooth",
    "Cigar-shaped smooth",
    "Barred spiral",
    "Unbarred tight spiral",
    "Unbarred loose spiral",
    "Edge-on without bulge",
    "Edge-on with bulge",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pt"

def build_model():
    """Same architecture as training. No ImageNet weights here, since
    we're about to overwrite them with the trained checkpoint anyway."""
    model = models.resnet50(weights = None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def load_model(path=MODEL_PATH):
    model = build_model().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    return model


# Eval transform. Resize(256) handles users uploading non-256x256 images.
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# Load once at startup as it's a heavy operation
print(f"Loading model on {DEVICE}...")
model = load_model()
print("Model loaded.")


@torch.no_grad()
def predict(image):
    """Take a PIL image, return {class_name: probability} for Gradio."""
    if image is None:
        return None
    # in case user uploads RGBA or grayscale
    image = image.convert("RGB")
    # Unsqueeze(N) adds a new dimension of size 1 at position N
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    # Softmax converts logits to percentages
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}


DESCRIPTION = """
# Galaxy Morphology Classifier

Upload an image of a galaxy and the model predicts which of 10
morphological classes it belongs to.

The model is a ResNet50 fine tuned on the
[Galaxy10 DECaLS benchmark dataset](https://astronn.readthedocs.io/en/latest/galaxy10.html)
(~17,700 images, crowdsourced Galaxy Zoo labels). It achieves a test
set macro F1 of 0.70 across the 10 classes.

**This is an educational portfolio project, not a scientific instrument.**
Real morphology classification for research requires careful treatment of
label noise, ensembling, and human expert review. The model also expects
inputs that look similar to the DECaLS training set: roughly centred
galaxies in colour cutouts.
"""


demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type = "pil", label = "Galaxy Image"),
    outputs = gr.Label(num_top_classes=3, label = "Top 3 predictions"),
    title = "Galaxy Morphology Classifier",
    description=DESCRIPTION,
    flagging_mode="never",
    examples=[f"examples/class_{i}.png" for i in range(10)],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)