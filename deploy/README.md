---
title: Galaxy Morphology Classifier
emoji: 🌌
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: ResNet50 classifying galaxies into 10 morphological types
---

# Galaxy Morphology Classifier

A ResNet50 fine tuned on the Galaxy10 DECaLS benchmark dataset to classify
galaxies into 10 morphological types. Test set macro F1 of 0.70 with
balanced accuracy of 0.72.

Two stage transfer learning: stage 1 trains the classifier head with the
ImageNet backbone frozen; stage 2 unfreezes layer4 and fine tunes end to
end at a smaller learning rate. Class imbalance is handled with inverse
frequency weighting in the cross entropy loss.

Source code and training pipeline: https://github.com/Arctxrus/galaxy-zoo-classifier

**Limitations.** This is an educational portfolio project, not a scientific
instrument. The Galaxy10 labels come from Galaxy Zoo crowdsourcing, which
has measurable label noise. Production scientific morphology classification
requires further validation, ensembling, and human expert review.
