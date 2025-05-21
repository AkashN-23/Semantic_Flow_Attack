# Adversarial Attack on Object Detection Model: Semantic Flow Attack

---

## ⚠️ IMPORTANT LEGAL & ETHICAL NOTICE

This repository contains a cutting-edge **Semantic Flow Adversarial Attack** framework designed to manipulate internal feature maps of deep neural networks, specifically targeting Faster R-CNN (FRCNN) with ResNet backbone.

**This code is for academic and research use only. Unauthorized cloning, copying, redistribution, or derivative work is strictly forbidden and subject to legal action, including criminal prosecution.**

If you use this code, you agree to abide by all applicable laws and respect ethical AI research standards.

---

## 🚀 Project Overview

This project implements a **Semantic Flow Attack** — an adversarial technique that modifies how the model perceives input feature maps internally to fool it at a semantic level, rather than just pixel-level perturbations.

### Model Used
- **Faster R-CNN** (FRCNN) with a **ResNet** backbone (feature extractor)

---

## 🛠️ Modules & Functionality

The project is modular, with each component addressing a core part of the attack pipeline:

- **Module 1: Initializing Faster R-CNN**  
  Load and prepare the pre-trained FRCNN ResNet model.

- **Module 2: Feature Hook**  
  Attach hooks to intercept intermediate feature maps during the forward pass.

- **Module 3: Calculating ΔF (Delta Feature Map)**  
  Compute perturbations on the feature maps to shift model perception.

- **Module 4: Applying ΔF**  
  Modify the feature maps by applying calculated perturbations.

- **Module 5: Iterative Semantic Flow Attack**  
  Repeatedly apply and optimize the perturbations to maximize adversarial effect.

- **Module 6: Visual Comparison**  
  Generate visualizations comparing clean vs. attacked feature maps and outputs.

- **Module 7: Evaluation Metrics**  
  Quantify the attack’s impact using relevant performance and robustness metrics.

---

## ⚙️ Requirements

- **Python** 3.9 or 3.10  
- **Libraries:**  
  - `torch`  
  - `torchvision`  
  - `torchaudio`  
  - `pillow`  
  - `matplotlib`  

---

## 🔄 How to Reproduce the Semantic Flow Attack

1. **Load a pre-trained FRCNN model** with ResNet backbone, compatible with feature extraction.  
2. **Intercept intermediate feature maps** at user-defined layers using feature hooks during forward propagation.  
3. **Generate a flow field** — a spatial vector field describing pixel shifts or semantic transformations within the feature map.  
4. **Warp the feature map** using the flow field (e.g., bilinear interpolation).  
5. **Feed the perturbed feature map** back into the model to complete forward pass.  
6. **Optimize the flow field** using gradient-based methods to maximize adversarial loss, such as targeted misclassification.

> *Note:* This is a **white-box attack** requiring access to the model internals.

---

## ⚡ Usage Example

```python
from semantic_flow_attack import FRCNNInitializer, FeatureHook, SemanticFlowAttack

# Initialize model
model = FRCNNInitializer.load_pretrained_frcnn_resnet()

# Hook features
hook = FeatureHook.attach(model, target_layer='layer3')

# Initialize attack
attack = SemanticFlowAttack(model, hook)

# Run attack iterations
adv_input = attack.run(input_image, target_label)

# Visualize
attack.visualize(adv_input, input_image)


🚫 Strict Warning
DO NOT clone, copy, redistribute, or repurpose this project without explicit written permission from the author.

Violators will face criminal prosecution and legal penalties under applicable copyright and cybercrime laws.
