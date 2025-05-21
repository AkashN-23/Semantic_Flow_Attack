# ===============================
# Module 3: Compute ΔF from Feature Space
# ===============================

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# -------- Load Model --------
device = torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

# -------- Hook Layer --------
feature_maps = {}

def hook_fn(module, input, output):
    output.retain_grad()  # Important: retain gradients on feature map
    feature_maps['feat'] = output

hook_handle = model.backbone.body.layer4.register_forward_hook(hook_fn)

# -------- Load Image --------
img_path = r'/home/akashnagarajan/CODING_AND_PROJECTS/semantic-flow-attack/dog.jpg'
img = Image.open(img_path).convert('RGB')
img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
img_tensor.requires_grad_(True)  # Enable gradients on input

# -------- Inference --------
outputs = model(img_tensor)

# -------- Adversarial Objective --------
# Let's try to "hide" detected objects by minimizing their confidence scores
# Take the negative sum of all detection scores
scores = outputs[0]['scores']
if len(scores) > 0:
    adv_loss = -torch.sum(scores)
else:
    adv_loss = torch.tensor(0.0, requires_grad=True)  # fallback

print("Adversarial loss (to suppress detections):", adv_loss.item())

# -------- Backward Pass --------
adv_loss.backward()

# -------- Get ΔF from feature map gradients --------
F_map = feature_maps['feat']
delta_F = F_map.grad  # This is ΔF
print("ΔF shape:", delta_F.shape)

# -------- Visualize the ΔF (just channel 0) --------
plt.imshow(delta_F[0, 0, :, :].detach().cpu(), cmap='bwr')
plt.title("ΔF - Semantic Flow Gradient (Channel 0)")
plt.colorbar()
plt.axis('off')
plt.show()

# -------- Clean up --------
hook_handle.remove()
