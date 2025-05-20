# ===============================
# Module 2: Feature Map Hooking
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
    feature_maps['feat'] = output

# Hook into ResNet Layer 4 (part of the backbone)
hook_handle = model.backbone.body.layer4.register_forward_hook(hook_fn)

# -------- Load Image --------
img_path = r'path of the image'
img = Image.open(img_path).convert('RGB')
img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

# -------- Inference (Forward Pass) --------
with torch.no_grad():
    outputs = model(img_tensor)

# -------- Access and Visualize Feature Map --------
F_map = feature_maps['feat']
print("Feature map shape:", F_map.shape)

# Visualize channel 0 of the feature map
plt.imshow(F_map[0, 0, :, :].cpu(), cmap='viridis')
plt.title("Faster R-CNN Feature Map - Channel 0")
plt.colorbar()
plt.axis('off')
plt.show()

# -------- Clean up the Hook --------
hook_handle.remove()
