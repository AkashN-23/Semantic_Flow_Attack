# ===============================
# Module 4: Apply ΔF to Input Image
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
    output.retain_grad()
    feature_maps['feat'] = output

hook_handle = model.backbone.body.layer4.register_forward_hook(hook_fn)

# -------- Load Image --------
img_path = r'/home/akashnagarajan/CODING_AND_PROJECTS/semantic-flow-attack/dog.jpg'
img = Image.open(img_path).convert('RGB')
img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
img_tensor.requires_grad_(True)

# -------- Inference --------
outputs = model(img_tensor)

# -------- Adversarial Loss --------
scores = outputs[0]['scores']
if len(scores) > 0:
    adv_loss = -torch.sum(scores)
else:
    adv_loss = torch.tensor(0.0, requires_grad=True)

print("Adversarial loss:", adv_loss.item())

# -------- Backpropagation --------
adv_loss.backward()

# -------- Get ΔF --------
delta_F = feature_maps['feat'].grad
print("ΔF shape:", delta_F.shape)

# -------- Get Gradient w.r.t. Input --------
grad_input = img_tensor.grad
print("Input gradient shape:", grad_input.shape)

# -------- Apply Perturbation --------
epsilon = 0.01  # You can tune this — higher means stronger attack
perturbed = img_tensor + epsilon * torch.sign(grad_input)
perturbed = torch.clamp(perturbed, 0, 1)  # keep in valid image range

# -------- Convert to Image and Save/Show --------
perturbed_img = perturbed.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

plt.imshow(perturbed_img)
plt.title("Adversarial Image - Module 4 Output")
plt.axis('off')
plt.show()

# -------- Clean up Hook --------
hook_handle.remove()
