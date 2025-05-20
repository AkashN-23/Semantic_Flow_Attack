# ===============================
# Module 5: Iterative SFA Loop
# ===============================

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# -------- Load Model --------
device = torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

# -------- Hook Setup --------
feature_maps = {}

def hook_fn(module, input, output):
    output.retain_grad()
    feature_maps['feat'] = output

hook_handle = model.backbone.body.layer4.register_forward_hook(hook_fn)

# -------- Load Image --------
img_path = r'path of the image'
img = Image.open(img_path).convert('RGB')
img_tensor_orig = F.to_tensor(img).unsqueeze(0).to(device)
img_tensor_adv = img_tensor_orig.clone().detach()

# -------- Attack Parameters --------
num_iters = 5        # Try increasing to 10–20 later
epsilon = 0.01       # Step size for each iteration

# -------- Iterative Attack Loop --------
for i in range(num_iters):
    print(f"\n[Iteration {i+1}]")

    img_tensor_adv.requires_grad_(True)
    feature_maps.clear()  # Clear old hook data

    outputs = model(img_tensor_adv)

    scores = outputs[0]['scores']
    if len(scores) > 0:
        adv_loss = -torch.sum(scores)
    else:
        adv_loss = torch.tensor(0.0, requires_grad=True)

    print("Adversarial loss:", adv_loss.item())

    model.zero_grad()
    adv_loss.backward()

    grad_input = img_tensor_adv.grad
    img_tensor_adv = img_tensor_adv + epsilon * torch.sign(grad_input)
    img_tensor_adv = torch.clamp(img_tensor_adv, 0, 1).detach()  # clip & stop grad

# -------- Visualize Final Adversarial Image --------
final_img = img_tensor_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()

plt.imshow(final_img)
plt.title("Final Adversarial Image After Iterative SFA")
plt.axis('off')
plt.show()

hook_handle.remove()

#Proof of Concept: Show Original vs Adversarial Image

# ---- Compare Original vs Adversarial Detections ----
print("\nOriginal detections:")
original_outputs = model(img_tensor_orig)
for box, label, score in zip(original_outputs[0]['boxes'], original_outputs[0]['labels'], original_outputs[0]['scores']):
    if score > 0.5:
        print(f"Label: {label.item()}, Score: {score.item():.2f}")

print("\nAdversarial detections:")
adv_outputs = model(img_tensor_adv)
for box, label, score in zip(adv_outputs[0]['boxes'], adv_outputs[0]['labels'], adv_outputs[0]['scores']):
    if score > 0.5:
        print(f"Label: {label.item()}, Score: {score.item():.2f}")

