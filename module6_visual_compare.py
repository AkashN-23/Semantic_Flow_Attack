# ===============================
# Module 6: Visual Comparison — Original vs Adversarial
# ===============================

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ---- COCO Labels ----
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ---- Helper Function to Draw Detections ----
def draw_boxes(image, outputs, threshold=0.5):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
        if score >= threshold:
            box = box.tolist()
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            draw.rectangle(box, outline='red', width=2)
            draw.text((box[0], box[1]-10), f"{label_name} {score:.2f}", fill='red', font=font)

    return image

# ---- Load Model ----
device = torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

# ---- Load Original Image ----
img_path = r'/home/akashnagarajan/CODING_AND_PROJECTS/Semantic Form Attack/dog.jpg'
img = Image.open(img_path).convert('RGB')
img_tensor_orig = F.to_tensor(img).unsqueeze(0).to(device)

# ---- Create Adversarial Image (Quick 5-iteration SFA) ----
img_tensor_adv = img_tensor_orig.clone().detach()

img_tensor_adv.requires_grad_(True)
for _ in range(5):
    outputs = model(img_tensor_adv)
    scores = outputs[0]['scores']
    if len(scores) > 0:
        adv_loss = -torch.sum(scores)
    else:
        adv_loss = torch.tensor(0.0, requires_grad=True)

    model.zero_grad()
    adv_loss.backward()
    grad_input = img_tensor_adv.grad
    img_tensor_adv = img_tensor_adv + 0.01 * torch.sign(grad_input)
    img_tensor_adv = torch.clamp(img_tensor_adv, 0, 1).detach()
    img_tensor_adv.requires_grad_(True)

# ---- Run Detections ----
outputs_orig = model(img_tensor_orig)
outputs_adv = model(img_tensor_adv)

# ---- Convert Tensors to PIL Images ----
img_orig_pil = F.to_pil_image(img_tensor_orig.squeeze(0).cpu())
img_adv_pil = F.to_pil_image(img_tensor_adv.squeeze(0).cpu())

# ---- Draw Detections ----
img_orig_drawn = draw_boxes(img_orig_pil, outputs_orig)
img_adv_drawn = draw_boxes(img_adv_pil, outputs_adv)

# ---- Display Side-by-Side ----
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_orig_drawn)
plt.title("Original Image — With Detections")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_adv_drawn)
plt.title("Adversarial Image — Detections Suppressed")
plt.axis('off')

plt.tight_layout()
plt.show()
