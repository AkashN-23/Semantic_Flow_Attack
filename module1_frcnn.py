# ===============================
# Module 1: Faster R-CNN Detection with Labels
# ===============================

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Load pre-trained Faster R-CNN model
device = torch.device('cpu')  # Using CPU for now
model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

# Load and preprocess the image
img_path = r'path of the image'
img = Image.open(img_path).convert('RGB')
img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    outputs = model(img_tensor)

# Check outputs
print(outputs)

# Draw detections with scores above threshold
draw = ImageDraw.Draw(img)
threshold = 0.5  # Confidence threshold
font = ImageFont.load_default()

for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
    if score >= threshold:
        box = box.tolist()
        draw.rectangle(box, outline='red', width=3)
        text = f"Label: {label.item()}, Conf: {score:.2f}"
        draw.text((box[0], box[1] - 10), text, fill='red', font=font)

# Show image with detections
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.title("Faster R-CNN Detections - Module 1 Working Check")
plt.show()
