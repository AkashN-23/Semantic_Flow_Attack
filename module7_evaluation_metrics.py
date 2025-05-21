import torch
import torch.nn.functional as Fnn
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import csv
from PIL import Image
from torchvision.transforms import functional as F
import torch

# Load both images
original_img = Image.open('dog.jpg').convert('RGB')
attacked_img = Image.open('test_images/attacked.jpg').convert('RGB')

# ----------- Load Model -----------
device = torch.device('cpu')
model = fasterrcnn_resnet50_fpn(weights="DEFAULT").eval().to(device)

# Transform to tensors
original_img_tensor = F.to_tensor(original_img).unsqueeze(0).to(device)
attacked_img_tensor = F.to_tensor(attacked_img).unsqueeze(0).to(device)

# ----------- Hook Feature Layer -----------
feature_maps = {}

def hook_fn(module, input, output):
    feature_maps['feat'] = output

handle = model.backbone.body.layer4.register_forward_hook(hook_fn)

# ----------- Metric Functions -----------

def l2_feature_distance(f1, f2):
    return torch.norm(f1 - f2).item()

def cosine_feature_similarity(f1, f2):
    f1_flat = f1.view(-1)
    f2_flat = f2.view(-1)
    return Fnn.cosine_similarity(f1_flat, f2_flat, dim=0).item()

def semantic_drift(delta_F):
    return delta_F.abs().mean().item()

def evaluate_attack(model, original_img_tensor, attacked_img_tensor):
    # -------- Original --------
    with torch.no_grad():
        _ = model(original_img_tensor)
    feat_original = feature_maps['feat'].detach().clone()

    # -------- Attacked --------
    with torch.no_grad():
        _ = model(attacked_img_tensor)
    feat_attacked = feature_maps['feat'].detach().clone()

    # -------- Compute Metrics --------
    l2 = l2_feature_distance(feat_original, feat_attacked)
    cosine = cosine_feature_similarity(feat_original, feat_attacked)
    drift = semantic_drift(feat_original - feat_attacked)

    return {
        'L2 Distance': l2,
        'Cosine Similarity': cosine,
        'Semantic Drift': drift
    }

# Assume you already have your image tensors:
# original_img_tensor = ...
# attacked_img_tensor = ...

results = evaluate_attack(model, original_img_tensor, attacked_img_tensor)

print("\n📊 Semantic Attack Evaluation Metrics")
for k, v in results.items():
    print(f"{k:<20}: {v:.4f}")

handle.remove()

def save_results(results, filepath="semantic_attack_metrics.csv"):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in results.items():
            writer.writerow([k, v])
