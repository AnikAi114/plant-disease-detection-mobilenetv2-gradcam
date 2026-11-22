# scripts/test_gradcam.py

import json
import torch
import cv2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import os


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        return heatmap.cpu().numpy(), class_idx

    def overlay_heatmap(self, heatmap, original_img, alpha=0.5):
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
        return overlayed


def load_model(weights_path, num_classes, device='cpu'):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size=224):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, np.array(image)


print("=" * 50)
print("Grad-CAM for Plant Disease Detection")
print("=" * 50)

with open("weights/classes.json", "r") as f:
    class_names = json.load(f)
print(f"Loaded {len(class_names)} classes")

model = load_model("weights/best_9948.pth", len(class_names), "cpu")
print("Model loaded successfully!")

target_layer = model.features[-1][-1]
print("Target layer ready")

grad_cam = GradCAM(model, target_layer)

img_path = input("\nEnter image path: ").strip()

if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
    exit()

input_tensor, original = preprocess_image(img_path)

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()
    prob = torch.softmax(output, dim=1)[0, pred].item()

print(f"\nPrediction: {class_names[pred]}")
print(f"Confidence: {prob * 100:.2f}%")

heatmap, _ = grad_cam.compute_heatmap(input_tensor, pred)
overlayed = grad_cam.overlay_heatmap(heatmap, original)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(heatmap, cmap="jet")
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis("off")

axes[2].imshow(overlayed)
axes[2].set_title(f"Model Focus: {class_names[pred]}\n({prob * 100:.1f}%)")
axes[2].axis("off")

plt.tight_layout()

# এই লাইনটাতে কোন args নেই - সরাসরি string
plt.savefig("results/gradcam_output.png", dpi=300)

print("\nSaved: results/gradcam_output.png")

plt.show()
print("\nDone!")