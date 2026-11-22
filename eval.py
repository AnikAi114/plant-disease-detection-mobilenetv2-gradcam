# eval.py
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import os
from pathlib import Path


# Dataset class
class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Load model from checkpoint
def load_model_from_checkpoint(weights_path, device='cpu'):
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)
    img_size = ckpt.get("img_size", 224)
    model_name = ckpt.get("model_name", "mobilenetv2")

    if model_name == "cnn":
        try:
            from models.baseline_cnn import SmallCNN
            model = SmallCNN(num_classes)
        except:
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    return model, classes, img_size


print("=" * 60)
print("🌿 PLANT DISEASE DETECTION - MODEL EVALUATION")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"📱 Device: {device}")

# Load model
print("📦 Loading model...")
model, classes, img_size = load_model_from_checkpoint("weights/best_9948.pth", device)
print(f"✅ Model loaded: {len(classes)} classes")

# Transform
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test data
print("📂 Loading test data...")
test_dataset = PlantDataset("data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"✅ Test samples: {len(test_dataset)}")

# Evaluate
print("\n🔄 Evaluating...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n" + "=" * 60)
print("📊 OVERALL METRICS")
print("=" * 60)
print(f"🎯 Accuracy:  {accuracy * 100:.4f}%")
print(f"📈 Precision: {precision * 100:.4f}%")
print(f"📉 Recall:    {recall * 100:.4f}%")
print(f"⭐ F1-Score:  {f1 * 100:.4f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(20, 16))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%', fontsize=14)

# Show only first 10 class names to avoid clutter
short_names = [c[:20] if len(c) > 20 else c for c in classes]
plt.xticks(np.arange(len(classes)), short_names, rotation=90, fontsize=8)
plt.yticks(np.arange(len(classes)), short_names, fontsize=8)

# Add numbers in cells
for i in range(len(classes)):
    for j in range(len(classes)):
        if cm[i, j] > 0:
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6)

plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig("results/cm.png", dpi=300, bbox_inches='tight')
print("\n✅ Confusion matrix saved: results/cm.png")

# Save metrics to file
with open("results/evaluation_metrics.txt", "w") as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL EVALUATION METRICS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Accuracy:  {accuracy * 100:.4f}%\n")
    f.write(f"Precision: {precision * 100:.4f}%\n")
    f.write(f"Recall:    {recall * 100:.4f}%\n")
    f.write(f"F1-Score:  {f1 * 100:.4f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(all_labels, all_preds, target_names=classes, digits=4))

print("✅ Metrics saved: results/evaluation_metrics.txt")
print("\n🎉 Evaluation complete!")