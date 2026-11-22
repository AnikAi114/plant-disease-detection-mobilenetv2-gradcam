
from pathlib import Path
import csv
import json

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ----------------------------------------
# Project paths - original files are read-only
# ----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_DIR / "data" / "test"

CHECKPOINT = PROJECT_DIR / "weights" / "best_9948.pth"

AUDIT_FILE = (
    PROJECT_DIR
    / "verification_results"
    / "test_duplicate_audit.csv"
)

OUTPUT_DIR = PROJECT_DIR / "verification_results"

# ----------------------------------------
# Safety checks
# ----------------------------------------

assert TEST_DIR.is_dir(), "Test dataset not found"
assert CHECKPOINT.is_file(), "Model checkpoint not found"
assert AUDIT_FILE.is_file(), "Duplicate audit file not found"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ----------------------------------------
# Load original model checkpoint
# ----------------------------------------

print("\nLoading original MobileNetV2 checkpoint...")

# Only load checkpoints created by you or a trusted source.
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

classes = checkpoint["classes"]
img_size = checkpoint.get("img_size", 224)
model_name = checkpoint.get("model_name", "mobilenetv2")

if model_name not in ("mobilenet", "mobilenetv2", "mobilenet_v2"):
    raise ValueError(
        f"Unexpected model architecture: {model_name}"
    )

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(classes)
)

model.load_state_dict(
    checkpoint["state_dict"],
    strict=True
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")
print("Number of classes:", len(classes))

# ----------------------------------------
# Load duplicate audit
# ----------------------------------------

excluded_paths = set()

with AUDIT_FILE.open(
    "r", encoding="utf-8", newline=""
) as file:
    reader = csv.DictReader(file)

    for row in reader:
        path = Path(row["test_image_path"])

        if not path.is_absolute():
            path = PROJECT_DIR / path

        excluded_paths.add(path.resolve())

print("\nExact-duplicate test paths:", len(excluded_paths))

# ----------------------------------------
# Image preprocessing
# ----------------------------------------

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------
# Filtered test dataset
# ----------------------------------------

class CleanTestDataset(Dataset):

    def __init__(self):
        self.samples = []
        self.transform = transform

        dataset_classes = sorted(
            p.name for p in TEST_DIR.iterdir()
            if p.is_dir()
        )

        if dataset_classes != sorted(classes):
            raise ValueError(
                "Checkpoint classes and test folders differ."
            )

        class_to_idx = {
            name: idx for idx, name in enumerate(classes)
        }

        extensions = {".jpg", ".jpeg", ".png"}

        for class_name in dataset_classes:
            class_dir = TEST_DIR / class_name

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in extensions:
                    continue

                if image_path.resolve() in excluded_paths:
                    continue

                self.samples.append((
                    image_path,
                    class_to_idx[class_name]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label


dataset = CleanTestDataset()

assert len(dataset) == 3100, (
    f"Expected 3100 images, found {len(dataset)}"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Filtered test images:", len(dataset))

# ----------------------------------------
# Run evaluation - no training
# ----------------------------------------

all_predictions = []
all_labels = []

print("\nRunning clean test evaluation...")

with torch.inference_mode():
    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(labels.tolist())

# ----------------------------------------
# Calculate metrics
# ----------------------------------------

accuracy = accuracy_score(
    all_labels, all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

print("\nCLEAN TEST EVALUATION RESULTS")
print("--------------------------------")
print("Test images:", len(all_labels))
print(f"Accuracy: {accuracy * 100:.4f}%")
print(f"Weighted Precision: {precision * 100:.4f}%")
print(f"Weighted Recall: {recall * 100:.4f}%")
print(f"Weighted F1: {f1 * 100:.4f}%")

# ----------------------------------------
# Save results to verification_results ONLY
# ----------------------------------------

metrics = {
    "checkpoint": CHECKPOINT.name,
    "test_images": len(all_labels),
    "excluded_exact_duplicate_images": len(excluded_paths),
    "accuracy": accuracy,
    "weighted_precision": precision,
    "weighted_recall": recall,
    "weighted_f1": f1,
    "note": "Exact-duplicate-filtered test evaluation"
}

with (
    OUTPUT_DIR / "clean_test_metrics.json"
).open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
)

with (
    OUTPUT_DIR / "clean_test_report.txt"
).open("w", encoding="utf-8") as file:
    file.write(report)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes)))
)

with (
    OUTPUT_DIR / "clean_test_confusion_matrix.csv"
).open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["actual/predicted"] + list(classes))

    for class_name, row in zip(classes, cm):
        writer.writerow([class_name] + row.tolist())

print("\nNew reports saved in verification_results/")
print("Original project files remain unchanged.")

