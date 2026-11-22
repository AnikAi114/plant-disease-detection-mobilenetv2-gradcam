# train_resnet_efficientnet.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ====================================

def get_model(model_name, num_classes=15):
    """Load ResNet50 or EfficientNet-B0"""
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def get_loaders():
    """Create data loaders"""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes


def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    """Train model and return best validation accuracy"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        val_acc = 100 * correct / total

        if val_acc > best_acc:
            best_acc = val_acc

        print(f"   Epoch {epoch:02d}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return best_acc


def measure_inference_time(model, num_runs=100):
    """Measure inference time (in milliseconds)"""
    model.eval()
    model.to(DEVICE)
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # Warmup
    for _ in range(10):
        _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy)
            end = time.time()
            times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def get_model_size_mb(model):
    """Get model size (in MB)"""
    torch.save(model.state_dict(), "temp_weights.pth")
    size_mb = os.path.getsize("temp_weights.pth") / (1024 * 1024)
    os.remove("temp_weights.pth")
    return size_mb


def main():
    print("=" * 60)
    print("🔄 TRAINING RESNET50 & EFFICIENTNET-B0")
    print("=" * 60)

    # Your MobileNetV2 data
    mobilenet_data = {
        "Model": "MobileNetV2 (Ours)",
        "Accuracy (%)": 99.48,
        "Size (MB)": 8.77,
        "Inference Time (ms)": 18.59
    }

    # Data loaders
    print("\n📂 Loading data...")
    train_loader, val_loader, test_loader, classes = get_loaders()
    num_classes = len(classes)
    print(f"   Classes: {num_classes}")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")

    results = [mobilenet_data]

    # Train ResNet50
    print("\n" + "=" * 60)
    print("🚀 Training RESNET50...")
    print("=" * 60)

    model = get_model("resnet50", num_classes)
    size_before = get_model_size_mb(model)
    print(f"   Model size (before training): {size_before:.2f} MB")

    start_train = time.time()
    accuracy = train_model(model, train_loader, val_loader)
    train_time = time.time() - start_train

    size_after = get_model_size_mb(model)
    inf_time_mean, inf_time_std = measure_inference_time(model)

    print(f"\n   ✅ ResNet50 Results:")
    print(f"   Validation Accuracy: {accuracy:.2f}%")
    print(f"   Model Size: {size_after:.2f} MB")
    print(f"   Inference Time: {inf_time_mean:.2f} ± {inf_time_std:.2f} ms")
    print(f"   Training Time: {train_time / 60:.2f} min")

    results.append({
        "Model": "ResNet50",
        "Accuracy (%)": round(accuracy, 2),
        "Size (MB)": round(size_after, 2),
        "Inference Time (ms)": round(inf_time_mean, 2)
    })

    # Train EfficientNet-B0
    print("\n" + "=" * 60)
    print("🚀 Training EFFICIENTNET-B0...")
    print("=" * 60)

    model = get_model("efficientnet_b0", num_classes)
    size_before = get_model_size_mb(model)
    print(f"   Model size (before training): {size_before:.2f} MB")

    start_train = time.time()
    accuracy = train_model(model, train_loader, val_loader)
    train_time = time.time() - start_train

    size_after = get_model_size_mb(model)
    inf_time_mean, inf_time_std = measure_inference_time(model)

    print(f"\n   ✅ EfficientNet-B0 Results:")
    print(f"   Validation Accuracy: {accuracy:.2f}%")
    print(f"   Model Size: {size_after:.2f} MB")
    print(f"   Inference Time: {inf_time_mean:.2f} ± {inf_time_std:.2f} ms")
    print(f"   Training Time: {train_time / 60:.2f} min")

    results.append({
        "Model": "EfficientNet-B0",
        "Accuracy (%)": round(accuracy, 2),
        "Size (MB)": round(size_after, 2),
        "Inference Time (ms)": round(inf_time_mean, 2)
    })

    # Final report
    df = pd.DataFrame(results)
    df = df.sort_values("Accuracy (%)", ascending=False)
    df.to_csv("comparison_final_report.csv", index=False)

    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON REPORT")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("📝 COPY THIS TABLE FOR YOUR PAPER:")
    print("=" * 60)
    print("\n| Model | Accuracy (%) | Size (MB) | Inference Time (ms) |")
    print("|-------|--------------|-----------|---------------------|")
    for _, row in df.iterrows():
        print(f"| {row['Model']} | {row['Accuracy (%)']} | {row['Size (MB)']} | {row['Inference Time (ms)']} |")

    print(f"\n✅ Report saved to: comparison_final_report.csv")
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()