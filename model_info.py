# model_info.py
import torch
import time
import os
import numpy as np
from pathlib import Path
from models.mobilenet_v2 import get_mobilenet_v2


def load_model_from_checkpoint(weights_path, device='cpu'):
    """Load model from checkpoint"""
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)
    model_name = ckpt.get("model_name", "mobilenet")

    if model_name == "cnn":
        from models.baseline_cnn import SmallCNN
        model = SmallCNN(num_classes)
    else:
        model = get_mobilenet_v2(num_classes, pretrained=False, freeze_backbone=False)

    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, classes


def get_model_size(weights_path):
    """Get model file size (in MB)"""
    size_bytes = os.path.getsize(weights_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def measure_inference_time(model, device='cpu', num_runs=100):
    """Measure inference time (in milliseconds)"""
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Warm up (to warm up GPU/CPU)
    for _ in range(10):
        _ = model(dummy_input)

    # Start timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)  # in milliseconds

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    return avg_time, std_time, min_time, max_time


def main():
    print("=" * 50)
    print("Model Size & Inference Time Analysis")
    print("=" * 50)

    weights_path = "weights/best_9948.pth"

    # 1. Model Size
    print("\n📦 1. Model Size:")
    size_mb = get_model_size(weights_path)
    print(f"   File size: {size_mb:.2f} MB")
    print(f"   File size: {size_mb * 1024:.2f} KB")

    # 2. Load model
    print("\n🔄 2. Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    model, classes = load_model_from_checkpoint(weights_path, device)
    print(f"   Classes: {len(classes)}")

    # 3. Inference Time
    print("\n⏱️ 3. Inference Time (100 runs):")
    avg_time, std_time, min_time, max_time = measure_inference_time(model, device, num_runs=100)
    print(f"   Average: {avg_time:.2f} ms")
    print(f"   Std Dev: {std_time:.2f} ms")
    print(f"   Min: {min_time:.2f} ms")
    print(f"   Max: {max_time:.2f} ms")

    # 4. Show GPU memory if available
    if device == 'cuda':
        print("\n🎮 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    # 5. Save data for comparison table
    print("\n" + "=" * 50)
    print("📊 Summary for Paper:")
    print("=" * 50)
    print(f"| Model Size | {size_mb:.2f} MB |")
    print(f"| Inference Time (avg) | {avg_time:.2f} ms |")
    print(f"| Device | {device.upper()} |")

    # Save to file
    with open("results/model_info.txt", "w") as f:
        f.write("Model Size & Inference Time Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model Size: {size_mb:.2f} MB\n")
        f.write(f"Inference Time (avg): {avg_time:.2f} ms\n")
        f.write(f"Inference Time (std): {std_time:.2f} ms\n")
        f.write(f"Device: {device.upper()}\n")
        f.write(f"Number of Classes: {len(classes)}\n")

    print("\n✅ Results saved to: results/model_info.txt")


if __name__ == "__main__":
    main()