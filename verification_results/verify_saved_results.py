"""Verify archived aggregate metrics without accessing private training images."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
cm = pd.read_csv(ROOT / "clean_test_confusion_matrix.csv", index_col=0)
metrics = json.loads((ROOT / "clean_test_metrics.json").read_text(encoding="utf-8"))
actual_n = int(cm.to_numpy().sum())
correct = int(np.trace(cm.to_numpy()))
accuracy = correct / actual_n
assert actual_n == int(metrics["test_images"]), "Image count mismatch"
assert abs(accuracy - float(metrics["accuracy"])) < 1e-12, "Accuracy mismatch"
print(f"Saved evaluation verified: {correct}/{actual_n} = {accuracy * 100:.4f}%")
print("This does not independently rerun the model or rule out near-duplicates.")
