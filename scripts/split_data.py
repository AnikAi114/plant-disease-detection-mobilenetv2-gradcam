# scripts/split_data.py
import os, shutil, random, json
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(42)

RAW_DIR = Path("data_raw/PlantVillage")
OUT_DIR = Path("data")
RATIOS = (0.70, 0.15, 0.15)  # train, val, test

def list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    return [p for p in folder.rglob("*") if p.suffix in exts]

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    assert RAW_DIR.exists(), f"RAW dataset not found at {RAW_DIR}"
    classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "class_to_idx.json").write_text(
        json.dumps({c:i for i, c in enumerate(classes)}, indent=2)
    )

    for cls in classes:
        src = RAW_DIR / cls
        imgs = list_images(src)
        if not imgs:
            print(f"[WARN] No images in {cls}"); continue

        tr, vt = train_test_split(imgs, test_size=(1-RATIOS[0]), random_state=42, shuffle=True)
        rel = RATIOS[1]/(RATIOS[1]+RATIOS[2])
        va, te = train_test_split(vt, test_size=(1-rel), random_state=42, shuffle=True)

        for p in tr: safe_copy(p, OUT_DIR/"train"/cls/p.name)
        for p in va: safe_copy(p, OUT_DIR/"val"/cls/p.name)
        for p in te: safe_copy(p, OUT_DIR/"test"/cls/p.name)

        print(f"[OK] {cls:40s} | Train: {len(tr):4d} | Val: {len(va):4d} | Test: {len(te):4d}")

    print("\n✅ Dataset splitting complete!")

if __name__ == "__main__":
    main()
