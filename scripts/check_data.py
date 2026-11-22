# scripts/check_data.py
from pathlib import Path

DATA = Path("data")

def count_files(d: Path):
    return sum(1 for p in d.iterdir() if p.is_file())

for split in ["train", "val", "test"]:
    d = DATA / split
    print(f"\n[{split.upper()}]")
    if not d.exists():
        print("  (missing)"); continue
    total = 0
    for cls in sorted([x for x in d.iterdir() if x.is_dir()]):
        n = count_files(cls); total += n
        print(f"{cls.name:40s}: {n:5d}")
    print(f"Total {split}: {total}")
