# train.py
import os, argparse, time, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from models.baseline_cnn import SmallCNN
from models.mobilenet_v2 import get_mobilenet_v2

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def get_loaders(data_dir, size=224, batch_size=32, augment=True):
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    train_tf = [transforms.Resize((size,size))]
    if augment:
        train_tf += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                     transforms.ColorJitter(0.1,0.1,0.1,0.05)]
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    val_tf = transforms.Compose([transforms.Resize((size,size)),
                                 transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose(train_tf))
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, train_ds.classes

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); y_true, y_pred = [], []; running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        running += loss.item() * x.size(0)
        y_true += y.detach().cpu().tolist()
        y_pred += out.argmax(1).detach().cpu().tolist()
    acc = accuracy_score(y_true, y_pred)
    return running/len(loader.dataset), acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); y_true, y_pred = [], []; running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x); loss = criterion(out, y)
        running += loss.item() * x.size(0)
        y_true += y.detach().cpu().tolist()
        y_pred += out.argmax(1).detach().cpu().tolist()
    acc = accuracy_score(y_true, y_pred)
    return running/len(loader.dataset), acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", choices=["cnn","mobilenet"], default="mobilenet")
    ap.add_argument("--pretrained", type=int, default=0)   # 1=use ImageNet (needs internet)
    ap.add_argument("--freeze_backbone", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--out_weights", default="weights/best_9948.pth")
    ap.add_argument("--log_csv", default="results/logs.csv")
    args = ap.parse_args()

    Path("weights").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, classes = get_loaders(args.data_dir, args.img_size, args.batch_size, bool(args.augment))
    num_classes = len(classes)

    if args.model == "cnn":
        model = SmallCNN(num_classes)
    else:
        model = get_mobilenet_v2(num_classes, pretrained=bool(args.pretrained), freeze_backbone=bool(args.freeze_backbone))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_acc = 0.0
    hist = {"epoch":[], "train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        hist["epoch"].append(epoch); hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss); hist["train_acc"].append(tr_acc); hist["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
            }, args.out_weights)
            with open("weights/classes.json","w") as f: json.dump(classes, f, indent=2)
            print(f" Saved best → {args.out_weights} (val_acc={val_acc:.4f})")

    # Save curves
    import pandas as pd
    df = pd.DataFrame(hist); df.to_csv(args.log_csv, index=False)
    plt.figure(); plt.plot(df["epoch"], df["train_loss"], label="train"); plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout(); plt.savefig("results/curves_loss.png"); plt.close()
    plt.figure(); plt.plot(df["epoch"], df["train_acc"], label="train"); plt.plot(df["epoch"], df["val_acc"], label="val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("acc"); plt.tight_layout(); plt.savefig("results/curves_acc.png"); plt.close()

    print(f" Done. Best val acc {best_val_acc:.4f}. Time {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
