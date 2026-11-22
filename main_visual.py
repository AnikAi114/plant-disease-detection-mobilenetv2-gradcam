import argparse, os
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import textwrap

# === Import your models ===
from models.baseline_cnn import SmallCNN
from models.mobilenet_v2 import get_mobilenet_v2


# ---------- Load trained model ----------
def load_model(weights_path="weights/best_9948.pth", device="cpu"):
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    model_name = ckpt.get("model_name", "mobilenet")

    if model_name == "cnn":
        model = SmallCNN(len(classes))
    else:
        model = get_mobilenet_v2(len(classes), pretrained=False, freeze_backbone=False)

    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, classes, img_size


# ---------- Image preprocessing ----------
def preprocess(img_path, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return img, tf(img).unsqueeze(0)


# ---------- Dynamic font + label drawing ----------
def _pick_font(img_w):
    """Pick best available font depending on OS"""
    size = max(18, img_w // 22)
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",  # Windows
        r"/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_label(img: Image.Image, text: str, place="top"):
    """Draw readable label with wrapping and transparency"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    draw = ImageDraw.Draw(img, "RGBA")
    font = _pick_font(img.width)

    # Auto-wrap long text
    max_chars = max(16, img.width // 12)
    lines = textwrap.wrap(text, width=max_chars)

    # Measure text box
    padding = 10
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    box_w = max(draw.textlength(ln, font=font) for ln in lines) + 2 * padding
    box_h = line_h * len(lines) + 2 * padding

    # Choose top or bottom
    x0 = 10
    y0 = 10 if place == "top" else (img.height - box_h - 10)

    # Draw transparent background
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 180))

    # Draw each text line
    y = y0 + padding
    for ln in lines:
        draw.text((x0 + padding, y), ln, fill=(50, 220, 90, 255), font=font)
        y += line_h

    return img.convert("RGB")


# ---------- Main function ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to test image")
    ap.add_argument("--weights", default="weights/best_9948.pth", help="Model weights path")
    ap.add_argument("--save_dir", default="results/predicted", help="Save folder")
    ap.add_argument("--place", choices=["top", "bottom"], default="top", help="Label position")
    ap.add_argument("--no_show", action="store_true", help="Do not open preview window")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, size = load_model(args.weights, device)

    # Load image
    orig_img, x = preprocess(args.img, size)
    x = x.to(device)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze()
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx])

    label = f"Prediction: {classes[idx]}  ({conf * 100:.2f}%)"
    vis = draw_label(orig_img.copy(), label, place=args.place)

    # Save result
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.save_dir) / (Path(args.img).stem + "_pred.jpg")
    vis.save(out_path)

    print(f" Prediction: {classes[idx]} (confidence {conf * 100:.2f}%)")
    print(f"💾 Saved → {out_path}")

    if not args.no_show:
        try:
            vis.show()
        except Exception:
            print("⚠️ Could not open preview window — image saved successfully.")


if __name__ == "__main__":
    main()
