# models/mobilenet_v2.py
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_mobilenet_v2(num_classes: int, pretrained: bool = False, freeze_backbone: bool = True):
    try:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v2(weights=weights)
    except Exception:
        # offline fallback
        model = mobilenet_v2(weights=None)

    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    return model
