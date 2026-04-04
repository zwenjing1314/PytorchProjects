from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

weights_enum = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
ckpt_name = Path(weights_enum.url).name  # fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
ckpt_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / ckpt_name

if ckpt_path.exists():
    print(f"Use local weights: {ckpt_path}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
else:
    print("Local weights not found, try downloading...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights_enum)