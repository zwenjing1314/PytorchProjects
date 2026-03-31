import matplotlib.pyplot as plt
from torchvision.io import read_image  # python 版本最好大于3.7,否则运行报错
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
BASE_PATH = Path(__file__).resolve().parents[1]

img_path = BASE_PATH / "d05_视觉图像检测基于Mask_R-CNN" / "data" / "PennFudanPed" / "PNGImages" / "FudanPed00046.png"
mask_path = BASE_PATH / "d05_视觉图像检测基于Mask_R-CNN" / "data" / "PennFudanPed" / "PedMasks" / "FudanPed00046_mask.png"


image = read_image(str(img_path))
mask = read_image(str(mask_path))

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))
# plt.show()

import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        """
        obj_ids[:, None, None]：把 obj_ids 从 [N] 变成 [N,1,1]，方便广播；
        与 mask 比较时自动广播，得到形状大致为 [N,H,W] 的布尔张量；
        每个 masks[i] 都是第 i 个目标的二值掩码（该目标像素为1，其它为0）；
        最后转成 uint8 便于后续检测/分割流程使用。
        """
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # 得到 h 和 w
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
