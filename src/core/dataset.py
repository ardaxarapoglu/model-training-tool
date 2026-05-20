"""PyTorch dataset and transform builder for froth images."""
import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomRotate90(torch.nn.Module):
    """Randomly rotate the image by exactly 0, 90, 180, or 270 degrees.

    No black corners because axis-aligned 90-degree rotations always produce
    a fully-filled rectangle.  Each call independently picks one of the four
    angles with equal probability, so over many epochs the model sees up to
    4x the effective variety per image.
    """

    def forward(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class FrothDataset(Dataset):
    """Map image files to their PB-concentration (regression) labels."""

    def __init__(self, samples: List[Tuple[str, float]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def collect_samples(experiments: list, split: str) -> List[Tuple[str, float]]:
    """Gather (image_path, pb_concentration) tuples for the requested split."""
    samples = []
    for exp in experiments:
        if exp.get("split") != split:
            continue
        for tf in exp.get("time_frames", []):
            folder = tf.get("folder_path", "")
            if not folder or not os.path.isdir(folder):
                continue
            pb = float(tf.get("pb_concentration", 0.0))
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    samples.append((os.path.join(folder, fname), pb))
    return samples


def build_transform(prep_cfg: dict, augment: bool = False) -> T.Compose:
    """Build a torchvision Compose transform from the preprocessing config."""
    steps = []

    # Crop
    crop = prep_cfg.get("crop", {})
    if crop.get("enabled", False):
        if crop.get("type") == "center":
            size = int(crop.get("center_size", 224))
            steps.append(T.CenterCrop(size))
        else:
            x, y = int(crop.get("x", 0)), int(crop.get("y", 0))
            w, h = int(crop.get("w", 224)), int(crop.get("h", 224))
            steps.append(T.Lambda(lambda img, _x=x, _y=y, _w=w, _h=h: img.crop((_x, _y, _x + _w, _y + _h))))

    # Resize
    rz = prep_cfg.get("resize", {})
    steps.append(T.Resize((int(rz.get("height", 224)), int(rz.get("width", 224)))))

    if augment:
        aug = prep_cfg.get("augmentation", {})

        if aug.get("random_resized_crop", {}).get("enabled", False):
            rrc = aug["random_resized_crop"]
            rz_cfg = prep_cfg.get("resize", {})
            steps.append(
                T.RandomResizedCrop(
                    (int(rz_cfg.get("height", 224)), int(rz_cfg.get("width", 224))),
                    scale=(float(rrc.get("scale_min", 0.8)), float(rrc.get("scale_max", 1.0))),
                )
            )

        if aug.get("h_flip", {}).get("enabled", False):
            steps.append(T.RandomHorizontalFlip(p=float(aug["h_flip"].get("p", 0.5))))

        if aug.get("v_flip", {}).get("enabled", False):
            steps.append(T.RandomVerticalFlip(p=float(aug["v_flip"].get("p", 0.5))))

        if aug.get("rotation", {}).get("enabled", False):
            steps.append(RandomRotate90())

        if aug.get("color_jitter", {}).get("enabled", False):
            cj = aug["color_jitter"]
            steps.append(
                T.ColorJitter(
                    brightness=float(cj.get("brightness", 0.2)),
                    contrast=float(cj.get("contrast", 0.2)),
                    saturation=float(cj.get("saturation", 0.2)),
                    hue=float(cj.get("hue", 0.05)),
                )
            )

        if aug.get("gaussian_blur", {}).get("enabled", False):
            ks = int(aug["gaussian_blur"].get("kernel_size", 5))
            if ks % 2 == 0:
                ks += 1
            steps.append(T.GaussianBlur(kernel_size=ks))

    steps.append(T.ToTensor())

    norm = prep_cfg.get("normalize", {})
    if norm.get("use_imagenet", True):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = norm.get("mean", [0.485, 0.456, 0.406])
        std = norm.get("std", [0.229, 0.224, 0.225])
    steps.append(T.Normalize(mean=mean, std=std))

    if augment:
        aug = prep_cfg.get("augmentation", {})
        if aug.get("random_erasing", {}).get("enabled", False):
            p = float(aug["random_erasing"].get("p", 0.2))
            steps.append(T.RandomErasing(p=p))

    return T.Compose(steps)
