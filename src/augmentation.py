"""SimCLR pretrain + supervised fine-tune augmentation pipelines.

NEVER add RandomHorizontalFlip — left/right hand is class-defining for State Farm.

Pretrain pipeline uses `torchvision.transforms.v2` with `v2.ToImage()` prefix so
it accepts BOTH PIL (JPEG fallback path) AND uint8 CHW tensor (memmap cache path).
"""
from typing import Tuple
import torch
import torchvision.transforms as T
from torchvision.transforms import v2
from PIL import Image

from .config import (
    IMG_SIZE, PRETRAIN_IMG_SIZE, PRETRAIN_IMG_H, PRETRAIN_IMG_W,
    IMAGENET_MEAN, IMAGENET_STD,
)


def build_pretrain_transform() -> v2.Compose:
    """SimCLR augmentation. Tuned for State Farm cabin scenes (not ImageNet).

    Pretrain at 160x120 (W x H, 4:3 aspect — matches dashcam source).
    Phase 2 fine-tune upscales back to IMG_SIZE (224) — FixRes trick.
    Blur kernel scaled to ~10% of pretrain resolution (15 px).

    Accepts PIL or uint8 CHW tensor (memmap cache). Output: float32 CHW tensor.
    """
    return v2.Compose([
        v2.ToImage(),  # no-op for tensors; PIL -> uint8 CHW tensor
        v2.RandomResizedCrop((PRETRAIN_IMG_H, PRETRAIN_IMG_W), scale=(0.5, 1.0), antialias=True),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))], p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_finetune_train_transform() -> T.Compose:
    """Lighter aug for supervised fine-tune."""
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform() -> T.Compose:
    """Deterministic eval transform (val + test) at IMG_SIZE (224)."""
    return T.Compose([
        T.Resize(int(IMG_SIZE * 256 / 224)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_pretrain_eval_transform() -> T.Compose:
    """Eval transform at 160x120 (W x H). Linear probe matches pretrain res."""
    # Resize short side, then center-crop to target H x W
    resize_short = int(PRETRAIN_IMG_H * 256 / 224)
    return T.Compose([
        T.Resize(resize_short),
        T.CenterCrop((PRETRAIN_IMG_H, PRETRAIN_IMG_W)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_tta_transforms() -> list:
    """5-crop TTA. NO horizontal flip."""
    resize = int(IMG_SIZE * 256 / 224)
    base_resize = T.Resize(resize)
    normalize = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def make(crop):
        return T.Compose([base_resize, crop, normalize])

    return [
        make(T.CenterCrop(IMG_SIZE)),
        make(T.Lambda(lambda img: img.crop((0, 0, IMG_SIZE, IMG_SIZE)))),
        make(T.Lambda(lambda img: img.crop((img.width - IMG_SIZE, 0, img.width, IMG_SIZE)))),
        make(T.Lambda(lambda img: img.crop((0, img.height - IMG_SIZE, IMG_SIZE, img.height)))),
        make(T.Lambda(lambda img: img.crop(
            (img.width - IMG_SIZE, img.height - IMG_SIZE, img.width, img.height)
        ))),
    ]


class ContrastiveViewGenerator:
    """Apply transform twice to produce a positive pair for SimCLR.

    Works for both PIL inputs (JPEG path) and uint8 CHW tensors (memmap path).
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img) -> Tuple:
        return self.transform(img), self.transform(img)
