# datasets/real_wrapper.py
# A "guaranteed-to-run" real-video wrapper with safe fallbacks.
#
# Why this exists:
# - Your configs include real_video.yaml / vlm_binding.yaml, but you may not have
#   an actual video dataset downloaded.
# - This file provides a unified Dataset interface that can:
#     (A) Load real frame folders if you have them
#     (B) Otherwise fall back to a synthetic "pseudo-video" dataset built from torchvision FakeData
#         (NO internet/download required) -> GUARANTEED to work.
#
# Output per sample:
#   {
#     "x": FloatTensor [T, C, H, W]  (in [0,1])
#   }
#
# Optional:
#   - returns "y" labels for debugging if source provides labels (not required by world model).
#
# Dependencies: torch, torchvision (already in requirements), PIL (via torchvision), numpy

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import FakeData
from PIL import Image


# -------------------------
# Helpers
# -------------------------
def _is_image_file(p: str) -> bool:
    p = p.lower()
    return p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".bmp") or p.endswith(".webp")


def _list_subdirs(root: str) -> List[str]:
    return sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])


def _list_frames(vid_dir: str) -> List[str]:
    files = sorted([p for p in glob.glob(os.path.join(vid_dir, "*")) if os.path.isfile(p) and _is_image_file(p)])
    return files


def _seeded_rng(seed: int, idx: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + int(idx) * 10007)


def _default_img_transform(image_size: int, channels: int) -> transforms.Compose:
    """
    Returns a transform producing float tensor in [0,1], shape [C,H,W].
    """
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    t: List[transforms.Transform] = []
    t.append(transforms.Resize((image_size, image_size)))
    if channels == 1:
        t.append(transforms.Grayscale(num_output_channels=1))
    else:
        t.append(transforms.Lambda(lambda im: im.convert("RGB")))
    t.append(transforms.ToTensor())  # [0,1]
    return transforms.Compose(t)


# -------------------------
# (A) Real frame-folder videos
# -------------------------
class FrameFolderVideoDataset(Dataset):
    """
    Expected directory structure:
      root/
        video_0001/
          000001.jpg
          000002.jpg
          ...
        video_0002/
          ...
    Each subfolder is treated as a video (ordered by filename sort).
    """

    def __init__(
        self,
        root: str,
        seq_len: int = 16,
        image_size: int = 64,
        channels: int = 3,
        stride: int = 1,
        random_start: bool = True,
        seed: int = 42,
    ):
        if not isinstance(root, str) or len(root) == 0:
            raise ValueError("root must be a non-empty path string")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"FrameFolderVideoDataset root not found: {root}")
        if not isinstance(seq_len, int) or seq_len <= 1:
            raise ValueError("seq_len must be int > 1")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be int > 0")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")

        self.root = root
        self.seq_len = seq_len
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.stride = stride
        self.random_start = bool(random_start)
        self.seed = int(seed)

        self.transform = _default_img_transform(self.image_size, self.channels)

        self.video_dirs = _list_subdirs(self.root)
        if len(self.video_dirs) == 0:
            raise FileNotFoundError(f"No subfolders (videos) found under: {self.root}")

        self.videos: List[List[str]] = []
        for vd in self.video_dirs:
            frames = _list_frames(vd)
            if len(frames) >= self.seq_len * self.stride:
                self.videos.append(frames)

        if len(self.videos) == 0:
            raise FileNotFoundError(
                f"Found video subfolders, but none have enough frames for seq_len={self.seq_len}, stride={self.stride}."
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frames = self.videos[int(idx)]
        rng = _seeded_rng(self.seed, idx)

        max_start = len(frames) - (self.seq_len - 1) * self.stride - 1
        if max_start < 0:
            # should not happen due to filtering
            start = 0
        else:
            start = int(rng.integers(0, max_start + 1)) if self.random_start else 0

        chosen = [frames[start + t * self.stride] for t in range(self.seq_len)]
        xs = []
        for p in chosen:
            img = Image.open(p)
            x = self.transform(img)  # [C,H,W] float in [0,1]
            xs.append(x)
        x_seq = torch.stack(xs, dim=0)  # [T,C,H,W]
        return {"x": x_seq}


# -------------------------
# (B) Guaranteed fallback: pseudo-videos from FakeData
# -------------------------
class PseudoVideoFromImages(Dataset):
    """
    Builds a "video" by taking a base image and applying deterministic per-frame augmentations.
    This is NOT a true video dataset, but it is perfect as a "real_video.yaml runs today" fallback.

    - Uses torchvision.datasets.FakeData by default (no download).
    - Sequence dynamics: small translations + rotations + color jitter (optional)
    """

    def __init__(
        self,
        size: int = 20000,
        seq_len: int = 16,
        image_size: int = 64,
        channels: int = 3,
        # augmentation magnitudes
        max_translate: float = 0.08,   # fraction of image size
        max_rotate: float = 12.0,      # degrees
        color_jitter: bool = True,
        noise_std: float = 0.0,
        seed: int = 42,
    ):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("size must be positive int")
        if not isinstance(seq_len, int) or seq_len <= 1:
            raise ValueError("seq_len must be int > 1")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")

        self.size = size
        self.seq_len = seq_len
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.max_translate = float(max_translate)
        self.max_rotate = float(max_rotate)
        self.color_jitter = bool(color_jitter)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

        # base dataset: FakeData returns PIL images by default if transform=None
        img_size = (self.channels, self.image_size, self.image_size)
        self.base = FakeData(
            size=self.size,
            image_size=img_size,
            num_classes=10,
            transform=None,
            target_transform=None,
            random_offset=0,
        )

        # base transform to tensor
        self.to_tensor = _default_img_transform(self.image_size, self.channels)

        # optional color jitter (applied after geometric transforms)
        self.jitter = (
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)
            if self.color_jitter and self.channels == 3
            else None
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = _seeded_rng(self.seed, idx)
        img, y = self.base[int(idx)]  # PIL image-like (actually produced by FakeData)
        if not isinstance(img, Image.Image):
            # FakeData can return tensors in some versions; normalize
            if isinstance(img, torch.Tensor):
                # [C,H,W] in [0,1] already
                base = img
                if base.shape[-1] != self.image_size or base.shape[-2] != self.image_size:
                    base = torchvision.transforms.functional.resize(base, [self.image_size, self.image_size])
            else:
                raise TypeError(f"Unexpected FakeData image type: {type(img)}")
        else:
            base = self.to_tensor(img)  # [C,H,W] float

        # Build sequence by per-frame deterministic transforms
        xs = []
        for t in range(self.seq_len):
            # deterministic parameters for this frame
            # small drifting motion
            frac = (t / max(1, self.seq_len - 1))
            dx = (rng.uniform(-1, 1) * self.max_translate) * (0.3 + 0.7 * frac)
            dy = (rng.uniform(-1, 1) * self.max_translate) * (0.3 + 0.7 * frac)
            ang = rng.uniform(-self.max_rotate, self.max_rotate) * (0.3 + 0.7 * frac)

            x = base
            # torchvision.functional.affine expects PIL or Tensor; handle tensor
            x = torchvision.transforms.functional.affine(
                x,
                angle=float(ang),
                translate=[int(dx * self.image_size), int(dy * self.image_size)],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                fill=0,
                center=None,
            )

            if self.jitter is not None:
                # jitter expects PIL; convert tensor->PIL->tensor safely
                pil = torchvision.transforms.functional.to_pil_image(x)
                pil = self.jitter(pil)
                x = torchvision.transforms.functional.to_tensor(pil)
                if self.channels == 1 and x.shape[0] != 1:
                    x = x[:1]

            if self.noise_std > 0:
                x = x + torch.randn_like(x) * float(self.noise_std)

            x = torch.clamp(x, 0.0, 1.0)
            xs.append(x)

        x_seq = torch.stack(xs, dim=0)  # [T,C,H,W]
        return {"x": x_seq, "y": torch.tensor(int(y), dtype=torch.long)}


# -------------------------
# Builders
# -------------------------
def build_real_video_datasets(cfg: Dict) -> Dict[str, Dataset]:
    """
    Build datasets dict for real_video.yaml.

    Supported sources:
      data.source: "frame_folder" | "fake"
        - frame_folder requires data.root to exist and contain subfolders of frames.
        - fake is guaranteed to work (no download).

    Returns: {"train":..., "val":..., "test":..., "test_ood"(optional)}
    """
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))

    source = str(data_cfg.get("source", "fake")).lower()
    seq_len = int(data_cfg.get("sequence", {}).get("length", 16))
    image_size = int(data_cfg.get("image_size", 64))
    channels = int(data_cfg.get("channels", 3))

    split_cfg = data_cfg.get("split", {})
    train_size = int(split_cfg.get("train_size", 20000))
    val_size = int(split_cfg.get("val_size", 2000))
    test_size = int(split_cfg.get("test_size", 2000))

    # Try real data if requested, else fallback to fake
    if source == "frame_folder":
        root = str(data_cfg.get("root", ""))
        stride = int(data_cfg.get("stride", 1))
        random_start = bool(data_cfg.get("random_start", True))

        # If it fails, we fall back to fake to GUARANTEE it runs.
        try:
            train = FrameFolderVideoDataset(
                root=root,
                seq_len=seq_len,
                image_size=image_size,
                channels=channels,
                stride=stride,
                random_start=random_start,
                seed=seed,
            )
            # for simplicity, reuse same dataset object for val/test with different seeds
            val = FrameFolderVideoDataset(
                root=root,
                seq_len=seq_len,
                image_size=image_size,
                channels=channels,
                stride=stride,
                random_start=random_start,
                seed=seed + 1337,
            )
            test = FrameFolderVideoDataset(
                root=root,
                seq_len=seq_len,
                image_size=image_size,
                channels=channels,
                stride=stride,
                random_start=random_start,
                seed=seed + 7331,
            )
            return {"train": train, "val": val, "test": test}
        except Exception as e:
            # fallback
            print(f"[real_wrapper] FrameFolderVideoDataset failed ({e}). Falling back to FakeData pseudo-video.")

    # Guaranteed fake fallback
    aug_cfg = data_cfg.get("augment", {})
    noise_std = float(aug_cfg.get("noise_std", 0.0))
    max_translate = float(aug_cfg.get("max_translate", 0.08))
    max_rotate = float(aug_cfg.get("max_rotate", 12.0))
    color_jitter = bool(aug_cfg.get("color_jitter", True))

    train = PseudoVideoFromImages(
        size=train_size,
        seq_len=seq_len,
        image_size=image_size,
        channels=channels,
        max_translate=max_translate,
        max_rotate=max_rotate,
        color_jitter=color_jitter,
        noise_std=noise_std,
        seed=seed,
    )
    val = PseudoVideoFromImages(
        size=val_size,
        seq_len=seq_len,
        image_size=image_size,
        channels=channels,
        max_translate=max_translate,
        max_rotate=max_rotate,
        color_jitter=color_jitter,
        noise_std=noise_std,
        seed=seed + 1337,
    )
    test = PseudoVideoFromImages(
        size=test_size,
        seq_len=seq_len,
        image_size=image_size,
        channels=channels,
        max_translate=max_translate,
        max_rotate=max_rotate,
        color_jitter=color_jitter,
        noise_std=noise_std,
        seed=seed + 7331,
    )

    # Optional OOD: stronger motion/noise
    ood_cfg = data_cfg.get("ood", {})
    out = {"train": train, "val": val, "test": test}
    if bool(ood_cfg.get("enabled", False)):
        test_ood = PseudoVideoFromImages(
            size=test_size,
            seq_len=seq_len,
            image_size=image_size,
            channels=channels,
            max_translate=float(ood_cfg.get("test_max_translate", max_translate * 1.8)),
            max_rotate=float(ood_cfg.get("test_max_rotate", max_rotate * 1.8)),
            color_jitter=bool(ood_cfg.get("test_color_jitter", color_jitter)),
            noise_std=float(ood_cfg.get("test_noise_std", max(noise_std, 0.02))),
            seed=seed + 99991,
        )
        out["test_ood"] = test_ood

    return out

