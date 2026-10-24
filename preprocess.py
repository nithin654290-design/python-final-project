"""
Data preprocessing and augmentation for HAM10000 dermoscopic images.

Provides:
  * DullRazor hair removal (black-hat morphology + inpainting)
  * CLAHE contrast enhancement
  * PyTorch Dataset / DataLoader builders with augmentation
  * Balanced class-weight computation for the imbalanced HAM10000 classes

Expected dataset layout (see README.md):

    data/
        HAM10000_metadata.csv
        images/
            ISIC_0024306.jpg
            ...
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

# Canonical class order used everywhere (model output index == position here).
CLASS_NAMES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_LABELS: Dict[str, str] = {
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

# Malignant / Pre-cancerous / Benign risk mapping.
CLASS_RISK: Dict[str, str] = {
    "mel": "Malignant",
    "bcc": "Malignant",
    "akiec": "Pre-cancerous",
    "bkl": "Benign",
    "df": "Benign",
    "nv": "Benign",
    "vasc": "Benign",
}

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image cleaning
# ---------------------------------------------------------------------------


def dull_razor(image_bgr: np.ndarray, kernel_size: int = 9, threshold: int = 10) -> np.ndarray:
    """Remove hair artifacts with the DullRazor algorithm.

    Steps: grayscale -> black-hat morphology (finds dark thin structures) ->
    binary threshold -> inpainting of the detected hair mask.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel (LAB)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def clean_image(image_bgr: np.ndarray) -> np.ndarray:
    """Full artifact-removal pipeline: DullRazor then CLAHE."""
    return apply_clahe(dull_razor(image_bgr))


def preprocess_for_inference(path_or_array) -> torch.Tensor:
    """Load an image (path, PIL image or BGR array) and return a 1x3xHxW tensor."""
    if isinstance(path_or_array, str):
        image_bgr = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
    elif isinstance(path_or_array, Image.Image):
        image_bgr = cv2.cvtColor(np.array(path_or_array.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image_bgr = np.asarray(path_or_array)

    cleaned = clean_image(image_bgr)
    rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = eval_transforms()(pil)
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HAM10000Dataset(Dataset):
    """Dermoscopic lesion dataset reading from a metadata dataframe."""

    def __init__(self, frame: pd.DataFrame, images_dir: str, transform=None, clean: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.clean = clean

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_path(self, image_id: str) -> str:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(self.images_dir, image_id + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Image not found for id '{image_id}' in {self.images_dir}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = self._resolve_path(str(row["image_id"]))
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to decode image: {path}")
        if self.clean:
            image_bgr = clean_image(image_bgr)
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil) if self.transform else transforms.ToTensor()(pil)
        return tensor, int(row["label"])


def load_metadata(data_dir: str) -> pd.DataFrame:
    """Read HAM10000_metadata.csv and attach integer labels."""
    csv_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    frame = pd.read_csv(csv_path)
    frame["dx"] = frame["dx"].str.lower().str.strip()
    unknown = sorted(set(frame["dx"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unexpected diagnosis values in metadata: {unknown}")
    frame["label"] = frame["dx"].map({name: i for i, name in enumerate(CLASS_NAMES)})
    return frame


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    num_workers: int = 2,
    subset_fraction: float = 1.0,
    clean: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create stratified train / val / test dataloaders."""
    images_dir = os.path.join(data_dir, "images")
    frame = load_metadata(data_dir)

    if subset_fraction < 1.0:
        frame, _ = train_test_split(
            frame,
            train_size=subset_fraction,
            stratify=frame["label"],
            random_state=seed,
        )

    train_frame, holdout = train_test_split(
        frame, test_size=val_size + test_size, stratify=frame["label"], random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    val_frame, test_frame = train_test_split(
        holdout, test_size=relative_test, stratify=holdout["label"], random_state=seed
    )

    train_loader = DataLoader(
        HAM10000Dataset(train_frame, images_dir, train_transforms(), clean),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        HAM10000Dataset(val_frame, images_dir, eval_transforms(), clean),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        HAM10000Dataset(test_frame, images_dir, eval_transforms(), clean),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def compute_class_weights(data_dir: str) -> torch.Tensor:
    """Balanced class weights: n_samples / (n_classes * count_per_class)."""
    frame = load_metadata(data_dir)
    counts = frame["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    total = float(counts.sum())
    weights = []
    for count in counts.tolist():
        weights.append(total / (len(CLASS_NAMES) * count) if count > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the preprocessing pipeline")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    meta = load_metadata(args.data_dir)
    print(f"Images: {len(meta)}")
    print(meta["dx"].value_counts())
    print("Class weights:", compute_class_weights(args.data_dir).tolist())
