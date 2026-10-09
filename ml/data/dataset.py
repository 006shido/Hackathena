"""
Real CelebA Paired Dataset & Preprocessing Layer for Neural Face Swapping
Builds genuine same-identity (ground truth reconstruction) and cross-identity training pairs.
Zero fake data generation, zero random image synthesis.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Safe PyTorch import
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:  # type: ignore
        """Fallback base class when PyTorch is not yet installed."""
        pass

from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    normalize_image_tensor,
    generate_facial_mask,
    get_canonical_landmarks
)


def validate_celeba_dataset(
    dataset_root: str,
    identity_file: str = "identity_CelebA.txt",
    images_dir: str = "img_align_celeba",
    quick_sample_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validates the local presence, integrity, and identity distributions of the CelebA dataset.

    Args:
        dataset_root: Path to the CelebA root directory.
        identity_file: Name of the identity mapping file.
        images_dir: Name of directory containing aligned face images.
        quick_sample_limit: Optional limit for testing without scanning all 202k files.

    Returns:
        report: Dictionary with dataset statistics and validation status.
    """
    root_path = Path(dataset_root)
    id_file_path = root_path / identity_file
    img_dir_path = root_path / images_dir

    report: Dict[str, Any] = {
        "dataset_root": str(root_path),
        "identity_file_path": str(id_file_path),
        "images_dir_path": str(img_dir_path),
        "is_valid": False,
        "total_references": 0,
        "existing_images": 0,
        "missing_images": 0,
        "total_identities": 0,
        "identities_with_single_image": 0,
        "identities_with_multiple_images": 0,
        "malformed_lines": 0,
        "errors": []
    }

    if not root_path.exists():
        report["errors"].append(
            f"Dataset root directory does not exist: '{root_path}'. "
            f"Please download CelebA and extract to '{root_path}'."
        )
        return report

    if not id_file_path.exists():
        report["errors"].append(
            f"Identity annotation file not found: '{id_file_path}'. "
            f"Please place 'identity_CelebA.txt' in '{root_path}'."
        )
        return report

    if not img_dir_path.exists():
        report["errors"].append(
            f"Images directory not found: '{img_dir_path}'. "
            f"Please place 'img_align_celeba/' in '{root_path}'."
        )
        return report

    identity_to_files: Dict[int, List[str]] = {}
    total_refs = 0
    malformed = 0
    missing = 0
    existing = 0

    with open(id_file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                malformed += 1
                continue

            filename, identity_str = parts[0], parts[1]
            try:
                identity_id = int(identity_str)
            except ValueError:
                malformed += 1
                continue

            total_refs += 1
            img_file = img_dir_path / filename

            if img_file.is_file():
                existing += 1
                if identity_id not in identity_to_files:
                    identity_to_files[identity_id] = []
                identity_to_files[identity_id].append(filename)
            else:
                missing += 1

            if quick_sample_limit and total_refs >= quick_sample_limit:
                break

    single_img_count = sum(1 for imgs in identity_to_files.values() if len(imgs) == 1)
    multi_img_count = sum(1 for imgs in identity_to_files.values() if len(imgs) > 1)

    report["total_references"] = total_refs
    report["existing_images"] = existing
    report["missing_images"] = missing
    report["total_identities"] = len(identity_to_files)
    report["identities_with_single_image"] = single_img_count
    report["identities_with_multiple_images"] = multi_img_count
    report["malformed_lines"] = malformed

    # Fail validation if missing files exceed 5% of references or if no multi-image identities exist
    if total_refs == 0:
        report["errors"].append("Annotation file is completely empty.")
    elif existing == 0:
        report["errors"].append(f"None of the {total_refs} referenced images exist inside '{img_dir_path}'.")
    elif (missing / total_refs) > 0.05:
        report["errors"].append(
            f"High missing image rate: {missing} out of {total_refs} files missing ({(missing/total_refs)*100:.1f}%)."
        )
    elif multi_img_count == 0:
        report["errors"].append(
            "No identities with multiple images found. Face-swap self-reconstruction requires >= 2 images per identity."
        )
    else:
        report["is_valid"] = True

    return report


class CelebAPairedDataset(Dataset):
    """
    Real CelebA Paired Dataset for Neural Face-Swap Training.

    Implements:
      1. Identity-disjoint train/validation partitioning (zero identity leakage).
      2. Same-identity sampling (Source=A, Target=B with id(A)==id(B), providing ground-truth supervision).
      3. Cross-identity sampling (Source from Person A, Target from Person B with id(A)!=id(B)).
      4. Landmark-derived soft facial region masks.
      5. [-1.0, 1.0] normalized RGB tensor outputs.
    """

    def __init__(
        self,
        dataset_root: str,
        identity_file: str = "identity_CelebA.txt",
        images_dir: str = "img_align_celeba",
        split: str = "train",
        train_ratio: float = 0.85,
        image_size: int = 128,
        same_identity_probability: float = 0.5,
        min_images_per_identity: int = 2,
        seed: int = 42,
        feather_radius: int = 5
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.id_file_path = self.dataset_root / identity_file
        self.img_dir_path = self.dataset_root / images_dir
        self.split = split.lower()
        self.train_ratio = train_ratio
        self.image_size = image_size
        self.same_identity_probability = same_identity_probability
        self.min_images_per_identity = min_images_per_identity
        self.seed = seed
        self.feather_radius = feather_radius

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"CelebA dataset not found at '{self.dataset_root}'. "
                f"Please place the CelebA dataset files under '{self.dataset_root}' "
                f"(expected '{identity_file}' and '{images_dir}/')."
            )

        if not self.id_file_path.exists():
            raise FileNotFoundError(f"Identity file not found at '{self.id_file_path}'.")

        if not self.img_dir_path.exists():
            raise FileNotFoundError(f"Image directory not found at '{self.img_dir_path}'.")

        self.preprocessor = RealFacePreprocessor(image_size=image_size, feather_radius=feather_radius)
        self.rng = random.Random(seed)

        # 1. Parse annotations and verify image existence on disk
        self.identity_to_images: Dict[int, List[Path]] = {}
        all_existing_images: List[Tuple[int, Path]] = []

        with open(self.id_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, id_str = parts[0], parts[1]
                try:
                    identity_id = int(id_str)
                except ValueError:
                    continue

                img_path = self.img_dir_path / filename
                if img_path.is_file():
                    if identity_id not in self.identity_to_images:
                        self.identity_to_images[identity_id] = []
                    self.identity_to_images[identity_id].append(img_path)
                    all_existing_images.append((identity_id, img_path))

        if len(self.identity_to_images) == 0:
            raise RuntimeError(f"No existing images found in CelebA directory '{self.img_dir_path}'.")

        # 2. Identity-Disjoint Train / Validation Split
        # Identities are sorted for determinism and partitioned so NO identity appears in both splits
        all_identities = sorted(list(self.identity_to_images.keys()))
        self.rng.shuffle(all_identities)

        num_train_ids = int(len(all_identities) * train_ratio)
        if self.split == "train":
            split_identities = set(all_identities[:num_train_ids])
        elif self.split in ("val", "validation"):
            split_identities = set(all_identities[num_train_ids:])
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train' or 'val'.")

        # Filter to split identities
        self.split_identity_to_images: Dict[int, List[Path]] = {
            i: self.identity_to_images[i]
            for i in split_identities
        }

        # Multi-image identities eligible for same-identity paired reconstruction
        self.multi_image_identities = [
            i for i, paths in self.split_identity_to_images.items()
            if len(paths) >= self.min_images_per_identity
        ]

        # Flat list of samples in this split for length and index tracking
        self.samples: List[Tuple[int, Path]] = [
            (i, path)
            for i, paths in self.split_identity_to_images.items()
            for path in paths
        ]

        if len(self.samples) == 0:
            raise RuntimeError(f"Split '{self.split}' has 0 samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Samples a real training pair:
          - Same-identity pair: with probability same_identity_probability
          - Cross-identity pair: with probability (1 - same_identity_probability)
        Rejects images where face detection fails and resamples a genuine pair.
        """
        max_attempts = 15
        for attempt in range(max_attempts):
            can_do_same_id = len(self.multi_image_identities) > 0
            is_same_identity = can_do_same_id and (random.random() < self.same_identity_probability)

            if is_same_identity:
                # SAME-IDENTITY: Select distinct images A and B from the same person
                identity_id = random.choice(self.multi_image_identities)
                img_list = self.split_identity_to_images[identity_id]
                source_path, target_path = random.sample(img_list, 2)
                source_id = identity_id
                target_id = identity_id
                pair_type = "same_identity"
            else:
                # CROSS-IDENTITY: Select Person A and Person B where id(A) != id(B)
                available_identities = list(self.split_identity_to_images.keys())
                if len(available_identities) < 2:
                    source_id = available_identities[0]
                    target_id = available_identities[0]
                    source_path = random.choice(self.split_identity_to_images[source_id])
                    target_path = source_path
                    pair_type = "same_identity"
                else:
                    source_id, target_id = random.sample(available_identities, 2)
                    source_path = random.choice(self.split_identity_to_images[source_id])
                    target_path = random.choice(self.split_identity_to_images[target_id])
                    pair_type = "cross_identity"

            try:
                # Preprocess both real images into normalized tensors [-1, 1], soft masks, and structural landmark maps
                source_tensor, source_mask = self.preprocessor.preprocess_aligned_celeba_image(str(source_path))
                target_tensor, target_mask, target_l_map, _ = self.preprocessor.preprocess_sample(str(target_path))
                break
            except Exception:
                # If face detection fails on a given photo, reject it and resample another genuine pair
                if attempt == max_attempts - 1:
                    raise

        # Convert to PyTorch tensors if torch is installed
        if TORCH_AVAILABLE:
            source_tensor = torch.from_numpy(source_tensor)
            target_tensor = torch.from_numpy(target_tensor)
            source_mask = torch.from_numpy(source_mask)
            target_mask = torch.from_numpy(target_mask)
            target_l_map = torch.from_numpy(target_l_map)

        return {
            "source": source_tensor,             # [3, H, W] in [-1.0, 1.0]
            "target": target_tensor,             # [3, H, W] in [-1.0, 1.0]
            "source_mask": source_mask,         # [1, H, W] in [0.0, 1.0]
            "target_mask": target_mask,         # [1, H, W] in [0.0, 1.0]
            "target_landmark_map": target_l_map,# [1, H, W] in [0.0, 1.0]
            "source_path": str(source_path),
            "target_path": str(target_path),
            "source_identity": source_id,
            "target_identity": target_id,
            "pair_type": pair_type              # "same_identity" or "cross_identity"
        }


def get_celeba_splits(
    dataset_root: str,
    identity_file: str = "identity_CelebA.txt",
    images_dir: str = "img_align_celeba",
    train_ratio: float = 0.85,
    image_size: int = 128,
    same_identity_probability: float = 0.5,
    seed: int = 42
) -> Tuple[CelebAPairedDataset, CelebAPairedDataset]:
    """
    Creates identity-disjoint train and validation dataset splits.
    """
    train_ds = CelebAPairedDataset(
        dataset_root=dataset_root,
        identity_file=identity_file,
        images_dir=images_dir,
        split="train",
        train_ratio=train_ratio,
        image_size=image_size,
        same_identity_probability=same_identity_probability,
        seed=seed
    )

    val_ds = CelebAPairedDataset(
        dataset_root=dataset_root,
        identity_file=identity_file,
        images_dir=images_dir,
        split="val",
        train_ratio=train_ratio,
        image_size=image_size,
        same_identity_probability=same_identity_probability,
        seed=seed
    )

    return train_ds, val_ds
