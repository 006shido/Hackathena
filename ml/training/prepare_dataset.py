#!/usr/bin/env python3
"""
CelebA Dataset Verification & Preprocessing Validation Utility for Hackathena Face-Swap
Inspects the local dataset, validates identity mappings, checks file integrity,
and optionally validates genuine facial landmark detection and mask preprocessing on real images.

Usage:
    python ml/training/prepare_dataset.py --dataset-root ml/data/celeba
    python ml/training/prepare_dataset.py --dataset-root ml/data/celeba --validate-preprocessing
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.dataset import validate_celeba_dataset
from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    NoFaceDetectedError,
    InvalidLandmarksError
)


def run_preprocessing_validation(
    dataset_root: str,
    images_dir: str = "img_align_celeba",
    model_asset_path: str = "client/public/models/face_landmarker.task",
    max_images: int = 100
) -> Dict[str, Any]:
    """
    Validates genuine facial landmark detection and mask preprocessing on real CelebA images.

    Reports:
      - total annotation images evaluated
      - images successfully detected
      - images with no detected face
      - images with multiple faces
      - images with invalid landmarks
      - detection success percentage

    Strictly does NOT generate replacement images for failures.
    """
    img_dir = Path(dataset_root) / images_dir
    image_paths = sorted(list(img_dir.glob("*.jpg"))) + sorted(list(img_dir.glob("*.png")))

    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]

    total_evaluated = len(image_paths)
    if total_evaluated == 0:
        return {
            "total_images": 0,
            "successfully_detected": 0,
            "no_face_detected": 0,
            "multiple_faces_detected": 0,
            "invalid_landmarks": 0,
            "success_rate_pct": 0.0,
            "error": "No image files found to validate."
        }

    try:
        preprocessor = RealFacePreprocessor(
            image_size=128,
            model_asset_path=model_asset_path
        )
    except Exception as e:
        return {
            "total_images": total_evaluated,
            "successfully_detected": 0,
            "no_face_detected": 0,
            "multiple_faces_detected": 0,
            "invalid_landmarks": 0,
            "success_rate_pct": 0.0,
            "error": str(e)
        }

    successfully_detected = 0
    no_face_detected = 0
    multiple_faces_detected = 0
    invalid_landmarks = 0

    print(f"Validating genuine landmark detection on {total_evaluated} real images...")

    for path in image_paths:
        try:
            tensor, mask, det = preprocessor.preprocess_image(str(path))
            successfully_detected += 1
            if det.num_faces_detected > 1:
                multiple_faces_detected += 1
        except NoFaceDetectedError:
            no_face_detected += 1
        except InvalidLandmarksError:
            invalid_landmarks += 1
        except Exception as e:
            # Other errors (e.g. corrupt image)
            invalid_landmarks += 1

    success_rate = (successfully_detected / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    return {
        "total_images": total_evaluated,
        "successfully_detected": successfully_detected,
        "no_face_detected": no_face_detected,
        "multiple_faces_detected": multiple_faces_detected,
        "invalid_landmarks": invalid_landmarks,
        "success_rate_pct": round(success_rate, 2),
        "error": None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify and inspect local CelebA dataset for neural face-swap training."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="ml/data/celeba",
        help="Path to the directory containing CelebA files (default: ml/data/celeba)"
    )
    parser.add_argument(
        "--identity-file",
        type=str,
        default="identity_CelebA.txt",
        help="Filename of identity annotations (default: identity_CelebA.txt)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="img_align_celeba",
        help="Subdirectory containing aligned face images (default: img_align_celeba)"
    )
    parser.add_argument(
        "--quick-sample",
        type=int,
        default=None,
        help="Optional: test only the first N lines for rapid verification"
    )
    parser.add_argument(
        "--validate-preprocessing",
        action="store_true",
        help="Run genuine landmark detection & mask validation on real CelebA images"
    )
    parser.add_argument(
        "--max-validation-images",
        type=int,
        default=100,
        help="Maximum real images to test during preprocessing validation (default: 100)"
    )
    parser.add_argument(
        "--model-asset-path",
        type=str,
        default="client/public/models/face_landmarker.task",
        help="Path to face_landmarker.task model asset (default: client/public/models/face_landmarker.task)"
    )

    args = parser.parse_args()

    print("====================================================================")
    print("      CelebA Dataset Validation & Integrity Check                   ")
    print("====================================================================")
    print(f"Target Root Directory: {args.dataset_root}")
    print(f"Identity Annotation:   {args.identity_file}")
    print(f"Images Subdirectory:   {args.images_dir}")
    print("--------------------------------------------------------------------")

    report = validate_celeba_dataset(
        dataset_root=args.dataset_root,
        identity_file=args.identity_file,
        images_dir=args.images_dir,
        quick_sample_limit=args.quick_sample
    )

    print(f"Total Image References:        {report['total_references']}")
    print(f"Existing Images Found:         {report['existing_images']}")
    print(f"Missing Images:                {report['missing_images']}")
    print(f"Unique Identities:             {report['total_identities']}")
    print(f"Identities (Multiple Photos):  {report['identities_with_multiple_images']} (eligible for same-ID pairs)")
    print(f"Identities (Single Photo):     {report['identities_with_single_image']}")
    print(f"Malformed Annotation Lines:    {report['malformed_lines']}")
    print("--------------------------------------------------------------------")

    if args.validate_preprocessing:
        print("\n====================================================================")
        print("      Real Facial Landmark & Preprocessing Validation               ")
        print("====================================================================")
        prep_report = run_preprocessing_validation(
            dataset_root=args.dataset_root,
            images_dir=args.images_dir,
            model_asset_path=args.model_asset_path,
            max_images=args.max_validation_images
        )
        if prep_report["error"]:
            print(f"[PREPROCESSING CHECK FAILED]: {prep_report['error']}")
        else:
            print(f"Total Annotation Images Evaluated: {prep_report['total_images']}")
            print(f"Images Successfully Detected:      {prep_report['successfully_detected']}")
            print(f"Images With No Detected Face:      {prep_report['no_face_detected']}")
            print(f"Images With Multiple Faces:        {prep_report['multiple_faces_detected']}")
            print(f"Images With Invalid Landmarks:     {prep_report['invalid_landmarks']}")
            print(f"Detection Success Percentage:      {prep_report['success_rate_pct']}%")
        print("--------------------------------------------------------------------")

    if report["is_valid"]:
        print("[PASS] Dataset Validation PASSED! All prerequisites are met for face-swap training.")
        sys.exit(0)
    else:
        print("[FAIL] Dataset Validation FAILED:")
        for err in report["errors"]:
            print(f"  * {err}")
        print("\nSetup Instructions:")
        print(f"  1. Download CelebA aligned dataset and identity annotations.")
        print(f"  2. Place 'identity_CelebA.txt' in: {args.dataset_root}/identity_CelebA.txt")
        print(f"  3. Place aligned images in:        {args.dataset_root}/{args.images_dir}/*.jpg")
        sys.exit(1)


if __name__ == "__main__":
    main()
