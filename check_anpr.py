"""
Temporary check script for TransitEye ANPR pipeline.

Usage:
    python check_anpr.py sample.jpg
"""

import json
import sys
from pathlib import Path

import cv2

from models.anpr.plate_detect import PlateDetector
from models.anpr.ocr import PlateOCR


def safe_crop(image, bbox):
    """
    Safely crop an image using detector bbox coordinates.

    bbox format:
        [x1, y1, x2, y2]
    """
    if image is None:
        return None

    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox

    # Sort coordinates in case they are reversed.
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))

    # Clamp to image boundaries.
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    # Reject empty crops.
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def main():
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else "sample.jpg")

    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    print(f"Loaded image: {image_path}")
    print(f"Image shape: {image.shape}")

    # --------------------------------------------------
    # 1. Plate detection
    # --------------------------------------------------
    detector = PlateDetector(conf_threshold=0.20)

    detections = detector.detect(image)

    print("\nDetector output:")
    print(json.dumps(detections, indent=2))

    if not detections:
        raise SystemExit("No license plates detected.")

    # Use the best detection.
    best_detection = detections[0]

    # --------------------------------------------------
    # 2. Crop detected plate
    # --------------------------------------------------
    plate_crop = safe_crop(image, best_detection["bbox"])

    if plate_crop is None:
        raise SystemExit("Plate crop is empty after bounding box clamping.")

    crop_path = Path("plate_crop_test.jpg")
    cv2.imwrite(str(crop_path), plate_crop)

    print(f"\nSaved plate crop to: {crop_path}")

    # --------------------------------------------------
    # 3. OCR
    # --------------------------------------------------
    ocr = PlateOCR(
        gpu=False,
        min_confidence=0.25,
        preprocess=True,
    )

    ocr_result = ocr.read(plate_crop)

    print("\nOCR result:")
    print(json.dumps(ocr_result, indent=2))

    # --------------------------------------------------
    # 4. Simple pipeline-style usage
    # --------------------------------------------------
    plate_text = ocr_result["plate_text"]
    confidence = ocr_result["confidence"]
    accepted = ocr_result["accepted"]

    print("\nSimple pipeline values:")
    print(f"plate_text: {plate_text}")
    print(f"confidence: {confidence}")
    print(f"accepted:   {accepted}")


if __name__ == "__main__":
    main()