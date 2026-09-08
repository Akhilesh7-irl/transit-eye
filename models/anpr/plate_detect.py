"""
models/anpr/plate_detect.py

License plate detection module using a pretrained YOLOv8 model.
Handles loading the model and extracting bounding boxes and confidence scores.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlateDetector:
    """
    A reusable license plate detector using YOLOv8.
    """

    def __init__(
        self, 
        model_path = "models/anpr/best.pt", 
        conf_threshold: float = 0.25,
        device: str | None = None
    ):
        """
        Initializes the plate detector and loads the YOLO model into memory.

        Args:
            model_path (str): Path to a local .pt file or a HuggingFace model ID.
            conf_threshold (float): Minimum confidence score to consider a detection valid.
            device (str | None): Device to run inference on (e.g., 'cpu', '0' for GPU). 
                                 Defaults to None (auto-select).
        """
        self.conf_threshold = conf_threshold
        self.device = device
        
        logger.info(f"Loading YOLO model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Detects license plates in a given image or vehicle crop.

        Args:
            image (np.ndarray): Input image in BGR format (OpenCV standard).

        Returns:
            list[dict]: A list of detections. Each dictionary contains:
                - 'bbox' (list[int]): [x1, y1, x2, y2] coordinates of the plate.
                - 'confidence' (float): Confidence score of the detection.
                - 'class_id' (int): Class index (usually 0 for plates).
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid image input. Expected a numpy ndarray.")
            return []

        if image.ndim != 3 or image.shape[2] != 3:
            logger.error("Invalid image shape. Expected a 3-channel BGR image.")
            return []

        try:
            # Run inference (verbose=False suppresses ultralytics console output)
            results = self.model(
                image, 
                conf=self.conf_threshold, 
                device=self.device, 
                verbose=False
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Extract coordinates and convert to standard Python ints
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                # Extract confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                detections.append({
                    "bbox": bbox,
                    "confidence": round(confidence, 4),
                    "class_id": class_id
                })

        # Sort by confidence descending (highest confidence first)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        return detections


if __name__ == "__main__":
    # Quick demo/test section
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test the Plate Detector on an image.")
    parser.add_argument("--image", type=str, default="test_image.jpg", help="Path to the test image.")
    parser.add_argument(
    "--model",
    type=str,
    default="models/anpr/best.pt",
    help="Path to the license plate YOLO model."
)
    args = parser.parse_args()

    # 1. Initialize Detector
    detector = PlateDetector(model_path=args.model, conf_threshold=0.3)

    # 2. Load Image
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error(f"Test image not found at {img_path}. Please provide a valid image path.")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        logger.error("Failed to read image. Check if the file is a valid image.")
        sys.exit(1)

    # 3. Run Detection
    plates = detector.detect(img)
    logger.info(f"Found {len(plates)} license plate(s).")

    # 4. Visualize Results
    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = plate["bbox"]
        conf = plate["confidence"]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Plate {i+1}: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Save output
    output_path = img_path.with_name(f"{img_path.stem}_detected{img_path.suffix}")
    cv2.imwrite(str(output_path), img)
    logger.info(f"Result saved to {output_path}")