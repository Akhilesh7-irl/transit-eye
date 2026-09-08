import cv2

from plate_detect import PlateDetector
from ocr import PlateOCR


IMAGE_PATH = "../../sample.png"
OUTPUT_PATH = "../../sample_anpr.jpg"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Could not read image: {IMAGE_PATH}")
        return

    detector = PlateDetector(
        model_path="best.pt",
        conf_threshold=0.25
    )

    ocr = PlateOCR(
        device="cpu"
    )

    detections = detector.detect(image)

    print(f"Found {len(detections)} plate(s)")

    for i, detection in enumerate(detections):

        x1, y1, x2, y2 = detection["bbox"]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        result = ocr.read_plate(plate_crop)

        plate_text = result["plate_text"]
        ocr_conf = result["confidence"]

        print(f"\nPlate {i + 1}")
        print(f"Detection confidence: {detection['confidence']}")
        print(f"Plate: {plate_text}")
        print(f"OCR confidence: {ocr_conf}")

        # Draw detection
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw OCR result
        label = f"{plate_text} ({ocr_conf:.2f})"

        cv2.putText(
            image,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imwrite(OUTPUT_PATH, image)

    print(f"\nSaved result to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()