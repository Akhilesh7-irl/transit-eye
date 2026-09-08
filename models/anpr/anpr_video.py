import cv2

from plate_detect import PlateDetector
from ocr import PlateOCR


VIDEO_PATH = "data/sample_clips/traffic.mp4"
OUTPUT_PATH = "data/sample_clips/anpr_output.mp4"

MAX_SECONDS = 30
OCR_EVERY_N_FRAMES = 5


def main():
    detector = PlateDetector(
        model_path="models/anpr/best.pt",
        conf_threshold=0.30
    )

    ocr = PlateOCR(
        device="cpu"
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = int(fps * MAX_SECONDS)

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_number = 0

    while frame_number < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Run detection on every frame
        detections = detector.detect(frame)

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # OCR only every few frames
            if frame_number % OCR_EVERY_N_FRAMES == 0:

                result = ocr.read_plate(plate_crop)

                plate_text = result["plate_text"]
                ocr_conf = result["confidence"]

                print(
                    f"Frame {frame_number}: "
                    f"{plate_text} "
                    f"({ocr_conf:.2f})"
                )

                label = f"{plate_text} ({ocr_conf:.2f})"

            else:
                label = "Plate detected"

            # Draw plate box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        writer.write(frame)

        # Show preview
        cv2.imshow("TransitEye ANPR", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print("Done.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()