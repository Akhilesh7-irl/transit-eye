import cv2
from ultralytics import YOLO

model = YOLO('models/traffic_density_model.pt')
VEHICLE_CLASSES = [0, 1, 2, 3, 4]

def classify_density(count):
    if count <= 8:
        return "LOW", (0, 255, 0)
    elif count <= 18:
        return "MEDIUM", (0, 255, 255)
    else:
        return "HIGH", (0, 0, 255)

VIDEO_PATH = "traffic_video_30s.mp4"
OUTPUT_PATH = "reports/density_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Could not open video file: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info - FPS: {fps}, Size: {width}x{height}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Video opened: {VIDEO_PATH}")
print(f"Will save output to: {OUTPUT_PATH}")
print("Press q to quit early")

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended")
        break

    results = model(frame, classes=VEHICLE_CLASSES, conf=0.4, verbose=False)
    vehicle_count = len(results[0].boxes)
    density_label, color = classify_density(vehicle_count)

    annotated_frame = results[0].plot()
    text = f"Vehicles: {vehicle_count}  Density: {density_label}"
    cv2.putText(annotated_frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    out.write(annotated_frame)
    frame_num += 1

    cv2.imshow("Traffic Density Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done - processed {frame_num} frames")
print(f"Saved output video to {OUTPUT_PATH}")