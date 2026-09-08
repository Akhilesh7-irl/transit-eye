import cv2

INPUT_PATH = "traffic_video.mp4"
OUTPUT_PATH = "traffic_video_30s.mp4"
DURATION_SECONDS = 30

cap = cv2.VideoCapture(INPUT_PATH)

if not cap.isOpened():
    print(f"❌ Could not open: {INPUT_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(fps * DURATION_SECONDS)

print(f"✅ Input opened — FPS: {fps}, Size: {width}x{height}")
print(f"Trimming to {DURATION_SECONDS} seconds ({max_frames} frames)...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0
while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Video ended before reaching target duration")
        break
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"✅ Done — saved {frame_count} frames to {OUTPUT_PATH}")