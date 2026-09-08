from ultralytics import YOLO

if __name__ == '__main__':
    # Load pre-trained YOLOv8 nano model
    model = YOLO('yolov8n.pt')

    # Train model on GPU
    results = model.train(
        data='data/auto_dataset/data.yaml',
        epochs=80,
        patience=20,
        imgsz=640,
        batch=16,       # Safe batch size for 6GB VRAM; change to 8 if VRAM fills up
        device=0,       # Targets your RTX 3050 GPU
        workers=2,      # Optimal data loader workers for Windows laptop
        project='runs/detect',
        name='indian_vehicles_v8'
    )