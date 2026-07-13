from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    source = "test.jpg"

    model.predict(
        source=source,
        imgsz=800,
        conf=0.25,
        save=True,
    )
