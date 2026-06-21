import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")

    model.val(
        data="data/water_floating_objects.yaml",
        split="val",
        imgsz=800,
        batch=8,
        iou=0.7,
        conf=0.001,
        workers=8,
    )
