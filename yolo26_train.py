import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # YOLO26 is the recommended default for this GitHub project.
    # To run YOLO11 variants, change MODEL_CFG to a file under ultralytics/cfg/models/11/.
    MODEL_CFG = "ultralytics/cfg/models/26/yolo26.yaml"
    DATA_CFG = r"D:\codex_project\yolo26_ACMIX\data\VisDrone-MOT.yaml"

    model = YOLO(MODEL_CFG)

    # Optional: initialize from official weights when available.
    model.load("yolo26n.pt")

    model.train(
        data=DATA_CFG,
        epochs=2,
        batch=8,
        imgsz=800,
        workers=8,
        device=0,
        optimizer="SGD",
        amp=True,
        cache=False,
    )
