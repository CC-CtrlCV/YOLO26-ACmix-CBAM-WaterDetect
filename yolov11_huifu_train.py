import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载中断时的检查点last.pt（路径根据实际训练文件夹修改）
    model = YOLO("runs/detect/train6/weights/last.pt")  # 假设第八次训练文件夹是exp8

    # 恢复训练：resume=True会自动读取last.pt中的训练状态
    results = model.train(
        data="data/data_SARD_1.yaml",
        epochs=200,  # 总轮次保持不变（模型会从已训练轮次继续累加）
        batch=8,
        imgsz=640,
        workers=8,
        device=0,
        optimizer="SGD",
        amp=True,
        cache=False,
        resume=True,  # 关键参数：启用恢复训练模式
    )
