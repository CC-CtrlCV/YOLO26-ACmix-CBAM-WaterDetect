# 操作说明文档

## 1. 环境配置

建议使用 Python 3.9 到 3.11，并优先使用虚拟环境。

### Conda

```bash
conda create -n water-yolo python=3.10 -y
conda activate water-yolo
pip install -r requirements.txt
```

### venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 GPU，请先安装与 CUDA 版本匹配的 PyTorch，再安装其余依赖。

## 2. 数据放在哪里

推荐把自己的数据放在项目外或项目内的 `datasets` 目录，例如：

```text
D:/datasets/water_floating_objects/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

图片和标签文件名需要一一对应：

```text
images/train/0001.jpg
labels/train/0001.txt
```

标签格式为 YOLO 检测格式：

```text
class_id x_center y_center width height
```

## 3. 修改数据集配置

打开：

```text
data/water_floating_objects.yaml
```

需要修改：

```yaml
path: D:/datasets/water_floating_objects
train: images/train
val: images/val
test: images/test

names:
  0: floating_object
```

如果你有多个类别，例如 bottle、foam、wood，需要改成：

```yaml
names:
  0: bottle
  1: foam
  2: wood
```

## 4. 修改模型类别数

打开：

```text
ultralytics/cfg/models/11/yolo11_ACmix_CBAM.yaml
```

修改：

```yaml
nc: 1
```

`nc` 必须等于数据集类别数量。训练时 Ultralytics 通常也会根据数据集 YAML 自动覆盖类别数，但建议保持一致，便于阅读和复现实验。

## 5. 训练模型

默认训练脚本：

```bash
python yolov11_train.py
```

常用参数在 `yolov11_train.py` 中修改：

| 参数 | 作用 |
| --- | --- |
| `data` | 数据集 YAML 路径 |
| `epochs` | 训练轮数 |
| `batch` | 批大小，显存不足时调小 |
| `imgsz` | 输入图像尺寸，水面小目标建议 800 或 960 |
| `device` | GPU 编号，CPU 使用 `"cpu"` |
| `optimizer` | 优化器，如 SGD、AdamW |
| `amp` | 自动混合精度，GPU 训练建议开启 |
| `cache` | 是否缓存数据，内存足够时可设为 True |

也可以使用命令行：

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_ACmix_CBAM.yaml data=data/water_floating_objects.yaml epochs=200 imgsz=800 batch=8 device=0
```

## 6. 恢复训练

如果训练中断，使用 last.pt 恢复：

```bash
yolo detect train model=runs/detect/train/weights/last.pt data=data/water_floating_objects.yaml resume=True
```

或者修改 `yolov11_huifu_train.py` 中的权重路径和数据路径。

## 7. 验证模型

```bash
python yolov11_val.py
```

需要修改 `yolov11_val.py`：

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

如果要测试 test 集，把：

```python
split="val"
```

改成：

```python
split="test"
```

## 8. 推理预测

```bash
python yolov11_predict.py
```

需要修改：

```python
model = YOLO("runs/detect/train/weights/best.pt")
source = "cat.jpg"
```

`source` 可以是：

- 单张图片：`test.jpg`
- 图片文件夹：`datasets/water_floating_objects/images/test`
- 视频：`test.mp4`
- 摄像头：`0`

命令行方式：

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=test.jpg imgsz=800 conf=0.25 save=True
```

预测结果默认保存到：

```text
runs/detect/predict/
```

## 9. 模型导出

导出 ONNX：

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=800
```

导出 OpenVINO：

```bash
yolo export model=runs/detect/train/weights/best.pt format=openvino imgsz=800
```

## 10. 常见问题

### 找不到数据集

检查 `data/water_floating_objects.yaml` 中的 `path` 是否真实存在。

### 显存不足

按顺序尝试：

```text
batch 8 -> 4 -> 2
imgsz 960 -> 800 -> 640
```

### 类别数量不对

确保：

- 数据集 YAML 的 `names` 数量正确。
- 模型 YAML 的 `nc` 正确。
- 标签里的 `class_id` 从 0 开始，不能超过类别数量。
