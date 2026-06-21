# YOLO26/YOLO11-ACmix-CBAM 水面漂浮物检测

本项目用于水面漂浮物目标检测，基于 Ultralytics YOLO26/YOLO11 二次开发，加入 ACmix、CBAM 与 P2 小目标检测层等改进结构，重点提升水面漂浮垃圾、泡沫、塑料瓶、木块等小目标在复杂水面背景中的检测效果。

本仓库建议作为最终 GitHub 项目发布，`yolov11_ACMIX` 仓库中的改进内容已经迁移到这里。

## 项目特点

- 同时支持 YOLO26 和 YOLO11 模型配置。
- YOLO26 保留 `end2end: True` 和 `reg_max: 1`，不破坏原生端到端检测头。
- 支持 ACmix、CBAM、P2 及其组合结构。
- 提供水面漂浮物数据集 YAML 模板。
- 提供训练、验证、推理脚本和 PyQt 图形化检测页面。
- 提供项目说明、操作说明、模型变体说明和面试问答文档。

## 推荐模型

轻量增强版本：

```text
ultralytics/cfg/models/26/yolo26_ACmix_CBAM.yaml
```

高召回小目标版本：

```text
ultralytics/cfg/models/26/yolo26_ACmix_CBAM_P2.yaml
```

YOLO11 对比版本：

```text
ultralytics/cfg/models/11/yolo11_ACmix_CBAM.yaml
ultralytics/cfg/models/11/yolo11_ACmix_CBAM_P2.yaml
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

训练：

```bash
python yolo26_train.py
```

验证：

```bash
python yolo26_val.py
```

推理：

```bash
python yolo26_predict.py
```

PyQt 页面：

```bash
python pyqt_detect.py
```

也可以直接使用 Ultralytics 命令：

```bash
yolo detect train model=ultralytics/cfg/models/26/yolo26_ACmix_CBAM.yaml data=data/water_floating_objects.yaml epochs=200 imgsz=800 batch=8 device=0
```

## 数据集格式

推荐目录结构：

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

标签格式为 YOLO 检测格式：

```text
class_id x_center y_center width height
```

修改数据集配置：

```text
data/water_floating_objects.yaml
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `yolo26_train.py` | 默认 YOLO26 训练脚本 |
| `yolo26_val.py` | 验证训练好的模型 |
| `yolo26_predict.py` | 图片、视频、文件夹推理 |
| `pyqt_detect.py` | PyQt 图形化检测页面 |
| `data/water_floating_objects.yaml` | 水面漂浮物数据集模板 |
| `ultralytics/nn/modules/ACmix.py` | ACmix 模块实现 |
| `ultralytics/nn/modules/conv.py` | CBAM、卷积和基础注意力模块 |
| `ultralytics/nn/tasks.py` | 模型 YAML 解析与自定义模块注册 |


## PyQt5 可视化界面结构

```text
yolo_water_detect/
  main.py                # GUI启动逻辑
  ui/main_ui.py          # 三分栏主界面、控件和事件绑定
  detect/yolo_infer.py   # YOLO推理封装
  detect/crop_grid.py    # 4×6分块裁剪推理
  detect/tracker.py      # 跟踪开关封装
  utils/export_tool.py   # 图片/裁剪/JSON/CSV/视频导出
  utils/qss_style.py     # 现代化浅蓝工业风QSS样式
```
## 文档

- 项目技术与算法说明：[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- 环境配置和操作说明：[docs/OPERATION_GUIDE.md](docs/OPERATION_GUIDE.md)
- PyQt 页面使用说明：[docs/PYQT_GUIDE.md](docs/PYQT_GUIDE.md)
- 目录与代码文件说明：[docs/FOLDER_AND_FILE_GUIDE.md](docs/FOLDER_AND_FILE_GUIDE.md)
- 简历和面试问答参考：[docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md)
- YOLO26/YOLO11 模型变体说明：[docs/MODEL_VARIANTS.md](docs/MODEL_VARIANTS.md)

## 许可证

本项目基于 Ultralytics 源码二次开发，原项目采用 AGPL-3.0 License。发布到 GitHub 时请保留 `LICENSE`、来源说明和相关版权信息。

