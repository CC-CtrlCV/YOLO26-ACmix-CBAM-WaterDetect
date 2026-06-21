# 目录和代码文件说明

本文档用于帮助使用者快速理解项目中每个主要文件夹和代码文件的作用。由于本仓库保留了较完整的 Ultralytics 源码，`ultralytics`、`docs/en`、`examples` 中存在大量官方框架文件；普通使用者优先关注“项目核心文件”即可。

## 顶层目录

| 路径 | 说明 |
| --- | --- |
| `.github/` | GitHub Issue、PR 和 CI 工作流配置，发布开源项目时使用 |
| `.idea/` | JetBrains/PyCharm 项目配置，不影响运行 |
| `data/` | 数据集 YAML 配置文件，用户主要修改这里 |
| `docker/` | Ultralytics 官方 Docker 构建文件 |
| `docs/` | 项目说明、操作说明、API 文档和原 Ultralytics 文档 |
| `examples/` | Ultralytics 官方示例，包括 ONNX、OpenCV、TFLite 等推理示例 |
| `runs/` | 训练、验证、推理输出目录 |
| `tests/` | Ultralytics 官方测试文件 |
| `ultralytics/` | 核心框架源码，模型、训练器、数据加载、工具函数都在这里 |

## 顶层文件

| 文件 | 说明 |
| --- | --- |
| `README.md` | 本项目 GitHub 首页说明 |
| `README.zh-CN.md` | 原 Ultralytics 中文 README，可作为框架参考 |
| `requirements.txt` | 运行本项目的 Python 依赖 |
| `pyproject.toml` | Python 包构建和依赖元信息 |
| `LICENSE` | AGPL-3.0 许可证 |
| `CITATION.cff` | 引用信息 |
| `CONTRIBUTING.md` | 原 Ultralytics 贡献指南 |
| `yolo11n.pt` | YOLO11n 预训练权重 |
| `cat.jpg` | 示例图片 |
| `ceshi.py` | 简单模型加载测试脚本 |
| `yolov11_train.py` | 本项目推荐训练脚本 |
| `yolov11_val.py` | 验证脚本 |
| `yolov11_predict.py` | 推理脚本 |
| `yolov11_huifu_train.py` | 恢复训练脚本 |
| `jixu_train.py` | 另一个恢复训练示例脚本，保留用于参考 |
| `pyqt_detect.py` | PyQt 图形界面检测脚本 |

## data 文件夹

| 文件 | 说明 |
| --- | --- |
| `data/water_floating_objects.yaml` | 水面漂浮物数据集模板，推荐使用 |
| `data/VisDrone-MOT.yaml` | VisDrone-MOT 数据集示例配置，原实验参考 |

## docs 文件夹

| 文件或目录 | 说明 |
| --- | --- |
| `docs/PROJECT_OVERVIEW.md` | 项目技术和算法说明 |
| `docs/OPERATION_GUIDE.md` | 环境配置、训练、验证、推理操作说明 |
| `docs/PYQT_GUIDE.md` | PyQt 页面使用说明 |
| `docs/FOLDER_AND_FILE_GUIDE.md` | 当前文件，解释目录和文件用途 |
| `docs/en/` | Ultralytics 官方英文文档 |
| `docs/overrides/` | MkDocs 文档站样式、脚本和模板 |
| `docs/build_docs.py` | 构建文档站脚本 |
| `docs/build_reference.py` | 生成 API reference 文档脚本 |
| `docs/model_data.py` | 文档中的模型数据表辅助脚本 |

## ultralytics 核心目录

| 路径 | 说明 |
| --- | --- |
| `ultralytics/cfg/` | 默认配置、模型 YAML、数据集 YAML、跟踪器 YAML |
| `ultralytics/data/` | 数据集加载、增强、转换、标注工具 |
| `ultralytics/engine/` | 训练器、验证器、预测器、导出器等核心引擎 |
| `ultralytics/models/` | YOLO、SAM、RT-DETR、FastSAM 等模型任务封装 |
| `ultralytics/nn/` | 神经网络结构解析和模块实现 |
| `ultralytics/solutions/` | 计数、热力图、测速、区域检测等应用方案 |
| `ultralytics/trackers/` | ByteTrack、BoT-SORT 等多目标跟踪模块 |
| `ultralytics/utils/` | 日志、指标、损失、NMS、绘图、下载等工具函数 |
| `ultralytics/assets/` | 官方示例图片 |

## 关键模型配置文件

| 文件 | 说明 |
| --- | --- |
| `ultralytics/cfg/models/11/yolo11.yaml` | 原始 YOLO11 检测模型 |
| `ultralytics/cfg/models/11/yolo11_ACmix.yaml` | P3 小目标分支加入 ACmix 的模型 |
| `ultralytics/cfg/models/11/yolo11_CBAM.yaml` | P3 小目标分支加入 CBAM 的模型 |
| `ultralytics/cfg/models/11/yolo11_P2.yaml` | 新增 P2 小目标检测层的模型 |
| `ultralytics/cfg/models/11/yolo11_ACmix_CBAM.yaml` | 推荐模型，P3 分支加入 ACmix 和 CBAM |
| `ultralytics/cfg/models/11/yolo11_ACmix_P2.yaml` | ACmix 与 P2 检测层组合模型 |
| `ultralytics/cfg/models/11/yolo11_CBAM_P2.yaml` | CBAM 与 P2 检测层组合模型 |
| `ultralytics/cfg/models/11/yolo11_ACmix_CBAM_P2.yaml` | ACmix、CBAM 与 P2 检测层组合模型 |
| `ultralytics/cfg/models/11/MEM-YOLO11.yaml` | 其他实验模型配置，非本项目默认路线 |
| `ultralytics/cfg/models/v8/yolov8_ACmix_*.yaml` | YOLOv8 上的 ACmix 实验配置，保留参考 |

## 关键 Python 代码文件

| 文件 | 说明 |
| --- | --- |
| `ultralytics/nn/tasks.py` | 解析模型 YAML，构建 PyTorch 网络；已加入 ACmix 和 CBAM 支持 |
| `ultralytics/nn/modules/ACmix.py` | ACmix 模块实现 |
| `ultralytics/nn/modules/conv.py` | Conv、DWConv、CBAM、ChannelAttention、SpatialAttention 等模块 |
| `ultralytics/nn/modules/block.py` | C2f、C3k2、SPPF、C2PSA 等 YOLO 基础结构 |
| `ultralytics/nn/modules/head.py` | Detect、Segment、Pose、OBB 等检测头 |
| `ultralytics/engine/trainer.py` | 训练流程主逻辑 |
| `ultralytics/engine/validator.py` | 验证流程主逻辑 |
| `ultralytics/engine/predictor.py` | 推理流程主逻辑 |
| `ultralytics/utils/loss.py` | YOLO 检测、分割、姿态等损失函数 |
| `ultralytics/utils/metrics.py` | mAP、Precision、Recall 等指标 |
| `ultralytics/utils/ops.py` | NMS、坐标转换、box 操作等 |
| `ultralytics/data/augment.py` | Mosaic、MixUp、HSV、随机仿射等数据增强 |
| `ultralytics/data/dataset.py` | YOLO 数据集读取和标签处理 |

## 普通使用者建议阅读顺序

1. `README.md`
2. `docs/OPERATION_GUIDE.md`
3. `data/water_floating_objects.yaml`
4. `yolov11_train.py`
5. `yolov11_predict.py`
6. `docs/PYQT_GUIDE.md`

## 开发者建议阅读顺序

1. `ultralytics/cfg/models/11/yolo11_ACmix_CBAM.yaml`
2. `ultralytics/nn/modules/ACmix.py`
3. `ultralytics/nn/modules/conv.py`
4. `ultralytics/nn/tasks.py`
5. `ultralytics/engine/trainer.py`
