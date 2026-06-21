# YOLO26/YOLO11 模型变体说明

本项目同时支持 YOLO26 和 YOLO11。推荐优先使用 YOLO26 变体；YOLO11 变体保留用于消融实验和与旧版本对比。

## YOLO26 变体

所有 YOLO26 配置位于：

```text
ultralytics/cfg/models/26/
```

| YAML 文件 | 改动 | 检测层 | 说明 |
| --- | --- | --- | --- |
| `yolo26.yaml` | 原始 YOLO26 | P3/P4/P5 | YOLO26 基线，端到端 NMS-free |
| `yolo26-p2.yaml` | 官方 P2 结构 | P2/P3/P4/P5 | 小目标召回更强，计算量更高 |
| `yolo26_CBAM.yaml` | P3 加 CBAM | P3/P4/P5 | 轻量注意力增强 |
| `yolo26_ACmix.yaml` | P3 使用 ACmix | P3/P4/P5 | 小目标特征增强 |
| `yolo26_ACmix_CBAM.yaml` | P3 使用 ACmix + CBAM | P3/P4/P5 | 推荐轻量增强版本 |
| `yolo26_CBAM_P2.yaml` | P2/P3 加 CBAM | P2/P3/P4/P5 | 高分辨率分支注意力增强 |
| `yolo26_ACmix_P2.yaml` | P3 使用 ACmix + P2 | P2/P3/P4/P5 | 小目标召回优先 |
| `yolo26_ACmix_CBAM_P2.yaml` | ACmix + CBAM + P2 | P2/P3/P4/P5 | 高召回版本，计算量最高 |

## YOLO11 变体

所有 YOLO11 配置位于：

```text
ultralytics/cfg/models/11/
```

| YAML 文件 | 改动 | 检测层 |
| --- | --- | --- |
| `yolo11.yaml` | 原始 YOLO11 | P3/P4/P5 |
| `yolo11_ACmix.yaml` | P3 使用 ACmix | P3/P4/P5 |
| `yolo11_CBAM.yaml` | P3 加 CBAM | P3/P4/P5 |
| `yolo11_P2.yaml` | 新增 P2 | P2/P3/P4/P5 |
| `yolo11_ACmix_CBAM.yaml` | ACmix + CBAM | P3/P4/P5 |
| `yolo11_ACmix_P2.yaml` | ACmix + P2 | P2/P3/P4/P5 |
| `yolo11_CBAM_P2.yaml` | CBAM + P2 | P2/P3/P4/P5 |
| `yolo11_ACmix_CBAM_P2.yaml` | ACmix + CBAM + P2 | P2/P3/P4/P5 |

## 推荐实验顺序

```text
yolo26.yaml
yolo26_CBAM.yaml
yolo26_ACmix.yaml
yolo26_ACmix_CBAM.yaml
yolo26-p2.yaml
yolo26_CBAM_P2.yaml
yolo26_ACmix_P2.yaml
yolo26_ACmix_CBAM_P2.yaml
```

如果需要和旧版本对比，再跑对应 YOLO11 变体。

## 推荐训练命令

轻量增强版本：

```bash
yolo detect train model=ultralytics/cfg/models/26/yolo26_ACmix_CBAM.yaml data=data/water_floating_objects.yaml epochs=200 imgsz=800 batch=8 device=0
```

高召回版本：

```bash
yolo detect train model=ultralytics/cfg/models/26/yolo26_ACmix_CBAM_P2.yaml data=data/water_floating_objects.yaml epochs=200 imgsz=800 batch=8 device=0
```

显存不足时优先降低：

```text
batch: 8 -> 4 -> 2
imgsz: 960 -> 800 -> 640
```
