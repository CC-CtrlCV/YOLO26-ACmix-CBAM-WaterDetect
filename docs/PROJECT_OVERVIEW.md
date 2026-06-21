# 项目说明文档

## 项目目标

本项目面向水面漂浮物检测任务，例如水面垃圾、漂浮瓶罐、泡沫、漂浮木块、漂浮障碍物等。水面场景通常存在反光、波纹、阴影、尺度变化大、目标面积小、背景干扰强等问题，因此本项目在 YOLO11 的小目标检测分支上加入 ACmix 和 CBAM 注意力增强。

## 使用技术

- Python：项目主语言。
- PyTorch：深度学习训练和推理框架。
- Ultralytics YOLO：检测框架、训练器、验证器、推理器和模型导出工具。
- OpenCV：图像读取、结果可视化和 PyQt 页面显示。
- PyQt5：可选图形界面，用于选择模型和图片并查看检测结果。
- YAML：模型结构和数据集路径配置。

## 模型结构

推荐模型配置文件：

```text
ultralytics/cfg/models/11/yolo11_ACmix_CBAM.yaml
```

模型整体仍采用 YOLO11 的 Backbone、Neck 和 Detect Head。主要改动集中在 P3/8 小目标分支：

1. Backbone 提取 P3、P4、P5 多尺度特征。
2. Head 进行上采样和特征融合。
3. P3 小目标分支使用 ACmix 替换原始 C3k2。
4. ACmix 输出后串联 CBAM。
5. Detect 头输出 P3、P4、P5 三尺度预测。

## ACmix 算法

ACmix 是一种混合注意力与卷积的特征增强模块。该模块同时保留两条路径：

- 注意力路径：通过 Q、K、V 和位置编码建模局部区域上下文。
- 卷积路径：基于 Q、K、V 生成卷积特征，再经过深度卷积提取局部纹理。

两条路径最后通过可学习权重融合。对水面漂浮物检测而言，ACmix 有助于在波纹、反光和复杂背景中提取更稳定的小目标特征。

实现文件：

```text
ultralytics/nn/modules/ACmix.py
```

## CBAM 算法

CBAM 是 Convolutional Block Attention Module，由两个部分组成：

- Channel Attention：判断哪些通道更重要。
- Spatial Attention：判断图像空间位置中哪些区域更重要。

本项目把 CBAM 放在 P3 小目标分支 ACmix 后面，用于进一步突出疑似漂浮物区域，抑制水面背景噪声。

实现文件：

```text
ultralytics/nn/modules/conv.py
```

## 适用场景

- 河道、湖面、水库、海面漂浮垃圾检测。
- 无人机、岸边摄像头、船载摄像头采集图像。
- 小目标、远距离目标、强背景干扰目标检测。

## 输出结果

训练完成后，结果默认保存在：

```text
runs/detect/train/
```

常用输出包括：

- `weights/best.pt`：验证指标最优权重。
- `weights/last.pt`：最后一轮权重。
- `results.png`：训练曲线。
- `confusion_matrix.png`：混淆矩阵。
- `val_batch*_pred.jpg`：验证集预测可视化。
