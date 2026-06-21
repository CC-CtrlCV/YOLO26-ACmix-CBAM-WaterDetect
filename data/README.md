# 数据集配置说明

本文件夹存放数据集 YAML 配置

## 推荐文件

```text
water_floating_objects.yaml
```

这是水面漂浮物检测任务的模板。

## 推荐数据目录

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

## YAML 示例

```yaml
path: D:/datasets/water_floating_objects
train: images/train
val: images/val
test: images/test

names:
  0: floating_object
```

## 标签格式

每张图片对应一个同名 `.txt` 标签文件：

```text
class_id x_center y_center width height
```

所有坐标都是相对图片宽高的归一化值。

## 多类别示例

```yaml
names:
  0: bottle
  1: foam
  2: wood
  3: plastic_bag
```

如果类别数改变，请同步检查模型 YAML 中的 `nc`。
