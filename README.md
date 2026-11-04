深度学习目标检测YOLOV8模型如何处理应用遥感卫星图像松材线虫目标检测数据集 林业病虫害监测、智慧林业、生态环保等场景
@[toc]

遥感卫星图像松材线虫目标检测数据集
共2500组图像（512*512尺寸），10GB数据量，卫星分辨率0.75m，图像提供16bit的tif格式图像（4波段）以及8bit的jpg格式（rgb三波段）
采用yolo格式标注
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bfd6062ba5fa4f05903b4f7dd854673e.png)


以下是针对该数据集的完整 **YOLOv8 训练与推理方案**，包含：

- 数据集结构说明  
- 多光谱（4波段）与 RGB（3波段）输入处理  
- `data.yaml` 配置文件  
- **YOLOv8 详细训练代码（支持 16bit TIF + 8bit JPG）**  
- 自动标注可视化、模型导出等工具  

---

## 🌲 松材线虫病害松树检测数据集概览

| 项目 | 说明 |
| :--- | :--- |
| **任务类型** | 单类别目标检测（病害松树） |
| **图像总数** | 2,500 张（512×512） |
| **数据量** | ~10 GB |
| **图像格式** |  
| &nbsp;&nbsp;• 多光谱 | 16bit `.tif`（4波段：R, G, B, NIR）  
| &nbsp;&nbsp;• 可见光 | 8bit `.jpg`（RGB 三波段）  
| **标注格式** | YOLO `.txt`（已标注） |
| **空间分辨率** | 0.75 米/像素 |
| **应用场景** | 林业巡检、病虫害预警、无人机/卫星遥感AI分析 |

---

## 📂 推荐目录结构

```bash
pine_disease_dataset/
│
├── images/
│   ├── train/
│   │   ├── rgb/          # 8bit JPG
│   │   └── ms/           # 16bit TIF（4波段）
│   ├── val/
│   │   ├── rgb/
│   │   └── ms/
│   └── test/
│       ├── rgb/
│       └── ms/
│
├── labels/
│   ├── train/
│   └── val/
│
├── data.yaml
└── README.md
```

> ✅ 标注文件 `.txt` 对应图像文件名（不含扩展名）

---

## 📄 `data.yaml` 配置文件

```yaml
# data.yaml
train: ./pine_disease_dataset/images/train/ms
val: ./pine_disease_dataset/images/val/ms

nc: 1
names: ['infected_pine']
```

> 💡 若使用 RGB 模式，将 `train` 和 `val` 路径改为 `.../train/rgb`

---

## 🛠 一、多光谱图像处理（4波段 TIF → 3通道输入）

由于 YOLO 默认输入为 3 通道，需对 4 波段（R, G, B, NIR）进行融合。

### ✅ 方法1：合成 NDVI + RGB（推荐）

```python
# preprocess_ms.py
import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
import os

def load_4band_tif(tif_path):
    """加载 4 波段 TIF 图像"""
    img = tiff.imread(tif_path)  # shape: (4, H, W)
    r = img[0].astype(np.float32)
    g = img[1].astype(np.float32)
    b = img[2].astype(np.float32)
    nir = img[3].astype(np.float32)
    return r, g, b, nir

def compute_ndvi(nir, red):
    """计算 NDVI 指数"""
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
    return ndvi.astype(np.uint8)

def create_false_color(r, g, b, nir):
    """合成假彩色图像（NIR → Red）"""
    false_color = np.stack([nir, r, g], axis=-1)
    false_color = cv2.normalize(false_color, None, 0, 255, cv2.NORM_MINMAX)
    return false_color.astype(np.uint8)

# 示例：转换一批图像
input_dir = "pine_disease_dataset/images/train/ms"
output_dir = "pine_disease_dataset/images/train/ms_rgb"

os.makedirs(output_dir, exist_ok=True)

for tif_file in os.listdir(input_dir):
    if tif_file.endswith(".tif"):
        r, g, b, nir = load_4band_tif(os.path.join(input_dir, tif_file))
        # 使用假彩色（NIR 作为红通道）
        fused = create_false_color(r, g, b, nir)
        fused = cv2.cvtColor(fused, cv2.COLOR_RGB2BGR)  # OpenCV 格式
        cv2.imwrite(os.path.join(output_dir, tif_file.replace(".tif", ".jpg")), fused)

print("✅ 多光谱图像转换完成！")
```

---

## 🚀 二、YOLOv8 训练代码（支持多光谱/RGB）

```python
# train_pine.py
from ultralytics import YOLO
import torch
import os

# ==================== 配置参数 ====================
MODEL_TYPE = 'yolov8m.pt'           # 推荐：yolov8m（精度高）
DATA_CONFIG = 'pine_disease_dataset/data.yaml'
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 512                      # 与图像尺寸一致
PROJECT_NAME = 'pine_disease_detection'
EXPERIMENT_NAME = 'yolov8m_512_pine_ms'

os.makedirs(f"runs/train/{EXPERIMENT_NAME}", exist_ok=True)

# ==================== 设备检查 ====================
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")

# ==================== 加载模型 ====================
model = YOLO(MODEL_TYPE)

# ==================== 开始训练 ====================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,
    save_period=10,
    device=DEVICE,
    workers=4,
    optimizer='AdamW',
    lr0=1e-3,
    lrf=1e-2,
    momentum=0.937,
    weight_decay=1e-4,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    close_mosaic=10,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.3,
    scale=0.7,
    shear=2.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    patience=20,
)
```

---

## 🔍 三、验证与评估

```python
# validate.py
from ultralytics import YOLO

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')
metrics = model.val(
    data=DATA_CONFIG,
    split='val',
    imgsz=IMG_SIZE,
    conf=0.001,
    iou=0.5,
    device=DEVICE,
    save_json=True,
)

print(f"📊 mAP@0.5: {metrics.box.map:.4f}")
print(f"📊 mAP@0.5:0.95: {metrics.box.map50:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")
```

---

## 🔦 四、推理代码（支持 TIF/JPG）

```python
# predict.py
from ultralytics import YOLO
import cv2
import numpy as np
import tifffile as tiff

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')

# 支持 JPG 和 TIF
source = 'pine_disease_dataset/images/test/ms'

def load_image(path):
    if path.endswith('.tif'):
        r, g, b, _ = load_4band_tif(path)
        img = np.stack([r, g, b], axis=-1)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(path)

# 自定义推理
results = model.predict(
    source=source,
    imgsz=IMG_SIZE,
    conf=0.3,
    iou=0.5,
    device=DEVICE,
    show=False,
    save=True,
    project='runs/predict/pine_result',
    name='result',
    line_width=2,
)

for r in results:
    boxes = r.boxes
    print(f"🖼️ 图像: {r.path}, 检测到 {len(boxes)} 棵病害松树")
```

---

## 📦 五、模型导出（ONNX / TensorRT）

```python
# export_model.py
from ultralytics import YOLO

model = YOLO('runs/train/yolov8m_512_pine_ms/weights/best.pt')

# 导出为 ONNX
model.export(format='onnx', opset=12, input_shape=[1, 3, 512, 512])
```

---


