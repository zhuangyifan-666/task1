# 车辆距离检测与警告系统

本项目实现了一个基于YOLOv10目标检测和三角测量原理的车辆距离检测与警告系统。该系统能够在图像或视频流中检测车辆和行人，计算它们与摄像机的距离，并在物体过近时提供警告。

## 功能特点

- 使用YOLOv10进行目标检测
- 使用三角测量原理计算距离
- 基于距离阈值的视觉警告
- 摄像机校准工具，用于精确测量距离
- 批处理功能，用于分析多张图像

## 核心文件说明

本项目包含三个主要的Python文件，每个文件负责系统的不同功能：

### distance_detection.py

这是系统的核心文件，实现了距离检测和警告功能：

- **主要功能**：
  - 使用YOLOv10模型检测图像或视频流中的物体（车辆、行人等）
  - 计算检测到的物体与摄像机之间的距离
  - 根据距离阈值提供视觉警告
  - 支持实时视频处理和单张图像处理

- **关键组件**：
  - `calculate_distance()`: 使用三角相似原理计算物体距离
  - `process_frame()`: 处理单帧图像，进行目标检测和距离计算
  - `main()`: 主函数，处理视频流或单张图像

- **特点**：
  - 支持多种物体类型（汽车、行人、卡车等）
  - 提供不同级别的距离警告（安全、警告、危险）
  - 可通过关闭窗口、按ESC键或'q'键退出程序

### calibrate_camera.py

这是摄像机校准工具，用于提高距离测量的准确性：

- **主要功能**：
  - 允许用户在图像中选择已知尺寸和距离的物体
  - 计算摄像机的焦距参数
  - 保存校准数据供距离检测使用

- **关键组件**：
  - `calibrate_focal_length()`: 根据已知参数计算焦距
  - `select_object()`: 交互式界面，让用户在图像中选择物体
  - `main()`: 处理命令行参数并执行校准过程

- **特点**：
  - 支持命令行参数或交互式输入
  - 提供直观的图形界面进行物体选择
  - 生成校准数据文件供后续使用

### batch_process.py

这是批处理工具，用于处理多张图像：

- **主要功能**：
  - 批量处理指定目录中的图像
  - 应用相同的距离检测算法到所有图像
  - 将处理结果保存到输出目录

- **关键组件**：
  - `process_directory()`: 处理整个目录中的图像
  - `main()`: 处理命令行参数并执行批处理

- **特点**：
  - 支持处理大量图像
  - 可限制处理的图像数量
  - 提供处理时间统计
  - 自动创建输出目录

## 系统要求

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy

## 安装方法

1. 克隆此仓库：
```
git clone <仓库URL>
cd vehicle-distance-detection
```

2. 安装所需的包：
```
pip install ultralytics torch torchvision opencv-python numpy
```

3. 下载YOLOv10模型：
```
# 运行脚本时会自动下载模型
```

## 使用方法

### 摄像机校准

为了进行精确的距离测量，您应该首先校准摄像机：

```
python calibrate_camera.py --image <校准图像路径> --distance <已知距离> --width <已知宽度>
```

示例：
```
python calibrate_camera.py --image road_datas/images/training/1.jpg --distance 10 --width 1.8
```

如果您不提供参数，脚本将提示您输入它们并在图像中选择一个物体。

### 单张图像或网络摄像头处理

要处理单张图像或使用网络摄像头：

```
python distance_detection.py
```

该脚本将尝试使用您的网络摄像头。如果无法访问网络摄像头，它将处理数据集中的样本图像。

#### 处理单张图像的详细步骤

如果您想要处理特定的单张图像，可以按照以下步骤操作：

1. **修改distance_detection.py文件**：
   - 打开`distance_detection.py`文件
   - 找到`main()`函数中的以下代码段：
   ```python
   # 使用网络摄像头
   try:
       cap = cv2.VideoCapture(0)
       if not cap.isOpened():
           raise Exception("Could not open webcam")
   except Exception as e:
       print(f"Error opening webcam: {e}")
       print("Trying to use a sample image instead...")
       # 使用数据集中的样本图像
       sample_img_path = "road_datas/images/training/1.jpg"
   ```
   - 将`sample_img_path`变量修改为您想要处理的图像路径，例如：
   ```python
   sample_img_path = "您的图像路径.jpg"
   ```

2. **运行脚本**：
   ```
   python distance_detection.py
   ```

3. **查看结果**：
   - 处理后的图像将显示在一个窗口中
   - 检测到的物体会用边界框标记出来
   - 每个物体的距离信息会显示在边界框上方
   - 如果有物体距离过近，顶部会显示警告信息
   - 处理后的图像也会保存为`result.jpg`

4. **关闭程序**：
   - 按任意键关闭图像窗口
   - 或者点击窗口的关闭按钮(X)

#### 另一种方法：修改代码以接受命令行参数

您也可以修改`distance_detection.py`文件，使其接受命令行参数来指定图像路径：

1. **添加命令行参数支持**：
   - 在`distance_detection.py`文件顶部导入`argparse`模块
   - 在`main()`函数中添加参数解析代码
   - 使用解析的参数来确定图像路径

2. **使用命令行运行**：
   ```
   python distance_detection.py --image 您的图像路径.jpg
   ```

这样您就可以方便地处理任何单张图像，而无需每次都修改源代码。

### 批处理

要处理多张图像：

```
python batch_process.py --input <输入目录> --output <输出目录> --model <模型路径> --limit <最大图像数>
```

示例：
```
python batch_process.py --input road_datas/images/training --output output --limit 10
```

## 工作原理

### 距离计算

系统使用三角相似原理计算距离：

```
距离 = (实际宽度 * 焦距) / 感知宽度
```

其中：
- 实际宽度：物体的已知宽度（以米为单位）
- 焦距：摄像机的焦距（通过校准确定）
- 感知宽度：图像中物体的像素宽度

### 警告阈值

系统根据以下阈值提供警告：
- 警告：距离 < 10米
- 危险：距离 < 5米

这些阈值可以在`distance_detection.py`文件中调整。

## 自定义设置

### 调整警告阈值

您可以在`distance_detection.py`中调整警告阈值：

```python
WARNING_THRESHOLD = 10  # 距离小于此值时发出警告
DANGER_THRESHOLD = 5    # 距离小于此值时发出危险警告
```

### 添加物体类型

要添加更多用于距离计算的物体类型，请更新`distance_detection.py`中的`KNOWN_WIDTH`字典：

```python
KNOWN_WIDTH = {
    'car': 1.8,  # 平均汽车宽度（米）
    'person': 0.5,  # 平均人宽度（米）
    'truck': 2.5,  # 平均卡车宽度（米）
    # 在此添加更多物体
}
```

## 数据集

系统使用road_datas数据集，其中包含：
- 训练图像：road_datas/images/training/
- 验证图像：road_datas/images/validation/
- 训练标注：road_datas/annotations/training/
- 验证标注：road_datas/annotations/validation/

## 局限性

- 距离计算假设摄像机已正确校准
- 距离测量的准确性取决于目标检测的准确性
- 系统假设物体相对于摄像机在平面上
- 三角测量方法在物体直接位于摄像机前方时效果最佳

## 未来改进

- 实现立体视觉以获得更准确的距离测量
- 添加跟踪功能以稳定随时间变化的距离测量
- 与GPS数据集成以获得绝对定位
- 实现音频警告
- 基于相对运动添加更复杂的警告逻辑
