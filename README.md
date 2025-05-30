<<<<<<< HEAD
# YOLO-
基于pyside6的yolo可视化检测界面
=======
# YOLO目标检测系统

基于YOLOv8的目标检测系统，支持图片、视频和摄像头检测。

## 功能特性

- **图片检测**: 支持JPG、PNG、BMP等格式
- **视频检测**: 支持MP4、AVI、MOV等格式，带播放控制
- **摄像头检测**: 实时摄像头检测
- **结果可视化**: 检测框、标签、置信度显示
- **数据导出**: 检测结果导出为CSV格式
- **参数调节**: 置信度、IoU阈值等参数设置

## 环境要求

- Python 3.8+
- Windows/Linux/macOS

## 安装部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd yolo-detection-system
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动程序
```bash
# Windows
run.bat

# Linux/macOS
chmod +x run.sh
./run.sh

# 或直接运行
python main.py
```

## 使用说明

### 基本操作流程

1. **加载模型**: 点击"加载模型"按钮，选择YOLO模型文件（.pt或.onnx格式）
2. **调整参数**: 在左侧参数面板设置置信度阈值、IoU阈值等
3. **选择检测类型**:
   - 图片检测: 点击"检测图片"，选择图片文件
   - 视频检测: 点击"检测视频"，选择视频文件
   - 摄像头检测: 点击"摄像头检测"开始实时检测
4. **查看结果**: 在右侧区域查看检测结果和数据统计

### 界面说明

- **左侧面板**: 模型设置、参数调节、检测控制
- **右侧主区域**:
  - 检测结果标签页: 显示检测结果和播放控制
  - 数据表标签页: 详细的检测数据统计
  - 系统信息标签页: 系统状态和设备信息

## 项目结构

```
yolo-detection-system/
├── main.py                    # 主程序入口
├── config.py                  # 配置文件
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目文档
├── run.bat                    # Windows启动脚本
├── run.sh                     # Linux/macOS启动脚本
├── core/                      # 核心模块
│   ├── detector.py            # YOLO检测器
│   └── utils.py               # 工具函数
├── ui/                        # 用户界面
│   ├── main_window.py         # 主窗口
│   └── widgets/               # 界面组件
│       ├── image_viewer.py    # 图片查看器
│       ├── video_player.py    # 视频播放器
│       ├── detection_table.py # 检测数据表
│       ├── parameter_panel.py # 参数面板
│       └── smart_detection_viewer.py # 智能检测查看器
└── examples/                  # 示例代码
```

## 支持格式

### 模型文件
- PyTorch模型: .pt
- ONNX模型: .onnx

### 媒体文件
- 图片: .jpg, .jpeg, .png, .bmp, .tiff
- 视频: .mp4, .avi, .mov, .mkv

## 常见问题

**Q: 程序启动失败，提示模块缺失**
A: 重新安装依赖包 `pip install -r requirements.txt`

**Q: 模型加载失败**
A: 检查模型文件格式（.pt或.onnx），确保路径中不包含中文字符

**Q: 摄像头无法打开**
A: 检查摄像头是否被其他程序占用，尝试更换摄像头索引

**Q: 检测速度慢**
A: 使用较小的模型（如yolov8n.pt），启用GPU加速

## 许可证

MIT License
>>>>>>> e337e3d (Initial commit)
