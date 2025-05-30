"""
YOLO目标检测系统配置文件

包含系统的配置参数，包括检测参数、界面配置、文件格式支持等。

配置类别：
- 检测参数：置信度、IoU阈值等
- 文件格式：支持的图片、视频、模型文件格式
- 界面配置：窗口大小、面板宽度等
- 输出配置：默认输出目录和格式设置
- 摄像头配置：摄像头相关参数设置
"""

# 默认检测参数
DEFAULT_DETECTION_PARAMS = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'max_det': 1000,
    'device': 'auto',
    'show_labels': True,
    'show_conf': True,
    'show_boxes': True
}

# 支持的文件格式
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
SUPPORTED_MODEL_FORMATS = ['.pt', '.onnx']

# UI配置
UI_CONFIG = {
    'window_title': 'YOLO目标检测系统 v2.0.0',
    'window_size': (1400, 900),
    'left_panel_width': 350,
    'min_left_panel_width': 300,
    'max_left_panel_width': 400
}

# 输出配置
OUTPUT_CONFIG = {
    'default_output_dir': 'output',
    'default_image_format': 'jpg',
    'default_video_format': 'mp4'
}

# 摄像头配置
CAMERA_CONFIG = {
    'default_camera_id': 0,
    'fps_limit': 30
}
