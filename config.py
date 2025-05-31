# YOLO目标检测系统配置文件

# 默认检测参数
DEFAULT_DETECTION_PARAMS = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'max_det': 1000,
    'device': 'auto',
    'show_labels': True,
    'show_conf': True,
    'show_boxes': True,
    'show_masks': True,
    'show_keypoints': True
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
