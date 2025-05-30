"""
工具函数模块
"""
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def check_cuda_availability() -> Tuple[bool, str]:
    """
    检查CUDA是否可用
    
    Returns:
        Tuple[bool, str]: (是否可用, 设备信息)
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        return True, f"CUDA可用 - {device_count}个GPU设备 ({device_name})"
    else:
        return False, "CUDA不可用，将使用CPU"


def get_device() -> str:
    """
    获取推荐的设备
    
    Returns:
        str: 设备名称 ('cuda' 或 'cpu')
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def validate_model_file(file_path: str) -> bool:
    """
    验证模型文件是否有效
    
    Args:
        file_path: 模型文件路径
        
    Returns:
        bool: 是否有效
    """
    if not os.path.exists(file_path):
        return False
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in ['.pt', '.onnx']


def validate_media_file(file_path: str) -> bool:
    """
    验证媒体文件是否有效
    
    Args:
        file_path: 媒体文件路径
        
    Returns:
        bool: 是否有效
    """
    if not os.path.exists(file_path):
        return False
    
    file_ext = Path(file_path).suffix.lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    return file_ext in image_exts + video_exts


def is_image_file(file_path: str) -> bool:
    """
    判断是否为图片文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为图片
    """
    file_ext = Path(file_path).suffix.lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    return file_ext in image_exts


def is_video_file(file_path: str) -> bool:
    """
    判断是否为视频文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为视频
    """
    file_ext = Path(file_path).suffix.lower()
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    return file_ext in video_exts


def resize_image_keep_ratio(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    保持宽高比缩放图片
    
    Args:
        image: 输入图片
        target_size: 目标尺寸 (width, height)
        
    Returns:
        np.ndarray: 缩放后的图片
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    
    # 计算新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图片
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建目标尺寸的画布
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 计算居中位置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # 将缩放后的图片放到画布中央
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def format_detection_info(results) -> str:
    """
    格式化检测结果信息
    
    Args:
        results: YOLO检测结果
        
    Returns:
        str: 格式化的信息
    """
    if not results or len(results) == 0:
        return "未检测到目标"
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return "未检测到目标"
    
    # 统计各类别数量
    class_counts = {}
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # 格式化输出
    info_lines = [f"检测到 {len(result.boxes)} 个目标:"]
    for class_name, count in class_counts.items():
        info_lines.append(f"  {class_name}: {count}个")
    
    return "\n".join(info_lines)


def create_output_dir(base_dir: str = "output") -> str:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录名
        
    Returns:
        str: 创建的目录路径
    """
    output_dir = Path(base_dir)
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)
