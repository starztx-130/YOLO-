"""
YOLO检测器核心模块

提供基于YOLOv8的目标检测功能，支持图片和视频检测。

主要功能：
- 模型加载：支持PyTorch (.pt) 和 ONNX (.onnx) 格式
- 参数配置：置信度、IoU阈值等参数设置
- 图片检测：单张图片的目标检测
- 视频检测：视频文件的逐帧检测
- 设备支持：自动检测并支持CPU/GPU加速

使用示例：
    detector = YOLODetector()
    detector.load_model('yolov8n.pt')
    results = detector.detect_image('image.jpg')
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Generator
from ultralytics import YOLO
import torch
import onnxruntime as ort
from .utils import get_device, validate_model_file


class YOLODetector:
    """YOLO检测器类"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.device = get_device()
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_det = 1000
        self.classes = None  # 指定检测的类别
        self.model_type = None  # 'pt' 或 'onnx'

    def load_model(self, model_path: str, device: str = None) -> bool:
        """
        加载YOLO模型

        Args:
            model_path: 模型文件路径
            device: 设备类型 ('cpu' 或 'cuda')

        Returns:
            bool: 是否加载成功
        """
        try:
            if not validate_model_file(model_path):
                raise ValueError(f"无效的模型文件: {model_path}")

            self.model_path = model_path
            self.device = device or self.device

            # 判断模型类型
            file_ext = Path(model_path).suffix.lower()
            self.model_type = file_ext[1:]  # 去掉点号

            if self.model_type == 'pt':
                # 加载PyTorch模型
                self.model = YOLO(model_path)
                # 设置设备
                if hasattr(self.model.model, 'to'):
                    self.model.model.to(self.device)
            elif self.model_type == 'onnx':
                # 加载ONNX模型
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(model_path, providers=providers)
            else:
                raise ValueError(f"不支持的模型格式: {file_ext}")

            return True

        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None
            return False

    def set_parameters(self, conf: float = None, iou: float = None,
                      max_det: int = None, classes: List[int] = None):
        """
        设置检测参数

        Args:
            conf: 置信度阈值
            iou: IoU阈值
            max_det: 最大检测数量
            classes: 指定检测的类别ID列表
        """
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou
        if max_det is not None:
            self.max_det = max_det
        if classes is not None:
            self.classes = classes

    def detect_image(self, image: Union[str, np.ndarray]) -> Optional[object]:
        """
        检测图片

        Args:
            image: 图片路径或numpy数组

        Returns:
            检测结果
        """
        if self.model is None:
            raise ValueError("模型未加载")

        try:
            if self.model_type == 'pt':
                # 使用ultralytics进行检测
                results = self.model(
                    image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    classes=self.classes,
                    device=self.device
                )
                return results
            else:
                # ONNX模型检测逻辑
                return self._detect_with_onnx(image)

        except Exception as e:
            print(f"检测失败: {str(e)}")
            return None

    def detect_video(self, video_path: str, output_path: str = None) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        检测视频

        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径

        Yields:
            Tuple[np.ndarray, int]: (检测后的帧, 帧数)
        """
        if self.model is None:
            raise ValueError("模型未加载")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 设置输出视频
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 检测当前帧
                results = self.detect_image(frame)

                if results and len(results) > 0:
                    # 绘制检测结果
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame

                # 保存帧
                if output_path:
                    out.write(annotated_frame)

                frame_count += 1

                # 可以在这里添加进度回调
                yield annotated_frame, frame_count

            cap.release()
            if output_path:
                out.release()

        except Exception as e:
            print(f"视频检测失败: {str(e)}")
            raise

    def _detect_with_onnx(self, image: Union[str, np.ndarray]):
        """
        使用ONNX模型进行检测

        Args:
            image: 输入图片

        Returns:
            检测结果
        """
        # 这里需要实现ONNX模型的检测逻辑
        # 由于ONNX模型的输入输出格式可能不同，这里提供基础框架
        if isinstance(image, str):
            image = cv2.imread(image)

        # 预处理图片
        input_tensor = self._preprocess_for_onnx(image)

        # 运行推理
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})

        # 后处理结果
        results = self._postprocess_onnx_output(outputs, image.shape)

        return results

    def _preprocess_for_onnx(self, image: np.ndarray) -> np.ndarray:
        """
        ONNX模型预处理
        """
        # 基础预处理，具体实现需要根据模型要求调整
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def _postprocess_onnx_output(self, outputs, original_shape):
        """
        ONNX模型后处理
        """
        # 基础后处理框架，具体实现需要根据模型输出格式调整
        # 这里返回一个简化的结果格式
        return outputs

    def get_model_info(self) -> dict:
        """
        获取模型信息

        Returns:
            dict: 模型信息
        """
        if self.model is None:
            return {}

        info = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'max_det': self.max_det
        }

        if self.model_type == 'pt' and hasattr(self.model, 'names'):
            info['classes'] = self.model.names

        return info

    def is_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            bool: 是否已加载
        """
        return self.model is not None
