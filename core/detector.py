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
        self.task_type = None  # 任务类型：'detect', 'segment', 'pose', 'classify', 'obb'

        # 显示选项
        self.show_labels = True
        self.show_conf = True
        self.show_boxes = True
        self.show_masks = True  # 显示分割掩码
        self.show_keypoints = True  # 显示关键点
        self.show_obb = True  # 显示定向边界框

        # 颜色管理
        self.class_colors = {}  # 类别颜色缓存
        self._init_class_colors()

    def _init_class_colors(self):
        """初始化类别颜色"""
        # 预定义一些美观的颜色（BGR格式）
        self.predefined_colors = [
            (255, 0, 0),     # 红色
            (0, 255, 0),     # 绿色
            (0, 0, 255),     # 蓝色
            (255, 255, 0),   # 青色
            (255, 0, 255),   # 洋红色
            (0, 255, 255),   # 黄色
            (128, 0, 128),   # 紫色
            (255, 165, 0),   # 橙色
            (0, 128, 128),   # 青绿色
            (128, 128, 0),   # 橄榄色
            (255, 192, 203), # 粉色
            (0, 191, 255),   # 深天蓝色
            (50, 205, 50),   # 酸橙绿
            (255, 20, 147),  # 深粉色
            (30, 144, 255),  # 道奇蓝
            (255, 69, 0),    # 红橙色
            (34, 139, 34),   # 森林绿
            (138, 43, 226),  # 蓝紫色
            (255, 140, 0),   # 深橙色
            (220, 20, 60),   # 深红色
        ]

    def _get_class_color(self, class_id: int) -> tuple:
        """获取类别对应的固定颜色"""
        if class_id not in self.class_colors:
            # 如果类别ID在预定义颜色范围内，使用预定义颜色
            if class_id < len(self.predefined_colors):
                self.class_colors[class_id] = self.predefined_colors[class_id]
            else:
                # 否则使用基于类别ID的确定性颜色生成
                import hashlib
                # 使用类别ID生成确定性的颜色
                hash_obj = hashlib.md5(str(class_id).encode())
                hash_hex = hash_obj.hexdigest()
                # 从哈希值提取RGB分量
                r = int(hash_hex[0:2], 16)
                g = int(hash_hex[2:4], 16)
                b = int(hash_hex[4:6], 16)
                self.class_colors[class_id] = (b, g, r)  # BGR格式

        return self.class_colors[class_id]

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

                # 识别任务类型
                self._detect_task_type()

            elif self.model_type == 'onnx':
                # 加载ONNX模型
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(model_path, providers=providers)
                # ONNX模型任务类型需要从文件名或其他方式推断
                self.task_type = 'detect'  # 默认为检测任务
            else:
                raise ValueError(f"不支持的模型格式: {file_ext}")

            return True

        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None
            return False

    def _detect_task_type(self):
        """通过模型结构检测任务类型"""
        try:
            if self.model_type == 'pt' and hasattr(self.model, 'model'):
                # 方法1：通过模型结构分析检测头类型
                task_type = self._analyze_model_structure()
                if task_type:
                    self.task_type = task_type
                    print(f"通过模型结构检测到任务类型: {self.task_type}")
                    return

                # 方法2：通过模型任务属性检测
                task_type = self._check_model_task_attribute()
                if task_type:
                    self.task_type = task_type
                    print(f"通过模型任务属性检测到任务类型: {self.task_type}")
                    return

                # 方法3：通过测试推理检测输出结构
                task_type = self._detect_by_inference()
                if task_type:
                    self.task_type = task_type
                    print(f"通过推理输出检测到任务类型: {self.task_type}")
                    return

                # 默认为检测任务
                self.task_type = 'detect'
                print(f"无法确定任务类型，默认为: {self.task_type}")

            elif self.model_type == 'onnx':
                # ONNX模型通过输出节点分析
                self.task_type = self._analyze_onnx_outputs()
                print(f"ONNX模型检测到任务类型: {self.task_type}")
            else:
                self.task_type = 'detect'

        except Exception as e:
            print(f"任务类型检测失败: {e}")
            self.task_type = 'detect'

    def _analyze_model_structure(self):
        """分析PyTorch模型结构确定任务类型"""
        try:
            model = self.model.model

            # 获取模型的最后一层（检测头）
            detect_layer = None

            # 方法1：通过model.model[-1]获取（YOLOv8标准结构）
            if hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
                try:
                    detect_layer = model.model[-1]
                except:
                    pass

            # 方法2：通过detect属性获取
            if detect_layer is None and hasattr(model, 'detect'):
                detect_layer = model.detect

            # 方法3：遍历模块查找检测头
            if detect_layer is None:
                try:
                    for name, module in model.named_modules():
                        module_name = module.__class__.__name__.lower()
                        if any(keyword in module_name for keyword in ['detect', 'head', 'output']):
                            detect_layer = module
                            break
                except:
                    pass

            # 方法4：获取最后一个模块
            if detect_layer is None:
                try:
                    modules = list(model.modules())
                    detect_layer = modules[-1] if modules else None
                except:
                    pass

            if detect_layer is None:
                return None

            # 检查检测头的类名
            layer_name = detect_layer.__class__.__name__.lower()
            print(f"检测头类型: {detect_layer.__class__.__name__}")

            # 根据检测头类型判断任务
            if 'segment' in layer_name or 'mask' in layer_name:
                return 'segment'
            elif 'pose' in layer_name or 'keypoint' in layer_name:
                return 'pose'
            elif 'classify' in layer_name or 'cls' in layer_name:
                return 'classify'
            elif 'obb' in layer_name or 'oriented' in layer_name:
                return 'obb'
            elif 'detect' in layer_name:
                return 'detect'

            # 检查检测头的输出数量和结构
            if hasattr(detect_layer, 'nc'):  # 类别数
                nc = detect_layer.nc
            elif hasattr(detect_layer, 'num_classes'):
                nc = detect_layer.num_classes
            else:
                nc = None

            # 检查是否有分割头
            if hasattr(detect_layer, 'nm'):  # 掩码数量
                return 'segment'

            # 检查是否有关键点
            if hasattr(detect_layer, 'nkpt') or hasattr(detect_layer, 'num_keypoints'):
                return 'pose'

            # 检查输出维度
            if hasattr(detect_layer, 'anchors') and hasattr(detect_layer, 'strides'):
                # 标准检测头
                return 'detect'

            return None

        except Exception as e:
            print(f"模型结构分析失败: {e}")
            return None

    def _check_model_task_attribute(self):
        """检查模型的任务属性"""
        try:
            # 检查ultralytics模型的task属性
            if hasattr(self.model, 'task'):
                task = self.model.task
                print(f"模型任务属性: {task}")

                # 映射ultralytics的任务名称
                task_mapping = {
                    'detect': 'detect',
                    'segment': 'segment',
                    'pose': 'pose',
                    'classify': 'classify',
                    'obb': 'obb',
                    'detection': 'detect',
                    'segmentation': 'segment',
                    'classification': 'classify',
                    'oriented': 'obb'
                }

                return task_mapping.get(task.lower(), None)

            # 检查模型配置中的任务信息
            if hasattr(self.model, 'cfg') and self.model.cfg:
                cfg = self.model.cfg
                if isinstance(cfg, dict) and 'task' in cfg:
                    task = cfg['task']
                    print(f"模型配置任务: {task}")
                    return task.lower() if task else None

            return None

        except Exception as e:
            print(f"模型任务属性检查失败: {e}")
            return None

    def _detect_by_inference(self):
        """通过测试推理检测任务类型"""
        try:
            # 创建小尺寸测试图像以加快推理速度
            test_image = np.zeros((320, 320, 3), dtype=np.uint8)

            # 运行推理
            results = self.model(test_image, verbose=False)

            if results and len(results) > 0:
                result = results[0]

                # 检查输出结构确定任务类型
                has_boxes = hasattr(result, 'boxes') and result.boxes is not None
                has_masks = hasattr(result, 'masks') and result.masks is not None
                has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
                has_probs = hasattr(result, 'probs') and result.probs is not None
                has_obb = hasattr(result, 'obb') and result.obb is not None

                print(f"推理输出结构 - boxes: {has_boxes}, masks: {has_masks}, keypoints: {has_keypoints}, probs: {has_probs}, obb: {has_obb}")

                # 根据输出结构判断任务类型
                if has_probs and not has_boxes and not has_obb:
                    return 'classify'
                elif has_masks and has_boxes:
                    return 'segment'
                elif has_keypoints and has_boxes:
                    return 'pose'
                elif has_obb:
                    return 'obb'
                elif has_boxes:
                    return 'detect'

            return None

        except Exception as e:
            print(f"推理检测失败: {e}")
            return None

    def _analyze_onnx_outputs(self):
        """分析ONNX模型输出确定任务类型"""
        try:
            if not hasattr(self.model, 'get_outputs'):
                return 'detect'

            outputs = self.model.get_outputs()
            output_names = [output.name for output in outputs]
            output_shapes = [output.shape for output in outputs]

            print(f"ONNX输出节点: {output_names}")
            print(f"ONNX输出形状: {output_shapes}")

            # 根据输出节点数量和名称判断
            if len(outputs) == 1:
                # 单输出通常是分类
                output_shape = output_shapes[0]
                if len(output_shape) == 2:  # [batch, classes]
                    return 'classify'
                else:
                    return 'detect'
            elif len(outputs) >= 2:
                # 多输出可能是分割、姿态估计或定向边框
                for name in output_names:
                    name_lower = name.lower()
                    if 'mask' in name_lower or 'segment' in name_lower:
                        return 'segment'
                    elif 'keypoint' in name_lower or 'pose' in name_lower:
                        return 'pose'
                    elif 'obb' in name_lower or 'oriented' in name_lower:
                        return 'obb'

                return 'detect'

            return 'detect'

        except Exception as e:
            print(f"ONNX输出分析失败: {e}")
            return 'detect'

    def set_parameters(self, conf: float = None, iou: float = None,
                      max_det: int = None, classes: List[int] = None,
                      show_labels: bool = None, show_conf: bool = None,
                      show_boxes: bool = None, show_masks: bool = None,
                      show_keypoints: bool = None, show_obb: bool = None):
        """
        设置检测参数

        Args:
            conf: 置信度阈值
            iou: IoU阈值
            max_det: 最大检测数量
            classes: 指定检测的类别ID列表
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
            show_boxes: 是否显示边界框
            show_masks: 是否显示分割掩码
            show_keypoints: 是否显示关键点
            show_obb: 是否显示定向边界框
        """
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou
        if max_det is not None:
            self.max_det = max_det
        if classes is not None:
            self.classes = classes
        if show_labels is not None:
            self.show_labels = show_labels
        if show_conf is not None:
            self.show_conf = show_conf
        if show_boxes is not None:
            self.show_boxes = show_boxes
        if show_masks is not None:
            self.show_masks = show_masks
        if show_keypoints is not None:
            self.show_keypoints = show_keypoints
        if show_obb is not None:
            self.show_obb = show_obb

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

                # 使用自定义绘制方法根据显示选项绘制结果
                annotated_frame = self.plot_results(results, frame)

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
            'task_type': self.task_type,
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

    def plot_results(self, results, image: np.ndarray) -> np.ndarray:
        """
        根据显示选项和任务类型绘制检测结果

        Args:
            results: 检测结果
            image: 原始图像

        Returns:
            np.ndarray: 绘制后的图像
        """
        if not results or len(results) == 0:
            return image

        # 检查是否有任何显示选项开启
        has_display_options = (self.show_boxes or self.show_labels or self.show_conf or
                              self.show_masks or self.show_keypoints or self.show_obb)
        if not has_display_options:
            return image

        # 复制图像以避免修改原图
        annotated_image = image.copy()
        result = results[0]

        # 根据任务类型绘制不同的结果
        if self.task_type == 'classify':
            return self._plot_classification_results(result, annotated_image)
        elif self.task_type == 'segment':
            return self._plot_segmentation_results(result, annotated_image)
        elif self.task_type == 'pose':
            return self._plot_pose_results(result, annotated_image)
        elif self.task_type == 'obb':
            return self._plot_obb_results(result, annotated_image)
        else:  # 'detect' 或其他
            return self._plot_detection_results(result, annotated_image)

    def _plot_detection_results(self, result, image: np.ndarray) -> np.ndarray:
        """绘制目标检测结果"""
        if not hasattr(result, 'boxes') or result.boxes is None:
            return image

        boxes = result.boxes
        if not hasattr(boxes, 'xyxy'):
            return image

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None
        names = getattr(self.model, 'names', {}) if self.model_type == 'pt' else {}

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)

            # 获取类别对应的颜色
            if cls is not None:
                class_id = int(cls[i])
                color = self._get_class_color(class_id)
            else:
                color = (0, 255, 0)  # 默认绿色

            # 绘制边界框
            if self.show_boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签和置信度
            if (self.show_labels or self.show_conf) and conf is not None and cls is not None:
                label_parts = []
                if self.show_labels:
                    class_name = names.get(int(cls[i]), f'class_{int(cls[i])}')
                    label_parts.append(class_name)
                if self.show_conf:
                    label_parts.append(f'{conf[i]:.2f}')

                if label_parts:
                    label = ' '.join(label_parts)
                    self._draw_label(image, label, (x1, y1), color)

        return image

    def _plot_segmentation_results(self, result, image: np.ndarray) -> np.ndarray:
        """绘制实例分割结果"""
        # 先绘制分割掩码
        if self.show_masks and hasattr(result, 'masks') and result.masks is not None:
            try:
                masks = result.masks.data.cpu().numpy()
                image_height, image_width = image.shape[:2]

                # 获取类别信息用于颜色分配
                cls = None
                if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result.boxes, 'cls'):
                    cls = result.boxes.cls.cpu().numpy()

                for i, mask in enumerate(masks):
                    # 获取对应的类别ID和固定颜色
                    if cls is not None and i < len(cls):
                        class_id = int(cls[i])
                        color = self._get_class_color(class_id)
                    else:
                        # 如果没有类别信息，使用实例索引生成固定颜色
                        color = self._get_class_color(i)

                    # 调整掩码尺寸到图像尺寸
                    if mask.shape != (image_height, image_width):
                        mask_resized = cv2.resize(mask.astype(np.float32),
                                                (image_width, image_height),
                                                interpolation=cv2.INTER_LINEAR)
                    else:
                        mask_resized = mask

                    # 创建彩色掩码
                    colored_mask = np.zeros_like(image)
                    mask_binary = mask_resized > 0.5

                    # 安全地应用掩码
                    if mask_binary.shape[:2] == image.shape[:2]:
                        colored_mask[mask_binary] = color
                        # 叠加到图像上
                        image = cv2.addWeighted(image, 1.0, colored_mask, 0.3, 0)
                    else:
                        print(f"警告：掩码尺寸 {mask_binary.shape} 与图像尺寸 {image.shape[:2]} 不匹配")

            except Exception as e:
                print(f"绘制分割掩码时出错: {e}")
                # 如果掩码绘制失败，继续绘制边界框

        # 然后绘制边界框和标签（如果有的话）
        if hasattr(result, 'boxes') and result.boxes is not None:
            image = self._plot_detection_results(result, image)

        return image

    def _plot_pose_results(self, result, image: np.ndarray) -> np.ndarray:
        """绘制姿态估计结果"""
        # 先绘制边界框和标签
        if hasattr(result, 'boxes') and result.boxes is not None:
            image = self._plot_detection_results(result, image)

        # 绘制关键点
        if self.show_keypoints and hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

            # COCO姿态关键点连接
            skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]

            for person_kpts in keypoints:
                # 绘制关键点
                for i, (x, y, conf) in enumerate(person_kpts):
                    if conf > 0.5:  # 置信度阈值
                        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), -1)

                # 绘制骨架连接
                for connection in skeleton:
                    kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1
                    if (kpt1_idx < len(person_kpts) and kpt2_idx < len(person_kpts) and
                        person_kpts[kpt1_idx][2] > 0.5 and person_kpts[kpt2_idx][2] > 0.5):
                        pt1 = (int(person_kpts[kpt1_idx][0]), int(person_kpts[kpt1_idx][1]))
                        pt2 = (int(person_kpts[kpt2_idx][0]), int(person_kpts[kpt2_idx][1]))
                        cv2.line(image, pt1, pt2, (255, 0, 0), 2)

        return image

    def _plot_obb_results(self, result, image: np.ndarray) -> np.ndarray:
        """绘制定向边界框检测结果"""
        if not hasattr(result, 'obb') or result.obb is None:
            return image

        obb = result.obb
        if not hasattr(obb, 'xyxyxyxy'):
            return image

        # 获取定向边界框的四个顶点坐标
        xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
        conf = obb.conf.cpu().numpy() if hasattr(obb, 'conf') else None
        cls = obb.cls.cpu().numpy() if hasattr(obb, 'cls') else None
        names = getattr(self.model, 'names', {}) if self.model_type == 'pt' else {}

        for i in range(len(xyxyxyxy)):
            # 获取四个顶点坐标
            points = xyxyxyxy[i].reshape(4, 2).astype(int)

            # 获取类别对应的颜色
            if cls is not None:
                class_id = int(cls[i])
                color = self._get_class_color(class_id)
            else:
                color = (0, 255, 0)  # 默认绿色

            # 绘制定向边界框
            if self.show_obb:
                # 绘制四条边
                for j in range(4):
                    pt1 = tuple(points[j])
                    pt2 = tuple(points[(j + 1) % 4])
                    cv2.line(image, pt1, pt2, color, 2)

                # 可选：填充半透明区域
                # overlay = image.copy()
                # cv2.fillPoly(overlay, [points], color)
                # image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)

            # 绘制标签和置信度
            if (self.show_labels or self.show_conf) and conf is not None and cls is not None:
                label_parts = []
                if self.show_labels:
                    class_name = names.get(int(cls[i]), f'class_{int(cls[i])}')
                    label_parts.append(class_name)
                if self.show_conf:
                    label_parts.append(f'{conf[i]:.2f}')

                if label_parts:
                    label = ' '.join(label_parts)
                    # 在第一个顶点位置绘制标签
                    label_pos = tuple(points[0])
                    self._draw_label(image, label, label_pos, color)

        return image

    def _plot_classification_results(self, result, image: np.ndarray) -> np.ndarray:
        """绘制分类结果"""
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            names = getattr(self.model, 'names', {}) if self.model_type == 'pt' else {}

            # 获取top-5预测
            top5_indices = np.argsort(probs)[-5:][::-1]

            y_offset = 30
            for i, idx in enumerate(top5_indices):
                if idx < len(names):
                    class_name = names[idx]
                    confidence = probs[idx]
                    text = f"{class_name}: {confidence:.3f}"

                    cv2.putText(image, text, (10, y_offset + i * 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image

    def _draw_label(self, image: np.ndarray, label: str, position: tuple, color: tuple = (0, 255, 0)):
        """绘制标签"""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 绘制文本背景
        cv2.rectangle(image,
                     (x, y - text_height - baseline - 5),
                     (x + text_width, y),
                     color, -1)

        # 绘制文本（使用白色或黑色，根据背景颜色自动选择）
        text_color = self._get_text_color(color)
        cv2.putText(image, label,
                   (x, y - baseline - 2),
                   font, font_scale, text_color, thickness)

    def _get_text_color(self, bg_color: tuple) -> tuple:
        """根据背景颜色选择合适的文本颜色"""
        # 计算背景颜色的亮度
        brightness = (bg_color[0] * 0.299 + bg_color[1] * 0.587 + bg_color[2] * 0.114)
        # 如果背景较暗，使用白色文本；如果背景较亮，使用黑色文本
        return (255, 255, 255) if brightness < 128 else (0, 0, 0)