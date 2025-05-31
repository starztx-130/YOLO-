# YOLO模型转换器
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
import torch
from .utils import validate_model_file, get_device


class ModelConverter:
    # YOLO模型转换器类
    
    # 支持的导出格式配置
    EXPORT_FORMATS = {
        'onnx': {
            'name': 'ONNX',
            'extension': '.onnx',
            'description': '跨平台推理格式，支持多种推理引擎',
            'supported_args': ['imgsz', 'half', 'dynamic', 'simplify', 'opset', 'batch']
        },
        'openvino': {
            'name': 'OpenVINO',
            'extension': '_openvino_model/',
            'description': 'Intel硬件优化格式，支持CPU/GPU/VPU推理',
            'supported_args': ['imgsz', 'half', 'dynamic', 'int8', 'batch', 'data']
        },
        'ncnn': {
            'name': 'NCNN',
            'extension': '_ncnn_model/',
            'description': '移动端优化推理框架，支持ARM/x86平台',
            'supported_args': ['imgsz', 'half', 'batch']
        }
    }
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.device = get_device()
        
        # 转换参数
        self.export_params = {
            'imgsz': 640,           # 输入图像尺寸
            'half': False,          # FP16量化
            'int8': False,          # INT8量化
            'dynamic': False,       # 动态输入尺寸
            'simplify': True,       # 简化ONNX模型
            'opset': None,          # ONNX操作集版本
            'workspace': None,      # TensorRT工作空间大小(GB)
            'batch': 1,             # 批处理大小
            'device': None,         # 导出设备
            'data': None,           # 量化校准数据集
            'keras': False,         # Keras格式
            'optimize': False,      # 移动端优化
        }
        
    def load_model(self, model_path: str) -> bool:
        # 加载要转换的模型
        try:
            if not validate_model_file(model_path):
                raise ValueError(f"无效的模型文件: {model_path}")
                
            # 检查是否为PyTorch模型
            if not model_path.endswith('.pt'):
                raise ValueError("模型转换器仅支持PyTorch (.pt) 格式的源模型")
                
            self.model_path = model_path
            self.model = YOLO(model_path)
            
            return True
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None
            return False
            
    def get_supported_formats(self) -> Dict[str, Dict]:
        # 获取支持的导出格式
        return self.EXPORT_FORMATS.copy()
        
    def set_export_params(self, **kwargs) -> None:
        # 设置导出参数
        for key, value in kwargs.items():
            if key in self.export_params:
                self.export_params[key] = value
                
    def get_export_params(self) -> Dict:
        # 获取当前导出参数
        return self.export_params.copy()
        
    def validate_export_params(self, format_key: str) -> Tuple[bool, str]:
        # 验证导出参数是否适用于指定格式
        if format_key not in self.EXPORT_FORMATS:
            return False, f"不支持的导出格式: {format_key}"
            
        format_info = self.EXPORT_FORMATS[format_key]
        supported_args = format_info.get('supported_args', [])
        
        # 检查参数兼容性
        warnings = []

        # INT8和half精度冲突检查
        if self.export_params.get('int8') and self.export_params.get('half'):
            if format_key == 'onnx':
                warnings.append("ONNX格式不支持同时启用INT8和FP16")

        # 动态尺寸支持检查
        if self.export_params.get('dynamic') and 'dynamic' not in supported_args:
            warnings.append(f"{format_info['name']}格式不支持动态输入尺寸")

        if warnings:
            return False, "; ".join(warnings)

        return True, ""
        
    def get_output_path(self, format_key: str, output_dir: str = None) -> str:
        # 生成输出文件路径
        if not self.model_path:
            raise ValueError("未加载模型")
            
        model_name = Path(self.model_path).stem
        format_info = self.EXPORT_FORMATS[format_key]
        extension = format_info['extension']
        
        if output_dir is None:
            output_dir = Path(self.model_path).parent / "converted_models"
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"{model_name}{extension}"
        return str(output_path)

    def convert_model(self, format_key: str, output_path: str = None,
                     progress_callback=None) -> Tuple[bool, str, str]:
        # 转换模型到指定格式
        if self.model is None:
            return False, "", "未加载模型"

        # 验证导出参数
        is_valid, error_msg = self.validate_export_params(format_key)
        if not is_valid:
            return False, "", error_msg

        try:
            # 生成输出路径
            if output_path is None:
                output_path = self.get_output_path(format_key)

            # 准备导出参数
            export_kwargs = self._prepare_export_kwargs(format_key)

            if progress_callback:
                progress_callback(10, "准备导出参数...")

            # 执行导出
            start_time = time.time()

            if progress_callback:
                progress_callback(20, f"开始转换为{self.EXPORT_FORMATS[format_key]['name']}格式...")

            # 调用ultralytics的export方法
            exported_model = self.model.export(format=format_key, **export_kwargs)

            if progress_callback:
                progress_callback(90, "验证转换结果...")

            # 验证输出文件
            if isinstance(exported_model, str):
                actual_output_path = exported_model
            else:
                actual_output_path = output_path

            if not os.path.exists(actual_output_path):
                return False, "", f"转换失败：输出文件不存在 {actual_output_path}"

            end_time = time.time()
            conversion_time = end_time - start_time

            if progress_callback:
                progress_callback(100, f"转换完成，耗时 {conversion_time:.2f} 秒")

            return True, actual_output_path, f"转换成功，耗时 {conversion_time:.2f} 秒"

        except Exception as e:
            error_msg = f"转换失败: {str(e)}"
            if progress_callback:
                progress_callback(0, error_msg)
            return False, "", error_msg

    def _prepare_export_kwargs(self, format_key: str) -> Dict[str, Any]:
        # 准备导出参数
        format_info = self.EXPORT_FORMATS[format_key]
        supported_args = format_info.get('supported_args', [])

        # 过滤支持的参数
        export_kwargs = {}
        for arg in supported_args:
            if arg in self.export_params and self.export_params[arg] is not None:
                export_kwargs[arg] = self.export_params[arg]

        # 特殊处理
        if format_key == 'openvino' and self.export_params.get('int8'):
            # OpenVINO INT8需要校准数据
            if not export_kwargs.get('data'):
                export_kwargs['data'] = 'coco8.yaml'  # 默认数据集

        # 设置设备
        if 'device' not in export_kwargs:
            export_kwargs['device'] = self.device

        return export_kwargs

    def get_model_info(self) -> Dict[str, Any]:
        # 获取当前模型信息
        if self.model is None:
            return {}

        try:
            info = {
                'model_path': self.model_path,
                'model_name': Path(self.model_path).name if self.model_path else "",
                'model_size': self._get_file_size(self.model_path) if self.model_path else 0,
                'device': self.device,
                'task': getattr(self.model, 'task', 'unknown'),
                'names': getattr(self.model, 'names', {}),
                'num_classes': len(getattr(self.model, 'names', {}))
            }

            # 尝试获取模型参数数量
            if hasattr(self.model, 'model'):
                try:
                    total_params = sum(p.numel() for p in self.model.model.parameters())
                    info['parameters'] = total_params
                except:
                    info['parameters'] = 0
            else:
                info['parameters'] = 0

            return info

        except Exception as e:
            print(f"获取模型信息失败: {e}")
            return {}

    def _get_file_size(self, file_path: str) -> int:
        # 获取文件大小（字节）
        try:
            return os.path.getsize(file_path)
        except:
            return 0

    def estimate_conversion_time(self, format_key: str) -> str:
        # 估算转换时间
        if format_key == 'onnx':
            return "约 30秒 - 2分钟"
        elif format_key == 'openvino':
            return "约 1-3分钟"
        elif format_key == 'ncnn':
            return "约 1-5分钟"
        else:
            return "约 1-5分钟"

    def get_format_recommendations(self) -> Dict[str, List[str]]:
        # 获取格式推荐
        return {
            '跨平台部署': ['onnx'],
            'Intel硬件优化': ['openvino'],
            '移动端/嵌入式设备': ['ncnn'],
            'CPU推理优化': ['openvino'],
            'ARM设备部署': ['ncnn'],
            '通用推理引擎': ['onnx']
        }
