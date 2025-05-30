"""
自动安装脚本
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n{description}...")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 失败: {e}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False


def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python版本过低，需要Python 3.8+")
        return False
    
    print("✓ Python版本符合要求")
    return True


def install_requirements():
    """安装依赖包"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("✗ 未找到requirements.txt文件")
        return False
    
    # 升级pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "升级pip"):
        print("警告: pip升级失败，继续安装依赖包...")
    
    # 安装依赖包
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "安装依赖包")


def check_cuda():
    """检查CUDA支持"""
    print("\n检查CUDA支持...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA可用 - {device_count}个GPU设备 ({device_name})")
            return True
        else:
            print("! CUDA不可用，将使用CPU模式")
            return False
    except ImportError:
        print("! 无法导入torch，请检查安装")
        return False


def install_cuda_pytorch():
    """安装CUDA版本的PyTorch"""
    print("\n是否要安装CUDA版本的PyTorch以获得GPU加速？")
    print("1. CUDA 11.8")
    print("2. CUDA 12.1") 
    print("3. 跳过")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        command = f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        return run_command(command, "安装CUDA 11.8版本的PyTorch")
    elif choice == "2":
        command = f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        return run_command(command, "安装CUDA 12.1版本的PyTorch")
    else:
        print("跳过CUDA PyTorch安装")
        return True


def create_directories():
    """创建必要的目录"""
    print("\n创建必要的目录...")
    
    directories = ["output", "models", "examples"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"✓ 创建目录: {directory}")
        else:
            print(f"✓ 目录已存在: {directory}")
    
    return True


def download_sample_model():
    """下载示例模型"""
    print("\n是否要下载YOLOv8n示例模型？(约6MB)")
    choice = input("下载示例模型？(y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        try:
            from ultralytics import YOLO
            print("正在下载YOLOv8n模型...")
            model = YOLO('yolov8n.pt')
            print("✓ 示例模型下载完成")
            return True
        except Exception as e:
            print(f"✗ 模型下载失败: {e}")
            return False
    else:
        print("跳过模型下载")
        return True


def test_installation():
    """测试安装"""
    print("\n测试安装...")
    
    try:
        # 测试导入主要模块
        import PySide6
        import ultralytics
        import cv2
        import numpy
        import torch
        import PIL
        
        print("✓ 所有依赖包导入成功")
        
        # 测试检测器
        from core.detector import YOLODetector
        from core.utils import check_cuda_availability
        
        detector = YOLODetector()
        cuda_available, cuda_info = check_cuda_availability()
        
        print(f"✓ 检测器创建成功")
        print(f"✓ CUDA状态: {cuda_info}")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("YOLO目标检测系统安装脚本")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        input("按回车键退出...")
        return 1
    
    # 安装依赖包
    if not install_requirements():
        print("依赖包安装失败，请手动安装")
        input("按回车键退出...")
        return 1
    
    # 检查CUDA
    cuda_available = check_cuda()
    
    # 可选安装CUDA版本PyTorch
    if not cuda_available:
        install_cuda_pytorch()
    
    # 创建目录
    create_directories()
    
    # 下载示例模型
    download_sample_model()
    
    # 测试安装
    if test_installation():
        print("\n" + "=" * 50)
        print("✓ 安装完成！")
        print("\n使用方法:")
        print("1. 运行 'python main.py' 启动GUI程序")
        print("2. 或者双击 'run.bat' (Windows) / 'run.sh' (Linux/Mac)")
        print("3. 查看 'examples/' 目录中的示例代码")
        print("4. 阅读 'README.md' 了解详细使用说明")
    else:
        print("\n" + "=" * 50)
        print("✗ 安装测试失败，请检查错误信息")
    
    input("\n按回车键退出...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
