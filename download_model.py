"""
下载YOLO模型脚本
"""
import os

# 设置环境变量解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

def download_models():
    """下载常用的YOLO模型"""
    models = {
        'yolov8n.pt': 'YOLOv8 Nano - 最小最快',
        'yolov8s.pt': 'YOLOv8 Small - 平衡速度和精度',
        'yolov8m.pt': 'YOLOv8 Medium - 较高精度',
    }

    print("开始下载YOLO模型...")
    print("=" * 50)

    for model_name, description in models.items():
        print(f"\n正在下载 {model_name} ({description})")
        try:
            model = YOLO(model_name)
            print(f"✓ {model_name} 下载完成")

            # 检查文件是否存在
            if os.path.exists(model_name):
                file_size = os.path.getsize(model_name) / (1024 * 1024)  # MB
                print(f"  文件大小: {file_size:.1f} MB")

        except Exception as e:
            print(f"✗ {model_name} 下载失败: {e}")

    print("\n" + "=" * 50)
    print("模型下载完成！")
    print("您现在可以在GUI中加载这些模型文件。")

if __name__ == "__main__":
    download_models()
