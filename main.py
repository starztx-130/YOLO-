"""
YOLO目标检测系统主程序

基于YOLOv8的目标检测系统，支持图片、视频和摄像头检测。
提供图形界面和完整的检测功能。

主要功能：
- 图片检测：支持常见图片格式
- 视频检测：支持视频文件播放和检测
- 摄像头检测：实时摄像头检测
- 结果可视化：检测框、标签、置信度显示
- 数据导出：检测结果导出为CSV

技术栈：
- YOLOv8: 目标检测模型
- PySide6: GUI框架
- OpenCV: 图像处理
- NumPy: 数值计算

版本: 2.0.0
许可: MIT License
"""
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

# 设置环境变量解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))



from ui.main_window import MainWindow


def check_dependencies():
    """检查依赖包是否安装"""
    missing_packages = []

    try:
        import PySide6
    except ImportError:
        missing_packages.append("PySide6")

    try:
        import ultralytics
    except ImportError:
        missing_packages.append("ultralytics")

    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")

    try:
        import torch
    except ImportError:
        missing_packages.append("torch")

    try:
        import PIL
    except ImportError:
        missing_packages.append("Pillow")

    if missing_packages:
        error_msg = "缺少以下依赖包，请先安装:\n\n"
        for package in missing_packages:
            error_msg += f"pip install {package}\n"
        error_msg += "\n或者运行: pip install -r requirements.txt"

        print(error_msg)
        return False

    return True


def setup_application():
    """设置应用程序"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("YOLO目标检测系统")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("YOLO Detection")

    # 设置应用程序样式
    app.setStyle("Fusion")

    # 设置高DPI支持
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    return app


def main():
    """主函数"""
    # 检查依赖
    if not check_dependencies():
        input("按回车键退出...")
        return 1

    # 创建应用程序
    app = setup_application()

    try:
        # 创建主窗口
        window = MainWindow()
        window.show()

        # 运行应用程序
        return app.exec()

    except Exception as e:
        error_msg = f"程序启动失败:\n{str(e)}"
        print(error_msg)

        # 如果QApplication已经创建，显示错误对话框
        try:
            QMessageBox.critical(None, "启动错误", error_msg)
        except:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
