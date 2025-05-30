"""
图片查看器组件
"""
import cv2
import numpy as np
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from typing import Optional


class ImageViewer(QScrollArea):
    """图片查看器组件"""

    # 信号
    imageClicked = Signal(QPoint)  # 图片点击信号

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建图片标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setText("请选择图片或视频文件")
        self.image_label.setScaledContents(False)  # 不自动缩放内容

        # 设置滚动区域
        self.setWidget(self.image_label)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)

        # 存储原始图片
        self.original_image = None
        self.current_pixmap = None
        self.scale_factor = 1.0
        self.auto_fit = True  # 自动适应窗口

        # 鼠标事件
        self.image_label.mousePressEvent = self._on_image_clicked

        # 监听窗口大小变化
        self.resizeEvent = self._on_resize

    def set_image(self, image: np.ndarray, is_rgb: bool = False):
        """
        设置显示的图片

        Args:
            image: 图片数组
            is_rgb: 是否已经是RGB格式，默认False（BGR格式）
        """
        if image is None:
            self.clear_image()
            return

        # 保存原始图片
        self.original_image = image.copy()

        # 转换为RGB格式
        if len(image.shape) == 3:
            if is_rgb:
                rgb_image = image  # 已经是RGB格式，不需要转换
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转RGB
        else:
            rgb_image = image

        # 转换为QImage
        h, w = rgb_image.shape[:2]
        if len(rgb_image.shape) == 3:
            bytes_per_line = 3 * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

        # 转换为QPixmap并显示
        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap = pixmap

        # 自动适应窗口大小
        if self.auto_fit:
            self.fit_to_window()
        else:
            self._update_display()

    def set_pixmap(self, pixmap: QPixmap):
        """
        直接设置QPixmap

        Args:
            pixmap: QPixmap对象
        """
        self.current_pixmap = pixmap
        self._update_display()

    def clear_image(self):
        """清除图片"""
        self.original_image = None
        self.current_pixmap = None
        self.image_label.clear()
        self.image_label.setText("请选择图片或视频文件")

    def zoom_in(self):
        """放大图片"""
        self.scale_factor *= 1.25
        self._update_display()

    def zoom_out(self):
        """缩小图片"""
        self.scale_factor /= 1.25
        self._update_display()

    def reset_zoom(self):
        """重置缩放"""
        self.scale_factor = 1.0
        self._update_display()

    def fit_to_window(self):
        """适应窗口大小"""
        if self.current_pixmap is None:
            return

        # 获取可用空间，留出一些边距
        available_size = self.viewport().size()
        margin = 20  # 边距
        available_width = available_size.width() - margin
        available_height = available_size.height() - margin

        pixmap_size = self.current_pixmap.size()

        # 计算缩放比例
        if available_width > 0 and available_height > 0:
            scale_x = available_width / pixmap_size.width()
            scale_y = available_height / pixmap_size.height()
            self.scale_factor = min(scale_x, scale_y)  # 允许放大和缩小

            # 设置最小和最大缩放限制
            self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        else:
            self.scale_factor = 1.0

        self._update_display()

    def _update_display(self):
        """更新显示"""
        if self.current_pixmap is None:
            return

        # 应用缩放
        scaled_pixmap = self.current_pixmap.scaled(
            self.current_pixmap.size() * self.scale_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    def _on_image_clicked(self, event):
        """处理图片点击事件"""
        if self.current_pixmap is None:
            return

        # 计算在原始图片中的坐标
        click_pos = event.pos()

        # 考虑缩放因子
        original_pos = QPoint(
            int(click_pos.x() / self.scale_factor),
            int(click_pos.y() / self.scale_factor)
        )

        self.imageClicked.emit(original_pos)

    def _on_resize(self, event):
        """处理窗口大小变化"""
        # 调用父类的resizeEvent
        super().resizeEvent(event)

        # 如果开启了自动适应，重新调整图片大小
        if self.auto_fit and self.current_pixmap is not None:
            self.fit_to_window()

    def set_auto_fit(self, auto_fit: bool):
        """设置是否自动适应窗口"""
        self.auto_fit = auto_fit
        if self.auto_fit and self.current_pixmap is not None:
            self.fit_to_window()

    def get_current_image(self) -> Optional[np.ndarray]:
        """
        获取当前显示的图片

        Returns:
            当前图片的numpy数组
        """
        return self.original_image.copy() if self.original_image is not None else None

    def save_image(self, file_path: str) -> bool:
        """
        保存当前图片

        Args:
            file_path: 保存路径

        Returns:
            bool: 是否保存成功
        """
        if self.current_pixmap is None:
            return False

        try:
            return self.current_pixmap.save(file_path)
        except Exception as e:
            print(f"保存图片失败: {str(e)}")
            return False


class VideoFrameViewer(ImageViewer):
    """视频帧查看器"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_info = ""

    def set_frame(self, frame: np.ndarray, frame_number: int, total_frames: int):
        """
        设置视频帧

        Args:
            frame: 视频帧
            frame_number: 当前帧号
            total_frames: 总帧数
        """
        self.set_image(frame)
        self.frame_info = f"帧 {frame_number}/{total_frames}"

    def get_frame_info(self) -> str:
        """获取帧信息"""
        return self.frame_info
