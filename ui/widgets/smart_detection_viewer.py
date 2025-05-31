"""
智能检测结果查看器
自动识别图片源和视频流，智能切换显示模式
"""
import cv2
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QSlider, QLabel, QFrame, QStackedWidget)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QIcon
from .image_viewer import ImageViewer
from .video_player import VideoPlayer, VideoDetectionThread
import time
import os
from pathlib import Path


class SmartDetectionViewer(QWidget):
    """智能检测结果查看器"""

    # 信号
    frameChanged = Signal(object, np.ndarray)  # 帧变化信号 (检测结果, 图片)
    progressChanged = Signal(int)  # 进度变化信号
    detectionFinished = Signal()  # 检测完成信号
    detectionError = Signal(str)  # 检测错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detector = None
        self.current_source = None
        self.source_type = None  # 'image', 'video', 'camera'
        self.video_thread = None
        self.current_original_image = None  # 保存原始图像用于重新绘制
        self.current_results = None  # 保存当前检测结果

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 图片控制面板（始终显示）
        image_controls = QHBoxLayout()

        self.fit_window_btn = QPushButton("适应窗口")
        self.fit_window_btn.setMaximumWidth(80)
        image_controls.addWidget(self.fit_window_btn)

        self.zoom_in_btn = QPushButton("放大")
        self.zoom_in_btn.setMaximumWidth(60)
        image_controls.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("缩小")
        self.zoom_out_btn.setMaximumWidth(60)
        image_controls.addWidget(self.zoom_out_btn)

        self.reset_zoom_btn = QPushButton("原始大小")
        self.reset_zoom_btn.setMaximumWidth(80)
        image_controls.addWidget(self.reset_zoom_btn)

        self.auto_fit_cb = QPushButton("自动适应")
        self.auto_fit_cb.setCheckable(True)
        self.auto_fit_cb.setChecked(True)
        self.auto_fit_cb.setMaximumWidth(80)
        image_controls.addWidget(self.auto_fit_cb)

        image_controls.addStretch()
        layout.addLayout(image_controls)

        # 主显示区域
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer)

        # 视频控制面板（初始隐藏）
        self.video_control_frame = QFrame()
        self.video_control_frame.setMaximumHeight(80)
        self.video_control_frame.setStyleSheet("QFrame { border: 1px solid gray; background-color: #f0f0f0; }")
        self.video_control_frame.setVisible(False)

        video_control_layout = QVBoxLayout(self.video_control_frame)

        # 进度条
        progress_layout = QHBoxLayout()

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(100)
        progress_layout.addWidget(self.time_label)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        progress_layout.addWidget(self.progress_slider)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        progress_layout.addWidget(self.frame_label)

        video_control_layout.addLayout(progress_layout)

        # 播放控制按钮
        button_layout = QHBoxLayout()

        self.play_pause_btn = QPushButton("播放")
        self.play_pause_btn.setMaximumWidth(60)
        button_layout.addWidget(self.play_pause_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMaximumWidth(60)
        button_layout.addWidget(self.stop_btn)

        self.prev_frame_btn = QPushButton("上一帧")
        self.prev_frame_btn.setMaximumWidth(80)
        button_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("下一帧")
        self.next_frame_btn.setMaximumWidth(80)
        button_layout.addWidget(self.next_frame_btn)

        # 播放速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))

        self.speed_0_5x_btn = QPushButton("0.5x")
        self.speed_0_5x_btn.setMaximumWidth(50)
        speed_layout.addWidget(self.speed_0_5x_btn)

        self.speed_1x_btn = QPushButton("1x")
        self.speed_1x_btn.setMaximumWidth(40)
        self.speed_1x_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        speed_layout.addWidget(self.speed_1x_btn)

        self.speed_2x_btn = QPushButton("2x")
        self.speed_2x_btn.setMaximumWidth(40)
        speed_layout.addWidget(self.speed_2x_btn)

        button_layout.addLayout(speed_layout)
        button_layout.addStretch()

        video_control_layout.addLayout(button_layout)
        layout.addWidget(self.video_control_frame)

        # 初始化状态
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.is_paused = False

    def connect_signals(self):
        """连接信号"""
        # 图片控制信号
        self.fit_window_btn.clicked.connect(self.fit_to_window)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        self.auto_fit_cb.toggled.connect(self.toggle_auto_fit)

        # 视频控制信号
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)

        # 进度条信号
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        self.progress_slider.valueChanged.connect(self.on_slider_value_changed)

        # 播放速度信号
        self.speed_0_5x_btn.clicked.connect(lambda: self.set_playback_speed(0.5))
        self.speed_1x_btn.clicked.connect(lambda: self.set_playback_speed(1.0))
        self.speed_2x_btn.clicked.connect(lambda: self.set_playback_speed(2.0))

    def set_detector(self, detector):
        """设置检测器"""
        self.detector = detector

    def load_source(self, source_path):
        """加载检测源（自动识别类型）"""
        self.current_source = source_path

        # 自动识别源类型
        if isinstance(source_path, (int, str)) and str(source_path).isdigit():
            # 摄像头
            self.source_type = 'camera'
            return self._setup_camera_mode()
        elif isinstance(source_path, str):
            # 文件路径
            if self._is_image_file(source_path):
                self.source_type = 'image'
                return self._setup_image_mode(source_path)
            elif self._is_video_file(source_path):
                self.source_type = 'video'
                return self._setup_video_mode(source_path)

        return False

    def _is_image_file(self, file_path):
        """判断是否为图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        return Path(file_path).suffix.lower() in image_extensions

    def _is_video_file(self, file_path):
        """判断是否为视频文件"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        return Path(file_path).suffix.lower() in video_extensions

    def _setup_image_mode(self, image_path):
        """设置图片模式"""
        try:
            # 隐藏视频控制面板
            self.video_control_frame.setVisible(False)

            # 加载并显示图片
            image = cv2.imread(image_path)
            if image is not None:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.image_viewer.set_image(rgb_image)
                return True

        except Exception as e:
            print(f"加载图片失败: {e}")

        return False

    def _setup_video_mode(self, video_path):
        """设置视频模式"""
        try:
            # 显示视频控制面板
            self.video_control_frame.setVisible(True)

            # 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 显示第一帧
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.image_viewer.set_image(rgb_frame, is_rgb=True)  # 指明图像已经是RGB格式

                cap.release()

                # 更新UI
                self.progress_slider.setMaximum(self.total_frames - 1)
                self.update_time_display(0, fps)
                self.update_frame_display(0)

                return True

        except Exception as e:
            print(f"加载视频失败: {e}")

        return False

    def _setup_camera_mode(self):
        """设置摄像头模式"""
        try:
            # 显示视频控制面板
            self.video_control_frame.setVisible(True)

            # 测试摄像头
            cap = cv2.VideoCapture(int(self.current_source))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.image_viewer.set_image(rgb_frame, is_rgb=True)  # 指明图像已经是RGB格式
                cap.release()

                # 摄像头模式的特殊设置
                self.total_frames = 0  # 摄像头没有固定帧数
                self.progress_slider.setMaximum(100)
                self.time_label.setText("实时")
                self.frame_label.setText("摄像头")

                return True

        except Exception as e:
            print(f"打开摄像头失败: {e}")

        return False

    def start_detection(self):
        """开始检测"""
        if not self.detector or self.current_source is None:
            return False

        if self.source_type == 'image':
            return self._detect_image()
        elif self.source_type in ['video', 'camera']:
            return self._start_video_detection()

        return False

    def _detect_image(self):
        """检测图片"""
        try:
            image = cv2.imread(self.current_source)
            if image is None:
                self.detectionError.emit("无法读取图片文件")
                return False

            # 保存原始图像
            self.current_original_image = image.copy()

            # 进行检测
            results = self.detector.detect_image(image)

            # 保存检测结果
            self.current_results = results

            # 使用自定义绘制方法根据显示选项绘制结果
            annotated_image = self.detector.plot_results(results, image)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # 显示结果（指明图像已经是RGB格式）
            self.image_viewer.set_image(annotated_image, is_rgb=True)

            # 发送信号，包含任务类型
            self.frameChanged.emit(results, annotated_image)
            self.progressChanged.emit(100)
            self.detectionFinished.emit()

            return True

        except Exception as e:
            self.detectionError.emit(str(e))
            return False

    def _start_video_detection(self):
        """开始视频检测"""
        if self.video_thread and self.video_thread.isRunning():
            self.stop_playback()

        # 创建检测线程
        self.video_thread = VideoDetectionThread(self.detector, self.current_source)
        self.video_thread.frameReady.connect(self.on_frame_ready)
        self.video_thread.progressUpdate.connect(self.on_progress_update)
        self.video_thread.finished.connect(self.on_playback_finished)
        self.video_thread.error.connect(self.on_playback_error)

        # 启动线程
        self.video_thread.start()
        self.is_playing = True
        self.is_paused = False
        self.play_pause_btn.setText("暂停")

        return True

    def stop_detection(self):
        """停止检测"""
        if self.source_type in ['video', 'camera']:
            self.stop_playback()
        # 图片检测无需停止操作

    def refresh_display(self, update_table=True):
        """刷新显示（当显示选项改变时重新绘制检测结果）

        Args:
            update_table: 是否更新数据表，默认True。当仅刷新显示选项时设为False
        """
        if self.source_type == 'image' and self.current_original_image is not None and self.current_results is not None:
            try:
                # 使用当前的显示选项重新绘制检测结果
                annotated_image = self.detector.plot_results(self.current_results, self.current_original_image)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                # 更新显示
                self.image_viewer.set_image(annotated_image, is_rgb=True)

                # 根据参数决定是否发送信号更新其他组件
                if update_table:
                    # 正常检测时，发送信号更新数据表等组件
                    self.frameChanged.emit(self.current_results, annotated_image)
                # 仅刷新显示选项时，不发送信号，避免重复添加数据表记录

            except Exception as e:
                print(f"刷新显示失败: {e}")
        elif self.source_type == 'video' and hasattr(self, 'current_frame'):
            # 对于视频，重新处理当前帧
            try:
                self._refresh_video_frame(update_table)
            except Exception as e:
                print(f"刷新视频帧失败: {e}")
        # 对于摄像头，暂时不支持实时刷新（实时流无法重新处理）

    def _refresh_video_frame(self, update_table=True):
        """刷新视频当前帧

        Args:
            update_table: 是否更新数据表，默认True。当仅刷新显示选项时设为False
        """
        if self.source_type == 'video' and self.current_source:
            cap = cv2.VideoCapture(self.current_source)
            if cap.isOpened():
                # 跳转到当前帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = cap.read()
                if ret:
                    # 进行检测
                    results = self.detector.detect_image(frame)

                    # 使用当前显示选项绘制结果
                    annotated_frame = self.detector.plot_results(results, frame)
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # 更新显示
                    self.image_viewer.set_image(annotated_frame, is_rgb=True)

                    # 根据参数决定是否发送信号更新其他组件
                    if update_table:
                        # 正常检测时，发送信号更新数据表等组件
                        self.frameChanged.emit(results, annotated_frame)
                    # 仅刷新显示选项时，不发送信号，避免重复添加数据表记录

                cap.release()

    # 图片控制方法
    def fit_to_window(self):
        """适应窗口"""
        self.image_viewer.fit_to_window()
        self.auto_fit_cb.setChecked(False)

    def zoom_in(self):
        """放大"""
        self.image_viewer.zoom_in()
        self.auto_fit_cb.setChecked(False)

    def zoom_out(self):
        """缩小"""
        self.image_viewer.zoom_out()
        self.auto_fit_cb.setChecked(False)

    def reset_zoom(self):
        """重置缩放"""
        self.image_viewer.reset_zoom()
        self.auto_fit_cb.setChecked(False)

    def toggle_auto_fit(self, checked):
        """切换自动适应"""
        self.image_viewer.set_auto_fit(checked)
        if checked:
            self.image_viewer.fit_to_window()

    # 视频控制方法
    def toggle_play_pause(self):
        """切换播放/暂停"""
        if not self.is_playing:
            self.start_detection()
        else:
            if self.is_paused:
                self.resume_playback()
            else:
                self.pause_playback()

    def pause_playback(self):
        """暂停播放"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.pause()
            self.is_paused = True
            self.play_pause_btn.setText("继续")

    def resume_playback(self):
        """继续播放"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.resume()
            self.is_paused = False
            self.play_pause_btn.setText("暂停")

    def stop_playback(self):
        """停止播放"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait(3000)

        self.is_playing = False
        self.is_paused = False
        self.play_pause_btn.setText("播放")
        self.detectionFinished.emit()

    def prev_frame(self):
        """上一帧"""
        if self.current_frame > 0:
            self.seek_to_frame(self.current_frame - 1)

    def next_frame(self):
        """下一帧"""
        if self.current_frame < self.total_frames - 1:
            self.seek_to_frame(self.current_frame + 1)

    def seek_to_frame(self, frame_number):
        """跳转到指定帧"""
        if self.source_type == 'video' and 0 <= frame_number < self.total_frames:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.seek(frame_number)
            else:
                self.show_frame(frame_number)

    def show_frame(self, frame_number):
        """显示指定帧"""
        if self.source_type == 'video':
            cap = cv2.VideoCapture(self.current_source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.image_viewer.set_image(rgb_frame, is_rgb=True)  # 指明图像已经是RGB格式
                    self.current_frame = frame_number
                    self.update_frame_display(frame_number)

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    self.update_time_display(frame_number, fps)
                cap.release()

    def set_playback_speed(self, speed):
        """设置播放速度"""
        # 重置按钮样式
        for btn in [self.speed_0_5x_btn, self.speed_1x_btn, self.speed_2x_btn]:
            btn.setStyleSheet("")

        # 高亮当前速度按钮
        if speed == 0.5:
            self.speed_0_5x_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        elif speed == 1.0:
            self.speed_1x_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        elif speed == 2.0:
            self.speed_2x_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")

    # 进度条控制
    def on_slider_pressed(self):
        """进度条按下"""
        if self.is_playing and not self.is_paused:
            self.pause_playback()

    def on_slider_released(self):
        """进度条释放"""
        if self.source_type == 'video':
            frame_number = self.progress_slider.value()
            self.seek_to_frame(frame_number)

    def on_slider_value_changed(self, value):
        """进度条值变化"""
        if self.source_type == 'video' and (not self.is_playing or self.is_paused):
            self.show_frame(value)

    # 事件处理
    def on_frame_ready(self, results, image, current_frame, total_frames, fps):
        """处理新帧"""
        self.image_viewer.set_image(image, is_rgb=True)  # 指明图像已经是RGB格式
        self.current_frame = current_frame
        self.total_frames = total_frames

        # 更新进度条和显示
        if self.total_frames > 0:
            self.progress_slider.setValue(current_frame)
            self.update_frame_display(current_frame)
            self.update_time_display(current_frame, fps)
        elif self.source_type == 'camera':
            # 摄像头模式显示实时状态
            self.time_label.setText("实时")
            self.frame_label.setText("摄像头")

        # 发送信号
        self.frameChanged.emit(results, image)

    def on_progress_update(self, progress):
        """进度更新"""
        self.progressChanged.emit(progress)

    def on_playback_finished(self):
        """播放完成"""
        self.is_playing = False
        self.is_paused = False
        self.play_pause_btn.setText("播放")
        self.detectionFinished.emit()

    def on_playback_error(self, error_msg):
        """播放错误"""
        self.is_playing = False
        self.is_paused = False
        self.play_pause_btn.setText("播放")
        self.detectionError.emit(error_msg)

    def update_time_display(self, frame_number, fps):
        """更新时间显示"""
        if fps > 0 and self.total_frames > 0:
            current_seconds = frame_number / fps
            total_seconds = self.total_frames / fps

            current_time = f"{int(current_seconds // 60):02d}:{int(current_seconds % 60):02d}"
            total_time = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"

            self.time_label.setText(f"{current_time} / {total_time}")

    def update_frame_display(self, frame_number):
        """更新帧数显示"""
        if self.total_frames > 0:
            self.frame_label.setText(f"{frame_number} / {self.total_frames}")

    def get_source_type(self):
        """获取当前源类型"""
        return self.source_type

    def get_current_image(self):
        """获取当前显示的图片"""
        return self.image_viewer.get_current_image()
