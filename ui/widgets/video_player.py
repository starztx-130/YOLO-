"""
视频播放器组件
支持进度条拖动、暂停、继续等功能
"""
import cv2
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QSlider, QLabel, QFrame)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QIcon
from .image_viewer import ImageViewer
import time


class VideoDetectionThread(QThread):
    """视频检测线程"""

    # 信号
    frameReady = Signal(object, np.ndarray, int, int, float)  # 检测结果, 图片, 当前帧, 总帧数, FPS
    progressUpdate = Signal(int)  # 进度更新
    finished = Signal()  # 完成信号
    error = Signal(str)  # 错误信号

    def __init__(self, detector, video_path):
        super().__init__()
        self.detector = detector
        self.video_path = video_path
        self.is_running = True
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.seek_frame = -1  # 跳转到指定帧

    def run(self):
        """运行检测"""
        try:
            # 判断是否为摄像头（数字或数字字符串）
            is_camera = isinstance(self.video_path, int) or (isinstance(self.video_path, str) and self.video_path.isdigit())

            if is_camera:
                cap = cv2.VideoCapture(int(self.video_path))
                if not cap.isOpened():
                    self.error.emit("无法打开摄像头")
                    return

                # 摄像头模式
                self.total_frames = 0  # 摄像头没有固定帧数
                fps = 30  # 摄像头默认30fps
                frame_delay = 1.0 / fps

                while self.is_running and cap.isOpened():
                    # 暂停处理
                    while self.is_paused and self.is_running:
                        time.sleep(0.1)

                    if not self.is_running:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        continue  # 摄像头读取失败时继续尝试

                    # 检测当前帧
                    results = self.detector.detect_image(frame)

                    if results and len(results) > 0:
                        annotated_frame = results[0].plot()
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    else:
                        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 发送帧数据（摄像头模式current_frame保持为0）
                    self.frameReady.emit(results, annotated_frame, 0, 0, fps)

                    # 摄像头模式进度固定为50%
                    self.progressUpdate.emit(50)

                    # 控制播放速度
                    time.sleep(frame_delay)
            else:
                # 视频文件模式
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.error.emit("无法打开视频文件")
                    return

                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30  # 默认30fps

                while self.is_running and cap.isOpened():
                    # 检查是否需要跳转
                    if self.seek_frame >= 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                        self.current_frame = self.seek_frame
                        self.seek_frame = -1

                    # 暂停处理
                    while self.is_paused and self.is_running:
                        time.sleep(0.1)

                    if not self.is_running:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break  # 视频结束

                    # 检测当前帧
                    results = self.detector.detect_image(frame)

                    if results and len(results) > 0:
                        annotated_frame = results[0].plot()
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    else:
                        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 发送帧数据
                    self.frameReady.emit(results, annotated_frame, self.current_frame, self.total_frames, fps)

                    # 更新进度
                    progress = int((self.current_frame / self.total_frames) * 100) if self.total_frames > 0 else 0
                    self.progressUpdate.emit(progress)

                    self.current_frame += 1

                    # 控制播放速度
                    time.sleep(frame_delay)

            cap.release()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def pause(self):
        """暂停"""
        self.is_paused = True

    def resume(self):
        """继续"""
        self.is_paused = False

    def stop(self):
        """停止"""
        self.is_running = False

    def seek(self, frame_number):
        """跳转到指定帧"""
        self.seek_frame = frame_number


class VideoPlayer(QWidget):
    """视频播放器组件"""

    # 信号
    frameChanged = Signal(object, np.ndarray)  # 帧变化信号
    progressChanged = Signal(int)  # 进度变化信号
    playbackFinished = Signal()  # 播放完成信号
    playbackError = Signal(str)  # 播放错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detector = None
        self.video_thread = None
        self.video_path = None
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.is_paused = False

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 视频显示区域
        self.video_viewer = ImageViewer()
        layout.addWidget(self.video_viewer)

        # 控制面板
        control_frame = QFrame()
        control_frame.setMaximumHeight(80)
        control_frame.setStyleSheet("QFrame { border: 1px solid gray; background-color: #f0f0f0; }")
        control_layout = QVBoxLayout(control_frame)

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

        control_layout.addLayout(progress_layout)

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

        control_layout.addLayout(button_layout)
        layout.addWidget(control_frame)

        # 初始状态
        self.set_controls_enabled(False)

    def connect_signals(self):
        """连接信号"""
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)

        # 进度条拖动
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        self.progress_slider.valueChanged.connect(self.on_slider_value_changed)

        # 播放速度
        self.speed_0_5x_btn.clicked.connect(lambda: self.set_playback_speed(0.5))
        self.speed_1x_btn.clicked.connect(lambda: self.set_playback_speed(1.0))
        self.speed_2x_btn.clicked.connect(lambda: self.set_playback_speed(2.0))

    def set_detector(self, detector):
        """设置检测器"""
        self.detector = detector

    def load_video(self, video_path):
        """加载视频"""
        self.video_path = video_path

        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 显示第一帧
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_viewer.set_image(rgb_frame)

            cap.release()

            # 更新UI
            self.progress_slider.setMaximum(self.total_frames - 1)
            self.update_time_display(0, fps)
            self.update_frame_display(0)
            self.set_controls_enabled(True)

            return True

        return False

    def start_detection(self):
        """开始检测播放"""
        if not self.detector or not self.video_path:
            return False

        if self.video_thread and self.video_thread.isRunning():
            self.stop_playback()

        # 创建检测线程
        self.video_thread = VideoDetectionThread(self.detector, self.video_path)
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

    def toggle_play_pause(self):
        """切换播放/暂停"""
        if not self.is_playing:
            # 开始播放
            self.start_detection()
        else:
            # 暂停/继续
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
        self.playbackFinished.emit()

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
        if 0 <= frame_number < self.total_frames:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.seek(frame_number)
            else:
                # 如果没有在播放，直接显示该帧
                self.show_frame(frame_number)

    def show_frame(self, frame_number):
        """显示指定帧"""
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_viewer.set_image(rgb_frame)
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

    def on_slider_pressed(self):
        """进度条按下"""
        if self.is_playing and not self.is_paused:
            self.pause_playback()

    def on_slider_released(self):
        """进度条释放"""
        frame_number = self.progress_slider.value()
        self.seek_to_frame(frame_number)

    def on_slider_value_changed(self, value):
        """进度条值变化"""
        if not self.is_playing or self.is_paused:
            self.show_frame(value)

    def on_frame_ready(self, results, image, current_frame, total_frames):
        """处理新帧"""
        self.video_viewer.set_image(image)
        self.current_frame = current_frame
        self.total_frames = total_frames

        # 更新进度条
        self.progress_slider.setValue(current_frame)
        self.update_frame_display(current_frame)

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
        self.playbackFinished.emit()

    def on_playback_error(self, error_msg):
        """播放错误"""
        self.is_playing = False
        self.is_paused = False
        self.play_pause_btn.setText("播放")
        self.playbackError.emit(error_msg)

    def update_time_display(self, frame_number, fps):
        """更新时间显示"""
        if fps > 0:
            current_seconds = frame_number / fps
            total_seconds = self.total_frames / fps

            current_time = f"{int(current_seconds // 60):02d}:{int(current_seconds % 60):02d}"
            total_time = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"

            self.time_label.setText(f"{current_time} / {total_time}")

    def update_frame_display(self, frame_number):
        """更新帧数显示"""
        self.frame_label.setText(f"{frame_number} / {self.total_frames}")

    def set_controls_enabled(self, enabled):
        """设置控件启用状态"""
        self.play_pause_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.prev_frame_btn.setEnabled(enabled)
        self.next_frame_btn.setEnabled(enabled)
        self.progress_slider.setEnabled(enabled)
        self.speed_0_5x_btn.setEnabled(enabled)
        self.speed_1x_btn.setEnabled(enabled)
        self.speed_2x_btn.setEnabled(enabled)
