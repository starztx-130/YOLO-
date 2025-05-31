# YOLO检测系统主窗口
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QSplitter, QMenuBar, QMenu, QStatusBar, QToolBar,
                               QFileDialog, QMessageBox, QProgressBar, QLabel,
                               QPushButton, QGroupBox, QTextEdit, QTabWidget,
                               QApplication, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QIcon, QPixmap

# 导入自定义组件
from .widgets.image_viewer import ImageViewer, VideoFrameViewer
from .widgets.parameter_panel import ParameterPanel
from .widgets.detection_table import DetectionTable
from .widgets.video_player import VideoPlayer
from .widgets.smart_detection_viewer import SmartDetectionViewer
from .widgets.converter_widget import ConverterWidget
from core.detector import YOLODetector
from core.utils import (check_cuda_availability, validate_model_file,
                       validate_media_file, is_image_file, is_video_file,
                       format_detection_info, create_output_dir)


class DetectionThread(QThread):
    """检测线程"""

    # 信号
    resultReady = Signal(object, np.ndarray)  # 检测结果和图片
    progressUpdate = Signal(int)  # 进度更新
    finished = Signal()  # 完成信号
    error = Signal(str)  # 错误信号

    def __init__(self, detector, source, source_type):
        super().__init__()
        self.detector = detector
        self.source = source
        self.source_type = source_type  # 'image', 'video', 'camera'
        self.is_running = True

    def run(self):
        """运行检测"""
        try:
            if self.source_type == 'image':
                self._detect_image()
            elif self.source_type == 'video':
                self._detect_video()
            elif self.source_type == 'camera':
                self._detect_camera()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        """停止检测"""
        self.is_running = False

    def _detect_image(self):
        # 检测图片
        image = cv2.imread(self.source)
        if image is None:
            self.error.emit("无法读取图片文件")
            return

        results = self.detector.detect_image(image)

        if results and len(results) > 0:
            # 绘制检测结果
            annotated_image = results[0].plot()
            # 转换BGR到RGB
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        else:
            annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.resultReady.emit(results, annotated_image)
        self.progressUpdate.emit(100)

    def _detect_video(self):
        # 检测视频
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error.emit("无法打开视频文件")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 检测当前帧
            results = self.detector.detect_image(frame)

            if results and len(results) > 0:
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            else:
                annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.resultReady.emit(results, annotated_frame)

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            self.progressUpdate.emit(progress)

        cap.release()

    def _detect_camera(self):
        # 检测摄像头
        cap = cv2.VideoCapture(0)  # 默认摄像头
        if not cap.isOpened():
            self.error.emit("无法打开摄像头")
            return

        frame_count = 0
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # 检测当前帧
            results = self.detector.detect_image(frame)

            if results and len(results) > 0:
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            else:
                annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.resultReady.emit(results, annotated_frame)

            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧更新一次进度
                self.progressUpdate.emit(50)  # 摄像头检测显示50%进度

        cap.release()


class MainWindow(QMainWindow):
    # 主窗口类

    def __init__(self):
        super().__init__()

        # 初始化检测器
        self.detector = YOLODetector()
        self.detection_thread = None

        # 当前文件路径
        self.current_file = None
        self.current_results = None
        self.current_original_image = None  # 保存原始图像用于重新绘制

        # 设置窗口
        self.setWindowTitle("YOLO目标检测系统 v2.0.0")
        self.setGeometry(100, 100, 1400, 900)

        # 初始化UI
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        self.connect_signals()

        # 检查CUDA可用性
        self.check_system_info()

    def setup_ui(self):
        # 设置UI界面
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧面板 - 参数设置
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # 模型加载组
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)

        self.load_model_btn = QPushButton("加载模型")
        self.model_info_label = QLabel("未加载模型")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: gray; font-size: 12px;")

        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.model_info_label)
        left_layout.addWidget(model_group)

        # 参数面板
        self.parameter_panel = ParameterPanel()
        left_layout.addWidget(self.parameter_panel)

        # 检测控制组
        control_group = QGroupBox("检测控制")
        control_layout = QVBoxLayout(control_group)

        self.detect_image_btn = QPushButton("检测图片")
        self.detect_video_btn = QPushButton("检测视频")
        self.detect_camera_btn = QPushButton("摄像头检测")
        self.stop_detection_btn = QPushButton("停止检测")

        # 初始状态下禁用检测按钮，直到加载模型
        self.detect_image_btn.setEnabled(False)
        self.detect_video_btn.setEnabled(False)
        self.detect_camera_btn.setEnabled(False)
        self.stop_detection_btn.setEnabled(False)

        control_layout.addWidget(self.detect_image_btn)
        control_layout.addWidget(self.detect_video_btn)
        control_layout.addWidget(self.detect_camera_btn)
        control_layout.addWidget(self.stop_detection_btn)
        left_layout.addWidget(control_group)

        splitter.addWidget(left_panel)

        # 右侧面板 - 显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 智能检测结果标签页（合并图片和视频功能）
        self.smart_viewer = SmartDetectionViewer()
        self.tab_widget.addTab(self.smart_viewer, "检测结果")

        # 检测表格标签页
        self.detection_table = DetectionTable()
        self.tab_widget.addTab(self.detection_table, "实时数据表")

        # 检测信息标签页
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        self.detection_info = QTextEdit()
        self.detection_info.setMaximumHeight(200)
        self.detection_info.setPlaceholderText("检测信息将在这里显示...")
        info_layout.addWidget(QLabel("检测信息:"))
        info_layout.addWidget(self.detection_info)

        # 系统信息
        self.system_info = QTextEdit()
        self.system_info.setMaximumHeight(150)
        self.system_info.setReadOnly(True)
        info_layout.addWidget(QLabel("系统信息:"))
        info_layout.addWidget(self.system_info)

        info_layout.addStretch()
        self.tab_widget.addTab(info_widget, "系统信息")

        # 模型转换器标签页
        self.converter_widget = ConverterWidget()
        self.tab_widget.addTab(self.converter_widget, "模型转换")

        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setSizes([350, 1050])
        main_layout.addWidget(splitter)

    def setup_menu(self):
        # 设置菜单栏
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        open_action = QAction("打开文件", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("保存结果", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)

        export_table_action = QAction("导出检测数据", self)
        export_table_action.setShortcut("Ctrl+E")
        export_table_action.triggered.connect(self.detection_table.export_to_csv)
        file_menu.addAction(export_table_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 模型菜单
        model_menu = menubar.addMenu("模型")

        load_model_action = QAction("加载模型", self)
        load_model_action.triggered.connect(self.load_model)
        model_menu.addAction(load_model_action)

        model_menu.addSeparator()

        convert_model_action = QAction("模型转换", self)
        convert_model_action.setShortcut("Ctrl+M")
        convert_model_action.triggered.connect(self.show_model_converter)
        model_menu.addAction(convert_model_action)



        # 视图菜单
        view_menu = menubar.addMenu("视图")

        zoom_in_action = QAction("放大", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.smart_viewer.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("缩小", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.smart_viewer.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_window_action = QAction("适应窗口", self)
        fit_window_action.setShortcut("Ctrl+0")
        fit_window_action.triggered.connect(self.smart_viewer.fit_to_window)
        view_menu.addAction(fit_window_action)

        reset_zoom_action = QAction("原始大小", self)
        reset_zoom_action.setShortcut("Ctrl+1")
        reset_zoom_action.triggered.connect(self.smart_viewer.reset_zoom)
        view_menu.addAction(reset_zoom_action)

        # 数据菜单
        data_menu = menubar.addMenu("数据")

        clear_table_action = QAction("清空检测表格", self)
        clear_table_action.setShortcut("Ctrl+Delete")
        clear_table_action.triggered.connect(self.detection_table.clear_table)
        data_menu.addAction(clear_table_action)

        # 添加自动清空选项
        self.auto_clear_action = QAction("新检测时自动清空数据表", self)
        self.auto_clear_action.setCheckable(True)
        self.auto_clear_action.setChecked(True)  # 默认开启
        data_menu.addAction(self.auto_clear_action)

        data_menu.addSeparator()

        show_table_action = QAction("显示数据表", self)
        show_table_action.setShortcut("Ctrl+T")
        show_table_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))  # 数据表是第二个标签页
        data_menu.addAction(show_table_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self):
        # 设置工具栏
        toolbar = self.addToolBar("主工具栏")

        # 打开文件
        open_action = QAction("打开", self)
        open_action.setToolTip("打开图片或视频文件")
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        # 加载模型
        model_action = QAction("模型", self)
        model_action.setToolTip("加载YOLO模型")
        model_action.triggered.connect(self.load_model)
        toolbar.addAction(model_action)

        toolbar.addSeparator()

        # 检测按钮
        detect_action = QAction("检测", self)
        detect_action.setToolTip("开始检测")
        detect_action.triggered.connect(self.start_detection)
        toolbar.addAction(detect_action)

        # 停止按钮
        stop_action = QAction("停止", self)
        stop_action.setToolTip("停止检测")
        stop_action.triggered.connect(self.stop_detection)
        toolbar.addAction(stop_action)

    def setup_statusbar(self):
        # 设置状态栏
        self.status_bar = self.statusBar()

        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # 设备信息标签
        self.device_label = QLabel()
        self.status_bar.addPermanentWidget(self.device_label)

    def connect_signals(self):
        # 连接信号
        # 按钮信号
        self.load_model_btn.clicked.connect(self.load_model)
        self.detect_image_btn.clicked.connect(lambda: self.detect_file('image'))
        self.detect_video_btn.clicked.connect(lambda: self.detect_file('video'))
        self.detect_camera_btn.clicked.connect(self.detect_camera)
        self.stop_detection_btn.clicked.connect(self.stop_detection)

        # 参数面板信号
        self.parameter_panel.parametersChanged.connect(self.update_detector_parameters)
        self.parameter_panel.parametersChanged.connect(self.refresh_detection_display)

        # 检测表格信号
        self.detection_table.detectionSelected.connect(self.on_detection_selected)

        # 智能检测查看器信号
        self.smart_viewer.frameChanged.connect(self.on_detection_result)
        self.smart_viewer.progressChanged.connect(self.on_progress_update)
        self.smart_viewer.detectionFinished.connect(self.on_detection_finished)
        self.smart_viewer.detectionError.connect(self.on_detection_error)

    def check_system_info(self):
        # 检查系统信息
        cuda_available, cuda_info = check_cuda_availability()

        system_info = f"系统信息:\n"
        system_info += f"CUDA状态: {cuda_info}\n"
        system_info += f"推荐设备: {'CUDA' if cuda_available else 'CPU'}\n"

        self.system_info.setText(system_info)
        self.device_label.setText(f"设备: {'CUDA' if cuda_available else 'CPU'}")

    def load_model(self):
        """加载模型"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "选择YOLO模型文件",
            "",
            "模型文件 (*.pt *.onnx);;PyTorch模型 (*.pt);;ONNX模型 (*.onnx)"
        )

        if file_path:
            self.status_label.setText("正在加载模型...")
            QApplication.processEvents()

            # 获取设备设置
            params = self.parameter_panel.get_parameters()
            device = params['device']
            if device == 'auto':
                device = 'cuda' if check_cuda_availability()[0] else 'cpu'

            # 加载模型
            success = self.detector.load_model(file_path, device)

            if success:
                model_info = self.detector.get_model_info()
                info_text = f"模型: {Path(file_path).name}\n"
                info_text += f"类型: {model_info.get('model_type', 'unknown').upper()}\n"
                info_text += f"任务: {model_info.get('task_type', 'unknown').upper()}\n"
                info_text += f"设备: {model_info.get('device', 'unknown').upper()}"

                self.model_info_label.setText(info_text)
                self.model_info_label.setStyleSheet("color: green; font-size: 12px;")

                # 更新类别列表
                if 'classes' in model_info:
                    self.parameter_panel.set_classes(model_info['classes'])

                self.status_label.setText("模型加载成功")

                # 启用检测按钮
                self.detect_image_btn.setEnabled(True)
                self.detect_video_btn.setEnabled(True)
                self.detect_camera_btn.setEnabled(True)

            else:
                self.model_info_label.setText("模型加载失败")
                self.model_info_label.setStyleSheet("color: red; font-size: 12px;")
                self.status_label.setText("模型加载失败")
                QMessageBox.warning(self, "错误", "模型加载失败，请检查文件格式和路径")

    def open_file(self):
        """打开文件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "选择图片或视频文件",
            "",
            "媒体文件 (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov);;图片文件 (*.jpg *.jpeg *.png *.bmp);;视频文件 (*.mp4 *.avi *.mov)"
        )

        if file_path and validate_media_file(file_path):
            self.current_file = file_path

            if is_image_file(file_path):
                # 显示图片
                image = cv2.imread(file_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.image_viewer.set_image(rgb_image)
                self.status_label.setText(f"已加载图片: {Path(file_path).name}")
            else:
                # 显示视频第一帧
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.image_viewer.set_image(rgb_frame)
                cap.release()
                self.status_label.setText(f"已加载视频: {Path(file_path).name}")

    def detect_file(self, file_type):
        """检测文件"""
        if not self.detector.is_loaded():
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        if file_type == 'image':
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self,
                "选择图片文件",
                "",
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
            )
        else:  # video
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self,
                "选择视频文件",
                "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)"
            )

        if file_path and os.path.exists(file_path):
            self.current_file = file_path
            self.start_smart_detection(file_path)

    def detect_camera(self):
        """摄像头检测"""
        if not self.detector.is_loaded():
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        self.start_smart_detection(0)

    def start_smart_detection(self, source):
        """开始智能检测（自动识别源类型）"""
        # 更新检测器参数
        self.update_detector_parameters()

        # 设置检测器到智能查看器
        self.smart_viewer.set_detector(self.detector)

        # 加载源
        if self.smart_viewer.load_source(source):
            # 根据用户设置决定是否清空检测表格
            source_type = self.smart_viewer.get_source_type()
            if self.auto_clear_action.isChecked():
                self.detection_table.clear_table()

            # 切换到检测结果标签页
            self.tab_widget.setCurrentIndex(0)

            # 更新UI状态
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.stop_detection_btn.setEnabled(True)
            self.detect_image_btn.setEnabled(False)
            self.detect_video_btn.setEnabled(False)
            self.detect_camera_btn.setEnabled(False)

            # 根据源类型显示不同状态
            if source_type == 'image':
                self.status_label.setText("图片已加载，点击开始检测")
                # 图片模式直接开始检测
                self.smart_viewer.start_detection()
            elif source_type == 'video':
                self.status_label.setText("视频已加载，点击播放开始检测")
            elif source_type == 'camera':
                self.status_label.setText("摄像头已连接，点击播放开始检测")
        else:
            QMessageBox.warning(self, "错误", "无法加载检测源")

    def start_detection(self):
        """开始检测当前文件"""
        if not self.detector.is_loaded():
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        if not self.current_file:
            QMessageBox.warning(self, "警告", "请先选择文件")
            return

        if is_image_file(self.current_file):
            self.start_detection_with_source(self.current_file, 'image')
        elif is_video_file(self.current_file):
            self.start_detection_with_source(self.current_file, 'video')

    def start_detection_with_source(self, source, source_type):
        """使用指定源开始检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.stop_detection()

        # 更新检测器参数
        self.update_detector_parameters()

        # 根据用户设置决定是否清空检测表格
        if self.auto_clear_action.isChecked():
            self.detection_table.clear_table()

        # 创建检测线程
        self.detection_thread = DetectionThread(self.detector, source, source_type)
        self.detection_thread.resultReady.connect(self.on_detection_result)
        self.detection_thread.progressUpdate.connect(self.on_progress_update)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)

        # 更新UI状态
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.stop_detection_btn.setEnabled(True)
        self.detect_image_btn.setEnabled(False)
        self.detect_video_btn.setEnabled(False)
        self.detect_camera_btn.setEnabled(False)

        self.status_label.setText(f"正在检测{source_type}...")

        # 启动检测
        self.detection_thread.start()



    def stop_detection(self):
        """停止检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait(3000)  # 等待3秒

        # 停止智能查看器
        self.smart_viewer.stop_detection()

        self.on_detection_finished()

    def update_detector_parameters(self):
        """更新检测器参数"""
        if not self.detector.is_loaded():
            return

        params = self.parameter_panel.get_parameters()

        # 设置设备
        device = params['device']
        if device == 'auto':
            device = 'cuda' if check_cuda_availability()[0] else 'cpu'

        # 更新检测参数
        self.detector.set_parameters(
            conf=params['conf_threshold'],
            iou=params['iou_threshold'],
            max_det=params['max_det'],
            classes=params['classes'],
            show_labels=params['show_labels'],
            show_conf=params['show_conf'],
            show_boxes=params['show_boxes'],
            show_masks=params['show_masks'],
            show_keypoints=params['show_keypoints']
        )

    def on_detection_result(self, results, image):
        """处理检测结果"""
        self.current_results = results
        # 注意：这里的image已经是处理后的RGB图像，我们需要从智能查看器获取原始图像

        # 更新检测信息
        info_text = format_detection_info(results)
        self.detection_info.setText(info_text)

        # 添加检测结果到表格
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 精确到毫秒

        # 获取当前任务类型
        task_type = getattr(self.detector, 'task_type', 'detect')
        self.detection_table.add_detections(results, current_time, task_type)

        # 不强制切换标签页，让用户自由查看任何标签页

    def refresh_detection_display(self):
        """刷新检测结果显示（当显示选项改变时）"""
        # 只刷新显示效果，不更新数据表
        if hasattr(self.smart_viewer, 'refresh_display'):
            self.smart_viewer.refresh_display(update_table=False)

    def on_progress_update(self, progress):
        """更新进度"""
        self.progress_bar.setValue(progress)

    def on_detection_finished(self):
        """检测完成"""
        self.progress_bar.setVisible(False)
        self.stop_detection_btn.setEnabled(False)
        self.detect_image_btn.setEnabled(True)
        self.detect_video_btn.setEnabled(True)
        self.detect_camera_btn.setEnabled(True)
        self.status_label.setText("检测完成")

    def on_detection_error(self, error_msg):
        """处理检测错误"""
        QMessageBox.critical(self, "检测错误", f"检测过程中发生错误:\n{error_msg}")
        self.status_label.setText("检测失败")

    def on_detection_selected(self, detection_id):
        """处理检测结果选中事件"""
        # 这里可以添加高亮显示选中检测结果的逻辑
        # 例如在图像上绘制特殊边框等
        self.status_label.setText(f"已选中检测结果 ID: {detection_id}")

        # 切换到图像显示标签页以查看选中的检测结果
        self.tab_widget.setCurrentIndex(0)





    def save_result(self):
        """保存检测结果"""
        current_image = self.smart_viewer.get_current_image()
        if current_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "保存检测结果",
            "detection_result.jpg",
            "图片文件 (*.jpg *.png *.bmp)"
        )

        if file_path:
            # 转换RGB到BGR用于保存
            bgr_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(file_path, bgr_image)

            if success:
                self.status_label.setText(f"结果已保存: {Path(file_path).name}")
                QMessageBox.information(self, "成功", "检测结果保存成功")
            else:
                QMessageBox.warning(self, "错误", "保存失败")

    def show_model_converter(self):
        """显示模型转换器"""
        # 切换到模型转换标签页
        self.tab_widget.setCurrentWidget(self.converter_widget)
        self.status_label.setText("模型转换器已打开")

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            "YOLO目标检测系统\n\n"
            "基于PySide6和Ultralytics开发\n"
            "支持图片、视频和实时摄像头检测\n"
            "支持PyTorch (.pt) 和 ONNX (.onnx) 模型\n"
            "支持模型格式转换功能\n\n"
            "版本: 2.0.0"
        )

    def closeEvent(self, event):
        """关闭事件"""
        if self.detection_thread and self.detection_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "检测正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.stop_detection()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


