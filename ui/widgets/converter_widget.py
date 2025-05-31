"""
模型转换器界面组件

提供YOLO模型格式转换的图形界面。

主要功能：
- 模型选择和加载
- 转换格式选择
- 参数配置界面
- 转换进度显示
- 结果展示和验证
"""
import os
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QGroupBox, QLabel, QPushButton, QComboBox,
                               QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
                               QTextEdit, QProgressBar, QFileDialog, QMessageBox,
                               QTabWidget, QTableWidget, QTableWidgetItem,
                               QSplitter, QFrame)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon

from core.model_converter import ModelConverter


class ConversionThread(QThread):
    """模型转换线程"""
    
    progress_updated = Signal(int, str)  # 进度, 状态信息
    conversion_finished = Signal(bool, str, str)  # 成功, 输出路径, 消息
    
    def __init__(self, converter, format_key, output_path=None):
        super().__init__()
        self.converter = converter
        self.format_key = format_key
        self.output_path = output_path
        
    def run(self):
        """执行转换"""
        def progress_callback(progress, message):
            self.progress_updated.emit(progress, message)
            
        success, output_path, message = self.converter.convert_model(
            self.format_key, self.output_path, progress_callback
        )
        
        self.conversion_finished.emit(success, output_path, message)


class ConverterWidget(QWidget):
    """模型转换器主界面"""
    
    def __init__(self):
        super().__init__()
        self.converter = ModelConverter()
        self.conversion_thread = None
        
        self.setup_ui()
        self.connect_signals()
        self.update_ui_state()
        
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # 右侧信息面板
        right_panel = self.create_info_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        # 模型路径选择
        path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择要转换的PyTorch模型文件 (.pt)")
        self.browse_model_btn = QPushButton("浏览...")
        self.browse_model_btn.setFixedWidth(80)
        
        path_layout.addWidget(QLabel("模型文件:"))
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(self.browse_model_btn)
        model_layout.addLayout(path_layout)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setEnabled(False)
        model_layout.addWidget(self.load_model_btn)
        
        layout.addWidget(model_group)
        
        # 转换设置组
        convert_group = QGroupBox("转换设置")
        convert_layout = QVBoxLayout(convert_group)
        
        # 目标格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("目标格式:"))
        self.format_combo = QComboBox()
        self.format_combo.setEnabled(False)
        format_layout.addWidget(self.format_combo)
        convert_layout.addLayout(format_layout)
        
        # 格式描述
        self.format_desc_label = QLabel()
        self.format_desc_label.setWordWrap(True)
        self.format_desc_label.setStyleSheet("color: #666; font-size: 12px;")
        convert_layout.addWidget(self.format_desc_label)
        
        # 参数配置
        params_group = QGroupBox("参数配置")
        params_layout = QGridLayout(params_group)
        
        # 图像尺寸
        params_layout.addWidget(QLabel("图像尺寸:"), 0, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        params_layout.addWidget(self.imgsz_spin, 0, 1)
        
        # 批处理大小
        params_layout.addWidget(QLabel("批处理大小:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(1)
        params_layout.addWidget(self.batch_spin, 1, 1)
        
        # FP16量化
        self.half_check = QCheckBox("启用FP16量化")
        params_layout.addWidget(self.half_check, 2, 0, 1, 2)
        
        # INT8量化
        self.int8_check = QCheckBox("启用INT8量化")
        params_layout.addWidget(self.int8_check, 3, 0, 1, 2)
        
        # 动态输入尺寸
        self.dynamic_check = QCheckBox("动态输入尺寸")
        params_layout.addWidget(self.dynamic_check, 4, 0, 1, 2)
        
        # 简化模型（ONNX）
        self.simplify_check = QCheckBox("简化模型（ONNX）")
        self.simplify_check.setChecked(True)
        params_layout.addWidget(self.simplify_check, 5, 0, 1, 2)
        
        convert_layout.addWidget(params_group)
        
        # 输出路径
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("输出路径（留空自动生成）")
        self.browse_output_btn = QPushButton("浏览...")
        self.browse_output_btn.setFixedWidth(80)
        
        output_layout.addWidget(QLabel("输出路径:"))
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.browse_output_btn)
        convert_layout.addLayout(output_layout)
        
        layout.addWidget(convert_group)
        
        # 转换控制
        control_group = QGroupBox("转换控制")
        control_layout = QVBoxLayout(control_group)
        
        # 转换按钮
        self.convert_btn = QPushButton("开始转换")
        self.convert_btn.setEnabled(False)
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.convert_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("请选择并加载模型")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)
        
        layout.addWidget(control_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
        
    def create_info_panel(self) -> QWidget:
        """创建信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 模型信息标签页
        model_info_widget = QWidget()
        model_info_layout = QVBoxLayout(model_info_widget)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(200)
        model_info_layout.addWidget(QLabel("模型信息:"))
        model_info_layout.addWidget(self.model_info_text)
        
        tab_widget.addTab(model_info_widget, "模型信息")
        
        # 格式支持标签页
        format_info_widget = QWidget()
        format_info_layout = QVBoxLayout(format_info_widget)
        
        self.format_table = QTableWidget()
        self.format_table.setColumnCount(3)
        self.format_table.setHorizontalHeaderLabels(["格式", "扩展名", "描述"])
        self.format_table.horizontalHeader().setStretchLastSection(True)
        format_info_layout.addWidget(QLabel("支持的转换格式:"))
        format_info_layout.addWidget(self.format_table)
        
        tab_widget.addTab(format_info_widget, "支持格式")
        
        # 转换日志标签页
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(QLabel("转换日志:"))
        log_layout.addWidget(self.log_text)
        
        tab_widget.addTab(log_widget, "转换日志")
        
        layout.addWidget(tab_widget)
        
        # 初始化格式表格
        self.populate_format_table()

        return panel

    def connect_signals(self):
        """连接信号"""
        self.model_path_edit.textChanged.connect(self.on_model_path_changed)
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        self.load_model_btn.clicked.connect(self.load_model)
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        self.browse_output_btn.clicked.connect(self.browse_output_path)
        self.convert_btn.clicked.connect(self.start_conversion)

        # 参数变化信号
        self.imgsz_spin.valueChanged.connect(self.update_export_params)
        self.batch_spin.valueChanged.connect(self.update_export_params)
        self.half_check.toggled.connect(self.update_export_params)
        self.int8_check.toggled.connect(self.update_export_params)
        self.dynamic_check.toggled.connect(self.update_export_params)
        self.simplify_check.toggled.connect(self.update_export_params)

    def populate_format_table(self):
        """填充格式表格"""
        formats = self.converter.get_supported_formats()
        self.format_table.setRowCount(len(formats))

        for row, (key, info) in enumerate(formats.items()):
            self.format_table.setItem(row, 0, QTableWidgetItem(info['name']))
            self.format_table.setItem(row, 1, QTableWidgetItem(info['extension']))
            self.format_table.setItem(row, 2, QTableWidgetItem(info['description']))

    def update_format_combo(self):
        """更新格式下拉框"""
        self.format_combo.clear()
        formats = self.converter.get_supported_formats()

        for key, info in formats.items():
            self.format_combo.addItem(f"{info['name']} ({info['extension']})", key)

    def on_model_path_changed(self):
        """模型路径改变"""
        path = self.model_path_edit.text().strip()
        self.load_model_btn.setEnabled(bool(path and os.path.exists(path)))

    def browse_model_file(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型文件", "", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_output_path(self):
        """浏览输出路径"""
        if self.format_combo.currentData():
            format_key = self.format_combo.currentData()
            format_info = self.converter.get_supported_formats()[format_key]

            if format_info['extension'].endswith('/'):
                # 目录格式
                dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
                if dir_path:
                    self.output_path_edit.setText(dir_path)
            else:
                # 文件格式
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "保存转换后的模型", "",
                    f"{format_info['name']} (*{format_info['extension']})"
                )
                if file_path:
                    self.output_path_edit.setText(file_path)

    def load_model(self):
        """加载模型"""
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return

        self.status_label.setText("正在加载模型...")
        self.load_model_btn.setEnabled(False)

        # 加载模型
        success = self.converter.load_model(model_path)

        if success:
            self.status_label.setText("模型加载成功")
            self.update_format_combo()
            self.format_combo.setEnabled(True)
            self.update_model_info()
            self.log_message(f"成功加载模型: {model_path}")
        else:
            self.status_label.setText("模型加载失败")
            QMessageBox.critical(self, "错误", "模型加载失败，请检查文件格式")
            self.log_message(f"模型加载失败: {model_path}")

        self.load_model_btn.setEnabled(True)
        self.update_ui_state()

    def update_model_info(self):
        """更新模型信息"""
        info = self.converter.get_model_info()
        if info:
            info_text = f"""
模型名称: {info.get('model_name', 'Unknown')}
模型路径: {info.get('model_path', 'Unknown')}
模型大小: {self.format_file_size(info.get('model_size', 0))}
任务类型: {info.get('task', 'Unknown')}
类别数量: {info.get('num_classes', 0)}
参数数量: {info.get('parameters', 0):,}
设备: {info.get('device', 'Unknown')}
"""
            self.model_info_text.setText(info_text.strip())
        else:
            self.model_info_text.setText("无法获取模型信息")

    def format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"

    def on_format_changed(self):
        """格式改变"""
        format_key = self.format_combo.currentData()
        if format_key:
            format_info = self.converter.get_supported_formats()[format_key]
            self.format_desc_label.setText(format_info['description'])

            # 更新时间估算
            time_estimate = self.converter.estimate_conversion_time(format_key)
            self.status_label.setText(f"预计转换时间: {time_estimate}")

            # 更新输出路径建议
            if self.converter.model_path:
                suggested_path = self.converter.get_output_path(format_key)
                if not self.output_path_edit.text():
                    self.output_path_edit.setText(suggested_path)

        self.update_ui_state()

    def update_export_params(self):
        """更新导出参数"""
        params = {
            'imgsz': self.imgsz_spin.value(),
            'batch': self.batch_spin.value(),
            'half': self.half_check.isChecked(),
            'int8': self.int8_check.isChecked(),
            'dynamic': self.dynamic_check.isChecked(),
            'simplify': self.simplify_check.isChecked(),
        }
        self.converter.set_export_params(**params)

    def update_ui_state(self):
        """更新界面状态"""
        has_model = self.converter.model is not None
        has_format = bool(self.format_combo.currentData())

        self.convert_btn.setEnabled(has_model and has_format)

        # 验证当前格式参数
        if has_model and has_format:
            format_key = self.format_combo.currentData()
            is_valid, error_msg = self.converter.validate_export_params(format_key)
            if not is_valid:
                self.convert_btn.setEnabled(False)
                self.status_label.setText(f"参数错误: {error_msg}")

    def start_conversion(self):
        """开始转换"""
        format_key = self.format_combo.currentData()
        output_path = self.output_path_edit.text().strip() or None

        if not format_key:
            QMessageBox.warning(self, "警告", "请选择转换格式")
            return

        # 更新导出参数
        self.update_export_params()

        # 验证参数
        is_valid, error_msg = self.converter.validate_export_params(format_key)
        if not is_valid:
            QMessageBox.warning(self, "参数错误", error_msg)
            return

        # 开始转换
        self.convert_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.conversion_thread = ConversionThread(self.converter, format_key, output_path)
        self.conversion_thread.progress_updated.connect(self.on_progress_updated)
        self.conversion_thread.conversion_finished.connect(self.on_conversion_finished)
        self.conversion_thread.start()

        self.log_message(f"开始转换为 {self.converter.get_supported_formats()[format_key]['name']} 格式...")

    def on_progress_updated(self, progress: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        self.log_message(f"[{progress}%] {message}")

    def on_conversion_finished(self, success: bool, output_path: str, message: str):
        """转换完成"""
        self.progress_bar.setVisible(False)
        self.convert_btn.setEnabled(True)

        if success:
            self.status_label.setText("转换完成")
            self.log_message(f"转换成功: {output_path}")
            self.log_message(message)

            # 显示成功对话框
            reply = QMessageBox.information(
                self, "转换成功",
                f"模型转换完成！\n\n输出路径: {output_path}\n\n{message}\n\n是否打开输出目录？",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.open_output_directory(output_path)
        else:
            self.status_label.setText("转换失败")
            self.log_message(f"转换失败: {message}")
            QMessageBox.critical(self, "转换失败", message)

        self.update_ui_state()

    def open_output_directory(self, file_path: str):
        """打开输出目录"""
        try:
            import subprocess
            import platform

            if os.path.isfile(file_path):
                directory = os.path.dirname(file_path)
            else:
                directory = file_path

            if platform.system() == "Windows":
                subprocess.run(["explorer", directory])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", directory])
            else:  # Linux
                subprocess.run(["xdg-open", directory])
        except Exception as e:
            self.log_message(f"无法打开目录: {e}")

    def log_message(self, message: str):
        """记录日志消息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
