"""
检测结果动态表格组件
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                               QTableWidgetItem, QHeaderView, QPushButton, QLabel,
                               QComboBox, QCheckBox, QGroupBox,
                               QAbstractItemView, QMenu)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QAction
import time


class DetectionTable(QWidget):
    """检测结果动态表格组件"""

    # 信号
    detectionSelected = Signal(int)  # 选中检测结果信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detection_data = []
        self.detection_id_counter = 1
        self.current_task_type = 'detect'  # 当前任务类型

        # 不同任务类型的表头定义
        self.task_headers = {
            'detect': [
                "ID", "类别", "置信度", "X1", "Y1", "X2", "Y2",
                "宽度", "高度", "面积", "中心X", "中心Y", "时间戳"
            ],
            'segment': [
                "ID", "类别", "置信度", "X1", "Y1", "X2", "Y2",
                "宽度", "高度", "面积", "掩码面积", "掩码像素", "时间戳"
            ],
            'pose': [
                "ID", "人体ID", "关键点数", "可见点数", "姿态置信度",
                "X1", "Y1", "X2", "Y2", "中心X", "中心Y", "时间戳"
            ],
            'classify': [
                "ID", "预测类别", "置信度", "Top2", "Top3", "Top4", "Top5", "时间戳"
            ],
            'obb': [
                "ID", "类别", "置信度", "P1_X", "P1_Y", "P2_X", "P2_Y",
                "P3_X", "P3_Y", "P4_X", "P4_Y", "旋转角度", "面积", "时间戳"
            ]
        }

        self.headers = self.task_headers[self.current_task_type]
        self.setup_ui()
        self.connect_signals()

        # 自动刷新定时器
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_statistics)
        self.refresh_timer.start(1000)  # 每秒更新一次统计信息

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 控制面板
        control_panel = QGroupBox("表格控制")
        control_layout = QHBoxLayout(control_panel)

        # 清空按钮
        self.clear_btn = QPushButton("清空表格")
        control_layout.addWidget(self.clear_btn)

        # 导出按钮
        self.export_btn = QPushButton("导出CSV")
        control_layout.addWidget(self.export_btn)

        # 自动滚动复选框
        self.auto_scroll_cb = QCheckBox("自动滚动")
        self.auto_scroll_cb.setChecked(True)
        control_layout.addWidget(self.auto_scroll_cb)

        # 最大行数设置
        control_layout.addWidget(QLabel("最大行数:"))
        self.max_rows_combo = QComboBox()
        self.max_rows_combo.addItems(["100", "500", "1000", "无限制"])
        self.max_rows_combo.setCurrentText("500")
        control_layout.addWidget(self.max_rows_combo)

        control_layout.addStretch()
        layout.addWidget(control_panel)

        # 统计信息面板
        stats_panel = QGroupBox("实时统计")
        stats_layout = QHBoxLayout(stats_panel)

        self.total_label = QLabel("总数: 0")
        self.avg_conf_label = QLabel("平均置信度: 0.000")
        self.classes_label = QLabel("类别: 0")

        stats_layout.addWidget(self.total_label)
        stats_layout.addWidget(self.avg_conf_label)
        stats_layout.addWidget(self.classes_label)
        stats_layout.addStretch()

        layout.addWidget(stats_panel)

        # 表格
        self.table = QTableWidget()

        # 设置表格列数和表头
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)

        # 设置表格属性
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)

        # 设置列宽
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)

        # 设置初始列宽
        self._setup_column_widths()

        layout.addWidget(self.table)

        # 创建右键菜单
        self.setup_context_menu()

    def setup_context_menu(self):
        """设置右键菜单"""
        self.context_menu = QMenu(self)

        # 复制行动作
        copy_action = QAction("复制行", self)
        copy_action.triggered.connect(self.copy_selected_row)
        self.context_menu.addAction(copy_action)

        # 删除行动作
        delete_action = QAction("删除行", self)
        delete_action.triggered.connect(self.delete_selected_row)
        self.context_menu.addAction(delete_action)

        self.context_menu.addSeparator()

        # 高亮显示动作
        highlight_action = QAction("在图像中高亮", self)
        highlight_action.triggered.connect(self.highlight_detection)
        self.context_menu.addAction(highlight_action)

    def connect_signals(self):
        """连接信号"""
        self.clear_btn.clicked.connect(self.clear_table)
        self.export_btn.clicked.connect(self.export_to_csv)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.cellClicked.connect(self.on_cell_clicked)

    def set_task_type(self, task_type: str):
        """设置任务类型并更新表格结构"""
        if task_type not in self.task_headers:
            task_type = 'detect'  # 默认为检测任务

        if task_type != self.current_task_type:
            self.current_task_type = task_type
            self.headers = self.task_headers[task_type]

            # 清空现有数据
            self.clear_table()

            # 更新表格结构
            self.table.setColumnCount(len(self.headers))
            self.table.setHorizontalHeaderLabels(self.headers)

            # 重新设置列宽
            self._setup_column_widths()

    def _setup_column_widths(self):
        """根据任务类型设置列宽"""
        column_widths = {
            'detect': [50, 80, 80, 60, 60, 60, 60, 60, 60, 80, 60, 60, 80],
            'segment': [50, 80, 80, 60, 60, 60, 60, 60, 60, 80, 80, 80, 80],
            'pose': [50, 60, 80, 80, 80, 60, 60, 60, 60, 60, 60, 80],
            'classify': [50, 100, 80, 80, 80, 80, 80, 80],
            'obb': [50, 80, 80, 60, 60, 60, 60, 60, 60, 60, 60, 80, 80, 80]
        }

        widths = column_widths.get(self.current_task_type, column_widths['detect'])
        for i, width in enumerate(widths):
            if i < len(self.headers):
                self.table.setColumnWidth(i, width)

    def add_detections(self, results, frame_time=None, task_type=None):
        """添加检测结果，支持多任务类型"""
        if not results or len(results) == 0:
            return

        result = results[0]

        # 自动检测任务类型
        if task_type is None:
            task_type = self._detect_task_type(result)

        # 如果任务类型改变，更新表格结构
        if task_type != self.current_task_type:
            self.set_task_type(task_type)

        if frame_time is None:
            import datetime
            frame_time = datetime.datetime.now().strftime("%H:%M:%S")

        # 检查最大行数限制
        self._check_max_rows()

        # 根据任务类型添加数据
        if task_type == 'detect':
            self._add_detection_data(result, frame_time)
        elif task_type == 'segment':
            self._add_segmentation_data(result, frame_time)
        elif task_type == 'pose':
            self._add_pose_data(result, frame_time)
        elif task_type == 'classify':
            self._add_classification_data(result, frame_time)
        elif task_type == 'obb':
            self._add_obb_data(result, frame_time)

        # 自动滚动到底部
        if self.auto_scroll_cb.isChecked():
            self.table.scrollToBottom()

    def _detect_task_type(self, result):
        """检测任务类型"""
        if hasattr(result, 'masks') and result.masks is not None:
            return 'segment'
        elif hasattr(result, 'keypoints') and result.keypoints is not None:
            return 'pose'
        elif hasattr(result, 'probs') and result.probs is not None:
            return 'classify'
        elif hasattr(result, 'obb') and result.obb is not None:
            return 'obb'
        else:
            return 'detect'

    def _check_max_rows(self):
        """检查最大行数限制"""
        max_rows_text = self.max_rows_combo.currentText()
        if max_rows_text != "无限制":
            max_rows = int(max_rows_text)
            current_rows = self.table.rowCount()

            # 如果超过限制，删除旧行
            if current_rows >= max_rows:
                rows_to_remove = min(50, current_rows - max_rows + 10)
                for _ in range(rows_to_remove):
                    self.table.removeRow(0)
                    if self.detection_data:
                        self.detection_data.pop(0)

    def clear_table(self):
        """清空表格"""
        self.table.setRowCount(0)
        self.detection_data.clear()
        self.detection_id_counter = 1
        self.update_statistics()

    def _add_detection_data(self, result, frame_time):
        """添加目标检测数据"""
        if not hasattr(result, 'boxes') or result.boxes is None:
            return

        for box in result.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 计算其他参数
            width = x2 - x1
            height = y2 - y1
            area = width * height
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 获取类别和置信度
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())

            # 创建行数据
            row_data = [
                self.detection_id_counter,  # ID
                class_name,                 # 类别
                f"{confidence:.3f}",        # 置信度
                f"{x1:.1f}",               # X1
                f"{y1:.1f}",               # Y1
                f"{x2:.1f}",               # X2
                f"{y2:.1f}",               # Y2
                f"{width:.1f}",            # 宽度
                f"{height:.1f}",           # 高度
                f"{area:.0f}",             # 面积
                f"{center_x:.1f}",         # 中心X
                f"{center_y:.1f}",         # 中心Y
                frame_time                 # 时间戳
            ]

            self._add_row_to_table(row_data, confidence)

    def _add_segmentation_data(self, result, frame_time):
        """添加实例分割数据"""
        import numpy as np

        if not hasattr(result, 'boxes') or result.boxes is None:
            return

        masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') and result.masks is not None else None

        for i, box in enumerate(result.boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 计算基本参数
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # 计算掩码参数
            mask_area = 0
            mask_pixels = 0
            if masks is not None and i < len(masks):
                try:
                    mask = masks[i]
                    mask_pixels = int(np.sum(mask > 0.5))
                    mask_area = mask_pixels  # 像素面积
                except Exception as e:
                    print(f"计算掩码参数时出错: {e}")
                    mask_area = 0
                    mask_pixels = 0

            # 获取类别和置信度
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())

            # 创建行数据
            row_data = [
                self.detection_id_counter,  # ID
                class_name,                 # 类别
                f"{confidence:.3f}",        # 置信度
                f"{x1:.1f}",               # X1
                f"{y1:.1f}",               # Y1
                f"{x2:.1f}",               # X2
                f"{y2:.1f}",               # Y2
                f"{width:.1f}",            # 宽度
                f"{height:.1f}",           # 高度
                f"{area:.0f}",             # 面积
                f"{mask_area:.0f}",        # 掩码面积
                f"{mask_pixels}",          # 掩码像素
                frame_time                 # 时间戳
            ]

            self._add_row_to_table(row_data, confidence)

    def _add_pose_data(self, result, frame_time):
        """添加姿态估计数据"""
        import numpy as np

        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return

        keypoints = result.keypoints.data.cpu().numpy()
        boxes = result.boxes if hasattr(result, 'boxes') and result.boxes is not None else None

        for i, person_kpts in enumerate(keypoints):
            # 计算关键点统计
            total_keypoints = len(person_kpts)
            visible_keypoints = np.sum(person_kpts[:, 2] > 0.5)  # 置信度 > 0.5的关键点
            pose_confidence = np.mean(person_kpts[:, 2])  # 平均关键点置信度

            # 获取边界框（如果有的话）
            if boxes is not None and hasattr(boxes, '__len__') and i < len(boxes):
                try:
                    if hasattr(boxes, 'xyxy'):
                        # 标准的boxes对象
                        box_data = boxes.xyxy[i] if hasattr(boxes.xyxy, '__getitem__') else boxes[i].xyxy[0]
                        x1, y1, x2, y2 = box_data.cpu().numpy()
                    else:
                        # 列表形式的boxes
                        x1, y1, x2, y2 = boxes[i].xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                except:
                    # 如果获取边界框失败，从关键点计算
                    valid_kpts = person_kpts[person_kpts[:, 2] > 0.5]
                    if len(valid_kpts) > 0:
                        x1, y1 = np.min(valid_kpts[:, :2], axis=0)
                        x2, y2 = np.max(valid_kpts[:, :2], axis=0)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                    else:
                        x1 = y1 = x2 = y2 = center_x = center_y = 0
            else:
                # 从关键点计算边界框
                valid_kpts = person_kpts[person_kpts[:, 2] > 0.5]
                if len(valid_kpts) > 0:
                    x1, y1 = np.min(valid_kpts[:, :2], axis=0)
                    x2, y2 = np.max(valid_kpts[:, :2], axis=0)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                else:
                    x1 = y1 = x2 = y2 = center_x = center_y = 0

            # 创建行数据
            row_data = [
                self.detection_id_counter,      # ID
                f"人体{i+1}",                   # 人体ID
                f"{total_keypoints}",           # 关键点数
                f"{visible_keypoints}",         # 可见点数
                f"{pose_confidence:.3f}",       # 姿态置信度
                f"{x1:.1f}",                   # X1
                f"{y1:.1f}",                   # Y1
                f"{x2:.1f}",                   # X2
                f"{y2:.1f}",                   # Y2
                f"{center_x:.1f}",             # 中心X
                f"{center_y:.1f}",             # 中心Y
                frame_time                     # 时间戳
            ]

            self._add_row_to_table(row_data, pose_confidence)

    def _add_classification_data(self, result, frame_time):
        """添加分类数据"""
        import numpy as np

        if not hasattr(result, 'probs') or result.probs is None:
            return

        probs = result.probs.data.cpu().numpy()
        names = result.names if hasattr(result, 'names') else {}

        # 获取top-5预测
        top5_indices = np.argsort(probs)[-5:][::-1]

        # 创建行数据
        row_data = [self.detection_id_counter]  # ID

        for i, idx in enumerate(top5_indices):
            if i == 0:
                # 第一个是预测类别
                class_name = names.get(idx, f"类别{idx}")
                confidence = probs[idx]
                row_data.extend([class_name, f"{confidence:.3f}"])
            else:
                # 其他是top2-5
                class_name = names.get(idx, f"类别{idx}")
                confidence = probs[idx]
                row_data.append(f"{class_name}({confidence:.3f})")

        # 补齐到8列（如果top5不足）
        while len(row_data) < 7:
            row_data.append("-")

        row_data.append(frame_time)  # 时间戳

        self._add_row_to_table(row_data, probs[top5_indices[0]])

    def _add_obb_data(self, result, frame_time):
        """添加定向边界框数据"""
        import numpy as np

        if not hasattr(result, 'obb') or result.obb is None:
            return

        obb = result.obb
        if not hasattr(obb, 'xyxyxyxy'):
            return

        xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
        conf = obb.conf.cpu().numpy() if hasattr(obb, 'conf') else None
        cls = obb.cls.cpu().numpy() if hasattr(obb, 'cls') else None

        for i in range(len(xyxyxyxy)):
            # 获取四个顶点坐标
            points = xyxyxyxy[i].reshape(4, 2)

            # 计算旋转角度（基于第一条边的角度）
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            angle = np.degrees(np.arctan2(dy, dx))

            # 计算面积（使用叉积公式）
            def polygon_area(vertices):
                n = len(vertices)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += vertices[i][0] * vertices[j][1]
                    area -= vertices[j][0] * vertices[i][1]
                return abs(area) / 2.0

            area = polygon_area(points)

            # 获取类别和置信度
            if cls is not None and conf is not None:
                class_id = int(cls[i])
                class_name = result.names[class_id] if hasattr(result, 'names') else f'class_{class_id}'
                confidence = float(conf[i])
            else:
                class_name = 'unknown'
                confidence = 0.0

            # 创建行数据
            row_data = [
                self.detection_id_counter,  # ID
                class_name,                 # 类别
                f"{confidence:.3f}",        # 置信度
                f"{points[0][0]:.1f}",     # P1_X
                f"{points[0][1]:.1f}",     # P1_Y
                f"{points[1][0]:.1f}",     # P2_X
                f"{points[1][1]:.1f}",     # P2_Y
                f"{points[2][0]:.1f}",     # P3_X
                f"{points[2][1]:.1f}",     # P3_Y
                f"{points[3][0]:.1f}",     # P4_X
                f"{points[3][1]:.1f}",     # P4_Y
                f"{angle:.1f}°",           # 旋转角度
                f"{area:.0f}",             # 面积
                frame_time                 # 时间戳
            ]

            self._add_row_to_table(row_data, confidence)

    def _add_row_to_table(self, row_data, confidence_value):
        """添加行到表格"""
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtGui import QColor

        # 添加到表格
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        for col, value in enumerate(row_data):
            item = QTableWidgetItem(str(value))

            # 根据置信度设置背景色（第2列或第4列可能是置信度）
            if (self.current_task_type in ['detect', 'segment', 'obb'] and col == 2) or \
               (self.current_task_type == 'pose' and col == 4) or \
               (self.current_task_type == 'classify' and col == 2):
                conf_val = confidence_value
                if conf_val >= 0.8:
                    item.setBackground(QColor(200, 255, 200))  # 高置信度 - 浅绿色
                elif conf_val >= 0.5:
                    item.setBackground(QColor(255, 255, 200))  # 中等置信度 - 浅黄色
                else:
                    item.setBackground(QColor(255, 220, 220))  # 低置信度 - 浅红色

            self.table.setItem(row_position, col, item)

        # 保存到内部数据
        self.detection_data.append(row_data)
        self.detection_id_counter += 1

    def update_statistics(self):
        """更新统计信息"""
        if not self.detection_data:
            self.total_label.setText("总数: 0")
            self.avg_conf_label.setText("平均置信度: 0.000")
            self.classes_label.setText("类别: 0")
            return

        # 统计各类别数量
        class_counts = {}
        confidence_sum = 0

        for row in self.detection_data:
            class_name = row[1]
            confidence = float(row[2])

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_sum += confidence

        avg_confidence = confidence_sum / len(self.detection_data)

        self.total_label.setText(f"总数: {len(self.detection_data)}")
        self.avg_conf_label.setText(f"平均置信度: {avg_confidence:.3f}")
        self.classes_label.setText(f"类别: {len(class_counts)}")

    def export_to_csv(self):
        """导出为CSV文件"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import csv

        if not self.detection_data:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出检测结果", "detection_results.csv", "CSV文件 (*.csv)"
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)

                    # 写入表头
                    writer.writerow(self.headers)

                    # 写入数据
                    for row_data in self.detection_data:
                        writer.writerow(row_data)

                QMessageBox.information(self, "成功", f"数据已导出到: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def show_context_menu(self, position):
        """显示右键菜单"""
        if self.table.itemAt(position):
            self.context_menu.exec_(self.table.mapToGlobal(position))

    def copy_selected_row(self):
        """复制选中行"""
        current_row = self.table.currentRow()
        if current_row >= 0 and current_row < len(self.detection_data):
            row_data = self.detection_data[current_row]
            text = '\t'.join(str(item) for item in row_data)

            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(text)

    def delete_selected_row(self):
        """删除选中行"""
        current_row = self.table.currentRow()
        if current_row >= 0 and current_row < len(self.detection_data):
            self.table.removeRow(current_row)
            del self.detection_data[current_row]
            self.update_statistics()

    def highlight_detection(self):
        """高亮显示选中的检测结果"""
        current_row = self.table.currentRow()
        if current_row >= 0 and current_row < len(self.detection_data):
            detection_id = self.detection_data[current_row][0]
            self.detectionSelected.emit(detection_id)

    def on_cell_clicked(self, row, column):
        """处理单元格点击"""
        if row >= 0 and row < len(self.detection_data):
            detection_id = self.detection_data[row][0]
            self.detectionSelected.emit(detection_id)

    def get_detection_count(self):
        """获取检测数量"""
        return len(self.detection_data)

    def get_class_statistics(self):
        """获取类别统计"""
        class_counts = {}
        for row in self.detection_data:
            class_name = row[1]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts
