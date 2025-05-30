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
        self.headers = [
            "ID", "类别", "置信度", "X1", "Y1", "X2", "Y2",
            "宽度", "高度", "面积", "中心X", "中心Y", "时间戳"
        ]
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
        column_widths = [50, 80, 80, 60, 60, 60, 60, 60, 60, 80, 60, 60, 80]
        for i, width in enumerate(column_widths):
            if i < len(self.headers):
                self.table.setColumnWidth(i, width)

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

    def add_detections(self, results, frame_time=None):
        """添加检测结果"""
        if not results or len(results) == 0:
            return

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return

        if frame_time is None:
            import datetime
            frame_time = datetime.datetime.now().strftime("%H:%M:%S")

        # 检查最大行数限制
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

        # 添加新检测结果
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

            # 添加到表格
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)

            for col, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))

                # 根据置信度设置背景色
                if col == 2:  # 置信度列
                    conf_val = float(value)
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

        # 自动滚动到底部
        if self.auto_scroll_cb.isChecked():
            self.table.scrollToBottom()

    def clear_table(self):
        """清空表格"""
        self.table.setRowCount(0)
        self.detection_data.clear()
        self.detection_id_counter = 1
        self.update_statistics()

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
