"""
参数设置面板组件
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QDoubleSpinBox, QSpinBox, QCheckBox,
                               QComboBox, QGroupBox, QPushButton, QListWidget,
                               QListWidgetItem)
from PySide6.QtCore import Qt, Signal
from typing import List, Dict, Any


class ParameterPanel(QWidget):
    """参数设置面板"""
    
    # 信号
    parametersChanged = Signal(dict)  # 参数改变信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 检测参数组
        detection_group = QGroupBox("检测参数")
        detection_layout = QVBoxLayout(detection_group)
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.0)
        self.conf_spinbox.setSingleStep(0.01)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setDecimals(2)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spinbox)
        detection_layout.addLayout(conf_layout)
        
        # IoU阈值
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU阈值:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.01, 1.0)
        self.iou_spinbox.setSingleStep(0.01)
        self.iou_spinbox.setValue(0.45)
        self.iou_spinbox.setDecimals(2)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_spinbox)
        detection_layout.addLayout(iou_layout)
        
        # 最大检测数量
        max_det_layout = QHBoxLayout()
        max_det_layout.addWidget(QLabel("最大检测数:"))
        self.max_det_spinbox = QSpinBox()
        self.max_det_spinbox.setRange(1, 10000)
        self.max_det_spinbox.setValue(1000)
        max_det_layout.addWidget(self.max_det_spinbox)
        max_det_layout.addStretch()
        detection_layout.addLayout(max_det_layout)
        
        layout.addWidget(detection_group)
        
        # 设备选择组
        device_group = QGroupBox("设备设置")
        device_layout = QVBoxLayout(device_group)
        
        device_select_layout = QHBoxLayout()
        device_select_layout.addWidget(QLabel("计算设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        device_select_layout.addWidget(self.device_combo)
        device_select_layout.addStretch()
        device_layout.addLayout(device_select_layout)
        
        layout.addWidget(device_group)
        
        # 类别选择组
        classes_group = QGroupBox("类别筛选")
        classes_layout = QVBoxLayout(classes_group)
        
        # 全选/取消全选按钮
        select_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.deselect_all_btn = QPushButton("取消全选")
        select_buttons_layout.addWidget(self.select_all_btn)
        select_buttons_layout.addWidget(self.deselect_all_btn)
        select_buttons_layout.addStretch()
        classes_layout.addLayout(select_buttons_layout)
        
        # 类别列表
        self.classes_list = QListWidget()
        self.classes_list.setMaximumHeight(200)
        classes_layout.addWidget(self.classes_list)
        
        layout.addWidget(classes_group)
        
        # 显示选项组
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)
        
        self.show_labels_cb = QCheckBox("显示标签")
        self.show_labels_cb.setChecked(True)
        display_layout.addWidget(self.show_labels_cb)
        
        self.show_conf_cb = QCheckBox("显示置信度")
        self.show_conf_cb.setChecked(True)
        display_layout.addWidget(self.show_conf_cb)
        
        self.show_boxes_cb = QCheckBox("显示边界框")
        self.show_boxes_cb.setChecked(True)
        display_layout.addWidget(self.show_boxes_cb)
        
        layout.addWidget(display_group)
        
        layout.addStretch()
    
    def connect_signals(self):
        """连接信号"""
        # 置信度阈值同步
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_spinbox.setValue(v / 100.0)
        )
        self.conf_spinbox.valueChanged.connect(
            lambda v: self.conf_slider.setValue(int(v * 100))
        )
        
        # IoU阈值同步
        self.iou_slider.valueChanged.connect(
            lambda v: self.iou_spinbox.setValue(v / 100.0)
        )
        self.iou_spinbox.valueChanged.connect(
            lambda v: self.iou_slider.setValue(int(v * 100))
        )
        
        # 参数改变信号
        self.conf_spinbox.valueChanged.connect(self._emit_parameters_changed)
        self.iou_spinbox.valueChanged.connect(self._emit_parameters_changed)
        self.max_det_spinbox.valueChanged.connect(self._emit_parameters_changed)
        self.device_combo.currentTextChanged.connect(self._emit_parameters_changed)
        
        # 类别选择
        self.select_all_btn.clicked.connect(self._select_all_classes)
        self.deselect_all_btn.clicked.connect(self._deselect_all_classes)
        self.classes_list.itemChanged.connect(self._emit_parameters_changed)
        
        # 显示选项
        self.show_labels_cb.toggled.connect(self._emit_parameters_changed)
        self.show_conf_cb.toggled.connect(self._emit_parameters_changed)
        self.show_boxes_cb.toggled.connect(self._emit_parameters_changed)
    
    def set_classes(self, classes: Dict[int, str]):
        """
        设置可选择的类别
        
        Args:
            classes: 类别字典 {id: name}
        """
        self.classes_list.clear()
        
        for class_id, class_name in classes.items():
            item = QListWidgetItem(f"{class_id}: {class_name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, class_id)
            self.classes_list.addItem(item)
    
    def get_selected_classes(self) -> List[int]:
        """
        获取选中的类别ID列表
        
        Returns:
            List[int]: 选中的类别ID
        """
        selected_classes = []
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            if item.checkState() == Qt.Checked:
                class_id = item.data(Qt.UserRole)
                selected_classes.append(class_id)
        return selected_classes
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取当前参数设置
        
        Returns:
            Dict[str, Any]: 参数字典
        """
        selected_classes = self.get_selected_classes()
        
        return {
            'conf_threshold': self.conf_spinbox.value(),
            'iou_threshold': self.iou_spinbox.value(),
            'max_det': self.max_det_spinbox.value(),
            'device': self.device_combo.currentText(),
            'classes': selected_classes if len(selected_classes) < self.classes_list.count() else None,
            'show_labels': self.show_labels_cb.isChecked(),
            'show_conf': self.show_conf_cb.isChecked(),
            'show_boxes': self.show_boxes_cb.isChecked()
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        设置参数
        
        Args:
            params: 参数字典
        """
        if 'conf_threshold' in params:
            self.conf_spinbox.setValue(params['conf_threshold'])
        
        if 'iou_threshold' in params:
            self.iou_spinbox.setValue(params['iou_threshold'])
        
        if 'max_det' in params:
            self.max_det_spinbox.setValue(params['max_det'])
        
        if 'device' in params:
            index = self.device_combo.findText(params['device'])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
        
        if 'show_labels' in params:
            self.show_labels_cb.setChecked(params['show_labels'])
        
        if 'show_conf' in params:
            self.show_conf_cb.setChecked(params['show_conf'])
        
        if 'show_boxes' in params:
            self.show_boxes_cb.setChecked(params['show_boxes'])
    
    def _select_all_classes(self):
        """全选类别"""
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            item.setCheckState(Qt.Checked)
    
    def _deselect_all_classes(self):
        """取消全选类别"""
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            item.setCheckState(Qt.Unchecked)
    
    def _emit_parameters_changed(self):
        """发射参数改变信号"""
        self.parametersChanged.emit(self.get_parameters())
