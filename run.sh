#!/bin/bash

# YOLO目标检测系统启动脚本
# 版本: 2.0.0

echo "========================================"
echo "    YOLO目标检测系统 v2.0.0"
echo "========================================"
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[错误] 未检测到Python，请先安装Python 3.8+"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
fi

# 确定Python命令
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

echo "[信息] Python环境检查通过"

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

echo "[信息] 启动YOLO目标检测系统..."
echo

# 启动程序
$PYTHON_CMD main.py

# 程序结束后处理
if [ $? -ne 0 ]; then
    echo
    echo "[错误] 程序运行出错，错误代码: $?"
    echo "请检查错误信息并重试"
else
    echo
    echo "[信息] 程序正常退出"
fi

echo "按任意键退出..."
read -n 1
