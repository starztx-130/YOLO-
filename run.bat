@echo off
echo ========================================
echo    YOLO目标检测系统 v2.0.0
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [信息] Python环境检查通过

REM 设置环境变量
set KMP_DUPLICATE_LIB_OK=TRUE

echo [信息] 启动YOLO目标检测系统...
echo.

REM 启动程序
python main.py

REM 程序结束后暂停
if errorlevel 1 (
    echo.
    echo [错误] 程序运行出错，错误代码: %errorlevel%
    echo 请检查错误信息并重试
) else (
    echo.
    echo [信息] 程序正常退出
)

pause
