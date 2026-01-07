@echo off
echo ================================================================================
echo                    FUSION DETECTION SYSTEM
echo             Pothole + Road Sign Detection
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if required packages are installed
echo Checking dependencies...
python -c "import cv2, numpy, ultralytics, torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some dependencies are missing!
    echo Installing required packages...
    echo.
    pip install ultralytics opencv-python numpy torch
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK!
echo.

REM Run the quick start script
echo ================================================================================
echo Starting Fusion Detection...
echo ================================================================================
echo.

python run_fusion.py

if %errorlevel% neq 0 (
    echo.
    echo ================================================================================
    echo ERROR: Detection failed
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Detection completed successfully!
echo ================================================================================
pause
