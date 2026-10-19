@echo off
chcp 65001 >nul
title VisoClean (No DFM) - Masked Face Clean Pipeline

set "PYTHON_EXE=D:\portable-files\python\python.exe"
set "PROJECT_DIR=%~dp0"
set "HF_HOME=J:\SEC\hf_cache"

if not "%~1"=="" (
    "%PYTHON_EXE%" "%PROJECT_DIR%run_pipeline.py" --video "%~1"
) else (
    "%PYTHON_EXE%" "%PROJECT_DIR%run_pipeline.py"
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Pipeline exited with error code %ERRORLEVEL%.
)
echo.
pause
