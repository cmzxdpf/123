@echo off
setlocal EnableDelayedExpansion

:: 1. UTF-8
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: 2. Base Paths & Python Interpreter
set "BASE_DIR=%~dp0"
set "PORTABLE_DIR=D:\portable-files"

if exist "%PORTABLE_DIR%\python\python.exe" (
    set "PYTHON_EXE=%PORTABLE_DIR%\python\python.exe"
) else if exist "D:\2\portable-files\venv\Scripts\python.exe" (
    set "PYTHON_EXE=D:\2\portable-files\venv\Scripts\python.exe"
) else (
    echo [ERROR] Python environment not found in %PORTABLE_DIR%\python\python.exe
    pause
    exit /b 1
)

:: 3. Proxy & Caches
set "HTTP_PROXY=http://127.0.0.1:7891"
set "HTTPS_PROXY=http://127.0.0.1:7891"
set "ALL_PROXY=http://127.0.0.1:7891"
set "NO_PROXY=localhost,127.0.0.1"

set "HF_HOME=J:\SEC\hf_cache"
set "TORCH_HOME=%BASE_DIR%torch_cache"
set "UV_CACHE_DIR=J:\SEC\uv_cache"
set "PIP_CACHE_DIR=J:\SEC\pip_cache"

echo [*] Python Interpreter : %PYTHON_EXE%
echo [*] SeC-4B HF Cache    : %HF_HOME%
echo [*] Proxy Active       : 127.0.0.1:7891
echo.

if "%~1"=="" (
    echo ===============================================================================
    echo SeC-4B Pure Video Mask Segmentation Runner
    echo ===============================================================================
    echo Usage examples:
    echo   run.bat --video "input.mp4" --output "output/sec4b"
    echo   run.bat --video "input.mp4" --start-frame 0 --end-frame 200 --pos "576,570"
    echo   run.bat --video "input.mp4" --preview-frame 0 --preview-out "preview.jpg"
    echo ===============================================================================
    cmd /k
) else (
    "%PYTHON_EXE%" "%BASE_DIR%sec4b_video_masks.py" %*
)
