@echo off
setlocal EnableDelayedExpansion

:: 1. Force UTF-8 Encoding
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: 2. Base directory and Python interpreter
set "BASE_DIR=%~dp0"
set "VM_PORTABLE=J:\VisoMaster\portable-files"

if exist "%VM_PORTABLE%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%VM_PORTABLE%\venv\Scripts\python.exe"
    echo [*] Using Python interpreter: %VM_PORTABLE%\venv\Scripts\python.exe
) else if exist "%BASE_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"
    echo [*] Using local venv: %BASE_DIR%venv\Scripts\python.exe
) else (
    echo [ERROR] No Python environment found! Please verify VisoMaster venv at J:\VisoMaster\portable-files\venv.
    pause
    exit /b 1
)

:: 3. Proxy Configuration (7891)
set "HTTP_PROXY=http://127.0.0.1:7891"
set "HTTPS_PROXY=http://127.0.0.1:7891"
set "ALL_PROXY=http://127.0.0.1:7891"
set "NO_PROXY=localhost,127.0.0.1"

:: 4. Cache Redirection to J: Drive (Zero C: Drive Pollution, Reuse J: weights)
if exist "%BASE_DIR%hf_cache\hub\models--OpenIXCLab--SeC-4B" (
    set "HF_HOME=%BASE_DIR%hf_cache"
) else if exist "J:\SEC_VideoMasks\hf_cache\hub\models--OpenIXCLab--SeC-4B" (
    set "HF_HOME=J:\SEC_VideoMasks\hf_cache"
) else if exist "J:\SEC\hf_cache\hub\models--OpenIXCLab--SeC-4B" (
    set "HF_HOME=J:\SEC\hf_cache"
) else (
    set "HF_HOME=%BASE_DIR%hf_cache"
)
set "TORCH_HOME=%BASE_DIR%torch_cache"
set "UV_CACHE_DIR=%BASE_DIR%uv_cache"
set "PIP_CACHE_DIR=%BASE_DIR%pip_cache"

echo [*] Proxy active: 127.0.0.1:7891
echo [*] HF Cache location: %HF_HOME%
echo.

if "%~1"=="" (
    echo ===============================================================================
    echo SeC-4B Pure Video Mask Segmentation Runner
    echo ===============================================================================
    echo Usage examples:
    echo   run.bat --video "J:\1\1.mp4" --output "J:\SEC_VideoMasks\output\sec4b"
    echo   run.bat --video "J:\1\1.mp4" --start-frame 0 --end-frame 200 --segment-length 400 --overlap 50
    echo   run.bat --video "J:\1\1_skin_5s_frames" --resume
    echo.
    echo Single-frame Preview:
    echo   run.bat --video "J:\1\1.mp4" --preview-frame 0 --preview-out "preview.jpg"
    echo ===============================================================================
    cmd /k
) else (
    "%PYTHON_EXE%" "%BASE_DIR%sec4b_video_masks.py" %*
)
