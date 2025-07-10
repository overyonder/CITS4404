@echo off
@REM PyTorch Engine Runner for Windows
@REM This script sets up the environment and runs the application with PyTorch support

@REM echo Setting up PyTorch environment...

@REM Add NVIDIA CUDA Toolkit to PATH (most robust solution)
@REM set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%
@REM Add libtorch DLLs to PATH
@REM set PATH=C:\libtorch\lib;%PATH%

@REM Set environment variables for PyTorch
@REM set LIBTORCH=C:\libtorch
@REM set LIBTORCH_LIB=C:\libtorch
@REM set LIBTORCH_INCLUDE=C:\libtorch
set LIBTORCH_BYPASS_VERSION_CHECK=1

@REM Additional fixes for path issues in torch-sys build script
set LIBTORCH_USE_PYTORCH=1

echo Environment configured. Starting application...
echo.

@REM Check if arguments were provided
if "%1"=="" (
    echo Starting TUI mode with PyTorch support...
    cargo run --release --features torch
) else (
    echo Starting CLI mode with arguments: %*
    cargo run --release --features torch -- %*
)

echo.
echo Application finished.
