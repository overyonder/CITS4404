@echo off
REM PyTorch Engine Runner for Windows
REM This script sets up the environment and runs the application with PyTorch support

echo Setting up PyTorch environment...

REM Add libtorch DLLs to PATH
set PATH=%PATH%;C:\libtorch\lib

REM Set environment variables for PyTorch
set LIBTORCH=C:\libtorch
set LIBTORCH_LIB=C:\libtorch\lib
set LIBTORCH_INCLUDE=C:\libtorch\include
set LIBTORCH_BYPASS_VERSION_CHECK=1

REM Additional fixes for path issues in torch-sys build script
set LIBTORCH_USE_PYTORCH=0
set LIBTORCH_CXX11_ABI=1

echo Environment configured. Starting application...
echo.

REM Check if arguments were provided
if "%1"=="" (
    echo Starting TUI mode with PyTorch support...
    cargo run --release --features torch
) else (
    echo Starting CLI mode with arguments: %*
    cargo run --release --features torch -- %*
)

echo.
echo Application finished.
pause 