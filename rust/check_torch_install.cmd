@echo off
REM PyTorch Installation Diagnostic Script
REM This script checks if your libtorch installation is correct

echo PyTorch Installation Diagnostic
echo =================================
echo.

echo Checking directory structure...
if not exist "C:\libtorch\" (
    echo [ERROR] C:\libtorch\ directory not found
    echo Please extract libtorch to C:\libtorch
    goto :end
)
echo [OK] Base directory exists

if not exist "C:\libtorch\lib\" (
    echo [ERROR] C:\libtorch\lib\ directory not found
    goto :end
)
echo [OK] Lib directory exists

if not exist "C:\libtorch\include\" (
    echo [ERROR] C:\libtorch\include\ directory not found
    goto :end
)
echo [OK] Include directory exists

echo.
echo Checking required library files...
if not exist "C:\libtorch\lib\torch.lib" (
    echo [ERROR] torch.lib not found
    goto :end
)
echo [OK] torch.lib found

if not exist "C:\libtorch\lib\torch_cpu.lib" (
    echo [ERROR] torch_cpu.lib not found
    goto :end
)
echo [OK] torch_cpu.lib found

if not exist "C:\libtorch\lib\c10.lib" (
    echo [ERROR] c10.lib not found
    goto :end
)
echo [OK] c10.lib found

echo.
echo Checking required header files...
if not exist "C:\libtorch\include\torch\" (
    echo [ERROR] C:\libtorch\include\torch\ directory not found
    echo This indicates you may have downloaded the Python version instead of C++
    goto :end
)
echo [OK] Include/torch directory exists

if not exist "C:\libtorch\include\torch\torch.h" (
    echo [ERROR] torch.h not found in include/torch/
    echo This indicates incomplete or wrong installation
    goto :end
)
echo [OK] torch.h found

if not exist "C:\libtorch\include\torch\csrc\" (
    echo [ERROR] C:\libtorch\include\torch\csrc\ directory not found
    echo This indicates incomplete installation
    goto :end
)
echo [OK] Include/torch/csrc directory exists

echo.
echo Checking version...
if exist "C:\libtorch\build-version" (
    echo LibTorch version:
    type "C:\libtorch\build-version"
    echo.
    echo NOTE: tch 0.20.0 expects PyTorch 2.7.0
    echo If your version is different, you may need LIBTORCH_BYPASS_VERSION_CHECK=1
)

echo.
echo Checking environment variables...
if defined LIBTORCH (
    echo [OK] LIBTORCH=%LIBTORCH%
) else (
    echo [WARNING] LIBTORCH environment variable not set
)

if defined LIBTORCH_LIB (
    echo [OK] LIBTORCH_LIB=%LIBTORCH_LIB%
) else (
    echo [WARNING] LIBTORCH_LIB environment variable not set
)

echo.
echo Diagnosis Complete!
echo.
echo If all checks pass, try running: build_torch_clean.cmd
echo.

:end
pause 