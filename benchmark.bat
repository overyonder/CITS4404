@echo off
REM This script compiles and benchmarks the C++ and Rust versions using hyperfine.

REM Check for hyperfine
hyperfine --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO hyperfine is not installed or not in your PATH.
    ECHO Please install it by running: cargo install hyperfine
    GOTO :EOF
)

REM Set number of generations from command-line argument, default to 100
SET GENS=%1
IF "%GENS%"=="" SET GENS=100

ECHO [1/3] Compiling C++ version...
REM The -O3 flag is for maximum optimization.
g++ -O3 -o C++/Evolve.exe C++/Evolve.cpp C++/NeuralNet.cpp C++/Player.cpp C++/Pong.cpp
IF %ERRORLEVEL% NEQ 0 ( 
    ECHO C++ compilation failed.
    GOTO :EOF
)

ECHO [2/3] Compiling Rust version...
cd rust
cargo build --release
IF %ERRORLEVEL% NEQ 0 ( 
    ECHO Rust compilation failed.
    cd ..
    GOTO :EOF
)
cd ..

ECHO.
ECHO ========================================================
ECHO      RUNNING BENCHMARKS FOR %GENS% GENERATIONS
ECHO ========================================================
ECHO.

ECHO [3/3] Running hyperfine benchmark...
hyperfine --warmup 1 "C++\Evolve.exe %GENS%" "rust\target\release\rust.exe %GENS%"

ECHO.
ECHO Benchmarking complete.
