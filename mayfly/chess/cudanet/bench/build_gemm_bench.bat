@echo off
rem Builds bench\build\gemm_bench.exe, which times kernels_gemm.cu against cuBLASLt (see gemm_bench.cpp).
rem Runs from any directory. The kernels are read from chess\cudanet (gemm_bench --src).
rem   CUDA_KIT  the CUDA 12.8 headers, an environment variable (default %CUDA_PATH%, set by NVIDIA's installer)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
if "%CUDA_KIT%"=="" set CUDA_KIT=%CUDA_PATH%
set CF=/nologo /std:c++17 /O2 /EHsc /MD /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Zc:preprocessor /I"%CUDA_KIT%\include"
cl %CF% gemm_bench.cpp ..\cuda_dyn.cpp ..\nvrtc_loader.cpp ..\gemm.cpp ..\mmgemm.cpp /Fobuild\ /Fe:build\gemm_bench.exe kernel32.lib user32.lib advapi32.lib shell32.lib ole32.lib || exit /b 1
echo built bench\build\gemm_bench.exe
