@echo off
rem Builds build\cudanet.dll from every chess\cudanet\*.cpp except cudanet_backend.cpp (the search adapters, which go
rem into mayfly_cppsearch instead, see the top of cudanet_backend.cpp). No CUDA import library is linked, because
rem cuda_dyn loads nvcuda, cudart, cuBLASLt and NVRTC at runtime. The kernels are CUDA C compiled by NVRTC on first use,
rem so the kernels_*.cu files are copied beside the dll.
rem   CUDA_KIT  the CUDA 12.8 headers, an environment variable (default %CUDA_PATH%, set by NVIDIA's installer). Only
rem             include\ is read at build time.
setlocal enabledelayedexpansion
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
if "%CUDA_KIT%"=="" set CUDA_KIT=%CUDA_PATH%
set CF=/nologo /std:c++17 /O2 /EHsc /MD /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DCUDANET_BUILD /Zc:preprocessor /I"%CUDA_KIT%\include"
set SRC=
for %%f in (*.cpp) do if /i not "%%f"=="cudanet_backend.cpp" set SRC=!SRC! %%f
cl %CF% /c !SRC! /Fobuild\ || exit /b 1
set OBJ=
for %%f in (!SRC!) do set OBJ=!OBJ! build\%%~nf.obj
link /nologo /DLL /out:build\cudanet.dll !OBJ! kernel32.lib user32.lib advapi32.lib shell32.lib ole32.lib || exit /b 1
copy /y kernels_trunk.cu build\ >nul || exit /b 1
copy /y kernels_heads.cu build\ >nul || exit /b 1
copy /y kernels_fuse.cu build\ >nul || exit /b 1
copy /y kernels_gemm.cu build\ >nul || exit /b 1
echo built build\cudanet.dll
