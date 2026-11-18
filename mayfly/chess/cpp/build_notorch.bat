@echo off
rem Builds mayfly_cppsearch with the hand-written CUDA net only and no LibTorch: pyext, search2 and chess_core
rem (/O2 /fp:precise) plus chess\cudanet, linked against CPython 3.13 alone. The .pyd needs no PyTorch to load.
rem It runs versions sealed with engine "cpp" and cuda_net from their .cnw file; a LibTorch trace is refused.
rem cudanet loads the CUDA libraries at run time (MAYFLY_CUDA_BIN) and compiles its kernels then, so every
rem kernels_*.cu file is copied beside the .pyd. This is the module of the Windows engine download
rem (make_engine_download.py). build_rig.bat builds the module with both network backends.
rem Settings are environment variables.
rem   CUDA_KIT   the CUDA 12.8 headers cudanet builds against (default %CUDA_PATH%, set by NVIDIA's installer)
rem   PY         the CPython 3.13 install with include\ and libs\ (default the base prefix of `python` on PATH)
rem   OUT        the output folder (default build_notorch)
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul || call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d "%~dp0"
if "%OUT%"=="" set OUT=build_notorch
if not exist "%OUT%" mkdir "%OUT%"
if "%CUDA_KIT%"=="" set CUDA_KIT=%CUDA_PATH%
if "%PY%"=="" for /f "delims=" %%i in ('python -c "import sys; print(sys.base_prefix)"') do set PY=%%i
set CF=/nologo /std:c++17 /EHsc /MD /O2 /fp:precise /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DMAYFLY_CUDANET /I"%PY%\Include"
set NF=/nologo /std:c++17 /EHsc /MD /O2 /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Zc:preprocessor /I"%CUDA_KIT%\include"
cl %CF% /c chess_core.cpp search2.cpp pyext.cpp /Fo%OUT%\ || exit /b 1
cl %NF% /c ..\cudanet\cudanet_backend.cpp ..\cudanet\cudanet.cpp ..\cudanet\cuda_dyn.cpp ..\cudanet\nvrtc_loader.cpp ^
  ..\cudanet\gemm.cpp ..\cudanet\gemm_tune.cpp ..\cudanet\mmgemm.cpp ..\cudanet\weights.cpp /Fo%OUT%\ || exit /b 1
link /nologo /DLL /out:%OUT%\mayfly_cppsearch.cp313-win_amd64.pyd %OUT%\pyext.obj %OUT%\search2.obj %OUT%\chess_core.obj ^
  %OUT%\cudanet_backend.obj %OUT%\cudanet.obj %OUT%\cuda_dyn.obj %OUT%\nvrtc_loader.obj %OUT%\gemm.obj %OUT%\gemm_tune.obj ^
  %OUT%\mmgemm.obj %OUT%\weights.obj /LIBPATH:"%PY%\libs" python313.lib advapi32.lib shell32.lib ole32.lib || exit /b 1
for %%f in (..\cudanet\kernels_*.cu) do copy /y %%f %OUT%\ >nul || exit /b 1
echo built %OUT%\mayfly_cppsearch.cp313-win_amd64.pyd
