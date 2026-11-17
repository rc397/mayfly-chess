@echo off
rem Builds mayfly_cppsearch with the network in C++, %OUT%\mayfly_cppsearch.cp313-win_amd64.pyd from pyext, search2
rem and chess_core (/O2 /fp:precise), torchnet and net_backend2 (LibTorch with CUDA graphs, NET_GRAPH), linked against
rem CPython 3.13 and torch 2.10's import libraries. The .pyd loads only beside the torch it was linked against, after
rem `import torch`. It also carries the hand-tuned CUDA net (chess\cudanet). A .cnw file in the trace's place makes a
rem tree on it. cudanet loads every CUDA library at runtime, so it adds no import library. every kernels_*.cu
rem file is copied beside the .pyd, which is where CudaNet reads them from.
rem Settings are environment variables.
rem   TORCH_KIT  a folder with torch 2.10's include\, lib\ and cuda\include (required, see cudanet\bench_search.py)
rem   CUDA_SHIM  host-only stand-ins for CUDA headers nvcc ships (default ..\cudanet\cuda_shim)
rem   CUDA_KIT   the CUDA 12.8 headers cudanet builds against (default %CUDA_PATH%, set by NVIDIA's installer)
rem   PY         the CPython 3.13 install with include\ and libs\ (default the base prefix of `python` on PATH)
rem   OUT        the output folder (default build)
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul || call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if "%OUT%"=="" set OUT=build
if not exist "%OUT%" mkdir "%OUT%"
if "%TORCH_KIT%"=="" (echo set TORCH_KIT to a torch 2.10 kit & exit /b 1)
if "%CUDA_SHIM%"=="" set CUDA_SHIM=%~dp0..\cudanet\cuda_shim
if "%CUDA_KIT%"=="" set CUDA_KIT=%CUDA_PATH%
if "%PY%"=="" for /f "delims=" %%i in ('python -c "import sys; print(sys.base_prefix)"') do set PY=%%i
set T=%TORCH_KIT%
set CF=/nologo /std:c++17 /EHsc /MD /O2 /fp:precise /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DMAYFLY_TORCH /DMAYFLY_CUDANET /I"%PY%\Include"
set TF=/nologo /std:c++17 /EHsc /MD /O2 /W0 /bigobj /DNOMINMAX /D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR /DNET_GRAPH /I"%T%\include" /I"%T%\include\torch\csrc\api\include" /I"%CUDA_SHIM%" /I"%T%\cuda\include"
set NF=/nologo /std:c++17 /EHsc /MD /O2 /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Zc:preprocessor /I"%CUDA_KIT%\include"
cl %CF% /c chess_core.cpp search2.cpp pyext.cpp /Fo%OUT%\ || exit /b 1
cl %TF% /c torchnet.cpp net_backend2.cpp /Fo%OUT%\ || exit /b 1
cl %NF% /c ..\cudanet\cudanet_backend.cpp ..\cudanet\cudanet.cpp ..\cudanet\cuda_dyn.cpp ..\cudanet\nvrtc_loader.cpp ^
  ..\cudanet\gemm.cpp ..\cudanet\gemm_tune.cpp ..\cudanet\mmgemm.cpp ..\cudanet\weights.cpp /Fo%OUT%\ || exit /b 1
link /nologo /DLL /out:%OUT%\mayfly_cppsearch.cp313-win_amd64.pyd %OUT%\pyext.obj %OUT%\search2.obj %OUT%\chess_core.obj ^
  %OUT%\torchnet.obj %OUT%\net_backend2.obj %OUT%\cudanet_backend.obj %OUT%\cudanet.obj %OUT%\cuda_dyn.obj ^
  %OUT%\nvrtc_loader.obj %OUT%\gemm.obj %OUT%\gemm_tune.obj %OUT%\mmgemm.obj %OUT%\weights.obj /LIBPATH:"%PY%\libs" python313.lib /LIBPATH:"%T%\lib" ^
  torch.lib torch_cpu.lib torch_cuda.lib c10.lib c10_cuda.lib cudart.lib advapi32.lib shell32.lib ole32.lib ^
  /INCLUDE:?warp_size@cuda@at@@YAHXZ || exit /b 1
for %%f in (..\cudanet\kernels_*.cu) do copy /y %%f %OUT%\ >nul || exit /b 1
echo built %OUT%\mayfly_cppsearch.cp313-win_amd64.pyd
