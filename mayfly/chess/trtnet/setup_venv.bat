@echo off
rem Creates the TensorRT venv for chess\trtnet, kept apart from the main Python.
rem TensorRT 10 for CUDA 12 from PyPI (Windows wheels with the CUDA runtime linked in, the driver needs r528 or later),
rem onnx for the export, and torch for the CPU only (export_onnx.py and test_cudanet's float32 reference).
rem   TRT_VENV  where to put it, an environment variable (default trt_venv in this folder)
setlocal
if "%TRT_VENV%"=="" set TRT_VENV=%~dp0trt_venv
if not exist "%TRT_VENV%\Scripts\python.exe" python -m venv "%TRT_VENV%" || exit /b 1
"%TRT_VENV%\Scripts\python.exe" -m pip install "tensorrt-cu12==10.16.1.11" numpy chess onnx || exit /b 1
"%TRT_VENV%\Scripts\python.exe" -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu || exit /b 1
echo venv ready: %TRT_VENV%
