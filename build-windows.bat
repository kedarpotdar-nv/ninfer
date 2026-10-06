@echo off
rem Native Windows build of NInfer (ninfer-serve) with MSVC 2022 + CUDA 13.4, Ninja generator.
rem Usage: build-windows.bat [configure|build|test|all]   (default: all)
setlocal
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
call "%VSDEVCMD%" -arch=x64 -host_arch=x64 -vcvars_ver=14.41 >nul || exit /b 1
cd /d "%~dp0"
if "%MODE%"=="configure" goto configure
if "%MODE%"=="build" goto build
if "%MODE%"=="test" goto test
:configure
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_HOST_COMPILER=cl ^
  -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF ^
  -DNINFER_ENABLE_FFMPEG=OFF -DNINFER_ENABLE_CURL=OFF || exit /b 1
if "%MODE%"=="configure" exit /b 0
:build
cmake --build build-win --target ninfer-serve --parallel 8 -- -k 0 || exit /b 1
if "%MODE%"=="build" exit /b 0
:test
cmake --build build-win --parallel 8 -- -k 0 || exit /b 1
ctest --test-dir build-win --output-on-failure -j1
exit /b %errorlevel%
