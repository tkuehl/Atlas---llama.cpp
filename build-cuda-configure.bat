@echo off
REM Configure llama.cpp with CUDA via VS 2022 Build Tools + Ninja
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set "PATH=C:\Program Files\CMake\bin;C:\Users\tk199\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"

cd /d "%~dp0"
cmake -B build-cuda -G Ninja ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_ASM_MASM_COMPILER=ml64 ^
    -DCMAKE_ASM_COMPILER=cl ^
    -DCMAKE_POLICY_DEFAULT_CMP0194=NEW ^
    -DGGML_CUDA=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=120 ^
    -DLLAMA_CURL=OFF
