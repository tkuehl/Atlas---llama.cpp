@echo off
REM Build only the llama-quantize target (requires configure to have run)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set "PATH=C:\Program Files\CMake\bin;C:\Users\tk199\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"

cd /d "%~dp0"
cmake --build build-cuda --config Release --target llama-quantize -j 12
