@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\CMake\bin;C:\Users\tk199\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cd /d "%~dp0"
cmake --build build-cuda --config Release -j 8 --target llama-perplexity
if errorlevel 1 exit /b 3
echo BUILD_OK
