@echo off
setlocal

set APP_NAME=HAT
set ENTRY=src\gui.py
set DIST_DIR=dist
set BUILD_DIR=build

uv run pyinstaller ^
  --onedir ^
  --windowed ^
  --name "%APP_NAME%" ^
  --paths src ^
  --hidden-import onnxruntime ^
  "%ENTRY%"

echo.
echo Build complete:
echo %DIST_DIR%\%APP_NAME%\%APP_NAME%.exe
echo.
pause