@echo off
setlocal
set "MEASURE_PY="
cd /d "%~dp0"
for %%P in ("%USERPROFILE%\miniconda3\envs\OpenCV\python.exe" "%CONDA_PREFIX%\python.exe" "%~dp0.venv\Scripts\python.exe") do if exist "%%~P" call :probe "%%~P"
for /f "delims=" %%P in ('where python.exe 2^>nul ^| findstr /i /v "WindowsApps"') do if not defined MEASURE_PY call :probe "%%P"
if not defined MEASURE_PY (
  echo Python with opencv-python, numpy and tkinter was not found.
  echo Install the packages into a Python environment, then run this launcher again.
  pause
  exit /b 1
)
echo Python: %MEASURE_PY%
if /i "%~1"=="--check" exit /b 0
"%MEASURE_PY%" "%~dp0ruler_app.py"
if errorlevel 1 pause
exit /b
:probe
if defined MEASURE_PY exit /b
setlocal
set "PATH=%~dp1;%~dp1Library\bin;%PATH%"
"%~1" -c "import cv2,numpy,tkinter" >nul 2>nul
if errorlevel 1 (endlocal & exit /b)
endlocal & set "MEASURE_PY=%~1" & set "PATH=%~dp1;%~dp1Library\bin;%PATH%"
exit /b
