@echo off
setlocal
cd /d "%~dp0"
:: Single frame ONNX test for iOS verification
:: Usage: test_single_frame_onnx.bat <name> <audio_path> [asr_mode] [frame_index]
:: Example: test_single_frame_onnx.bat sanders dataset\aoc\aud.wav ave 8

set "name=%~1"
set "audio=%~2"
set "asr=%~3"
set "frame_idx=%~4"

if "%name%"=="" (
    echo [ERROR] Missing name parameter
    echo Usage: test_single_frame_onnx.bat ^<name^> ^<audio_path^> [asr_mode] [frame_index]
    echo Example: test_single_frame_onnx.bat sanders dataset\aoc\aud.wav ave 8
    exit /b 2
)

if "%audio%"=="" (
    echo [ERROR] Missing audio parameter
    echo Usage: test_single_frame_onnx.bat ^<name^> ^<audio_path^> [asr_mode] [frame_index]
    echo Example: test_single_frame_onnx.bat sanders dataset\aoc\aud.wav ave 8
    exit /b 2
)

if "%asr%"=="" set "asr=ave"
if "%frame_idx%"=="" set "frame_idx=8"

:: Derive audio directory for auto-detection
for %%I in ("%audio%") do set "aud_dir=%%~dpI"
if "%aud_dir:~-1%"=="\" set "aud_dir=%aud_dir:~0,-1%"

:: Check for flattened dist structure
set "dataset_root_flag="
if exist "%aud_dir%\models\generator.onnx" if exist "%aud_dir%\landmarks" (
    set "dataset_root_flag=--dataset_root "%aud_dir%""
    echo [INFO] Detected flattened dataset structure in %aud_dir%
)

:: Check for audio encoder
set "audio_enc_flag="
if defined dataset_root_flag if exist "%aud_dir%\models\audio_encoder.onnx" (
    set "audio_enc_flag=--audio_encoder_onnx "%aud_dir%\models\audio_encoder.onnx""
)

:: Activate virtual environment
call .\.venv312\Scripts\activate.bat >nul 2>&1 || (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

echo ============================================================
echo Single Frame ONNX Test
echo ============================================================
echo Name: %name%
echo Audio: %audio%
echo ASR Mode: %asr%
echo Frame Index: %frame_idx%
echo ============================================================

python test_single_frame_onnx.py --name %name% --audio_path "%audio%" --asr %asr% --frame_index %frame_idx% %dataset_root_flag% %audio_enc_flag%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] Test completed successfully
    echo ============================================================
    echo Check the output in dataset\%name%\test_output\
    echo - frame_%frame_idx%_tensors.json for statistics
    echo - tensors_npy\ folder for exact numpy arrays
    echo - *.jpg files for visual verification
) else (
    echo.
    echo ============================================================
    echo [ERROR] Test failed with code %ERRORLEVEL%
    echo ============================================================
)

exit /b %ERRORLEVEL%
