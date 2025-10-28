@echo off
setlocal
cd /d "%~dp0"
:: Single frame ONNX test using PREPROCESSED images + LIVE audio
:: Usage: test_single_frame_onnx_preprocessed.bat <name> [audio_path] [asr_mode] [frame_index]
:: Example: test_single_frame_onnx_preprocessed.bat sanders dataset\aoc\aud.wav ave 8
:: Example: test_single_frame_onnx_preprocessed.bat sanders "" ave 8  (uses default aud.wav)

set "name=%~1"
set "audio=%~2"
set "asr=%~3"
set "frame_idx=%~4"

if "%name%"=="" (
    echo [ERROR] Missing name parameter
    echo Usage: test_single_frame_onnx_preprocessed.bat ^<name^> [audio_path] [asr_mode] [frame_index]
    echo Example: test_single_frame_onnx_preprocessed.bat sanders dataset\aoc\aud.wav ave 8
    echo Example: test_single_frame_onnx_preprocessed.bat sanders ave 8  ^(uses dataset\sanders\aud.wav^)
    exit /b 2
)

:: Handle case where second param might be asr mode (no audio path)
if "%audio%"=="ave" set "asr=ave" & set "audio=" & set "frame_idx=%~3"
if "%audio%"=="hubert" set "asr=hubert" & set "audio=" & set "frame_idx=%~3"
if "%audio%"=="wenet" set "asr=wenet" & set "audio=" & set "frame_idx=%~3"

:: Set defaults
if "%asr%"=="" set "asr=ave"
if "%frame_idx%"=="" set "frame_idx=8"

:: Check for preprocessed image cache
if not exist "dataset\%name%\cache\" (
    echo [ERROR] Cache directory not found: dataset\%name%\cache\
    echo Run preprocessing first!
    exit /b 2
)

:: Set audio path flags
set "audio_flag="
if not "%audio%"=="" (
    if not "%audio%"=="" set "audio_flag=--audio_path "%audio%""
)

:: If no audio specified, check for default aud.wav
if "%audio%"=="" (
    if not exist "dataset\%name%\aud.wav" (
        echo [ERROR] No audio specified and default not found: dataset\%name%\aud.wav
        echo Usage: test_single_frame_onnx_preprocessed.bat ^<name^> [audio_path] [asr_mode] [frame_index]
        exit /b 2
    )
    echo [INFO] Using default audio: dataset\%name%\aud.wav
)

:: Activate virtual environment
call .\.venv312\Scripts\activate.bat >nul 2>&1 || (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

echo ============================================================
echo Single Frame ONNX Test - PREPROCESSED IMAGES + LIVE AUDIO
echo ============================================================
echo Name: %name%
echo Audio: %audio% ^(or default if empty^)
echo ASR Mode: %asr%
echo Frame Index: %frame_idx%
echo Using preprocessed image cache: dataset\%name%\cache\
echo ============================================================

python test_single_frame_onnx_preprocessed.py --name %name% --asr %asr% --frame_index %frame_idx% %audio_flag%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] Test completed successfully
    echo ============================================================
    echo Check the output in dataset\%name%\test_output_preprocessed\
    echo - frame_%frame_idx%_tensors.json for statistics
    echo - tensors_npy\ folder for exact numpy arrays
    echo - *.jpg files for visual verification
    echo.
    echo This test used:
    echo   - PREPROCESSED images from cache/ ^(exact training data^)
    echo   - LIVE audio processing from WAV file
) else (
    echo.
    echo ============================================================
    echo [ERROR] Test failed with code %ERRORLEVEL%
    echo ============================================================
)

exit /b %ERRORLEVEL%
