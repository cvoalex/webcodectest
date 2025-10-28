@echo off
REM PyTorch single-frame test
REM Usage: test_single_frame_pth.bat <name> [audio_path] <asr> <frame_index>
REM   Example: test_single_frame_pth.bat sanders ave 8
REM   Example: test_single_frame_pth.bat sanders "dataset\sanders\silence_1s.wav" ave 8

setlocal enabledelayedexpansion

set name=%~1
set audio=%~2
set asr=%~3
set frame=%~4

REM Check if audio is actually an asr mode (ave/hubert/wenet)
if "%audio%"=="ave" set "asr=ave" & set "audio=" & set "frame=%~3"
if "%audio%"=="hubert" set "asr=hubert" & set "audio=" & set "frame=%~3"
if "%audio%"=="wenet" set "asr=wenet" & set "audio=" & set "frame=%~3"

REM Set defaults
if "%asr%"=="" set asr=ave
if "%frame%"=="" set frame=8

REM Show what we're using
if "%audio%"=="" (
    echo [INFO] Using default audio: dataset\%name%\aud.wav
    set "audio_arg="
) else (
    set "audio_arg=--audio_path %audio%"
)

echo ============================================================
echo PyTorch Single Frame Test
echo ============================================================
echo Name: %name%
echo Audio: %audio% ^(or default if empty^)
echo ASR Mode: %asr%
echo Frame Index: %frame%
echo ============================================================
echo.

call .venv312\Scripts\activate.bat
python test_single_frame_pth.py --name %name% %audio_arg% --asr %asr% --frame_index %frame%
