@echo off
setlocal enabledelayedexpansion

:: CONFIGURATION
SET "VLC_PATH=C:\Program Files\VideoLAN\VLC\vlc.exe"
SET "PLAYLIST_DIR=playlists"

:: Prompt for playlist name
set /p "PLAYLIST_NAME=Enter playlist name: "
SET "PLAYLIST_FILE=%PLAYLIST_DIR%\%PLAYLIST_NAME%.txt"

:: Check if playlist file exists
if not exist "%PLAYLIST_FILE%" (
    echo Playlist not found: %PLAYLIST_FILE%
    pause
    exit /b
)

:: Check if VLC exists
if not exist "%VLC_PATH%" (
    echo VLC not found: %VLC_PATH%
    pause
    exit /b
)

:: Read and play each video in the playlist
for /f "usebackq tokens=* delims=" %%A in ("%PLAYLIST_FILE%") do (
    echo Playing: %%A
    "%VLC_PATH%" "%%A" --fullscreen --no-osd --no-video-title-show vlc://quit
)

endlocal
