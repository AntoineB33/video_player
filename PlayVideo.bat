@echo off

:: This script plays a video full-screen with no overlays using VLC.
:: The video file should be in the same folder as this .bat file.

:: --- CONFIGURATION ---
:: 1. Set the full path to your vlc.exe if it's not the default.
SET "VLC_PATH=C:\Program Files\VideoLAN\VLC\vlc.exe"

:: 2. Set the name of your video file.
SET "VIDEO_FILE=C:\Users\N6506\Downloads\Thunderbolts.2025.2160p.iT.WEB-DL.DV.HDR10+.ENG.LATINO.DDP5.1.Atmos.H265.MP4-BEN.THE.MEN\Thunderbolts.2025.2160p.iT.WEB-DL.DV.HDR10+[Ben The Men].mp4"
:: --- END CONFIGURATION ---


:: Check if VLC exists at the specified path
if not exist "%VLC_PATH%" (
    echo VLC Media Player not found at the specified path:
    echo %VLC_PATH%
    echo Please update the VLC_PATH in this script.
    pause
    exit /b
)

:: Check if the video file exists
if not exist "%VIDEO_FILE%" (
    echo Video file not found: %VIDEO_FILE%
    echo Please check the VIDEO_FILE path in this script.
    pause
    exit /b
)

:: Run VLC with options
:: --fullscreen : Starts in full-screen mode.
:: --no-osd : Disables all on-screen display (like the pause icon).
:: --no-video-title-show : Hides the video title at the start.
:: vlc://quit : Special command to make VLC close after playback finishes.
"%VLC_PATH%" "%VIDEO_FILE%" --fullscreen --no-osd --no-video-title-show vlc://quit