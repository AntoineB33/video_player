import os
import sys
import ctypes
import pickle
import random
from pynput import keyboard

# Set path to directory containing libvlc.dll
libvlc_dir = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(libvlc_dir)  # Windows 10+
ctypes.CDLL(os.path.join(libvlc_dir, "libvlc.dll"))

import tkinter as tk
from tkinter import messagebox
import vlc
import threading
import time
from generate_sortings import instr_struct
from encrypt import get_playlist_status
from config import DECRYPTED_MEDIA_PATH, PLAYLISTS_PATH, DEFAULT_MUSICS_FILE

# Windows API constants
GWL_STYLE = -16
WS_CURSOR = 0x0001  # Cursor visibility flag
MUSIC_VOLUME = 30  # Volume for background music (0-100)

class FullscreenPlayer:
    def __init__(self, master, playlist, musics, playlist_name):
        self.master = master
        self.playlist = playlist
        self.music_playlist = musics
        self.playlist_name = playlist_name
        self.playlist_index = 0
        self.seeking = False
        
        # NEW: Music-related attributes
        self.current_music_index = 0
        self.music_player = None
        self.music_instance = None
        self.video_has_audio = True
        self.music_enabled = True
        
        if not self.playlist:
            messagebox.showerror("Error", "Playlist is empty!")
            master.destroy()
            return

        master.attributes('-fullscreen', True)
        master.attributes('-topmost', True)
        master.configure(bg='black')
        self.video_frame = tk.Frame(master, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        self.hide_cursor()
        
        master.update()
        master.focus_force()

        vlc_options = [
            "--no-osd", "--no-video-title-show", "--quiet", "--no-embedded-video",
            "--avcodec-hw=dxva2", "--avcodec-fast", "--sout-avcodec-strict=-2",
            "--audio-desync=0", "--audio-replay-gain-mode=none", "--file-caching=300",
            "--network-caching=300", "--live-caching=300", "--drop-late-frames",
            "--skip-frames", "--avcodec-skiploopfilter=4", "--winrt-d3dcontext",
            "--d3d11", "--avcodec-threads=0", "--verbose=0"
        ]

        self.instance = vlc.Instance(" ".join(vlc_options))
        self.player = self.instance.media_player_new()
        self.player.set_hwnd(self.video_frame.winfo_id())

        # NEW: Setup music player
        self.setup_music_player()

        self.events = self.player.event_manager()
        self.events.event_attach(
            vlc.EventType.MediaPlayerEndReached,
            lambda e: self.master.after(0, self.next_video)
        )
        self.events.event_attach(vlc.EventType.MediaPlayerSeekableChanged, self.on_seekable_changed)
        self.events.event_attach(vlc.EventType.MediaPlayerPositionChanged, self.on_position_changed)
        
        # NEW: Event for detecting when video starts playing (to check audio)
        self.events.event_attach(vlc.EventType.MediaPlayerPlaying, self.on_video_playing)
        
        # --- Setup and start the global keyboard listener ---
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.load_video()
    
    # NEW: Setup music player
    def setup_music_player(self):
        """Initialize the music player and load music playlist"""
        try:
            if self.music_playlist:
                # Shuffle the music playlist for variety
                random.shuffle(self.music_playlist)
                
                # Create separate VLC instance for music
                music_options = [
                    "--intf=dummy", "--no-video", "--audio-filter=compressor",
                    "--compressor-attack=1.0", "--compressor-release=100.0",
                    "--compressor-ratio=4.0", "--compressor-threshold=-12.0",
                    "--quiet", "--verbose=0"
                ]
                self.music_instance = vlc.Instance(" ".join(music_options))
                self.music_player = self.music_instance.media_player_new()
                self.music_player.audio_set_volume(MUSIC_VOLUME)
                
                # Setup music events
                music_events = self.music_player.event_manager()
                music_events.event_attach(
                    vlc.EventType.MediaPlayerEndReached,
                    lambda e: self.master.after(0, self.next_music)
                )
                
                print(f"Loaded {len(self.music_playlist)} music tracks")
            else:
                print("Music playlist file is empty")
        except Exception as e:
            print(f"Error setting up music player: {e}")
    
    # NEW: Check if current video has audio
    def check_video_audio(self):
        """Check if the current video has audio tracks"""
        try:
            # Wait a moment for media to fully load
            time.sleep(0.5)
            audio_tracks = self.player.audio_get_track_count()
            return audio_tracks > 0
        except:
            return True  # Default to assuming video has audio
    
    # NEW: Event handler for when video starts playing
    def on_video_playing(self, event):
        """Called when video starts playing - check for audio and manage music"""
        def check_audio():
            self.video_has_audio = self.check_video_audio()
            if not self.video_has_audio and self.music_enabled and self.music_playlist:
                self.start_background_music()
            elif self.video_has_audio:
                self.stop_background_music()
        
        # Run audio check in a separate thread to avoid blocking
        threading.Thread(target=check_audio, daemon=True).start()
    
    # NEW: Start background music
    def start_background_music(self):
        """Start playing background music"""
        if not self.music_player or not self.music_playlist:
            return
        
        try:
            music_path = self.music_playlist[self.current_music_index]
            if os.path.exists(music_path):
                media = self.music_instance.media_new(music_path)
                self.music_player.set_media(media)
                self.music_player.play()
                print(f"Playing background music: {os.path.basename(music_path)}")
            else:
                print(f"Music file not found: {music_path}")
                self.next_music()
        except Exception as e:
            print(f"Error starting background music: {e}")
    
    # NEW: Stop background music
    def stop_background_music(self):
        """Stop playing background music"""
        if self.music_player:
            self.music_player.stop()
    
    # NEW: Go to next music track
    def next_music(self):
        """Move to next music track in playlist"""
        if not self.music_playlist:
            return
        
        self.current_music_index = (self.current_music_index + 1) % len(self.music_playlist)
        if not self.video_has_audio and self.music_enabled:
            self.start_background_music()
    
    # NEW: Toggle music on/off
    def toggle_music(self):
        """Toggle background music on/off"""
        self.music_enabled = not self.music_enabled
        if self.music_enabled and not self.video_has_audio:
            self.start_background_music()
        else:
            self.stop_background_music()
        print(f"Background music {'enabled' if self.music_enabled else 'disabled'}")
    
    # --- Global key press handler ---
    def on_press(self, key):
        try:
            # Handle character keys (for numbers 0-9)
            if '0' <= key.char <= '9':
                self.seek_to_percentage(int(key.char) / 10.0)
            elif key.char == 'n':
                self.next_video()
            elif key.char == 'p':
                self.prev_video()
            elif key.char == 'm':  # NEW: Toggle music
                self.toggle_music()
                
        except AttributeError:
            # Handle special keys (like arrow keys, escape, etc.)
            if key == keyboard.Key.esc:
                self.quit_player()
            elif key == keyboard.Key.right:
                self.seek_forward()
            elif key == keyboard.Key.left:
                self.seek_backward()

    def hide_cursor(self):
        try:
            hwnd = ctypes.windll.user32.GetParent(self.master.winfo_id())
            ctypes.windll.user32.ShowCursor(False)
            style = ctypes.windll.user32.GetWindowLongPtrW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style & ~WS_CURSOR)
        except Exception as e:
            print(f"Error hiding cursor: {e}")
    
    def show_cursor(self):
        try:
            hwnd = ctypes.windll.user32.GetParent(self.master.winfo_id())
            ctypes.windll.user32.ShowCursor(True)
            style = ctypes.windll.user32.GetWindowLongPtrW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style | WS_CURSOR)
        except Exception as e:
            print(f"Error showing cursor: {e}")
    
    def on_seekable_changed(self, event):
        pass
    
    def on_position_changed(self, event):
        if self.seeking:
            threading.Timer(0.1, self.finish_seek).start()
    
    def finish_seek(self):
        self.seeking = False

    def seek_forward(self):
        if self.player.is_playing():
            self.perform_seek(5000)

    def seek_backward(self):
        if self.player.is_playing():
            self.perform_seek(-5000)

    def perform_seek(self, offset_ms):
        self.seeking = True
        current_time = self.player.get_time()
        target_time = max(0, current_time + offset_ms)
        self.player.set_time(target_time)

    def seek_to_percentage(self, ratio):
        if self.player.is_playing():
            self.seeking = True
            self.player.set_position(ratio)

    def load_video(self):
        if not 0 <= self.playlist_index < len(self.playlist):
            return
        video_path = self.playlist[self.playlist_index]
        if not os.path.exists(video_path):
            messagebox.showwarning("File Missing", f"Video not found:\n{video_path}")
            return
        
        # NEW: Stop music when loading new video (will restart if needed)
        self.stop_background_music()
        
        media = self.instance.media_new(video_path)
        self.player.set_media(media)
        self.player.play()
        self.master.after(100, lambda: self.master.focus_force())

    def next_video(self):
        if self.playlist_index < len(self.playlist) - 1:
            self.playlist_index += 1
            self.load_video()

    def prev_video(self):
        if self.playlist_index > 0:
            self.playlist_index -= 1
            self.load_video()

    def quit_player(self):
        # --- Stop the listener and music for a clean exit ---
        self.listener.stop()
        if self.music_player:
            self.music_player.stop()
        self.player.stop()
        self.show_cursor()
        self.master.destroy()

# --- Entry Point ---
if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status()
    playlist_file = os.path.join(PLAYLISTS_PATH, playlist_name)
    playlist_all = playlists.get(playlist_name, [])
    if not playlist_all:
        messagebox.showerror("Error", "No valid media found in playlist!")
        sys.exit(1)
    playlist = [os.path.join(DECRYPTED_MEDIA_PATH, video) for video in playlist_all["available"]["media"]]
    musics = [os.path.join(DECRYPTED_MEDIA_PATH, music) for music in playlist_all["available"]["musics"]]
    if not musics:
        if os.path.exists(DEFAULT_MUSICS_FILE):
            with open(DEFAULT_MUSICS_FILE, "r") as f:
                musics = [os.path.join(DECRYPTED_MEDIA_PATH, line.strip()) for line in f if line.strip()]
        else:
            print(f"Default music playlist file '{DEFAULT_MUSICS_FILE}' not found")
    root = tk.Tk()
    app = FullscreenPlayer(root, playlist, musics, playlist_name)
    root.mainloop()
    # Ensure cursor is restored and music is stopped even on error
    try:
        # Create a temporary window to safely show cursor
        temp = tk.Tk()
        temp.withdraw()
        ctypes.windll.user32.ShowCursor(True)
        temp.destroy()
    except:
        pass