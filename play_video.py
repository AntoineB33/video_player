import os
import sys
import ctypes
from encrypt import get_playlist_status
from config import DECRYPTED_MEDIA_PATH, PLAYLISTS_PATH
import pickle

# --- NEW: Import pynput ---
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

# Windows API constants
GWL_STYLE = -16
WS_CURSOR = 0x0001  # Cursor visibility flag

class FullscreenPlayer:
    def __init__(self, master, playlist):
        self.master = master
        self.playlist = playlist
        self.playlist_index = 0
        self.seeking = False
        
        if not self.playlist:
            messagebox.showerror("Error", "Playlist is empty!")
            master.destroy()
            return

        master.attributes('-fullscreen', True)
        master.attributes('-topmost', True)
        master.configure(bg='black')
        self.video_frame = tk.Frame(master, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- REMOVED: All self.master.bind() calls are no longer needed ---

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

        self.events = self.player.event_manager()
        self.events.event_attach(
            vlc.EventType.MediaPlayerEndReached,
            lambda e: self.master.after(0, self.next_video)
        )
        self.events.event_attach(vlc.EventType.MediaPlayerSeekableChanged, self.on_seekable_changed)
        self.events.event_attach(vlc.EventType.MediaPlayerPositionChanged, self.on_position_changed)
        
        # --- NEW: Setup and start the global keyboard listener ---
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.load_video()
    
    # --- NEW: Global key press handler ---
    def on_press(self, key):
        try:
            # Handle character keys (for numbers 0-9)
            if '0' <= key.char <= '9':
                self.seek_to_percentage(int(key.char) / 10.0)
            elif key.char == 'n':
                self.next_video()
            elif key.char == 'p':
                self.prev_video()
                
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

    def seek_forward(self): # MODIFIED: Removed 'event' parameter
        if self.player.is_playing():
            self.perform_seek(5000)

    def seek_backward(self): # MODIFIED: Removed 'event' parameter
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
        media = self.instance.media_new(video_path)
        self.player.set_media(media)
        self.player.play()
        self.master.after(100, lambda: self.master.focus_force())

    def next_video(self): # MODIFIED: Removed 'event' parameter
        if self.playlist_index < len(self.playlist) - 1:
            self.playlist_index += 1
            self.load_video()

    def prev_video(self): # MODIFIED: Removed 'event' parameter
        if self.playlist_index > 0:
            self.playlist_index -= 1
            self.load_video()

    def quit_player(self): # MODIFIED: Removed 'event' parameter
        # --- NEW: Stop the listener for a clean exit ---
        self.listener.stop()
        self.player.stop()
        self.show_cursor()
        self.master.destroy()

# --- Entry Point ---
if __name__ == "__main__":
    try:
        playlists, playlist_name = get_playlist_status()
        playlist_file = os.path.join(PLAYLISTS_PATH, playlist_name)
        playlist = playlists.get(playlist_name, [])
        if not playlist:
            messagebox.showerror("Error", "No valid media found in playlist!")
            sys.exit(1)
        playlist = [os.path.join(DECRYPTED_MEDIA_PATH, video) for video in playlist["media"]]
        root = tk.Tk()
        app = FullscreenPlayer(root, playlist)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error:\n{e}")
    finally:
        # Ensure cursor is restored even on error
        try:
            # Create a temporary window to safely show cursor
            temp = tk.Tk()
            temp.withdraw()
            ctypes.windll.user32.ShowCursor(True)
            temp.destroy()
        except:
            pass