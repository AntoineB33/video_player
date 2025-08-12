import os
import sys
import ctypes
from encrypt import get_playlist_status
from config import DECRYPTED_MEDIA_PATH, PLAYLISTS_PATH, DEFAULT_PLAYLIST_FILE
import pickle

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
        self.seeking = False  # Flag to track if we're currently seeking
        
        if not self.playlist:
            messagebox.showerror("Error", "Playlist is empty!")
            master.destroy()
            return

        master.attributes('-fullscreen', True)
        master.attributes('-topmost', True)
        master.configure(bg='black')
        self.video_frame = tk.Frame(master, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_frame.bind("<Escape>", self.quit_player)
        self.video_frame.bind("n", self.next_video)
        self.video_frame.bind("p", self.prev_video)
        self.video_frame.bind("<Right>", self.seek_forward)
        self.video_frame.bind("<Left>", self.seek_backward)
        for i in range(10):
            self.video_frame.bind(str(i), lambda event, x=i: self.seek_to_percentage(x / 10.0))
        
        # Hide cursor using Windows API
        self.hide_cursor()
        
        master.update()  # Force update to ensure frame is created
        master.focus_force()

        # Improved VLC configuration options for better seeking
        vlc_options = [
            "--no-osd",
            "--no-video-title-show",
            "--quiet",
            "--no-embedded-video",
            "--avcodec-hw=dxva2",
            "--avcodec-fast",
            "--sout-avcodec-strict=-2",
            "--audio-desync=0",
            "--audio-replay-gain-mode=none",
            "--file-caching=300",
            "--network-caching=300",
            "--live-caching=300",
            "--drop-late-frames",
            "--skip-frames",
            "--avcodec-skiploopfilter=4",
            "--winrt-d3dcontext",
            "--d3d11",
            "--avcodec-threads=0",
            "--verbose=0"
        ]

        self.instance = vlc.Instance(" ".join(vlc_options))
        self.player = self.instance.media_player_new()
        self.player.set_hwnd(self.video_frame.winfo_id())

        self.events = self.player.event_manager()
        self.events.event_attach(
            vlc.EventType.MediaPlayerEndReached,
            lambda e: self.master.after(0, self.next_video)
        )
        
        # Add event handlers for seeking
        self.events.event_attach(vlc.EventType.MediaPlayerSeekableChanged, self.on_seekable_changed)
        self.events.event_attach(vlc.EventType.MediaPlayerPositionChanged, self.on_position_changed)

        self.load_video()
    
    def hide_cursor(self):
        """Hide cursor using Windows API"""
        try:
            # Get the window handle
            hwnd = ctypes.windll.user32.GetParent(self.master.winfo_id())
            
            # Hide cursor using Windows API
            ctypes.windll.user32.ShowCursor(False)
            
            # Remove cursor style from window
            style = ctypes.windll.user32.GetWindowLongPtrW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style & ~WS_CURSOR)
        except Exception as e:
            print(f"Error hiding cursor: {e}")
    
    def show_cursor(self):
        """Show cursor using Windows API"""
        try:
            # Get the window handle
            hwnd = ctypes.windll.user32.GetParent(self.master.winfo_id())
            
            # Show cursor using Windows API
            ctypes.windll.user32.ShowCursor(True)
            
            # Restore cursor style to window
            style = ctypes.windll.user32.GetWindowLongPtrW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style | WS_CURSOR)
        except Exception as e:
            print(f"Error showing cursor: {e}")
    
    def on_seekable_changed(self, event):
        """Handle when seeking capability changes"""
        pass
    
    def on_position_changed(self, event):
        """Handle position changes - can be used to detect when seeking is complete"""
        if self.seeking:
            # Small delay to ensure audio catches up
            threading.Timer(0.1, self.finish_seek).start()
    
    def finish_seek(self):
        """Complete the seeking operation"""
        self.seeking = False

    def seek_forward(self, event):
        """Skip forward 5 seconds with improved sync"""
        if self.player.is_playing():
            self.perform_seek(5000)

    def seek_backward(self, event):
        """Skip backward 5 seconds with improved sync"""
        if self.player.is_playing():
            self.perform_seek(-5000)

    def perform_seek(self, offset_ms):
        """Perform seeking with audio/video sync handling"""
        self.seeking = True
        current_time = self.player.get_time()
        target_time = max(0, current_time + offset_ms)
        self.player.set_time(target_time)

    def seek_to_percentage(self, ratio):
        """Jump to a specific percentage of the video with better sync"""
        if self.player.is_playing():
            self.seeking = True
            self.player.set_position(ratio)

    def load_video(self):
        """Load and play the current video from the playlist"""
        if not 0 <= self.playlist_index < len(self.playlist):
            return
            
        video_path = self.playlist[self.playlist_index]
        if not os.path.exists(video_path):
            messagebox.showwarning("File Missing", f"Video not found:\n{video_path}")
            return
            
        media = self.instance.media_new(video_path)
        self.player.set_media(media)
        self.player.play()
        
        # Restore focus after a brief delay
        self.master.after(100, lambda: self.video_frame.focus_force())

    def next_video(self, event=None):
        """Load the next video in the playlist"""
        if self.playlist_index < len(self.playlist) - 1:
            self.playlist_index += 1
            self.load_video()
        else:
            messagebox.showinfo("End", "End of playlist reached")

    def prev_video(self, event=None):
        """Load the previous video in the playlist"""
        if self.playlist_index > 0:
            self.playlist_index -= 1
            self.load_video()

    def quit_player(self, event=None):
        self.player.stop()
        self.show_cursor()  # Restore cursor before exit
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
        # Use pickle to save default playlist name
        with open(DEFAULT_PLAYLIST_FILE, "wb") as f:
            pickle.dump(playlist_name, f)
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