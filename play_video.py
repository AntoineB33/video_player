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
        self.video_frame.config(cursor="none")
        master.config(cursor="none")

        master.update()  # Force update to ensure frame is created
        master.focus_force()

        # Improved VLC configuration options for better seeking
        vlc_options = [
            "--no-osd",
            "--no-video-title-show",
            "--quiet",
            "--no-embedded-video",
            
            # Hardware acceleration
            "--avcodec-hw=dxva2",
            
            # Seeking and sync improvements
            "--avcodec-fast",              # Fast decoding for seeking
            "--sout-avcodec-strict=-2",    # Allow experimental features
            "--audio-desync=0",            # Reset audio desync
            "--audio-replay-gain-mode=none", # Disable audio processing that can cause delays
            
            # Buffer and caching optimizations
            "--file-caching=300",          # Reduce file caching (default 1000ms)
            "--network-caching=300",       # Reduce network caching
            "--live-caching=300",          # Reduce live stream caching
            
            # Frame handling
            "--drop-late-frames",          # Drop frames that are too late
            "--skip-frames",               # Allow frame skipping
            "--avcodec-skiploopfilter=4",  # Skip loop filter for faster decoding
            
            # Video output optimizations
            "--winrt-d3dcontext",
            "--d3d11",
            
            # Seeking precision
            "--avcodec-threads=0",         # Use all CPU cores for decoding
            "--verbose=0"                  # Reduce logging overhead
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
        
        # Method 1: Basic seeking with sync reset
        self.player.set_time(target_time)
        
        # Method 2: Alternative - pause/seek/play for better sync (uncomment to try)
        # was_playing = self.player.is_playing()
        # if was_playing:
        #     self.player.pause()
        # self.player.set_time(target_time)
        # if was_playing:
        #     # Small delay before resuming
        #     threading.Timer(0.05, self.player.play).start()

    def seek_to_percentage(self, ratio):
        """Jump to a specific percentage of the video with better sync"""
        if self.player.is_playing():
            self.seeking = True
            
            # Method 1: Using set_position (often more accurate for large jumps)
            self.player.set_position(ratio)
            
            # Method 2: Alternative using time calculation (uncomment to try instead)
            # total_duration = self.player.get_length()
            # if total_duration > 0:
            #     target_time = int(total_duration * ratio)
            #     self.player.set_time(target_time)

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
        self.master.destroy()

# Alternative approach: Mute during seeking
class FullscreenPlayerWithMuting(FullscreenPlayer):
    """Alternative implementation that mutes audio during seeking"""
    
    def __init__(self, master, playlist):
        super().__init__(master, playlist)
        self.original_volume = 100
        self.mute_timer = None
    
    def perform_seek(self, offset_ms):
        """Seek with temporary audio muting"""
        # Store original volume and mute
        self.original_volume = self.player.audio_get_volume()
        self.player.audio_set_volume(0)
        
        # Perform the seek
        current_time = self.player.get_time()
        target_time = max(0, current_time + offset_ms)
        self.player.set_time(target_time)
        
        # Cancel any existing timer
        if self.mute_timer:
            self.mute_timer.cancel()
        
        # Restore audio after a short delay
        self.mute_timer = threading.Timer(0.3, self.restore_audio)
        self.mute_timer.start()
    
    def restore_audio(self):
        """Restore audio volume after seeking"""
        self.player.audio_set_volume(self.original_volume)
        self.mute_timer = None
    
    def seek_to_percentage(self, ratio):
        """Jump to percentage with muting"""
        # Store original volume and mute
        self.original_volume = self.player.audio_get_volume()
        self.player.audio_set_volume(0)
        
        # Perform the seek
        self.player.set_position(ratio)
        
        # Cancel any existing timer
        if self.mute_timer:
            self.mute_timer.cancel()
        
        # Restore audio after a longer delay for percentage jumps
        self.mute_timer = threading.Timer(0.5, self.restore_audio)
        self.mute_timer.start()

# Usage example:
# To use the muting version, replace FullscreenPlayer with FullscreenPlayerWithMuting
# player = FullscreenPlayerWithMuting(root, playlist)

# --- Entry Point ---
if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status()
    playlist_file = os.path.join(PLAYLISTS_PATH, playlist_name)
    try:
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