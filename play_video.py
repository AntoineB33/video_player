import os
import sys
import ctypes
from url_to_filename import url_to_filename
from config import MEDIA_PATH, PLAYLISTS_PATH, DEFAULT_PLAYLIST_FILE

# Set path to directory containing libvlc.dll
libvlc_dir = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(libvlc_dir)  # Windows 10+
ctypes.CDLL(os.path.join(libvlc_dir, "libvlc.dll"))

import tkinter as tk
from tkinter import messagebox
import vlc
import base64

class FullscreenPlayer:
    def __init__(self, master, playlist):
        self.master = master
        self.playlist = playlist
        self.playlist_index = 0
        
        if not self.playlist:
            messagebox.showerror("Error", "Playlist is empty!")
            master.destroy()
            return

        master.attributes('-fullscreen', True)
        master.attributes('-topmost', True)
        master.configure(bg='black')
        master.bind("<Escape>", self.quit_player)
        # Replace number key bindings for navigation
        master.bind("n", self.next_video)  # 'n' for next video
        master.bind("p", self.prev_video)  # 'p' for previous video
        master.bind("<Right>", self.seek_forward)
        master.bind("<Left>", self.seek_backward)
        
        # Add bindings for percentage seeking (0-9 keys)
        for i in range(10):
            master.bind(str(i), lambda event, x=i: self.seek_to_percentage(x / 10.0))

        self.video_frame = tk.Frame(master, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        master.update()  # Force update to ensure frame is created
        master.focus_force()

        # VLC configuration options
        vlc_options = [
            "--no-osd",
            "--no-video-title-show",
            "--quiet",
            "--no-embedded-video",
            
            "--avcodec-hw=dxva2",  # Windows hardware decoding

            # "--avcodec-hw=any --codec av1",

            # "--ffmpeg-hw",          # Enable FFmpeg hardware acceleration
            "--drop-late-frames",   # Prevent A/V desync
            "--skip-frames",        # Allow frame skipping during seeks

            "--winrt-d3dcontext",  # Windows 11 Direct3D11 optimization
            "--d3d11",             # Force Direct3D11 rendering
            # "--enable-sout",       # Enable streaming output (if needed)
            "--network-caching=1500"  # Adjust buffer for network streams
        ]

        self.instance = vlc.Instance(" ".join(vlc_options))
        self.player = self.instance.media_player_new()
        self.player.set_hwnd(self.video_frame.winfo_id())

        self.events = self.player.event_manager()
        self.events.event_attach(vlc.EventType.MediaPlayerEndReached, self.next_video)
        # self.print_decoder_info()  # Print decoder info for debugging

        self.load_video()
    
    def print_decoder_info(self):
        print(self.player.get_marquee())  # Check "decoder" in output
        print(self.player.video_get_decode_rate())  # Should be near frame rate

    def seek_forward(self, event):
        """Skip forward 5 seconds"""
        if self.player.is_playing():
            current_time = self.player.get_time()
            self.player.set_time(current_time + 5000)  # 5 seconds in ms

    def seek_backward(self, event):
        """Skip backward 5 seconds"""
        if self.player.is_playing():
            current_time = self.player.get_time()
            self.player.set_time(max(0, current_time - 5000))  # 5 seconds in ms

    def seek_to_percentage(self, ratio):
        """Jump to a specific percentage of the video"""
        if self.player.is_playing():
            total_duration = self.player.get_length()
            if total_duration > 0:
                target_time = int(total_duration * ratio)
                self.player.set_time(target_time)

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

def read_playlist(file_path):
    """Read video paths from a text file"""
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"Playlist file not found:\n{file_path}")
        return []
        
    with open(file_path, 'r') as f:
        return [
            os.path.join(MEDIA_PATH, url_to_filename(line.strip())) + ".mp4"
            for line in f.readlines()
            if line.strip()
        ]

# --- Entry Point ---
if __name__ == "__main__":
    # --- Ask user for playlist name via CMD ---
    print("available playlists:")
    all_playlists = os.listdir(PLAYLISTS_PATH)
    default_playlist = None
    with open(DEFAULT_PLAYLIST_FILE, "r") as f:
        default_playlist = f.read().strip()
        if default_playlist not in all_playlists:
            default_playlist = None
    for file in all_playlists:
        if file.endswith(".txt"):
            suffix = ''
            if_default = default_playlist is not None and file == default_playlist
            if if_default:
                suffix = " (default)"
            paths = read_playlist(os.path.join(PLAYLISTS_PATH, file))
            with open(os.path.join(PLAYLISTS_PATH, file), 'r') as f:
                videos = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip()
                ]
            for i, path in enumerate(paths):
                if not os.path.exists(path):
                    suffix += f"\n\t(video {i + 1} missing : {videos[i]})"
            print(f"{'>' if if_default else '-'} {file}{suffix}")
    playlist_name = input("Playlist file: ").strip()
    if not playlist_name:
        if default_playlist is not None:
            playlist_name = default_playlist

    if not playlist_name.endswith(".txt"):
        playlist_name += ".txt"

    playlist_file = os.path.join(PLAYLISTS_PATH, playlist_name)
    try:
        playlist = read_playlist(playlist_file)
        if not playlist:
            messagebox.showerror("Error", "No valid videos found in playlist!")
            sys.exit(1)
        with open(DEFAULT_PLAYLIST_FILE, "w") as f:
            f.write(playlist_name)
        root = tk.Tk()
        app = FullscreenPlayer(root, playlist)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error:\n{e}")