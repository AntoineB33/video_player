import os
import sys
import ctypes
from url_to_filename import url_to_filename

# Set path to directory containing libvlc.dll
libvlc_dir = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(libvlc_dir)  # Windows 10+
ctypes.CDLL(os.path.join(libvlc_dir, "libvlc.dll"))

import tkinter as tk
from tkinter import messagebox
import vlc
import base64

MEDIA_PATH = r"C:\Users\N6506\Home\health\entertainment\news_underground\mediaSorter\programs\mediaSorter_Python\data\media"

# --- Ask user for playlist name via CMD ---
print("Enter the name of the playlist file (e.g., hey2.txt):")
playlist_name = input("Playlist file: ").strip()

if not playlist_name.endswith(".txt"):
    playlist_name += ".txt"

PLAYLIST_FILE = os.path.join("playlists", playlist_name)


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
        master.bind("3", self.next_video)
        master.bind("1", self.prev_video)
        master.bind("<Right>", self.seek_forward)
        master.bind("<Left>", self.seek_backward)

        self.video_frame = tk.Frame(master, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        master.update()  # Force update to ensure frame is created
        master.focus_set()  # Ensure window has focus for key events

        vlc_options = "--no-osd --no-video-title-show --quiet"
        self.instance = vlc.Instance(vlc_options)
        self.player = self.instance.media_player_new()
        self.player.set_hwnd(self.video_frame.winfo_id())

        self.events = self.player.event_manager()
        self.events.event_attach(vlc.EventType.MediaPlayerEndReached, self.next_video)

        self.load_video()

        self.master.after(100, self.ensure_focus)
        self.ensure_focus_timer = None

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
    try:
        playlist = read_playlist(PLAYLIST_FILE)
        if not playlist:
            messagebox.showerror("Error", "No valid videos found in playlist!")
            sys.exit(1)
            
        root = tk.Tk()
        app = FullscreenPlayer(root, playlist)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error:\n{e}")