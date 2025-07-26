import os
import sys
import ctypes

# Set path to directory containing libvlc.dll
libvlc_dir = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(libvlc_dir)  # Windows 10+
ctypes.CDLL(os.path.join(libvlc_dir, "libvlc.dll"))

import tkinter as tk
from tkinter import messagebox
import vlc


# --- CONFIGURATION ---
# Replace "my_awesome_video.mp4" with the exact name of your video file.
VIDEO_FILE = r"C:\Users\N6506\Downloads\Thunderbolts.2025.2160p.iT.WEB-DL.DV.HDR10+.ENG.LATINO.DDP5.1.Atmos.H265.MP4-BEN.THE.MEN\Thunderbolts.2025.2160p.iT.WEB-DL.DV.HDR10+[Ben The Men].mp4"

# --- APPLICATION ---

class FullscreenPlayer:
    def __init__(self, master, video_path):
        self.master = master
        self.video_path = video_path

        # Check if the video file actually exists before starting
        if not os.path.exists(self.video_path):
            # Display an error pop-up and quit if the file is not found
            messagebox.showerror("File Not Found", 
                                 f"The video file '{self.video_path}' was not found in this directory.")
            self.master.destroy()
            return

        # Configure the main window to be completely borderless and fullscreen
        self.master.attributes('-fullscreen', True)
        self.master.attributes('-topmost', True)
        self.master.configure(bg='black')
        self.master.bind("<Escape>", self.quit_player) # Bind Escape key to the quit function

        # Create a frame to hold the video output
        self.video_frame = tk.Frame(self.master, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Set up the VLC instance and player
        # --no-osd disables the on-screen display (like volume bars)
        # --no-video-title-show disables the video title from appearing at the start
        vlc_options = "--no-osd --no-video-title-show --quiet"
        self.instance = vlc.Instance(vlc_options)
        self.player = self.instance.media_player_new()
        
        # Tell VLC to draw on our tkinter frame
        self.player.set_hwnd(self.video_frame.winfo_id())

        # Load the media
        self.media = self.instance.media_new(self.video_path)
        self.player.set_media(self.media)

        # Set up an event handler to close the app when the video finishes
        events = self.player.event_manager()
        events.event_attach(vlc.EventType.MediaPlayerEndReached, self.quit_player)

        # Start playback
        self.player.play()

    def quit_player(self, event=None):
        """Stops video playback and closes the application."""
        self.player.stop()
        self.master.destroy()

# --- Entry Point ---
if __name__ == "__main__":
    try:
        root = tk.Tk()
        # Hide the default empty window that appears for a split second
        root.withdraw() 
        player = FullscreenPlayer(root, VIDEO_FILE)
        # Only start the main loop if the player was initialized successfully
        if 'normal' == root.state():
             root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")