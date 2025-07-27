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
VIDEO_FILE = r"C:\Users\N6506\Downloads\Thunderbolts.2025.2160p.iT.WEB-DL.DV.HDR10+.ENG.LATINO.DDP5.1.Atmos.H265.MP4-BEN.THE.MEN\Thunderbolts.2025.2160p.iT.WEB-DL.DV.HDR10+[Ben The Men].mp4"

# Set path to directory containing libvlc.dll
libvlc_dir = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(libvlc_dir)
ctypes.CDLL(os.path.join(libvlc_dir, "libvlc.dll"))

class FullscreenPlayer:
    def __init__(self, master, video_path):
        self.master = master
        self.video_path = video_path

        if not os.path.exists(self.video_path):
            messagebox.showerror("File Not Found", f"Video file not found:\n{self.video_path}")
            master.destroy()
            return

        master.attributes('-fullscreen', True)
        master.attributes('-topmost', True)
        master.configure(bg='black')
        master.bind("<Escape>", self.quit_player)

        self.video_frame = tk.Frame(master, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        master.update()  # Force update to ensure frame is created

        vlc_options = "--no-osd --no-video-title-show --quiet"
        self.instance = vlc.Instance(vlc_options)
        self.player = self.instance.media_player_new()
        self.player.set_hwnd(self.video_frame.winfo_id())

        media = self.instance.media_new(self.video_path)
        self.player.set_media(media)

        events = self.player.event_manager()
        events.event_attach(vlc.EventType.MediaPlayerEndReached, self.quit_player)

        self.master.after(100, self.start_playback)  # Delay playback

    def start_playback(self):
        self.player.play()

    def quit_player(self, event=None):
        self.player.stop()
        self.master.destroy()

# --- Entry Point ---
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FullscreenPlayer(root, VIDEO_FILE)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error:\n{e}")
