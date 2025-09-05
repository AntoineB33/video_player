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
from PIL import Image, ImageTk, ImageSequence
from generate_sortings import instr_struct
from encrypt import get_playlist_status
from config import DECRYPTED_MEDIA_PATH, PLAYLISTS_PATH, DEFAULT_MUSICS_FILE

# Windows API constants
GWL_STYLE = -16
WS_CURSOR = 0x0001  # Cursor visibility flag
MUSIC_VOLUME = 30  # Volume for background music (0-100)
VOLUME_STEP = 5    # Volume change step (0-100)

# Image and GIF constants
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
GIF_EXTENSIONS = {'.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}

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
        
        # NEW: Volume control attributes
        self.video_volume = 100  # Video volume (0-100)
        self.music_volume = MUSIC_VOLUME  # Background music volume (0-100)
        
        # NEW: Image/GIF handling attributes
        self.current_media_type = None  # 'video', 'image', or 'gif'
        self.image_label = None
        self.gif_frames = []
        self.gif_frame_index = 0
        self.gif_duration = 0
        self.gif_animation_id = None
        self.gif_start_time = 0
        self.gif_total_duration = 0
        
        if not self.playlist:
            messagebox.showerror("Error", "Playlist is empty!")
            master.destroy()
            return



        # master.attributes('-fullscreen', True)
        # master.attributes('-topmost', True)
        # master.configure(bg='black')
        
        # --- Change: Make windowed instead of fullscreen ---
        window_width = 1280
        window_height = 720
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        master.geometry(f"{window_width}x{window_height}+{x}+{y}")
        master.configure(bg='black')
        # Remove fullscreen and topmost attributes
        # master.attributes('-fullscreen', True)
        # master.attributes('-topmost', True)



        
        # Create container for both video and image display
        self.media_frame = tk.Frame(master, bg='black')
        self.media_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame for VLC
        self.video_frame = tk.Frame(self.media_frame, bg='black')
        
        # Label for images and GIFs
        self.image_label = tk.Label(self.media_frame, bg='black')

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
        self.player.audio_set_volume(self.video_volume)  # Set initial video volume

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

        self.load_media()
    
    def get_media_type(self, file_path):
        """Determine the type of media file"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return 'image'
        elif ext in GIF_EXTENSIONS:
            return 'gif'
        elif ext in VIDEO_EXTENSIONS:
            return 'video'
        else:
            # Default to video for unknown extensions
            return 'video'
    
    def hide_video_show_image(self):
        """Hide video frame and show image label"""
        self.video_frame.pack_forget()
        self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def hide_image_show_video(self):
        """Hide image label and show video frame"""
        self.image_label.pack_forget()
        self.video_frame.pack(fill=tk.BOTH, expand=True)
    
    def stop_gif_animation(self):
        """Stop any running GIF animation"""
        if self.gif_animation_id:
            self.master.after_cancel(self.gif_animation_id)
            self.gif_animation_id = None
    
    def load_image(self, image_path):
        """Load and display a static image"""
        try:
            # Load and resize image to fit screen
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()
            
            img = Image.open(image_path)
            
            # Calculate scaling to fit screen while maintaining aspect ratio
            img_ratio = img.width / img.height
            screen_ratio = screen_width / screen_height
            
            if img_ratio > screen_ratio:
                # Image is wider relative to screen
                new_width = screen_width
                new_height = int(screen_width / img_ratio)
            else:
                # Image is taller relative to screen
                new_height = screen_height
                new_width = int(screen_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            self.hide_video_show_image()
            
            # Start or resume background music for static images
            self.play_or_resume_background_music()
            
            print(f"Loaded image: {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self.next_video()
    
    def load_gif(self, gif_path):
        """Load and display an animated GIF"""
        try:
            # Load GIF and extract frames
            img = Image.open(gif_path)
            self.gif_frames = []
            self.gif_total_duration = 0
            
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()
            
            # Calculate scaling
            img_ratio = img.width / img.height
            screen_ratio = screen_width / screen_height
            
            if img_ratio > screen_ratio:
                new_width = screen_width
                new_height = int(screen_width / img_ratio)
            else:
                new_height = screen_height
                new_width = int(screen_height * img_ratio)
            
            # Extract all frames
            for frame in ImageSequence.Iterator(img):
                # Get frame duration (in milliseconds)
                duration = frame.info.get('duration', 100)  # Default 100ms if not specified
                self.gif_total_duration += duration
                
                # Resize and convert frame
                frame = frame.convert('RGBA')
                frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(frame)
                
                self.gif_frames.append({
                    'image': photo,
                    'duration': duration
                })
            
            # Convert total duration to seconds
            self.gif_total_duration = self.gif_total_duration / 1000.0
            
            self.hide_video_show_image()
            
            # Start or resume background music for GIFs
            self.play_or_resume_background_music()
            
            # Start GIF animation
            self.gif_frame_index = 0
            self.gif_start_time = time.time()
            self.animate_gif()
            
            print(f"Loaded GIF: {os.path.basename(gif_path)} ({len(self.gif_frames)} frames, {self.gif_total_duration:.1f}s)")
            
        except Exception as e:
            print(f"Error loading GIF {gif_path}: {e}")
            self.next_video()
    
    def animate_gif(self):
        """Animate the GIF by cycling through frames"""
        if not self.gif_frames or self.current_media_type != 'gif':
            return
        
        # Display current frame
        current_frame = self.gif_frames[self.gif_frame_index]
        self.image_label.configure(image=current_frame['image'])
        
        # Move to next frame
        self.gif_frame_index = (self.gif_frame_index + 1) % len(self.gif_frames)
        
        # Check if we completed a full cycle
        if self.gif_frame_index == 0:
            elapsed_time = time.time() - self.gif_start_time
            
            # If GIF is 30 seconds or longer, or if we've been playing for more than 30 seconds, go to next media
            if self.gif_total_duration >= 30.0 or elapsed_time >= 30.0:
                print(f"GIF finished (duration: {self.gif_total_duration:.1f}s, played: {elapsed_time:.1f}s)")
                self.next_video()
                return
            
            # Otherwise, loop the animation
            self.gif_start_time = time.time()  # Reset timer for next loop
        
        # Schedule next frame
        duration = current_frame['duration']
        self.gif_animation_id = self.master.after(duration, self.animate_gif)
    
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
                self.music_player.audio_set_volume(self.music_volume)
                
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
            if not self.video_has_audio:
                self.master.after(0, self.play_or_resume_background_music)
            elif self.video_has_audio:
                self.master.after(0, self.pause_background_music)
        
        # Run audio check in a separate thread to avoid blocking
        threading.Thread(target=check_audio, daemon=True).start()
    
    # MODIFIED: Helper to load and play the current music track
    def _load_and_play_current_music_track(self):
        """Loads and plays the music track at the current index."""
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
                print(f"Music file not found, skipping: {music_path}")
                self.next_music()
        except Exception as e:
            print(f"Error loading music track: {e}")

    # MODIFIED: Start or resume background music
    def play_or_resume_background_music(self):
        """Starts a new music track or resumes a paused one."""
        if not self.music_player or not self.music_playlist or not self.music_enabled:
            return
        
        try:
            state = self.music_player.get_state()
            if state == vlc.State.Paused:
                self.music_player.play()
                print("Resuming background music.")
            elif state not in [vlc.State.Playing, vlc.State.Buffering]:
                self._load_and_play_current_music_track()
        except Exception as e:
            print(f"Error in play_or_resume_background_music: {e}")

    # MODIFIED: Pause background music
    def pause_background_music(self):
        """Pauses the background music if it is playing."""
        if self.music_player and self.music_player.is_playing():
            self.music_player.pause()
            print("Pausing background music.")
    
    # MODIFIED: Go to next music track
    def next_music(self):
        """Move to next music track in playlist and plays it."""
        if not self.music_playlist:
            return
        
        self.current_music_index = (self.current_music_index + 1) % len(self.music_playlist)
        self._load_and_play_current_music_track()
    
    # MODIFIED: Toggle music on/off
    def toggle_music(self):
        """Toggle background music on/off"""
        self.music_enabled = not self.music_enabled
        if self.music_enabled:
            # If toggling on, play only if the current media is silent.
            if self.current_media_type in ['image', 'gif'] or not self.video_has_audio:
                self.play_or_resume_background_music()
        else:
            # If toggling off, always pause.
            self.pause_background_music()
        print(f"Background music {'enabled' if self.music_enabled else 'disabled'}")
    
    # NEW: Volume control methods
    def volume_up(self):
        """Increase volume of currently playing audio"""
        if self.current_media_type == 'video' and self.video_has_audio:
            # Control video volume
            self.video_volume = min(100, self.video_volume + VOLUME_STEP)
            self.player.audio_set_volume(self.video_volume)
            print(f"Video volume: {self.video_volume}%")
        elif self.music_player and (self.current_media_type in ['image', 'gif'] or not self.video_has_audio):
            # Control background music volume
            self.music_volume = min(100, self.music_volume + VOLUME_STEP)
            self.music_player.audio_set_volume(self.music_volume)
            print(f"Background music volume: {self.music_volume}%")
    
    def volume_down(self):
        """Decrease volume of currently playing audio"""
        if self.current_media_type == 'video' and self.video_has_audio:
            # Control video volume
            self.video_volume = max(0, self.video_volume - VOLUME_STEP)
            self.player.audio_set_volume(self.video_volume)
            print(f"Video volume: {self.video_volume}%")
        elif self.music_player and (self.current_media_type in ['image', 'gif'] or not self.video_has_audio):
            # Control background music volume
            self.music_volume = max(0, self.music_volume - VOLUME_STEP)
            self.music_player.audio_set_volume(self.music_volume)
            print(f"Background music volume: {self.music_volume}%")
    
    # --- Global key press handler ---
    def on_press(self, key):
        try:
            # Handle character keys (for numbers 0-9)
            if '0' <= key.char <= '9' and self.current_media_type == 'video':
                self.seek_to_percentage(int(key.char) / 10.0)
            elif key.char == 'n':
                self.next_video()
            elif key.char == 'p':
                self.prev_video()
            elif key.char == 'm':  # NEW: Toggle music
                self.toggle_music()
                
        except AttributeError:
            if key == keyboard.Key.esc:
                self.quit_player()
            elif key == keyboard.Key.right and self.current_media_type == 'video':
                self.seek_forward()
            elif key == keyboard.Key.left and self.current_media_type == 'video':
                self.seek_backward()
            elif key == keyboard.Key.up:  # NEW: Volume up
                self.volume_up()
            elif key == keyboard.Key.down:  # NEW: Volume down
                self.volume_down()

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
        if self.player.is_playing() and self.current_media_type == 'video':
            self.perform_seek(5000)

    def seek_backward(self):
        if self.player.is_playing() and self.current_media_type == 'video':
            self.perform_seek(-5000)

    def perform_seek(self, offset_ms):
        if self.current_media_type == 'video':
            self.seeking = True
            current_time = self.player.get_time()
            target_time = max(0, current_time + offset_ms)
            self.player.set_time(target_time)

    def seek_to_percentage(self, ratio):
        if self.player.is_playing() and self.current_media_type == 'video':
            self.seeking = True
            self.player.set_position(ratio)

    def load_media(self):
        """Load current media (video, image, or GIF)"""
        if not 0 <= self.playlist_index < len(self.playlist):
            return
        
        media_path = self.playlist[self.playlist_index]
        if not os.path.exists(media_path):
            messagebox.showwarning("File Missing", f"Media not found:\n{media_path}")
            return
        
        # Stop any ongoing animations
        self.stop_gif_animation()
        # MODIFICATION: Do NOT stop music here, allow it to play between silent media
        
        new_media_type = self.get_media_type(media_path)
        
        # Stop the video player if we're switching to non-video media
        if self.current_media_type == 'video' and new_media_type != 'video':
            self.player.stop()
        
        # Determine media type and load accordingly
        self.current_media_type = new_media_type
        
        if self.current_media_type == 'image':
            self.load_image(media_path)
        elif self.current_media_type == 'gif':
            self.load_gif(media_path)
        else:  # video
            self.load_video(media_path)

    def load_video(self, video_path):
        """Load and play a video file"""
        self.hide_image_show_video()
        
        media = self.instance.media_new(video_path)
        self.player.set_media(media)
        self.player.play()
        self.master.after(100, lambda: self.master.focus_force())
        
        print(f"Loaded video: {os.path.basename(video_path)}")

    def next_video(self):
        if self.playlist_index < len(self.playlist) - 1:
            self.playlist_index += 1
            self.load_media()

    def prev_video(self):
        if self.playlist_index > 0:
            self.playlist_index -= 1
            self.load_media()

    def quit_player(self):
        # --- Stop the listener, animations, and music for a clean exit ---
        self.listener.stop()
        self.stop_gif_animation()
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