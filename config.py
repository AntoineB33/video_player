import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MEDIA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "mediaSorter", "media")
PLAYLISTS_PATH = os.path.join(BASE_DIR, "data", "playlists")
DEFAULT_PLAYLIST_FILE = os.path.join(BASE_DIR, "data", "default_playlist.txt")
