import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data")
# MEDIA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "mediaSorter", "media")
ENCRYPTED_MEDIA_PATH = os.path.join(DATA_PATH, "encrypted_media")
DECRYPTED_MEDIA_PATH = os.path.join(DATA_PATH, "decrypted_media")
KEY_PATH = os.path.join(DATA_PATH, "file.key")
PLAYLISTS_PATH = os.path.join(DATA_PATH, "playlists")
DEFAULT_PLAYLIST_FILE = os.path.join(DATA_PATH, "default_playlist.txt")
