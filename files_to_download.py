from config import ENCRYPTED_MEDIA_PATH, PLAYLISTS_PATH
import os
from encrypt import get_playlist_status


if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status()