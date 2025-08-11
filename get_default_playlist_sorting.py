import os
import pickle
import pyperclip
from encrypt import get_playlist_status
from config import DEFAULT_PLAYLIST_FILE, PLAYLISTS_PATH



if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status(True)
    with open(os.path.join(PLAYLISTS_PATH, playlist_name), "rb") as f:
        saved = pickle.load(f)
        pyperclip.copy('\n'.join(['\t'.join(row) for row in saved["output"]["new_table"]]))