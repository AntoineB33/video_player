from encrypt import get_playlist_status
from generate_sortings import instr_struct


if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status(only_numbers = True)
    while True:
        if playlists[playlist_name]["missing"]:
            input(f"Missing file: {playlists[playlist_name]['missing'][0]}")
        else:
            playlists, playlist_name = get_playlist_status(only_numbers = True)
