import pyperclip
from encrypt import get_playlist_status
from generate_sortings import instr_struct



if __name__ == "__main__":
    playlist_name = None
    while True:
        playlists, playlist_name = get_playlist_status(given_default=playlist_name, only_numbers=True)
        d = playlists[playlist_name]["missing"]
        if list := d["medium"] + d["music"]:
            pyperclip.copy(list[0])
            input(f"Missing file: {list[0]}")
        else:
            break
