from generate_sortings import solver, instr_struct
from encrypt import get_playlist_status

if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status(show_missings = False, get_all_new_table=True)
    if not playlists:
        input("Press Enter to exit.")
    else:
        solver(playlists[playlist_name], [], True, existing_pb=True)