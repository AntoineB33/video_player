from generate_sortings import solver
from encrypt import get_playlist_status

if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status(get_all_new_table=True)
    solver(playlists[playlist_name], [], True)