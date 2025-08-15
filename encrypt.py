from cryptography.fernet import Fernet
from pathlib import Path
import os
import pickle
from url_to_filename import filename_to_url
from config import KEY_PATH, PLAYLISTS_PATH, DEFAULT_PLAYLIST_FILE, ENCRYPTED_MEDIA_PATH, DECRYPTED_MEDIA_PATH

def load_key(key_path=KEY_PATH):
    return Path(key_path).read_bytes()

def encrypt_file(in_path, out_path, key_path=KEY_PATH):
    key = load_key(key_path)
    f = Fernet(key)
    data = Path(in_path).read_bytes()          # reads entire file
    token = f.encrypt(data)                    # encrypted + integrity-protected
    Path(out_path).write_bytes(token)
    os.remove(in_path)

def decrypt_file(in_path, out_path, key_path=KEY_PATH):
    key = load_key(key_path)
    f = Fernet(key)
    token = Path(in_path).read_bytes()
    data = f.decrypt(token)                    # verifies integrity before decrypting
    Path(out_path).write_bytes(data)
    os.remove(in_path)

def files_with_same_stem(base: Path):
    """
    Return all files in base.parent whose stem matches base.stem,
    regardless of extension (supports multi-extensions like .tar.gz).
    """
    parent = base.parent
    stem = base.stem
    # e.g. if base is /data/report, this matches /data/report.*, /data/report.*.* etc.
    # The simple and fast choice is just one dot: report.*
    matches = [p for p in parent.glob(f"{stem}.*") if p.is_file()]
    return matches

def get_playlist_status(errors, ask_if_no_default = False, get_all_videos = False, get_all_new_table = False):
    # --- Ask user for playlist name via CMD ---
    if not get_all_videos:
        print("available playlists:")
    os.makedirs(PLAYLISTS_PATH, exist_ok=True)
    os.makedirs(DECRYPTED_MEDIA_PATH, exist_ok=True)
    os.makedirs(ENCRYPTED_MEDIA_PATH, exist_ok=True)
    all_playlists = [os.path.basename(f) for f in os.listdir(PLAYLISTS_PATH)]
    if not all_playlists:
        print("No playlists found.")
        return {}, ""
    if not os.path.exists(DEFAULT_PLAYLIST_FILE):
        with open(DEFAULT_PLAYLIST_FILE, "wb") as f:
            pickle.dump(None, f)
    with open(DEFAULT_PLAYLIST_FILE, "rb") as f:
        default_playlist = pickle.load(f)
    
    if ask_if_no_default and default_playlist is not None and default_playlist in all_playlists:
        return {}, default_playlist
    playlists = {}
    playlist_infos = []
    for file in all_playlists:
        suffix = ''
        if_default = default_playlist is not None and file == default_playlist
        if if_default:
            suffix = " (default)"
        if not ask_if_no_default:
            playlists[file] = {"media":[], "not_decrypted": []}
            file_path = os.path.join(PLAYLISTS_PATH, file)
            if not os.path.exists(file_path):
                errors.append(f"Playlist file referenced but does not exist: {file_path}")
                return
            with open(file_path, 'rb') as f:
                saved = pickle.load(f)
                playlists[file].update(saved)
            paths = playlists[file]["data"]["urls"]
            for i, path in enumerate(paths):
                files_found = files_with_same_stem(Path(DECRYPTED_MEDIA_PATH) / path)
                if files_found:
                    playlists[file]["media"].append(files_found[0])
                else:
                    url = filename_to_url(path)
                    files_found = files_with_same_stem(Path(ENCRYPTED_MEDIA_PATH) / path)
                    if files_found:
                        playlists[file]["not_decrypted"].append(files_found[0])
                        suffix += f"\n\t(medium {i + 1} not decrypted: {url})"
                    else:
                        suffix += f"\n\t(medium {i + 1} missing: {url})"
        playlist_infos.append((file, suffix, if_default))

    if get_all_videos:
        return playlists, default_playlist

    for idx, (file, suffix, if_default) in enumerate(playlist_infos, 1):
        print(f"{idx}. {file}{suffix}")

    playlist_name = ""
    selection = input("Select playlist by number or enter name: ").strip()
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(playlist_infos):
            playlist_name = playlist_infos[idx][0]
    elif not selection:
        if default_playlist is not None and default_playlist in all_playlists:
            playlist_name = default_playlist
        else:
            playlist_name = all_playlists[0]
    else:
        if selection in all_playlists:
            playlist_name = selection
        else:
            for file in all_playlists:
                if file.startswith(selection):
                    playlist_name = file
                    break
    if not playlist_name:
        input("Error: No valid playlist selected!")
        exit(1)
    return playlists, playlist_name

if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status()
    playlist_file = os.path.join(PLAYLISTS_PATH, playlist_name)
    try:
        playlist = playlists.get(playlist_name, [])
        if not playlist:
            raise ValueError("No valid media found in playlist!")
        # Use pickle to save default playlist name
        with open(DEFAULT_PLAYLIST_FILE, "wb") as f:
            pickle.dump(playlist_name, f)
        # decrypt all files in the playlist
        for medium in playlist["not_decrypted"]:
            try:
                in_path = Path(ENCRYPTED_MEDIA_PATH) / medium.name
                out_path = Path(DECRYPTED_MEDIA_PATH) / medium.name
                decrypt_file(in_path, out_path)
                print(f"Decrypted {filename_to_url(medium.name)}")
            # except FileNotFoundError:
            except KeyboardInterrupt:
                input(f"Error: File not found for decryption: {medium}")
                break
    # except Exception as e:
    except KeyboardInterrupt as e:
        input(f"Error: Unexpected error:\n{e}")

    # decrypt_file(
    #     in_path=Path(MEDIA_PATH) / "aHR0cHM6Ly9lY2NoaS5pd2FyYS50di92aWRlb3Mvbm1sZ3lpNDRuZnBqM2EyYQ.mp4",
    #     out_path=Path(MEDIA_PATH) / "example_decrypted.mp4"
    # )
    # Uncomment below to

    # folder_path = r"C:\Users\N6506\Home\health\entertainment\news_underground\mediaSorter\media"
    # # encrypt all files from folder_path to MEDIA_PATH
    # for file in Path(folder_path).glob("*.*"):
    #     in_path = file
    #     out_path = Path(MEDIA_PATH) / file.name
    #     encrypt_file(in_path, out_path)
    #     print(f"Encrypted {in_path} to {out_path}")
