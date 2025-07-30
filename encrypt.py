from cryptography.fernet import Fernet
from pathlib import Path
import os
from url_to_filename import url_to_filename
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

def read_playlist(file_path):
    """Read video paths from a text file"""
    if not os.path.exists(file_path):
        input(f"Error: Playlist file does not exist: {file_path}")
        return []
        
    with open(file_path, 'r') as f:
        return [
            url_to_filename(line.strip())
            for line in f.readlines()
            if line.strip()
        ]

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

def get_playlist_status():
    # --- Ask user for playlist name via CMD ---
    print("available playlists:")
    all_playlists = [f for f in os.listdir(PLAYLISTS_PATH) if f.endswith(".txt")]
    default_playlist = None
    with open(DEFAULT_PLAYLIST_FILE, "r") as f:
        default_playlist = f.read().strip()
        if default_playlist not in all_playlists:
            default_playlist = None

    playlists = {}
    playlist_infos = []
    for file in all_playlists:
        suffix = ''
        if_default = default_playlist is not None and file == default_playlist
        if if_default:
            suffix = " (default)"
        paths = read_playlist(os.path.join(PLAYLISTS_PATH, file))
        playlists[file] = [Path(DECRYPTED_MEDIA_PATH) / path for path in paths]
        with open(os.path.join(PLAYLISTS_PATH, file), 'r') as f:
            videos = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]
        for i, path in enumerate(playlists[file]):
            files_found = files_with_same_stem(path)
            if files_found:
                playlists[file][i] = files_found[0]
            else:
                suffix += f"\n\t(video {i + 1} missing : {videos[i]})"
        playlist_infos.append((file, suffix, if_default))

    for idx, (file, suffix, if_default) in enumerate(playlist_infos, 1):
        print(f"{idx}. {file}{suffix}")

    playlist_name = ""
    selection = input("Select playlist by number or enter name: ").strip()
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(playlist_infos):
            playlist_name = playlist_infos[idx][0]
    elif not selection:
        if default_playlist is not None:
            playlist_name = default_playlist
    else:
        playlist_name = selection
        if not playlist_name.endswith(".txt"):
            playlist_name += ".txt"

    return playlists, playlist_name

if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status()
    playlist_file = os.path.join(PLAYLISTS_PATH, playlist_name)
    try:
        playlist = playlists.get(playlist_name, [])
        if not playlist:
            raise ValueError("No valid videos found in playlist!")
        with open(DEFAULT_PLAYLIST_FILE, "w") as f:
            f.write(playlist_name)
        # decrypt all files in the playlist
        for video in playlist:
            try:
                in_path = Path(ENCRYPTED_MEDIA_PATH) / video.name
                out_path = Path(DECRYPTED_MEDIA_PATH) / video.name
                decrypt_file(in_path, out_path)
                print(f"Decrypted {in_path} to {out_path}")
            except FileNotFoundError:
                print(f"Error: File not found for decryption: {video}")
    except Exception as e:
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
