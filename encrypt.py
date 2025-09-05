from cryptography.fernet import Fernet
from pathlib import Path
import os
import pickle
from url_to_filename import filename_to_url, url_to_filename
from generate_sortings import instr_struct
from config import KEY_PATH, PLAYLISTS_PATH, DEFAULT_PLAYLIST_FILE, ENCRYPTED_MEDIA_PATH, DECRYPTED_MEDIA_PATH, DEFAULT_MUSICS_FILE

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

def check_media(media_list, playlist, suffix, medium_type, only_numbers):
    for i, url in enumerate(media_list):
        if not url:
            continue
        path = url_to_filename(url)
        files_found = files_with_same_stem(Path(DECRYPTED_MEDIA_PATH) / path)
        if files_found:
            playlist["available"][medium_type].append(files_found[0])
        else:
            files_found = files_with_same_stem(Path(ENCRYPTED_MEDIA_PATH) / path)
            if files_found:
                playlist["not_decrypted"][medium_type].append(files_found[0])
                if not only_numbers:
                    suffix += f"\n\t({medium_type} {i + 1} not decrypted: {url})"
            else:
                playlist["missing"][medium_type].append(url)
                if not only_numbers:
                    suffix += f"\n\t({medium_type} {i + 1} missing: {url})"
    return suffix

def get_playlist_status(given_default = None, show_missings = True, get_all_videos = False, only_numbers = False):
    # --- Ask user for playlist name via CMD ---
    while True:
        os.makedirs(PLAYLISTS_PATH, exist_ok=True)
        os.makedirs(DECRYPTED_MEDIA_PATH, exist_ok=True)
        os.makedirs(ENCRYPTED_MEDIA_PATH, exist_ok=True)
        all_playlists = [os.path.basename(f) for f in os.listdir(PLAYLISTS_PATH) if f.endswith('.pkl')]
        if not all_playlists:
            print("No playlists found.")
            return {}, ""
        if not os.path.exists(DEFAULT_PLAYLIST_FILE):
            with open(DEFAULT_PLAYLIST_FILE, "wb") as f:
                pickle.dump(None, f)
        with open(DEFAULT_PLAYLIST_FILE, "rb") as f:
            default_playlist = pickle.load(f)
        
        playlists = {}
        playlist_infos = []
        for file in all_playlists:
            suffix = ''
            if_default = default_playlist is not None and file == default_playlist
            if if_default:
                suffix = " (default)"
            playlists[file] = {"available": {"media": [], "musics": []}, "not_decrypted": {"media": [], "musics": []}, "missing": {"media": [], "musics": []}}
            playlist = playlists[file]
            file_path = os.path.join(PLAYLISTS_PATH, file)
            with open(file_path, 'rb') as f:
                saved = pickle.load(f)
                playlist.update(saved)
            if show_missings:
                urls = playlist["data"]["urls"]
                if "output" in playlist:
                    urls = playlist["output"]["urls"]
                suffix = check_media(urls, playlist, suffix, "media", only_numbers)
                musics = playlist["musics"]
                if not musics:
                    with open(DEFAULT_MUSICS_FILE, "r") as f:
                        musics = [line.strip() for line in f if line.strip()]
                suffix = check_media(musics, playlist, suffix, "musics", only_numbers)
            if only_numbers:
                for media_type in ["media", "musics"]:
                    suffix += f"\t{media_type} missing: {len(playlist['missing'][media_type])}, not decrypted: {len(playlist['not_decrypted'][media_type])}"
            playlist_infos.append((file, suffix, if_default))

        if get_all_videos:
            return playlists, default_playlist

        ask_input = not given_default or given_default not in all_playlists
        if ask_input:
            print("available playlists:")
            for idx, (file, suffix, if_default) in enumerate(playlist_infos, 1):
                print(f"{idx}. {file}{suffix}")

            playlist_name = ""
            selection = input("Select playlist by number or enter name (type s to switch display option): ").strip()
            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(playlist_infos):
                    playlist_name = playlist_infos[idx][0]
            elif not selection:
                if default_playlist is not None and default_playlist in all_playlists:
                    playlist_name = default_playlist
                else:
                    playlist_name = all_playlists[0]
            elif selection.lower() == 's':
                only_numbers = not only_numbers
                continue
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
        else:
            playlist_name = given_default
        with open(DEFAULT_PLAYLIST_FILE, "wb") as f:
            pickle.dump(playlist_name, f)
        break
    return playlists, playlist_name

if __name__ == "__main__":
    playlists, playlist_name = get_playlist_status()
    playlist_file = os.path.join(PLAYLISTS_PATH, playlist_name)
    try:
        playlist = playlists.get(playlist_name, [])
        if not playlist:
            raise ValueError("No valid media found in playlist!")
        # decrypt all files in the playlist
        for d in playlist["not_decrypted"].values():
            for medium in d:
                try:
                    in_path = Path(ENCRYPTED_MEDIA_PATH) / medium.name
                    out_path = Path(DECRYPTED_MEDIA_PATH) / medium.name
                    decrypt_file(in_path, out_path)
                    print(f"Decrypted {filename_to_url(os.path.splitext(medium.name)[0])}")
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
