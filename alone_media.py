import os
from encrypt import get_playlist_status, filename_to_url
from config import ENCRYPTED_MEDIA_PATH, DECRYPTED_MEDIA_PATH

if __name__ == "__main__":
    playlists, _ = get_playlist_status(get_all_videos=True)
    playlists_videos = set()
    for playlist in playlists.values():
        for media_list in playlist.values():
            playlists_videos.update(media_list)
    stored_videos = {ENCRYPTED_MEDIA_PATH: os.listdir(ENCRYPTED_MEDIA_PATH), DECRYPTED_MEDIA_PATH: os.listdir(DECRYPTED_MEDIA_PATH)}
    for folder, videos in stored_videos.items():
        print(f"Folder: {os.path.basename(folder)}\n")
        for video in videos:
            if video not in playlists_videos:
                print(filename_to_url(os.path.splitext(video)[0]))
        print("\n\n")
    input("Press Enter to continue...")