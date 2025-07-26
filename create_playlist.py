import os

def create_playlist(playlist_name, video_paths):
    playlist_dir = "playlists"
    os.makedirs(playlist_dir, exist_ok=True)
    playlist_path = os.path.join(playlist_dir, f"{playlist_name}.txt")

    with open(playlist_path, 'w', encoding='utf-8') as f:
        for path in video_paths:
            f.write(path.strip() + '\n')
    print(f"Playlist '{playlist_name}' saved to {playlist_path}")

# Example usage:
if __name__ == "__main__":
    playlist_name = input("Enter playlist name: ")
    print("Enter full paths to video files (blank line to finish):")
    paths = []
    while True:
        path = input("Video path: ").strip()
        if not path:
            break
        if os.path.isfile(path):
            paths.append(path)
        else:
            print("File not found. Try again.")
    create_playlist(playlist_name, paths)
